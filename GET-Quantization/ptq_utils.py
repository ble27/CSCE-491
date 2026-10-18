import os
import sys
import io
import time
import copy
import yaml
import pickle
from GET.networks.embedding_translation import MainPipeline
from GET.valid import hd95
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import QuantStub, DeQuantStub, QConfig, HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from omegaconf import OmegaConf

try: 
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError:
    medpy_hd95 = None


# ------------------------------------------------------------------------------
# Target selection & Layer grouping
# ------------------------------------------------------------------------------

SKIP = {
    # VAE boundary layers
    "encoder.conv_in",
    "encoder.conv_out",
    "decoder.conv_in",
    "decoder.conv_out",
    "quant_conv",
    "post_quant_conv",
    # Mapper boundary & prediction heads
    "stem",
    "h0.2",
    "h1.2",
    "h2.2",
    "h3.2",
}

def is_target(name: str, m: nn.Module, skip_set=SKIP) -> bool:
    """
    Quantize only dense 2D convolutions (groups == 1) not in skip_set.
    Automatically excludes depthwise convolutions (groups == in_channels).
    """
    return isinstance(m, nn.Conv2d) and m.groups == 1 and name not in skip_set

def get_layer_group(model_tag: str, name: str) -> str:
    """
    Categorizes target convolution layers into functional groups for Stage B (W8A8) 
    sensitivity sweep.
    model_tag: 'vae' or 'mapper'
    """
    if model_tag == "vae":
        if "attn" in name or "mid.attn" in name:
            return "vae_attn_1x1"
        if "downsample" in name or "upsample" in name:
            return "vae_up_down"
        if name.startswith("encoder."):
            return "vae_enc_resnet"
        if name.startswith("decoder."):
            return "vae_dec_resnet"
        return "vae_other"
    else:
        if ".fuse." in name:
            return "mapper_fem_fuse"
        if ".br." in name:
            return "mapper_fem_br_pw" # pw = point-wise convolution with kernel_size of 1
        if "bottleneck" in name:
            return "mapper_ssa"
        if "msfa" in name:
            return "mapper_msfa"
        if any(k in name for k in ["enc", "dec"]):
            return "mapper_mbconv_pw"
        if name in ("s1", "s2", "s3"):
            return "mapper_skip_s"
        return "mapper_other"


# ------------------------------------------------------------------------------
# Model loader & Calibration DataLoader
# ------------------------------------------------------------------------------

def load_models(root_dir: str, config_rel_path: str = "GET/configs/bus_valid.yaml", device: str = "cpu"):
    """Loads MainPipeline (mapper), AutoencoderKL (vae), scale_factor, and resolved config."""
    get_dir = os.path.join(root_dir, "GET")
    if get_dir not in sys.path:
        sys.path.inser(0, get_dir)
    
    from ..GET.networks.embedding_translation import MainPipeline 
    from ..GET.networks.vae_setup.autoencoder import AutoencoderKL
    from ..GET.utils.tools import load_checkpoint

    cfg_path = os.path.join(root_dir, config_rel_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {} 
    
    # Resolve paths relative to GET/
    model_weight = os.path.join(get_dir, cfg["model_weight"].lstrip("./"))
    vae_cfg_path = os.path.join(get_dir, cfg["vae_config_path"].lstrip("./"))
    vae_ckpt     = os.path.join(get_dir, cfg["vae_ckpt"].lstrip("./"))
    pkl_path     = os.path.join(get_dir, cfg["pickle_file_path"].lstrip("./"))

    mapper = MainPipeline(
        in_channel=cfg["in_channel"],
        out_channels=cfg["out_channels"],
        ch=cfg["ch"],
        ch_mult=tuple(cfg["ch_mult"]),
    )
    mapper = load_checkpoint(mapper, model_weight)
    mapper.to(device).eval().requires_grad_(False)

    vae_cfg = OmegaConf.load(vae_cfg_path)
    vae = AutoencoderKL(**vae_cfg.first_stage_config.params)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu")["state_dict"], strict=True)
    vae.to(device).eval().requires_grad_(False)

    scale = float(vae_cfg.first_stage_config.scale_factor)
    cfg["resolved_pickle_path"] = pkl_path
    return vae, mapper, scale, cfg


def get_dataloaders(pkl_path: str, n_calib: int = 64, n_eval: int = None, seed: int = 2333):
    """
    Runs a sample set of images to determine the intermediate activation outputs to
    determine the proper range of the outputs, Scale, and Zero

    Returns (calib_reader, test_loader)
    calib_loader uses the 'train' split names but forces deterministic 'test' transforms.
    """
    from ..GET.utils.image_dataset import Image_Dataset

    # Calibration dataset: load 'train' split, override transform with deterministic resize
    calib_ds = Image_Dataset(pkl_path, stage="train")
    calib_ds.stage = "test"
    calib_ds.transform = calib_ds.get_transforms()

    rng = np.random.default_rng(seed)
    calib_indices = rng.choice(len(calib_ds), size=min(n_calib, len(calib_ds)), replace=False)
    calib_subset = Subset(calib_ds, calib_indices)
    calib_loader = DataLoader(calib_subset, batch_size=1, shuffle=False)

    # Evaluation dataset: 'test' split
    test_ds = Image_Dataset(pkl_path, stage="test")
    if n_eval is not None and n_eval < len(test_ds):
        test_ds = Subset(test_ds, list(range(n_eval)))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=1)

    return calib_loader, test_loader


# ------------------------------------------------------------------------------
# Evaluation (DSc, IoU, HD95, Latency, Size)
# ------------------------------------------------------------------------------

# DSc = Measure overlap area relative to the average size of both shapes (1: Perfect overlap, 0: No overlap)
# IoU  (Intersection over Union) = Measure overlap area relative to the total combined area of both shapes (1: Perfect overlap, 0: No overlap)
# HD95 = Measure boundary distance error between the predicted surface and true surface (0: Perfect match, >0: Higher error)

def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    # check if either mask is empty
    if (pred.sum() == 0) ^ (gt.sum() == 0):
        h, w = gt.shape
        return float(np.hypot(h, w))
    return float(medpy_hd95(pred, gt)) if medpy_hd95 is not None else float("nan")


@torch.no_grad()
def forward_pipeline(vae, mapper, img: torch.Tensor, scale: float) -> torch.Tensor:
    """Runs the 3-step GET inference path: encode -> latent mapper -> decode"""
    post = vae.encode(img) # turn image into an embedding
    mu, _ = post.mu_and_sigma()
    mu = scale * mu

    mu_pred = mapper(mu)["out"]

    z = (mu_pred / scale).clamp(-10, 10)
    seg = vae.decode(z).mean(1, True).clamp(-1, 1)
    seg = torch.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
    return ((seg + 1) * 0.5).clamp(0, 1)


@torch.no_grad()
def evaluate(vae, mapper, loader, scale: float, device: str = "cpu", warmup: int = 3) -> dict:
    """Evaluates DSc, IoU, HD95, and average latency (ms/img)"""
    dscs, ious, hd95s, times_ms = [], [], [], []
    is_cuda = torch.device(device).type == "cuda"

    for idx, batch in enumerate(loader):
        img = (2 * batch["img"] - 1 ).to(device)
        seg_raw = (batch["seg"]).permute(0,3,1,2 / 255.0).to(device)
        seg_bin = seg_raw.mean(1, True)

        if idx < warmup:
            _ = forward_pipeline(vae, mapper, img, scale)
        
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter() # start counter
    
        prob = forward_pipeline(vae, mapper, img, scale) # run pipeline

        if is_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000) # end counter

        pred = (prob >= 0.5).cpu().numpy().astype(np.uint8).squeeze() # remove dimension of size 1
        gt = (seg_bin >= 0.5).cpu().numpy().astype(np.uint8).squeeze()

        tn, fp, fn, tp = confusion_matrix(gt.ravel(), pred.ravel(), labels=[0,1]).ravel()
        denom_dsc = 2 * tp + fp + fn
        denom_iou = tp + fp + fn

        dscs.append(2 * tp / denom_dsc if denom_dsc else 0.0)
        ious.append(tp / denom_iou if denom_iou else 0.0)
        hd95s.append(compute_hd95(pred, gt))

        return {
            "DSC": float(np.mean(dscs)),
            "IoU": float(np.mean(ious)),
            "HD95": float(np.nanmean(hd95s)),
            "ms_per_img": float(np.mean(times_ms)),
        }


def get_state_dict_size_mb(model: nn.Module) -> float:
    """Measures serialized state_dict size in decimal MB (works for both FP32 and real INT8)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def estimate_int8_size_mb(model: nn.Module, skip_set=SKIP) -> float:
    """Theoretical weight footprint (decimal MB) when target Conv2d weights are stored as INT8."""
    total_bytes = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if is_target(name, m, skip_set):
                # 1 byte per weight (INT8) + FP32 per-channel scale + optional FP32 bias
                total_bytes += m.weight.numel() * 1 + m.out_channels * 4
            else:
                total_bytes += m.weight.numel() * 4
            if m.bias is not None:
                total_bytes += m.bias.numel() * 4
        elif isinstance(m, (nn.GroupNorm, nn.Linear)):
            for p in m.parameters(recurse=False):
                total_bytes += p.num_el() * 4
    return total_bytes / 1e6