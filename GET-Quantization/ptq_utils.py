import os
import sys
import io
import time
import copy
import yaml
import pickle
from GET.networks.embedding_translation import MainPipeline
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