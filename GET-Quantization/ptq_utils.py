import os
import sys
import io
import time
import copy
import yaml
import pickle
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