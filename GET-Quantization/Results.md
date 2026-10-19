# GET PTQ Results (BUSI)

**Date:** 2026-10-18

Post-training INT8 quantization of GET (VAE + latent mapper) on a Colab T4, batch size 1.

**Setup**
- Dataset: BUSI (stand-in for BUS). Calibration: 64 random train images. Evaluation: first 50 test images (all benign).
- Stages: A = weight-only fake-quant, B = simulated W8A8 (weights + activations), C = real INT8 with NVIDIA ModelOpt + TensorRT.
- Latency: ms per image, 3 warmup images. Only Stage C measures real INT8 speed; fake-quant rows (A, B) add rounding ops, so they are not faster than FP32.

## Summary

- INT8 post-training quantization with TensorRT works on GET with no retraining. DSC drops by 0.0034 (0.8402 to 0.8368).
- TensorRT INT8 runs at 22.7 ms/img versus 284.7 ms for FP32 PyTorch (about 12.5x). Most of that gain comes from TensorRT itself: TensorRT FP16 already reaches 28.0 ms. INT8 is a further 1.23x over FP16.
- Model size falls from 339 MB (FP32) to 89.5 MB (INT8 engines), about 3.8x.
- Weight-only INT8 is essentially lossless (within 0.0005 DSC of FP32), per-tensor or per-channel.
- Activation range estimation matters: percentile clipping loses 0.024 DSC (HD95 17.1), while min/max and MSE lose about 0.001.
- The layer-group sensitivity sweep found no group worth keeping in FP32 (worst group: 0.0011 DSC).

## Full results

| Config | Device | DSC | dDSC vs FP32 | IoU | HD95 | ms/img | MB |
|---|---|---|---|---|---|---|---|
| FP32 baseline | cuda | 0.8402 | 0.0000 | 0.7748 | 12.2933 | 284.665 | 339.05 |
| A per-tensor: mapper only | cuda | 0.8402 | -0.0000 | 0.7748 | 12.3378 | 295.622 | |
| A per-tensor: VAE only | cuda | 0.8407 | +0.0004 | 0.7754 | 12.3304 | 293.817 | |
| A per-tensor: VAE + mapper | cuda | 0.8404 | +0.0002 | 0.7751 | 12.3338 | 285.464 | 85.34 |
| A per-channel: mapper only | cuda | 0.8401 | -0.0001 | 0.7748 | 12.2917 | 283.719 | |
| A per-channel: VAE only | cuda | 0.8402 | -0.0000 | 0.7749 | 12.3177 | 285.852 | |
| A per-channel: VAE + mapper | cuda | 0.8403 | +0.0000 | 0.7750 | 12.2957 | 288.627 | 85.34 |
| B W8A8 minmax | cuda | 0.8390 | -0.0012 | 0.7732 | 12.3129 | 325.464 | 85.34 |
| B W8A8 percentile | cuda | 0.8161 | -0.0242 | 0.7517 | 17.0846 | 318.210 | 85.34 |
| B W8A8 mse | cuda | 0.8387 | -0.0016 | 0.7736 | 12.3933 | 320.405 | 85.34 |
| B W8A8 minmax + sweep-pruned skip | cuda | 0.8390 | -0.0012 | 0.7732 | 12.3129 | 319.394 | |
| C TensorRT fp16 | cuda | 0.8402 | -0.0000 | 0.7748 | 12.3074 | 27.954 | 174.26 |
| **C TensorRT int8** | cuda | 0.8368 | -0.0034 | 0.7723 | 11.9187 | 22.694 | 89.45 |

MB for A/B rows is the estimated INT8 weight footprint; the Stage C rows are the real engine sizes on disk.

## Sensitivity sweep (W8A8 minmax, one group quantized at a time)

| Layer group | DSC drop |
|---|---|
| mapper_mbconv_pw | +0.0011 |
| mapper_fem_br_pw | +0.0003 |
| vae_up_down | +0.0003 |
| vae_enc_resnet | +0.0003 |
| mapper_msfa | +0.0002 |
| vae_dec_resnet | +0.0001 |
| mapper_fem_fuse | +0.0000 |
| mapper_ssa | +0.0000 |
| vae_attn_1x1 | -0.0000 |
| mapper_skip_s | -0.0001 |

All groups are under the 0.01 tolerance, so none stay FP32 beyond the default boundary-layer skip list (`SKIP` in `ptq_utils.py`).

## Interpretation and limits

- Real INT8 (Stage C, -0.0034 DSC) loses more than the simulated W8A8 (-0.0012). TensorRT's kernels, layer fusion and Q/DQ placement differ from the fake-quant model, so the simulation approximates but does not predict the engine exactly.
- HD95 improving from 12.29 to 11.92 is within noise, not a real gain.
- Each difference of a few thousandths of DSC is small relative to the sampling noise of a 50-image evaluation set.

## Scope of the conclusions

The VAE + latent-mapper segmentation model converts to INT8 TensorRT engines with post-training quantization only, keeping DSC within about 0.4% of FP32 while reducing size about 3.8x and latency below TensorRT FP16.

Not covered by these results:
- **One dataset, and not BUS.** Results are on BUSI with BUSI-trained weights. Other modalities (GlaS, HAM10000, Kvasir) are untested.
- **Small, unbalanced test subset.** The first 50 test images are all benign (the full test split is 88 benign, 42 malignant). Malignant cases are typically harder, so this may understate the degradation.
- **No statistics.** Single run, no confidence intervals or per-image paired comparison.
- **One GPU, batch size 1.** Speedups are specific to the T4 and this setup; engines are not portable across GPUs.
- **Latency baseline.** The large FP32 speedup is mostly TensorRT versus eager PyTorch; the INT8-specific gain is the 1.23x over TensorRT FP16.

