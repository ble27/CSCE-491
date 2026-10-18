# GET Quantization Methodology & Parameter Analysis

## 1. Three-Question Framework for Selecting INT8 Quantization Targets

To decide which modules in the GET pipeline (`AutoencoderKL` + `MainPipeline`) should be quantized to INT8 vs. kept in FP16/FP32, each layer is evaluated against three criteria:

1. **Is the layer compute-bound (High MACs/FLOPs)?**
   - Layers operating at large spatial resolutions ($224 \times 224$, $112 \times 112$, $56 \times 56$) or large channel widths ($128 \to 512$) dominate execution time and benefit most from INT8 Tensor Core / VNNI acceleration.
2. **Does the layer contain a large share of weights (High Memory/VRAM)?**
   - Layers with large $C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w$ weight tensors dominate model size and memory bandwidth. Converting them from FP32 (4 bytes/param) or FP16 (2 bytes/param) to INT8 (1 byte/param) yields the largest memory footprint reduction.
3. **How sensitive is the layer to quantization noise (Accuracy Risk)?**
   - Layers with high redundancy (dense $3 \times 3$ and $1 \times 1$ convolutions with residual connections) tolerate INT8 rounding well.
   - Layers with low parameter redundancy (depthwise convolutions), non-linear dynamic ranges (attention $\text{MatMul}$ and $\text{Softmax}$, `GroupNorm`, `SiLU`), or boundary interfaces (first/last layers mapping directly to pixels or continuous latents) are sensitive and should remain in FP16/FP32.

---

## 2. Mathematical Formulas for Parameter & Memory Estimation

### A. 2D Convolution Parameter Formula
For any `nn.Conv2d(in_channels, out_channels, kernel_size, groups)` layer, the exact number of weight parameters is:

$$\text{Params}_{\text{Conv2d}} = \frac{C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w}{\text{groups}} + B$$

where $B = C_{\text{out}}$ if `bias=True`, else $0$.

* **Dense Convolution ($\text{groups} = 1$):** Scales **quadratically** with channel width ($C_{\text{in}} \times C_{\text{out}} \times K^2$).
* **Pointwise Convolution ($K_h = K_w = 1, \text{groups} = 1$):** Scales as $C_{\text{in}} \times C_{\text{out}}$.
* **Depthwise Convolution ($\text{groups} = C_{\text{in}} = C_{\text{out}}$):** Scales **linearly** with channel width ($C_{\text{in}} \times K^2$), resulting in very few parameters.

### B. Compute (MACs) Formula for 2D Convolution
For an output feature map of spatial size $H_{\text{out}} \times W_{\text{out}}$:

$$\text{MACs}_{\text{Conv2d}} = H_{\text{out}} \times W_{\text{out}} \times \frac{C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w}{\text{groups}}$$

### C. Checkpoint File Size to Total Parameters Formula
Because each FP32 parameter requires $4\text{ bytes}$ ($32\text{ bits}$) of storage, total parameter count and uncompressed checkpoint size are related by:

$$\text{Total Params} \approx \frac{\text{Checkpoint Size (Bytes)}}{4\text{ Bytes/Param}}$$

This is an upper bound: checkpoints also store pickle metadata, buffers/extra keys (e.g. EMA copies, `state_dict` wrappers), so the estimate slightly overshoots the true count (e.g. mapper: 4.34 MB $\to$ 1.085M estimated vs. 1.071M counted).

$$\text{Memory Footprint (MB)} = \frac{\text{Total Params} \times \text{Bytes per Element}}{10^6} \qquad \text{Memory Footprint (MiB)} = \frac{\text{Total Params} \times \text{Bytes per Element}}{1024^2}$$

*(where $\text{Bytes per Element} = 4$ for FP32, $2$ for FP16/BF16, and $1$ for INT8).* File sizes reported by `ls -l` / Finder are in decimal MB ($10^6$); use the MiB form only when comparing against binary units. Example: the VAE's 83.65M params are $334.6\text{ MB} = 319.1\text{ MiB}$ in FP32, $83.7\text{ MB}$ in INT8.

**Parameter count of a `ResnetBlock`** (`GroupNorm` $\to$ `conv1` $\to$ `GroupNorm` $\to$ `conv2`, plus a $1 \times 1$ `nin_shortcut` only when $C_{\text{in}} \neq C_{\text{out}}$):

$$\text{Params}_{\text{ResnetBlock}} = 9\,C_{\text{in}} C_{\text{out}} + 9\,C_{\text{out}}^2 + [C_{\text{in}} \neq C_{\text{out}}]\, C_{\text{in}} C_{\text{out}} + \text{biases} + \text{GroupNorm affine } (2 C_{\text{in}} + 2 C_{\text{out}})$$

---

## 3. Determining Parameter Distribution: `AutoencoderKL` vs. `MainPipeline`

### A. `AutoencoderKL` (Stable Diffusion First-Stage VAE)
From `GET/networks/vae_setup/diffusion_model.py`, both `Encoder` and `Decoder` use:
* Base channel: $ch = 128$
* Channel multipliers: $ch\_mult = (1, 2, 4, 4) \implies [128, 256, 512, 512]$
* Spatial resolutions: $224 \times 224 \to 112 \times 112 \to 56 \times 56 \to 28 \times 28$

Each `ResnetBlock` contains two dense $3 \times 3$ convolutions (`conv1`, `conv2`). A $512 \to 512$ block has:

$$\text{Params}_{\text{ResnetBlock}(512)} = 2 \times (512 \times 512 \times 3 \times 3) = 4,718,592 \approx 4.72\text{M params}$$

The `Encoder` has 10 `ResnetBlock`s (2 per level $\times$ 4 levels + 2 in `mid`); 6 of them output 512 channels, but the first block of level 2 is $256 \to 512$ (smaller, and has an `nin_shortcut`), so only 5 are full $512 \to 512$ blocks. The `Decoder` has 14 `ResnetBlock`s (3 per level $\times$ 4 levels + 2 in `mid`), 8 of which are $512 \to 512$. Adding the `AttnBlock`s (one in `mid` of each of Encoder/Decoder) and the `Downsample`/`Upsample` convs:

$$\text{Params}_{\text{VAE}} = \text{Params}_{\text{Encoder}} (34{,}163{,}592 \approx 34.2\text{M}) + \text{Params}_{\text{Decoder}} (49{,}490{,}179 \approx 49.5\text{M}) + \text{Params}_{\text{quant\_conv, post\_quant\_conv}} (92) = \mathbf{83{,}653{,}863 \approx 83.65\text{M params}}$$

Breakdown of the VAE by layer type (parameter counts derived from the layer definitions):

| Group | Params | Share of VAE |
| :--- | :---: | :---: |
| `ResnetBlock` `conv1`/`conv2`/`nin_shortcut` | 73.04M | 87.3% |
| `Downsample` (3 in Encoder, 3.10M) + `Upsample` (3 in Decoder, 5.31M) $3 \times 3$ convs | 8.41M | 10.0% |
| **Subtotal (quantization targets)** | **~81.4M** | **~97.4%** |
| `AttnBlock` $1 \times 1$ (`q,k,v,proj_out`; 2 blocks $\times$ $4 \times 512^2$) | 2.10M | 2.5% |
| Boundary convs (`conv_in`, `conv_out`, `quant_conv`, `post_quant_conv`) | ~0.06M | ~0.07% |
| `GroupNorm` affine + remaining biases | small | < 0.1% |

Verified via checkpoint size (`768-v-ema-first-stage-VAE.ckpt` $\approx 335\text{ MB}$ decimal $= 319\text{ MiB}$; the file is not in the repo, only its README, so this was computed from the code rather than from the file):

$$\frac{335\text{ MB}}{4\text{ Bytes/Param}} \approx 83.6\text{M params} \quad (\mathbf{\approx 98.7\%\text{ of total pipeline weights}})$$

### B. `MainPipeline` (GET Latent Mapper)
From `GET/configs/bus_valid.yaml` and `GET/networks/embedding_translation.py`:
* Base channel: $ch = 32$
* Channel multipliers: $ch\_mult = [1, 2, 3, 5] \implies [32, 64, 96, 160]$
* Spatial resolutions: $28 \times 28 \to 14 \times 14 \to 7 \times 7 \to 3 \times 3$

Instead of 512-channel dense $3 \times 3$ convolutions, `MainPipeline` uses narrow channels ($\le 160$) with `MBConv` ($1 \times 1$ pointwise + $3 \times 3$ depthwise), `SSA` attention at the bottleneck, and `FEM` blocks.

Parameter count derived from the layer definitions: **1,071,104 params** ($= 4.28\text{ MB}$ FP32). Checkpoint size (`Results/valid_bus/best_model.pth` $= 4{,}340{,}786\text{ bytes} \approx 4.34\text{ MB} = 4.14\text{ MiB}$; all five datasets use the same config and have near-identical sizes):

$$\frac{4.34\text{ MB}}{4\text{ Bytes/Param}} \approx 1.085\text{M params (upper bound)} \;\approx\; \mathbf{1.07\text{M params}} \quad (\mathbf{\approx 1.3\%\text{ of total pipeline weights}})$$

Breakdown of the mapper by layer type (share of 1,071,104 params):

| Group | Params | Share of Mapper |
| :--- | :---: | :---: |
| `FEM.fuse` dense $3 \times 3$ convs | 552,960 | 51.6% |
| `MBConv` pointwise $1 \times 1$ (expand + project) | 285,696 | 26.7% |
| `SSA` $1 \times 1$ (`q, k, v, proj`) | 102,400 | 9.6% |
| `FEM.br` pointwise $1 \times 1$ | 46,080 | 4.3% |
| `MSFA` (`proj` $1 \times 1$ 3,072 + `ref` $3 \times 3$ 36,864) | 39,936 | 3.7% |
| Depthwise $3 \times 3$ (`MBConv` 12,672 + `FEM.br` 6,048) | 18,720 | 1.7% |
| `GroupNorm` affine | 8,896 | 0.8% |
| Skip projections `s1`–`s3` ($1 \times 1$) | 7,168 | 0.7% |
| Heads `h1`–`h3` ($3 \times 3$) | 6,912 | 0.6% |
| Boundary: `stem` + head `h0` | 2,336 | 0.2% |

---

## 4. Summary of Target Regions for Quantization

| Module / Layer Group | Network | Weight Share | Compute Load | INT8 Risk | Decision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`ResnetBlock` (`conv1`, `conv2`, `nin_shortcut`) & `Upsample`/`Downsample` convs** | `AutoencoderKL` | **~97.4% of VAE (~81.4M)** | **Very High** | Low | **Quantize (INT8)** |
| **`FEM.fuse` (Dense $3\times 3$ convs)** | `MainPipeline` | **~51.6% of Mapper (553K)** | **High** | Low | **Quantize (INT8)** |
| **`MBConv` Pointwise ($1\times 1$ convs)** | `MainPipeline` | **~26.7% of Mapper (286K)** | **High** | Low | **Quantize (INT8)** |
| **`FEM.br` Pointwise ($1\times 1$ convs)** | `MainPipeline` | **~4.3% of Mapper (46K)** | **Moderate** | Low | **Quantize (INT8)** |
| **`MSFA` ($1\times 1$ `proj` & $3\times 3$ `ref`)** | `MainPipeline` | **~3.7% of Mapper (40K)** | **Moderate** | Low | **Quantize (INT8)** |
| **Attention $1\times 1$ Projections (`q, k, v, proj`)** | Both (`SSA` 102K = 9.6% of Mapper; `AttnBlock` 2.10M = 2.5% of VAE) | **~2.2M total** | **Moderate** | Low | **Quantize (INT8)** |
| **Depthwise $3\times 3$ Convs (`groups = c`)** | `MainPipeline` (`MBConv`, `FEM.br`) | **~1.7% of Mapper (~18.7K)** | **Very Low** | **High** | **Keep FP16/FP32** (or per-channel INT8) |
| **Attention `@` / `bmm` & `softmax`, `GroupNorm`, `SiLU`** | Both | **0 conv weights** (`GroupNorm` affine: ~0.8% of Mapper) | **Low** | **High** | **Keep FP16/FP32** |
| **Boundary Layers (`stem`, `h0`, `conv_in`, `conv_out`, `quant_conv`, `post_quant_conv`)** | Both | **~0.08% of pipeline (~65K)** | **Negligible** | **Very High** | **Keep FP16/FP32** |

*Not listed (small, quantize-or-keep at your discretion): skip projections `s1`–`s3` (0.7% of Mapper, $1 \times 1$) and output heads `h1`–`h3` (0.6% of Mapper, $3 \times 3$; they emit the final level predictions, so keeping them FP is safer).*

*Percentages are of the Mapper's 1,071,104 params unless stated; the VAE has 83,653,863 params. All counts were derived from the layer definitions in `embedding_translation.py` and `diffusion_model.py`, not from loaded checkpoints.*
