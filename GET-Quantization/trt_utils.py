"""
Stage C helpers: real INT8 on an NVIDIA GPU (Colab T4) through TensorRT.

Pipeline (each GET sub-network is handled separately so ONNX export stays simple):

    FP32 module --(ModelOpt PTQ)--> module with Q/DQ nodes --(torch.onnx)--> .onnx
                --(TensorRT builder)--> .plan engine --(TRTEngine)--> callable on CUDA tensors

Three sub-networks, matching GET/valid.py's inference path:
    EncoderWrap : image [1,3,224,224]  -> latent mean mu [1,4,28,28]   (VAE encoder + quant_conv)
    MapperWrap  : scaled mu            -> predicted latent [1,4,28,28]  (MainPipeline['out'])
    DecoderWrap : latent z             -> reconstructed image [1,3,224,224] (post_quant_conv + VAE decoder)

Only the ModelOpt / ONNX / TensorRT parts need the GPU stack; everything is imported lazily so
this file can be imported (and the ONNX export tested) on a CPU-only machine.
"""
import copy
import torch
import torch.nn as nn

from ptq_utils import SKIP, is_target


# ------------------------------------------------------------------------------
# 1. Sub-network wrappers (tensor in -> tensor out, which is what ONNX/TensorRT need)
# ------------------------------------------------------------------------------

class EncoderWrap(nn.Module):
    """image -> posterior mean. Attribute names (encoder, quant_conv) match the VAE, so SKIP names still apply."""
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        return torch.chunk(moments, 2, dim=1)[0]   # mean half (same as DiagonalGaussianDistribution.mean)


class MapperWrap(nn.Module):
    """MainPipeline returns a dict of 4 outputs; the pipeline only uses ['out']."""
    PREFIX = "mapper."   # names inside are "mapper.stem", ... -> strip this to compare against SKIP

    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper

    def forward(self, x):
        return self.mapper(x)["out"]


class DecoderWrap(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


def make_wrappers(vae, mapper):
    """Returns {'enc': ..., 'map': ..., 'dec': ...}. They share weights with vae/mapper (deepcopy before editing)."""
    return {"enc": EncoderWrap(vae), "map": MapperWrap(mapper), "dec": DecoderWrap(vae)}


# ------------------------------------------------------------------------------
# 2. Calibration data for each sub-network
# ------------------------------------------------------------------------------

@torch.no_grad()
def collect_stage_inputs(vae, mapper, loader, scale, device):
    """
    Runs the FP32 pipeline on the calibration images and records what each sub-network receives:
      enc <- preprocessed image, map <- scale*mu, dec <- clamp(mapper_out/scale).
    ModelOpt then calibrates every sub-network on realistic inputs.
    """
    out = {"enc": [], "map": [], "dec": []}
    for batch in loader:
        img = (2 * batch["img"] - 1).to(device)
        mu = scale * vae.encode(img).mean
        z = (mapper(mu)["out"] / scale).clamp(-10, 10)
        out["enc"].append(img.cpu())
        out["map"].append(mu.cpu())
        out["dec"].append(z.cpu())
    return out


# ------------------------------------------------------------------------------
# 3. ModelOpt: insert INT8 Q/DQ nodes (post-training, "max" calibration)
# ------------------------------------------------------------------------------

def _set_quantizer(cfg, pattern, enable, params=None):
    """
    Add one rule to a ModelOpt quant config. Later rules override earlier ones.
    Newer ModelOpt (>=0.4x) stores quant_cfg as a LIST of {"quantizer_name", "enable"/"cfg"} rules,
    older versions as a DICT {pattern: params}; support both.
    """
    qc = cfg["quant_cfg"]
    if isinstance(qc, list):
        rule = {"quantizer_name": pattern, "enable": enable}
        if params:
            rule["cfg"] = params
        qc.append(rule)
    else:
        qc[pattern] = {**(params or {}), "enable": enable}


def quantize_int8(wrapper, calib_inputs, device, skip_set=SKIP, prefix=""):
    """
    In-place PTQ of one wrapper with NVIDIA ModelOpt. Pass a deepcopy if you still need the FP32 one.
      - Target convs (is_target): INT8 weight (per-channel) + INT8 input (per-tensor) quantizers.
      - Every other conv (depthwise, boundary layers, anything in skip_set) is left in float.
      - GroupNorm / SiLU / softmax / matmul are never quantized by ModelOpt's default config.
    Rules are matched against exact quantizer names like "encoder.conv_in.input_quantizer", and are
    appended AFTER the defaults (the default config also disables some names such as "*proj_out.*",
    which is the VAE attention output conv, so we explicitly re-enable our targets).
    """
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for name, m in wrapper.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        rel = name[len(prefix):] if prefix and name.startswith(prefix) else name   # name as used in SKIP
        if is_target(rel, m, skip_set):
            _set_quantizer(cfg, f"{name}.weight_quantizer", True, {"num_bits": 8, "axis": 0})
            _set_quantizer(cfg, f"{name}.input_quantizer", True, {"num_bits": 8, "axis": None})
        else:
            _set_quantizer(cfg, f"{name}.weight_quantizer", False)
            _set_quantizer(cfg, f"{name}.input_quantizer", False)

    def forward_loop(model):
        with torch.no_grad():
            for x in calib_inputs:
                model(x.to(device))

    wrapper.to(device).eval()
    return mtq.quantize(wrapper, cfg, forward_loop)


# ------------------------------------------------------------------------------
# 4. ONNX export + TensorRT engine build
# ------------------------------------------------------------------------------

def export_onnx(wrapper, example, path, opset=17):
    """Static-shape ONNX export (batch 1). dynamo=False keeps the classic exporter, which handles Q/DQ ops."""
    wrapper.eval()
    kw = dict(input_names=["input"], output_names=["output"], opset_version=opset, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, example, path, dynamo=False, **kw)
        except TypeError:           # older torch has no `dynamo` argument
            torch.onnx.export(wrapper, example, path, **kw)
    return path


def build_engine(onnx_path, plan_path, int8=False, fp16=True, workspace_gb=2):
    """
    Build a serialized TensorRT engine.
      int8=True : use with the Q/DQ ONNX (explicit quantization, no calibrator needed)
      fp16=True : lets TensorRT run the non-quantized layers in half precision
    The FP16 baseline engine is built from the *unquantized* ONNX with int8=False.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()                      # TRT >= 10: explicit batch is the default
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed (see log above)")
    with open(plan_path, "wb") as f:
        f.write(plan)
    return plan_path


# ------------------------------------------------------------------------------
# 5. Run engines from PyTorch (CUDA tensors in, CUDA tensors out)
# ------------------------------------------------------------------------------

class TRTEngine:
    """Minimal single-input / single-output TensorRT runner (TensorRT >= 10 tensor-name API)."""
    def __init__(self, plan_path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if n != self.in_name)

    def __call__(self, x):
        x = x.contiguous().float()
        out = torch.empty(tuple(self.ctx.get_tensor_shape(self.out_name)), dtype=torch.float32, device=x.device)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)   # same stream as torch -> ordered
        return out


def make_trt_forward(enc, mapper, dec, scale):
    """
    Same math as ptq_utils.forward_pipeline, but each stage is a TensorRT engine.
    Returns fn(img) -> probability map, usable as evaluate(..., forward_fn=fn).
    """
    @torch.no_grad()
    def fn(img):
        mu = scale * enc(img)
        z = (mapper(mu) / scale).clamp(-10, 10)
        seg = dec(z).mean(1, True).clamp(-1, 1)
        seg = torch.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
        return ((seg + 1) * 0.5).clamp(0, 1)
    return fn
