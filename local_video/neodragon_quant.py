"""CPU-only INT8 adapters. Imported by the isolated inference runtime only.

The vendor source and original checkpoints stay unchanged. Mapping the source
checkpoint avoids a simultaneous source copy and converted model in RAM.
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn.functional as functional
from accelerate import init_empty_weights
from safetensors.torch import load_file


class Int8Linear(torch.nn.Module):
    def __init__(self, original, packed_placeholder=False):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        if packed_placeholder:
            with torch.device("cpu"):
                self.operation = torch.ao.nn.quantized.dynamic.Linear(
                    original.in_features, original.out_features, bias_=original.bias is not None)
        else:
            original.float()
            original.qconfig = torch.ao.quantization.per_channel_dynamic_qconfig
            self.operation = torch.ao.nn.quantized.dynamic.Linear.from_float(original)

    def forward(self, value):
        return self.operation(value.float()).to(value.dtype)


def quantize_linears(module, packed_placeholder=False):
    count = 0
    for key, child in list(module.named_children()):
        if type(child) is torch.nn.Linear:
            setattr(module, key, Int8Linear(child, packed_placeholder))
            count += 1
        else:
            count += quantize_linears(child, packed_placeholder)
    return count


def load_mapped_model(cls, folder: Path, dtype):
    """Load a Diffusers model by assigning mapped tensors into a meta skeleton."""
    from .readonly_weights import load_readonly
    with init_empty_weights():
        model = cls.from_config(cls.load_config(folder, local_files_only=True))
    # One read-only mapping: no copy-on-write commit for the weights (see readonly_weights).
    state = load_readonly(str(folder / "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(state, strict=True, assign=True)
    # Release the dictionary before quantization; otherwise it keeps every
    # original weight alive after its module has been replaced.
    del state
    gc.collect()
    # The selected Neo checkpoints are BF16. Converting the whole mapped model
    # here would defeat mapping, so convert only when the source dtype differs.
    if model.dtype != dtype:
        model.to(dtype=dtype)
    return model.eval()


class MappedLinearCPU(torch.nn.Module):
    """Keep exact BF16 weights mapped and widen only the current multiplication.

    This avoids quantizing activations and avoids a full FP32 model allocation.
    """
    def __init__(self, original):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight = torch.nn.Parameter(original.weight.detach(), requires_grad=False)
        self.bias = torch.nn.Parameter(original.bias.detach(), requires_grad=False) if original.bias is not None else None

    def forward(self, value):
        return functional.linear(value.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)


class MappedLinearGPU(MappedLinearCPU):
    """The original mapped BF16 matrix is multiplied on the OpenCL GPU."""
    def __init__(self, original, engine):
        super().__init__(original)
        self.engine = engine

    def forward(self, value):
        if self.weight.dtype != torch.bfloat16:
            raise ValueError("GPU 스트리밍은 고정된 BF16 원본 행렬만 사용합니다.")
        shape = value.shape[:-1]
        inputs = value.float().contiguous().reshape(-1, self.in_features)
        bits = self.weight.detach().view(torch.uint16).numpy()
        bias = self.bias.detach().float().numpy() if self.bias is not None else None
        result = self.engine.linear(inputs.numpy(), bits, bias)
        return torch.from_numpy(result).reshape(*shape, self.out_features)


class MappedLinearTorchGPU(MappedLinearCPU):
    """Exact BF16 weights, multiplied in FP32 on a PyTorch GPU (CUDA or XPU).

    Streamed mode keeps the weights mapped on the CPU side and uploads one matrix per call.
    Resident mode (discrete GPUs with enough free VRAM) uploads every matrix once.
    """
    stored = None
    stored_bias = None

    @staticmethod
    def bitwise_for(device):
        # The Iris Xe driver's dtype-cast kernels fail while integer copies, shifts, masks and
        # bit-reinterpreting views work, so Intel GPUs widen BF16 with bit operations.
        return device.type == "xpu"

    @classmethod
    def upload(cls, bf16_cpu, device, bitwise=None):
        bitwise = cls.bitwise_for(device) if bitwise is None else bitwise
        data = bf16_cpu.detach().contiguous()
        if bitwise and data.numel() % 2:
            return data.float().to(device)  # odd element count cannot pair into 32-bit words
        return data.reshape(-1).view(torch.int32).to(device) if bitwise else data.to(device)

    @classmethod
    def widen_stored(cls, stored, shape, bitwise):
        """BF16 -> FP32 on the GPU. The bit form is exact: two BF16 values share one 32-bit word and
        each is the top half of an FP32 word."""
        if stored.dtype == torch.float32:
            return stored.reshape(*shape)
        if not bitwise:
            return stored.float()
        even = (stored << 16).view(torch.float32)
        odd = (stored & -65536).view(torch.float32)
        return torch.stack((even, odd), dim=-1).reshape(*shape)

    @classmethod
    def widen(cls, bf16_cpu, device, bitwise=None):
        bitwise = cls.bitwise_for(device) if bitwise is None else bitwise
        return cls.widen_stored(cls.upload(bf16_cpu, device, bitwise), bf16_cpu.shape, bitwise)

    def pin(self, device):
        """Resident mode: move this matrix to the GPU once."""
        self.stored = self.upload(self.weight, device)
        self.stored_bias = self.bias.detach().float().to(device) if self.bias is not None else None

    def forward(self, value):
        device = value.device
        bitwise = self.bitwise_for(device)
        if self.stored is not None:
            weight, bias = self.widen_stored(self.stored, self.weight.shape, bitwise), self.stored_bias
        else:
            weight = self.widen(self.weight, device, bitwise)
            bias = self.bias.detach().float().to(device) if self.bias is not None else None
        return functional.linear(value.float(), weight, bias)


MappedLinearXPU = MappedLinearTorchGPU


class MappedConvTorchGPU(torch.nn.Module):
    """A Conv2d/Conv3d whose BF16 weight stays on the CPU side and is widened to FP32 on the GPU per call.

    Large video VAEs (LTX: 2.3 GB BF16, 4.6 GB in FP32) do not fit next to the transformer in an
    integrated GPU's shared memory when converted whole.
    """
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.weight_cpu = original.weight.detach()
        self.bias_cpu = original.bias.detach().float() if original.bias is not None else None
        del original.weight
        if original.bias is not None:
            del original.bias
        original.register_parameter("weight", None)
        original.register_parameter("bias", None)

    def forward(self, value):
        device = value.device
        weight = MappedLinearTorchGPU.widen(self.weight_cpu, device)
        bias = self.bias_cpu.to(device) if self.bias_cpu is not None else None
        return self.original._conv_forward(value.float(), weight, bias)


def stream_convs(module):
    count = 0
    for key, child in list(module.named_children()):
        if type(child) in (torch.nn.Conv2d, torch.nn.Conv3d):
            setattr(module, key, MappedConvTorchGPU(child))
            count += 1
        else:
            count += stream_convs(child)
    return count


def linear_weight_bytes(module):
    return sum(child.weight.numel() * child.weight.element_size() for child in module.modules() if type(child) is torch.nn.Linear)


def stream_linears(module, engine=None, device=None, resident=False):
    count = 0
    for key, child in list(module.named_children()):
        if type(child) is torch.nn.Linear:
            if device is not None and device.type != "cpu":
                wrapped = MappedLinearTorchGPU(child)
                if resident:
                    wrapped.pin(device)
                setattr(module, key, wrapped)
            else:
                setattr(module, key, MappedLinearCPU(child) if engine is None else MappedLinearGPU(child, engine))
            count += 1
        else:
            count += stream_linears(child, engine, device, resident)
    return count


def float_non_linear_parameters(module, device=None):
    for child in module.modules():
        if isinstance(child, (MappedLinearCPU, MappedConvTorchGPU)):
            continue
        for key, value in list(child.named_parameters(recurse=False)):
            if value.is_floating_point():
                value = value.float() if device is None else value.float().to(device)
                setattr(child, key, torch.nn.Parameter(value, requires_grad=False))
        for key, value in list(child.named_buffers(recurse=False)):
            if value.is_floating_point():
                setattr(child, key, value.float() if device is None else value.float().to(device))
