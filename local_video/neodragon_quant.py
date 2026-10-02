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
    with init_empty_weights():
        model = cls.from_config(cls.load_config(folder, local_files_only=True))
    state = load_file(str(folder / "diffusion_pytorch_model.safetensors"), device="cpu")
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


def stream_linears(module, engine=None):
    count = 0
    for key, child in list(module.named_children()):
        if type(child) is torch.nn.Linear:
            setattr(module, key, MappedLinearCPU(child) if engine is None else MappedLinearGPU(child, engine))
            count += 1
        else:
            count += stream_linears(child, engine)
    return count


def float_non_linear_parameters(module):
    for child in module.modules():
        if isinstance(child, MappedLinearCPU):
            continue
        for key, value in list(child.named_parameters(recurse=False)):
            if value.is_floating_point():
                setattr(child, key, torch.nn.Parameter(value.float(), requires_grad=False))
        for key, value in list(child.named_buffers(recurse=False)):
            if value.is_floating_point():
                setattr(child, key, value.float())
