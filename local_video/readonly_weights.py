"""Read-only, single-mapping safetensors loader.

`safetensors.torch.load_file` maps a file and Torch maps it again privately; on Windows each
copy-on-write view is charged to commit at full file size, so a 3.1 GB checkpoint can reserve
6.3 GB of commit (measured on this PC: commit stayed about 2x the private working set). Here the
file is mapped once with ACCESS_READ and every tensor is a view into it: Windows charges no commit
for those pages and can drop them under memory pressure, re-reading from disk.

The tensors are read-only. Writing to them crashes the process (access violation), so they are only
used for inference weights that are read, widened or converted into new tensors. Set
CVL_READONLY_WEIGHTS=0 to fall back to the library loader.
"""
from __future__ import annotations
import json
import math
import mmap
import os
import struct
import sys
import warnings

_MAPPINGS = []
_ORIGINAL = {}


def enabled():
    return os.environ.get("CVL_READONLY_WEIGHTS", "1") != "0"


def _dtypes(torch):
    return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool}


def load_readonly(filename, device="cpu"):
    """Same contract as safetensors.torch.load_file for CPU loads, without copy-on-write commit."""
    import torch
    if str(device) != "cpu" or not enabled():
        return _ORIGINAL.get("load_file", _library_load_file())(filename, device=device)
    with open(filename, "rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    size = len(mapping)
    if size < 8:
        raise ValueError(f"{filename}: not a safetensors file")
    header_length = struct.unpack("<Q", mapping[:8])[0]
    if 8 + header_length > size:
        raise ValueError(f"{filename}: header exceeds file size")
    header = json.loads(bytes(mapping[8:8 + header_length]))
    header.pop("__metadata__", None)
    base, view, dtypes = 8 + header_length, memoryview(mapping), _dtypes(torch)
    result, spans = {}, []
    for name, entry in header.items():
        dtype = dtypes.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{filename}: unsupported dtype {entry['dtype']} for {name}")
        start, end = entry["data_offsets"]
        count = math.prod(entry["shape"])
        if not 0 <= start <= end <= size - base or end - start != count * torch.empty((), dtype=dtype).element_size():
            raise ValueError(f"{filename}: invalid offsets for {name}")
        spans.append((start, end, name))
        if count == 0:
            result[name] = torch.empty(entry["shape"], dtype=dtype)
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not writable.*")
            result[name] = torch.frombuffer(view, dtype=dtype, count=count, offset=base + start).reshape(entry["shape"])
    spans.sort()
    for (_, previous_end, previous), (start, _, name) in zip(spans, spans[1:]):
        if start < previous_end:
            raise ValueError(f"{filename}: tensors {previous} and {name} overlap")
    _MAPPINGS.append(mapping)
    return result


def _library_load_file():
    from safetensors.torch import load_file
    return load_file


def install(transformers=False):
    """Route the libraries' safetensors loads (Diffusers, direct calls and optionally Transformers) through
    the read-only loader. Transformers binds the loader at import time, so it is imported only when asked."""
    if not enabled():
        return False
    import safetensors.torch as library
    _ORIGINAL.setdefault("load_file", library.load_file)
    library.load_file = load_readonly
    if transformers:
        import transformers.modeling_utils  # noqa: F401
    module = sys.modules.get("transformers.modeling_utils")
    if module is not None and hasattr(module, "safe_load_file"):
        module.safe_load_file = load_readonly
    return True
