"""Run one inference stage on a PyTorch GPU backend while logging attention calls and slow modules.

Usage: xpu_trace.py <runtime-dir> <stage-dir> <stage-name> [xpu|cuda]
"""
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import accelerator  # noqa: E402

root, output, stage_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
gpu_backend = sys.argv[4] if len(sys.argv) > 4 else "xpu"
accelerator.enable(root, gpu_backend)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
gpu = getattr(torch, gpu_backend)

original = F.scaled_dot_product_attention
calls = [0]


def traced(q, k, v, attn_mask=None, *args, **kwargs):
    started = time.monotonic()
    result = original(q, k, v, attn_mask, *args, **kwargs)
    gpu.synchronize()
    calls[0] += 1
    print(f"sdpa#{calls[0]} q={tuple(q.shape)} k={tuple(k.shape)} mask={None if attn_mask is None else (tuple(attn_mask.shape), str(attn_mask.dtype))} "
          f"{time.monotonic() - started:.2f}s", flush=True)
    return result


state = {"last": time.monotonic(), "count": 0}


def after_module(module, args, output):
    gpu.synchronize()
    now = time.monotonic()
    state["count"] += 1
    if now - state["last"] > 3:
        print(f"slow module #{state['count']} {type(module).__name__} {now - state['last']:.1f}s", flush=True)
    state["last"] = now


torch.nn.modules.module.register_module_forward_hook(after_module)
F.scaled_dot_product_attention = traced
torch.nn.functional.scaled_dot_product_attention = traced
from local_video import neodragon_stage  # noqa: E402

neodragon_stage.stage(root, output, stage_name)
