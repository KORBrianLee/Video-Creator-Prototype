"""PyTorch GPU backends chosen per computer: CUDA (NVIDIA), XPU (Intel iGPU/Arc), otherwise none.

Each backend lives in its own site folder beside the CPU runtime (runtime/<name>-site), so the
CPU runtime is never touched. A cached probe records whether the installed build really runs
kernels on this computer's GPU. GPUs without a backend here (e.g. AMD) keep the Vulkan and
OpenCL paths; a failing GPU stage is rerun on the CPU path by the caller.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = {
    "cuda": {"site": "cuda-site", "vendors": ("nvidia",)},
    "xpu": {"site": "xpu-site", "vendors": ("intel",)},
}
GPU_STAGES = {"video_infer", "video_decode"}

PROBE_SCRIPT = r"""
import json, sys, os
backend, site = sys.argv[1], sys.argv[2]
sys.path.insert(0, site)
libraries = os.path.join(site, "Library", "bin")
if os.path.isdir(libraries):
    os.environ["PATH"] = libraries + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(libraries)
import torch
module = getattr(torch, backend)
info = {"ok": False, "torch": torch.__version__}
if module.is_available():
    a = torch.randn(256, 256, device=backend); b = torch.randn(256, 256, device=backend)
    c = torch.nn.functional.linear(a, b)
    q = torch.randn(1, 2, 64, 32, device=backend)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    module.synchronize()
    free, total = (module.mem_get_info() if hasattr(module, "mem_get_info") else (0, 0))
    props = module.get_device_properties(0)
    info.update(ok=bool(torch.isfinite(c).all()), device=module.get_device_name(0),
                total_memory_bytes=int(getattr(props, "total_memory", total)), free_memory_bytes=int(free))
print(json.dumps(info))
"""


def backend_for_vendor(vendor):
    return next((name for name, spec in BACKENDS.items() if vendor in spec["vendors"]), None)


def site(root, backend):
    return Path(root) / "runtime" / BACKENDS[backend]["site"]


def python(root):
    return Path(root) / "runtime" / "neodragon-python" / "python.exe"


def enable(root, backend):
    """Make `import torch` resolve to the GPU build. Must run before torch is imported."""
    folder = site(root, backend)
    if not folder.is_dir():
        return False
    libraries = folder / "Library" / "bin"
    sys.path.insert(0, str(folder))
    if libraries.is_dir():
        os.environ["PATH"] = str(libraries) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(libraries))
    return True


def _record_path(root, backend):
    return Path(root) / "cache" / f"torch-probe-{backend}.json"


def _signature(root, backend):
    stamp = site(root, backend) / "torch" / "version.py"
    return stamp.stat().st_mtime if stamp.is_file() else 0


def probe(root, backend, retry_failed_after=3600.0):
    """Cached one-time check that the GPU build runs real kernels on this computer."""
    root = Path(root)
    folder = site(root, backend)
    if os.name != "nt" or not folder.is_dir() or not python(root).is_file():
        return {"ok": False, "reason": "gpu_runtime_not_installed"}
    signature = _signature(root, backend)
    try:
        record = json.loads(_record_path(root, backend).read_text(encoding="utf-8"))
        if record.get("signature") == signature and (record.get("ok") or time.time() - record.get("checked_at", 0) < retry_failed_after):
            return record
    except (OSError, ValueError):
        pass
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        run = subprocess.run([str(python(root)), "-B", "-c", PROBE_SCRIPT, backend, str(folder)], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=300, creationflags=flags)
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        record = json.loads(lines[-1]) if run.returncode == 0 and lines else {"ok": False, "reason": (run.stderr or run.stdout)[-300:]}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        record = {"ok": False, "reason": str(exc)[:300]}
    record.update(signature=signature, checked_at=time.time())
    _write_record(root, backend, record)
    return record


def _write_record(root, backend, record):
    path = _record_path(root, backend)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError:
        pass


def mark_failed(root, backend, reason):
    """A GPU stage failed at run time: stop choosing this backend until the next probe window."""
    _write_record(root, backend, {"ok": False, "reason": str(reason)[:300], "signature": _signature(root, backend),
                                  "checked_at": time.time(), "failed_at_runtime": True})


def choose(root, selected_device, backend_setting="auto"):
    """Pick the torch device for the selected GPU: {'torch_device', 'torch_backend', 'discrete', ...}."""
    cpu = {"torch_device": "cpu", "torch_backend": None, "discrete": False}
    if backend_setting == "cpu" or os.environ.get("CVL_NO_TORCH_GPU") or not selected_device:
        return cpu
    backend = backend_for_vendor(selected_device.get("vendor"))
    if backend is None:
        return {**cpu, "reason": f"no_torch_backend_for_{selected_device.get('vendor')}"}
    record = probe(root, backend)
    if not record.get("ok"):
        return {**cpu, "reason": record.get("reason", "probe_failed")[:200]}
    return {"torch_device": backend, "torch_backend": backend, "discrete": not selected_device.get("shared_memory", True),
            "device_name": record.get("device"), "total_memory_bytes": record.get("total_memory_bytes")}


def vram_free_bytes(torch, device):
    """Free dedicated memory of a discrete GPU, or None on shared-memory GPUs."""
    module = getattr(torch, device.type, None)
    if module is None or not hasattr(module, "mem_get_info"):
        return None
    try:
        return int(module.mem_get_info()[0])
    except RuntimeError:
        return None


def weights_resident(model_bytes, discrete, free_vram_bytes, activation_reserve=1.5 * 2**30, headroom=1.1):
    """Keep all linear weights on a discrete GPU when they fit with room for activations."""
    if not discrete or free_vram_bytes is None:
        return False
    return model_bytes * headroom + activation_reserve <= free_vram_bytes


def attention_budget_bytes(torch=None, device=None, discrete=False):
    """Bytes allowed for one attention score tile (its intermediates are about three times larger)."""
    from . import resources
    if discrete and torch is not None and device is not None:
        free = vram_free_bytes(torch, device)
        if free is not None:
            return int(min(1024 * resources.MIB, max(64 * resources.MIB, free * 0.1)))
    total, available = resources.memory_bytes()
    spare = available - resources.reserve_gib(total) * resources.GIB
    return int(min(384 * resources.MIB, max(48 * resources.MIB, spare * 0.08)))


def bound_attention_memory(torch, budget=None, device_types=("xpu", "cuda"), discrete=False):
    """Process attention in query chunks so score memory grows linearly with sequence length.

    A kernel that materializes the full T x T score matrix needs 3.3 GiB at T=6000. Every query
    row attends to the same keys, so chunking the queries gives identical results.
    """
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    if getattr(original, "_memory_bounded", False):
        return

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **extra):
        if query.device.type not in device_types or is_causal or query.dim() < 3:
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **extra)
        rows, keys = query.shape[-2], key.shape[-2]
        per_row = (query.numel() // (rows * query.shape[-1])) * keys * 4
        allowed = budget() if budget is not None else attention_budget_bytes(torch, query.device, discrete)
        step = max(16, int(allowed // max(1, per_row)))
        if step >= rows:
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, **extra)
        pieces = []
        for start in range(0, rows, step):
            mask = attn_mask
            if mask is not None and mask.dim() >= 2 and mask.shape[-2] == rows:
                mask = mask[..., start:start + step, :]
            pieces.append(original(query[..., start:start + step, :], key, value, attn_mask=mask,
                                   dropout_p=dropout_p, scale=scale, **extra))
        return torch.cat(pieces, dim=-2)

    attention._memory_bounded = True
    functional.scaled_dot_product_attention = attention


def patch_vendor_for_fp32_gpu(torch):
    """Neo's reference code builds a few small float64 tensors on the compute device.

    Iris Xe has no FP64 units. These tensors (sigmas, timesteps, rotary angles) are tiny, so they
    are built on the CPU in float64 exactly as the reference does and then moved to the GPU as
    float32. The vendor sources themselves stay unchanged.
    """
    import importlib
    scheduler = importlib.import_module("neodragon.pyramid_scheduler").PyramidFlowMatchEulerDiscreteScheduler

    def on_cpu_then_gpu(original):
        def wrapper(self, num_inference_steps, stage, device=None):
            value = original(self, num_inference_steps, stage, device=None)
            return value if device is None else value.to(device=device, dtype=torch.float32)
        return wrapper

    scheduler.get_stage_timesteps = on_cpu_then_gpu(scheduler.get_stage_timesteps)
    scheduler.get_stage_sigmas = on_cpu_then_gpu(scheduler.get_stage_sigmas)
    embedding = importlib.import_module("neodragon.pyramid_mmdit.modeling_embedding")
    original_rope = embedding.rope

    def rope(pos, dim, theta):
        return original_rope(pos.cpu(), dim, theta).to(pos.device)

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("neodragon") and getattr(module, "rope", None) is original_rope:
            module.rope = rope


def describe(root, selected_device, backend_setting="auto"):
    """Doctor view of the torch GPU choice, including why a GPU is not used."""
    choice = choose(root, selected_device, backend_setting)
    return {**choice, "backends_installed": [name for name in BACKENDS if site(root, name).is_dir()],
            "weights": "resident_when_vram_fits_else_streamed" if choice.get("discrete") else "streamed"}
