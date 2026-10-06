"""SkyReels-V2 Diffusion Forcing 1.3B: the long-video tier for NVIDIA GPUs with enough VRAM.

Diffusion Forcing generates a long clip in one run: each new block of frames is denoised while
attending to the overlapping end of the previous block, so 15 seconds is one continuous
generation, not joined clips. Settings follow the GPU's total VRAM (stable, so cache keys stay
stable). Weights are installed once from a hash-pinned lock and rewritten from FP32 to BF16.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

LOCK = "skyreels.lock.json"
MINIMUM_VRAM_GIB = 8.0
GIB = 2**30
# Frame counts the model card lists as aligned with training (10s=257, 15s=377); short clips use 24 fps x s + 1.
FRAMES = {2: 49, 4: 97, 8: 193, 10: 257, 15: 377}


def folder(root, lock):
    return Path(root) / "models" / lock["folder"]


def frames_for(duration):
    return FRAMES[int(duration)]


def settings_for(total_vram_bytes, duration, preset="preview"):
    """Generation settings from total VRAM. Lower tiers trade resolution and block length for memory."""
    vram = (total_vram_bytes or 0) / GIB
    if vram >= 20:
        width, height, base, text_device = 960, 544, 97, "cuda" if vram >= 28 else "cpu"
    elif vram >= 14:
        width, height, base, text_device = 960, 544, 77, "cpu"
    elif vram >= 11:
        width, height, base, text_device = 800, 448, 77, "cpu"
    else:
        width, height, base, text_device = 672, 384, 57, "cpu"
    frames = frames_for(duration)
    return {"width": width, "height": height, "frames": frames, "fps": 24,
            "base_num_frames": min(base, frames), "overlap_history": 17 if frames > base else None,
            "addnoise_condition": 20 if frames > base else 0, "ar_step": 0,
            "steps": 30 if preset == "quality" else 20, "guidance_scale": 5.0, "flow_shift": 5.0,
            "text_encoder_device": text_device, "vae_tiling": vram < 20}


def installed(root, source, lock_name=LOCK):
    """True when every locked file is present in its final form (verified, BF16 where required)."""
    lock = json.loads((Path(source) / lock_name).read_text(encoding="utf-8"))
    record_path = folder(root, lock) / "install-record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if record.get("revision") != lock["revision"]:
        return False
    return all(item["path"] in record.get("files", {}) and (folder(root, lock) / item["path"]).is_file() for item in lock["files"])


def convert_to_bf16(path):
    """Rewrite one FP32 safetensors shard as BF16 in place (same file name, so shard indexes stay valid)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    path = Path(path)
    tensors, metadata = {}, None
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            value = handle.get_tensor(key)
            tensors[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    pending = path.with_suffix(".bf16.partial")
    save_file(tensors, str(pending), metadata=metadata or {"format": "pt"})
    del tensors
    pending.replace(path)
    return path


def fix_index_total_size(index_path):
    """Shard indexes record the byte total; keep it consistent after the BF16 rewrite."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = {index_path.parent / name for name in set(index.get("weight_map", {}).values())}
    if all(shard.is_file() for shard in shards):
        index.setdefault("metadata", {})["total_size"] = sum(shard.stat().st_size for shard in shards) // 2 * 2
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--convert":
        print(json.dumps({"converted": str(convert_to_bf16(sys.argv[2]))}), flush=True)
    else:
        raise SystemExit("usage: skyreels.py --convert <shard.safetensors>")
