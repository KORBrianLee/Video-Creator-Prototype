"""Verify prepared INT8 weights without launching or importing PyTorch."""
from __future__ import annotations

import hashlib
import json
import re


def verify(root, app, is_cancelled):
    from .control import runtime_path, read_json, write_json
    from .backend import _cancel
    packed = runtime_path(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.pt")
    notice = runtime_path(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.provenance.json")
    if not packed.is_file() or not notice.is_file():
        return None
    _cancel(is_cancelled)
    try:
        record = read_json(notice)
        if not isinstance(record, dict):
            return None
        models = read_json(app / "neodragon.lock.json")
        source = next(item for item in models["files"]
                      if item["filename"].endswith("diffusion_transformer_320p/diffusion_pytorch_model.safetensors"))
        runtime = read_json(app / "neodragon-runtime.lock.json")
        version = next(item["version"] for item in runtime["packages"] if item["name"] == "torch")
        expected = record.get("sha256", "")
        if (record.get("conversion") != "neodragon-cpu-per-channel-int8-v2"
                or record.get("source_sha256") != source["sha256"]
                or record.get("source_revision") != source["revision"]
                or record.get("torch") != version
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            return None
        info = packed.stat()
        if not 0 < info.st_size < 3 * 2**30:
            return None
        fingerprint = {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
                       "ctime_ns": info.st_ctime_ns, "inode": info.st_ino, "expected_sha256": expected}
        stamp = runtime_path(root, "cache", "model-integrity", "neodragon-derived.json")
        try:
            saved = read_json(stamp)
        except (OSError, ValueError):
            saved = None
        if info.st_size <= 16 * 2**20 or saved != fingerprint:
            digest = hashlib.sha256()
            with packed.open("rb") as stream:
                while chunk := stream.read(8 * 2**20):
                    _cancel(is_cancelled)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                return None
            after = packed.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino):
                raise RuntimeError("확인 중 변환 모델이 변경됐습니다.")
            write_json(stamp, fingerprint)
        _cancel(is_cancelled)
        return {"sha256": expected, "bytes": info.st_size}
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None
