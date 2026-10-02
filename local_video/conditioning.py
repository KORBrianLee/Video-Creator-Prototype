"""Small, checked CPU conditioning caches; no model libraries in the server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from . import backend as media

ARTIFACTS = {"video_text": "video-text.pt", "video_encode": "video-first-latent.pt"}
MAXIMUM_BYTES = 32 * 2**20


def cache_key(app, work, stage, request):
    value = {"schema": "cpu-conditioning-v1", "stage": stage,
             "models": request["model_revision"], "threads": request["threads"]}
    # Both imports and the CPU quantizer can affect numerical results.
    digest = hashlib.sha256()
    for relative in ("local_video/neodragon_stage.py", "local_video/neodragon_quant.py",
                     "neodragon-runtime.lock.json"):
        digest.update((app / relative).read_bytes())
    value["implementation"] = digest.hexdigest()
    scene = json.loads((work / "request.json").read_text(encoding="utf-8"))
    if stage == "video_text":
        # Eval-only text encoding has no random sampling. Verify this property
        # against two seeds before accepting a rebuilt release.
        value["prompt"] = scene["prompt"]
        value["prompt_modifier"] = scene.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours")
    else:
        value.update(seed=scene["seed"], width=scene["width"], height=scene["height"])
        with (work / "first-frame.png").open("rb") as stream:
            value["image_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def restore(root, folder, work, key, filename, is_cancelled):
    media._cancel(is_cancelled)
    try:
        metadata = media._path_in_runtime(folder / "metadata.json", root)
        cached = media._path_in_runtime(folder / filename, root)
        destination = media._path_in_runtime(work / filename, root)
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        size = cached.stat().st_size
        if (record.get("state") != "complete" or record.get("cache_key") != key
                or not 0 < size <= MAXIMUM_BYTES or size != record.get("size")):
            return False
        # Check the copy that inference will actually consume, so a source
        # changed during copying cannot become a trusted conditioning file.
        shutil.copy2(cached, destination)
        with destination.open("rb") as stream:
            valid = hashlib.file_digest(stream, "sha256").hexdigest() == record.get("sha256")
        media._cancel(is_cancelled)
        if not valid:
            destination.unlink(missing_ok=True)
        return valid
    except (OSError, ValueError, KeyError):
        return False


def commit(root, folder, work, key, filename, is_cancelled):
    from .control import write_json
    media._cancel(is_cancelled)
    source = media._path_in_runtime(work / filename, root)
    if not 0 < source.stat().st_size <= MAXIMUM_BYTES:
        raise RuntimeError("조건 계산 파일 크기가 허용 범위를 벗어났습니다.")
    folder = media._path_in_runtime(folder, root)
    folder.mkdir(parents=True, exist_ok=True)
    pending = media._path_in_runtime(folder / (filename + "." + uuid.uuid4().hex + ".tmp"), root)
    destination = media._path_in_runtime(folder / filename, root)
    metadata = media._path_in_runtime(folder / "metadata.json", root)
    try:
        shutil.copy2(source, pending)
        with pending.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        media._cancel(is_cancelled)
        pending.replace(destination)
        write_json(metadata, {"state": "complete", "cache_key": key,
                              "size": destination.stat().st_size, "sha256": digest})
    finally:
        pending.unlink(missing_ok=True)


def run_cached(root, work, stage, request, progress, is_cancelled, runner):
    if stage not in ARTIFACTS:
        return runner(root, work, stage, request, progress, is_cancelled)
    started = time.monotonic()
    app = Path(__file__).resolve().parents[1]
    key = cache_key(app, work, stage, request)
    folder = media._path_in_runtime(root / "cache" / "conditioning" / stage / key, root)
    filename = ARTIFACTS[stage]
    if restore(root, folder, work, key, filename, is_cancelled):
        progress({"stage": stage + "_cache_reuse"})
        result = {"name": stage, "state": "completed", "cache_hit": True, "cache_key": key,
                  "elapsed_seconds": round(time.monotonic()-started, 3), "peak_working_set_bytes": 0}
        media._write_json(work / (stage + ".metrics.json"), result)
        return result
    result = runner(root, work, stage, request, progress, is_cancelled)
    commit(root, folder, work, key, filename, is_cancelled)
    return {**result, "cache_hit": False, "cache_key": key}
