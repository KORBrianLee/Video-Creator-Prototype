"""Precompute checked local scene conditioning, without training model weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import control, neodragon, site_profiles, conditioning
from installer import inside


def main():
    cfg, root = control.load_config()
    request = {**cfg, "threads": min(4, cfg["threads"]),
        "model_revision": control.normalize({"scenes": [{"prompt": "construction"}]})["model_revision"]}
    lock = control.EngineLock(inside(root, "engine.lock"))
    if not lock.acquire():
        raise RuntimeError("영상 작업이 실행 중입니다. 완료 후 현장 조건 계산을 준비하세요.")
    records = []
    try:
        for name, profile in site_profiles.PROFILES.items():
            work = inside(root, "datasets", "construction-v1", "conditioning", name)
            work.mkdir(parents=True, exist_ok=True)
            control.write_json(work / "request.json", {"prompt": profile["prompt"], "seed": 42, "threads": request["threads"],
                "prompt_modifier": profile.get("prompt_modifier", ""),
                "width": 512, "height": 320, "frames": 49})
            result = conditioning.run_cached(root, work, "video_text", request, lambda update: None, lambda: False, neodragon.run_stage)
            record = {**result, "id": name, "prompt_sha256": hashlib.sha256(profile["prompt"].encode()).hexdigest(),
                "state": "conditioning_ready_not_weight_training"}
            records.append(record)
            print(json.dumps({k: record[k] for k in ["id", "state", "cache_hit", "elapsed_seconds"]}, ensure_ascii=False), flush=True)
        control.write_json(inside(root, "datasets", "construction-v1", "conditioning", "manifest.json"), {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "profile_revision": site_profiles.revision(),
            "weights_finetuned": False, "model_revision": request["model_revision"], "profiles": records})
    finally:
        lock.release()


if __name__ == "__main__":
    main()
