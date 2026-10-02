"""Fresh CPU generation before/after conditioning reuse, with truthful timing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import control, neodragon
from local_video.backend import _path_in_runtime
from local_video.integrity import verify


def peak(result):
    return max(phase.get("peak_working_set_bytes", 0)
               for shot in result["shots"] for phase in shot["phases"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default=str(SOURCE / "examples" / "beach-request.json"))
    args = parser.parse_args()
    cfg, root = control.load_config()
    run_id = uuid.uuid4().hex[:10]
    folder = _path_in_runtime(root / "audits" / ("rebuild-0.5.0-" + run_id), root)
    folder.mkdir(parents=True)
    value = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    value.update(title="리빌드 CPU 비교 " + run_id, diagnostic=True)
    before = time.monotonic()
    submitted = control.submit(value)
    job_id = submitted["job_id"]
    print(json.dumps({"cold_job_id": job_id}, ensure_ascii=False), flush=True)
    while True:
        current = control.wait(job_id, 30)
        print(json.dumps({key: current[key] for key in ["job_id", "state", "stage", "error"] if key in current}, ensure_ascii=False), flush=True)
        if current["state"] in control.TERMINAL:
            break
        if time.monotonic()-before > 1800:
            control.cancel(job_id)
            raise RuntimeError("Rebuild benchmark timed out")
    if current["state"] != "completed":
        raise RuntimeError(str(current))
    cold = current["result"]
    assert not any(shot["cache_hit"] for shot in cold["shots"]), "A reused clip is not a fresh-generation benchmark"
    lock = control.EngineLock(_path_in_runtime(root / "engine.lock", root))
    deadline = time.monotonic()+15
    while not lock.acquire():
        if time.monotonic() > deadline:
            raise RuntimeError("Local engine is busy")
        time.sleep(0.2)
    try:
        request = control.normalize(value)
        verify(request, root, lambda update: None, lambda: False)
        # A separate shot-cache folder forces new temporal inference while
        # leaving the production cache and the model parameters untouched.
        warm = neodragon.generate_project(request, folder / "warm" / "output",
            folder / "warm" / "shots", root / "models",
            lambda update: print(json.dumps({"warm_stage": update.get("stage")}, ensure_ascii=False), flush=True) if "stage" in update else None,
            lambda: False)
        assert not any(shot["cache_hit"] for shot in warm["shots"])
        phases = {phase["name"]: phase for phase in warm["shots"][0]["phases"]}
        assert phases["video_text"]["cache_hit"] and phases["video_encode"]["cache_hit"]
        assert phases["video_pack"]["reused_derived_weights"]
        assert cold["video_sha256"] == warm["video_sha256"], "Warm conditioning changed the encoded result"

        # Prove the text-cache key can omit the random video seed. This extra
        # probe is not included in either full-generation timing above.
        seed_work = _path_in_runtime(folder / "text-seed-check", root)
        seed_work.mkdir()
        scene = request["scenes"][0]
        control.write_json(seed_work / "request.json", {"prompt": scene["prompt"], "seed": scene["seed"]+1,
            "threads": request["threads"], **cold["settings"]})
        neodragon.run_stage(root, seed_work, "video_text", request, lambda update: None, lambda: False)
        original_phase = next(phase for phase in cold["shots"][0]["phases"] if phase["name"] == "video_text")
        original_text = root / "cache" / "conditioning" / "video_text" / original_phase["cache_key"] / "video-text.pt"
        script = "import json,sys,torch; a=torch.load(sys.argv[1],map_location='cpu',weights_only=True); b=torch.load(sys.argv[2],map_location='cpu',weights_only=True); result={k:torch.equal(a[k],b[k]) for k in a}; print(json.dumps(result)); assert a.keys()==b.keys() and all(result.values())"
        checked = subprocess.run([str(root / "runtime" / "neodragon-python" / "python.exe"), "-B", "-c", script,
            str(original_text), str(seed_work / "video-text.pt")], cwd=root,
            env=neodragon.environment(root, request["threads"]), capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if checked.returncode:
            raise RuntimeError("Text seed equivalence failed: " + checked.stderr.decode("utf-8", errors="replace"))
        equality = json.loads(checked.stdout.decode("utf-8"))
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as cpu:
            cpu_name = winreg.QueryValueEx(cpu, "ProcessorNameString")[0].strip()
        image_cached = all(shot["phases"][0].get("cache_hit", False) for shot in cold["shots"] + warm["shots"])
        report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "server_version": "0.5.0", "run_directory": str(folder), "cold_job_id": job_id,
            "both_runs_have_fresh_temporal_inference": True, "first_image_cache_used": image_cached,
            "host_cpu": cpu_name, "cpu_threads": request["threads"], "target_gram_verified": False, "cuda_used": False,
            "baseline_0_4_host_cpu": "Intel Core i9-13900HX",
            "baseline_0_4_seconds": 128.640, "baseline_0_4_peak_bytes": 4237996032,
            "cold_conditioning_seconds": cold["elapsed_seconds"], "warm_conditioning_seconds": warm["elapsed_seconds"],
            "cold_peak_bytes": peak(cold), "warm_peak_bytes": peak(warm),
            "cold_settings": cold["settings"], "identical_cold_warm_video_sha256": cold["video_sha256"],
            "text_seed_tensor_equality": equality, "cold_result": cold, "warm_result": warm}
        control.write_json(folder / "benchmark.json", report)
        control.write_json(root / "audits" / "rebuild-benchmark.json", report)
        print(json.dumps({key: report[key] for key in ["passed", "cold_job_id", "cold_conditioning_seconds",
            "warm_conditioning_seconds", "cold_peak_bytes", "warm_peak_bytes", "text_seed_tensor_equality"]}, ensure_ascii=False), flush=True)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
