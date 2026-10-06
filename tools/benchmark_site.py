"""Run a real, guarded CPU site shot and keep dense frames for visual review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import control
from installer import inside


def review_sheet(video, target):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", input_params=["-threads", "2"])
    try:
        metadata = next(reader)
        frames = []
        for index, raw in enumerate(reader):
            if index % 6 == 0:
                frames.append((index, Image.frombytes("RGB", tuple(metadata["size"]), raw).resize((256, 160))))
        sheet = Image.new("RGB", (256*4, 184*((len(frames)+3)//4)), "#171a21")
        draw = ImageDraw.Draw(sheet)
        for tile, (index, frame) in enumerate(frames):
            x, y = tile % 4 * 256, tile // 4 * 184
            sheet.paste(frame, (x, y))
            draw.text((x+6, y+163), f"frame {index}  {index/metadata['fps']:.2f}s", fill="white")
        sheet.save(target, quality=94)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default=str(SOURCE / "examples" / "construction-request.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--duration", type=int, choices=[2, 4, 8, 10, 15])
    args = parser.parse_args()
    _, root = control.load_config()
    value = control.read_json(Path(args.request))
    if args.seed is not None:
        value["seed"] = args.seed
    if args.duration is not None:
        value["duration_seconds"] = args.duration
    value["diagnostic"] = True
    submitted = control.submit(value)
    job_id = submitted["job_id"]
    print(json.dumps({"job_id": job_id}, ensure_ascii=False), flush=True)
    deadline = time.monotonic()+1800
    while True:
        current = control.wait(job_id, 30)
        print(json.dumps({key: current[key] for key in ["job_id", "state", "stage", "error", "working_set_gib", "peak_working_set_gib"] if key in current}, ensure_ascii=False), flush=True)
        if current["state"] in control.TERMINAL:
            break
        if time.monotonic() > deadline:
            control.cancel(job_id)
            raise RuntimeError("Site benchmark timed out")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job_id,
              "state": current["state"], "target_gram_verified": False, "cuda_used": False,
              "weights_finetuned": False, "visual_review": "pending", "result": current.get("result"), "error": current.get("error")}
    folder = inside(root, "audits", "site-"+job_id)
    folder.mkdir(exist_ok=True)
    if current["state"] == "completed":
        result = current["result"]
        if result.get("continuous"):
            assert result["edit_count"] == 0 and len(result["shots"]) == 1
        review_sheet(result["video_path"], inside(root, "audits", "site-"+job_id, "dense-review.jpg"))
        report["fresh_temporal_inference"] = not any(shot["cache_hit"] for shot in result["shots"])
        report["peak_working_set_gib"] = round(max(phase.get("peak_working_set_bytes", 0)
            for shot in result["shots"] for phase in shot["phases"])/2**30, 4)
    control.write_json(folder / "benchmark.json", report)
    print(json.dumps({key: report[key] for key in ["state", "job_id", "fresh_temporal_inference", "peak_working_set_gib", "error"] if key in report}, ensure_ascii=False), flush=True)
    if current["state"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
