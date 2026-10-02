"""Experimental, offline CPU probe. Each model phase runs in a separate process."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside

STAGES = ["first_text", "first_latent", "first_decode", "video_text", "video_encode", "video_pack", "video_infer", "video_decode", "safety"]


from local_video.neodragon_stage import stage

def run(root, request):
    from local_video.backend import _memory
    output = inside(root, "audits", "neodragon-cpu-probe-" + uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    environment = {**os.environ, "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                   "HF_HOME": str(root / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TORCH_HOME": str(root / "cache" / "torch"),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    python = inside(root, "runtime", "neodragon-python", "python.exe")
    report = {"state": "running", "device": "cpu", "threads": 4, "visual_quality": "not_reviewed",
              "target_gram_verified": False, "maximum_inference_working_set_gib": 5.0,
              "maximum_one_time_conversion_working_set_gib": 6.5,
              "experimental_start_free_ram_gib": 3.0, "reserve_system_ram_gib": 1.0, "stages": [],
              "output_dir": str(output), "probe_pid": os.getpid(),
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.monotonic()
    report_path = output / "benchmark.json"
    print(json.dumps({"output": str(output), "stage": "starting_cpu_probe"}), flush=True)
    try:
        stages = list(STAGES)
        if request.get("first_image"):
            from PIL import Image
            source = Path(request["first_image"]).resolve(strict=True)
            if not source.is_relative_to(root):
                raise ValueError("First image must be inside the D runtime")
            with Image.open(source) as frame:
                frame.convert("RGB").resize((request["width"], request["height"])).save(output / "first-frame.png")
            stages = stages[3:]
        if request.get("cpu_precision") == "bf16_stream":
            stages.remove("video_pack")
        for name in stages:
            _, available = _memory(os.getpid())
            conversion = name == "video_pack"
            # These phases use mapped weights and release models between processes.
            # The live 1GiB reserve and working-set cap remain enforced every 250ms.
            minimum = 3.0
            if available < minimum * 2**30:
                raise RuntimeError(f"Available RAM is below this phase's {minimum}GiB start policy")
            cap = 6.5 if conversion else 5.0
            phase = {"name": name, "state": "running", "peak_working_set_bytes": 0,
                     "working_set_limit_gib": cap, "one_time_conversion": conversion}
            report["stages"].append(phase)
            phase_start = time.monotonic()
            with (output / (name + ".log")).open("wb") as log:
                process = subprocess.Popen([str(python), str(SOURCE / "local_video" / "neodragon_stage.py"), "--runtime-dir", str(root),
                    "--stage", name, "--output", str(output)], cwd=output, env=environment,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                phase["process_id"] = process.pid
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                try:
                    while process.poll() is None:
                        if (output / "cancel.request").exists():
                            raise RuntimeError("Research probe cancelled")
                        working, free = _memory(process.pid)
                        phase["peak_working_set_bytes"] = max(phase["peak_working_set_bytes"], working)
                        if working > cap * 2**30 or free < 1 * 2**30:
                            raise RuntimeError("Research phase exceeded the RAM protection policy")
                        if time.monotonic() - phase_start > 1800:
                            raise RuntimeError("CPU research phase timed out")
                        time.sleep(0.25)
                    if process.returncode != 0:
                        raise RuntimeError(name + " failed; see its saved log")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
            phase.update(state="completed", elapsed_seconds=round(time.monotonic()-phase_start, 3))
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(phase), flush=True)
        from PIL import Image
        paths = sorted(output.glob("frame-*.png"))
        indices = [round(i * (len(paths)-1) / 11) for i in range(12)]
        sheet = Image.new("RGB", (request["width"] * 4, request["height"] * 3))
        for slot, index in enumerate(indices):
            with Image.open(paths[index]) as frame:
                sheet.paste(frame, ((slot % 4)*request["width"], (slot // 4)*request["height"]))
        sheet.save(output / "contact-sheet.jpg", quality=88)
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", "24", "-i", str(output / "frame-%03d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output / "video.mp4")],
                       cwd=output, env=environment, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.run([ffmpeg, "-v", "error", "-i", str(output / "video.mp4"), "-f", "null", "-"],
                       cwd=output, env=environment, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        report["state"] = "encoded_requires_visual_review"
    except BaseException as exc:
        report.update(state="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic()-started, 3)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=__import__("local_video.storage", fromlist=["default_runtime"]).default_runtime())
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output")
    parser.add_argument("--prompt", default="Gentle ocean waves roll onto a sandy beach in daylight, a static camera watches the moving water.")
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first-image")
    parser.add_argument("--cpu-precision", choices=["int8", "bf16_stream"], default="int8")
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    if args.stage:
        output = Path(args.output).resolve(strict=True)
        if not output.is_relative_to(inside(root, "audits")):
            raise ValueError("Research output must be inside the D audit directory")
        stage(root, output, args.stage)
    else:
        if args.frames < 9 or args.frames > 49 or (args.frames - 1) % 8 or args.width % 32 or args.height % 32:
            raise ValueError("Probe uses 8n+1 frames and spatial sizes divisible by 32")
        run(root, {"prompt": args.prompt, "frames": args.frames, "width": args.width, "height": args.height, "seed": args.seed,
                   "first_image": args.first_image, "cpu_precision": args.cpu_precision})


if __name__ == "__main__":
    main()
