"""Diagnostic only: verify the suspected scheduler mismatch without editing models."""
import json
import struct
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_video import backend, control

request = control.normalize({"preset": "smoke", "seed": 42, "scenes": [{"id": "shore", "prompt": "Ocean waves rolling onto a sandy beach, blue sea, warm morning sunlight, continuous natural motion, static wide shot"}]})
root = Path(request["_runtime_dir"])
variant = sys.argv[1] if len(sys.argv) > 1 else "disk-fa"
if variant not in {"disk-fa", "cpu", "disk", "cpu-native", "cpu-native-tae", "cpu-native-monolithic", "stock", "f16", "older", "context16", "haswell"}:
    raise ValueError("Unknown diagnostic variant")
work = root / "audits" / ("lightning-linear-alpha-probe" if variant == "disk-fa" else "lightning-linear-alpha-probe-" + variant)
work.mkdir(parents=True, exist_ok=True)
alpha = 1.0
alphas = []
for index in range(1000):
    alpha *= 1.0 - (0.00085 + (0.012 - 0.00085) * index / 999)
    alphas.append(alpha)
header = json.dumps({"alphas_cumprod": {"dtype": "F32", "shape": [1000], "data_offsets": [0, 4000]}}, separators=(",", ":")).encode()
header += b" " * ((-len(header)) % 8)
alpha_path = work / "linear-alphas.safetensors"
alpha_path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<1000f", *alphas))
(work / "prompt.txt").write_text(request["scenes"][0]["prompt"], encoding="utf-8")
(work / "negative.txt").write_text("", encoding="utf-8")
binary, assignment, help_text, identity = backend._engine(request, root)
if variant == "older":
    binary = root / "engines" / "diagnostic-841" / "sd-cli.exe"
    help_text = subprocess.run([str(binary), "--help"], capture_output=True, text=True, check=True).stdout
if variant == "haswell":
    binary = root / "engines" / "diagnostic-haswell" / "sd-cli.exe"
paths = backend._model_paths(request["_model_spec"], root / "models", "lightning")
settings = dict(backend.PRESETS["lightning"]["smoke"])
if variant == "stock":
    settings.update(frames=4, steps=20)
if variant == "f16":
    settings.update(frames=4)
if variant == "haswell":
    settings.update(frames=4)
if variant == "context16":
    settings.update(frames=16)
command = backend.build_command(binary, assignment, "lightning", paths, settings, request["scenes"][0], work, 4, request["_model_spec"]["sampling"])
command += ["--embeddings-connectors", str(alpha_path)]
if variant in {"cpu", "disk", "cpu-native", "cpu-native-tae", "cpu-native-monolithic", "stock", "f16", "older", "context16", "haswell"}:
    command.remove("--diffusion-fa")
if variant in {"cpu", "cpu-native", "cpu-native-tae", "cpu-native-monolithic", "stock", "f16", "older", "context16", "haswell"}:
    command[command.index("--params-backend")+1] = "cpu"
if variant in {"cpu-native-tae", "cpu-native-monolithic", "stock", "f16", "older", "context16", "haswell"}:
    command += ["--taesd", str(root / "models" / "taesd_sd15.safetensors")]
if variant in {"cpu-native-monolithic", "f16", "context16", "haswell"}:
    command += ["--disable-segmented-compute"]
if variant == "f16":
    command += ["--type", "f16"]
if variant == "stock":
    position = command.index("--sigmas")
    del command[position:position+2]
    command[command.index("--motion-module")+1] = str(root / "models" / "v3_sd15_mm.ckpt")
    command[command.index("--cfg-scale")+1] = "8"
    command += ["--scheduler", "discrete", "--disable-segmented-compute"]
if variant == "older":
    for flag, count in (("--disable-prefetch", 1), ("--conditioning-cache-size", 2)):
        position = command.index(flag)
        del command[position:position+count]
    command += ["--max-vram", "0"]
command += ["--verbose"]
backend._validate_flags(command, help_text)
result = backend._infer(command, work, request, lambda update: print(json.dumps(update), flush=True), lambda: False, 1, 1, settings["steps"])
video = work / "video.mp4"
backend._encode_frames(work, video, settings, lambda: False)
result["verification"] = backend.verify_video(video, settings)
backend._assemble([{"id": "shore", "seed": 42, "video_path": str(video)}], work / "review.mp4", work / "contact-sheet.jpg", settings, lambda: False)
backend._write_json(work / "benchmark.json", result)
print(json.dumps(result), flush=True)
