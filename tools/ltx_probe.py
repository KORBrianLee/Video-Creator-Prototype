"""Time LTX-Video rendering on this computer's GPU, optionally without the text encoder.

Usage: ltx_probe.py <runtime-dir> <first-frame.png> <seconds> <out-dir> [--zero-text | --prompt TEXT]
With --zero-text the transformer and VAE run on blank conditioning, which checks kernels, speed and
memory before the 18 GB text encoder is installed (the frames themselves are not meaningful).
"""
import argparse
import json
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import ltx, ltx_stage  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("runtime")
parser.add_argument("first_frame")
parser.add_argument("seconds", type=int)
parser.add_argument("out")
parser.add_argument("--zero-text", action="store_true")
parser.add_argument("--prompt", default="")
parser.add_argument("--backend", default="xpu")
args = parser.parse_args()
root, out = Path(args.runtime), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch, device = ltx_stage.open_gpu(root, {"torch_device": args.backend})
settings = ltx.settings_for(args.seconds)
started = time.monotonic()
if args.zero_text:
    embeds = torch.zeros(1, settings["max_sequence_length"], 4096).to(device)
    mask = torch.ones(1, settings["max_sequence_length"]).to(device)
else:
    embeds, mask = ltx_stage.encode_text(torch, device, ltx_stage.model_folder(root), args.prompt, settings["max_sequence_length"])
text_seconds = time.monotonic() - started
frames, info = ltx_stage.render(torch, device, root, Path(args.first_frame), settings, embeds, mask, 42, False)
for index in range(0, len(frames), max(1, len(frames) // 8)):
    frames[index].save(out / f"sample-{index:03d}.png")
print(json.dumps({"seconds": args.seconds, "frames": len(frames), "text_seconds": round(text_seconds, 1),
                  "total_seconds": round(time.monotonic() - started, 1), **info}), flush=True)
