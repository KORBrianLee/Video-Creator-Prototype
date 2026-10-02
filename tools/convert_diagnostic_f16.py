"""Diagnostic only. Separate dequantized model; production original is preserved."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_video import backend, control

request = control.normalize({"preset": "smoke", "scenes": [{"prompt": "Ocean waves at a sunny beach"}]})
root = Path(request["_runtime_dir"])
work = root / "audits" / "diagnostic-f16-conversion"
work.mkdir(parents=True, exist_ok=True)
target = root / "models" / "stable-diffusion-v1-5-lightning-dequant-f16.gguf"
if target.exists():
    raise FileExistsError("Do not overwrite diagnostic evidence")
request["maximum_working_set_gib"] = 7.0  # Engineering trial on 32GiB host, never Gram default.
command = [str(root / "engines" / "cpu" / "sd-cli.exe"), "--mode", "convert", "--model",
           str(root / "models" / request["_model_spec"]["prepared_base"]["filename"]),
           "--type", "f16", "--tensor-type-rules", "^alphas_cumprod$=f32", "--threads", "1",
           "--output", str(target), "--verbose"]
result = backend._infer(command, work, request, lambda update: print(json.dumps(update), flush=True),
                        lambda: False, 1, 1, 1)
backend._write_json(work / "benchmark.json", result)
print(json.dumps(result), flush=True)
