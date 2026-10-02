"""Compare the official parallel and sequential causal decoder numerically."""
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside

root = d_root(r"D:\CursorVideoLocal")
sys.path.insert(0, str(inside(root, "engines", "experimental-neodragon")))
import torch
from neodragon.asymmetric_causal_video_vae.decoder import TAEHVDecoder, apply_model_with_memblocks

torch.set_num_threads(4)
torch.manual_seed(123)
assert torch.version.cuda is None
decoder = TAEHVDecoder(n_f=(8, 8, 4, 4)).eval()
values = torch.randn(1, 3, 16, 3, 5)
with torch.inference_mode():
    parallel = apply_model_with_memblocks(decoder.blocks, values.clone(), True, False)
    sequential = apply_model_with_memblocks(decoder.blocks, values.clone(), False, False)
    error = float((parallel-sequential).abs().max())
assert torch.allclose(parallel, sequential, atol=1e-6, rtol=1e-5), error
report = {"passed": True, "device": "cpu", "parallel_shape": list(parallel.shape),
          "sequential_shape": list(sequential.shape), "maximum_absolute_error": error,
          "scope": "official decoder control flow with small feature dimensions; not a realism test"}
inside(root, "audits", "neodragon-decoder-equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
