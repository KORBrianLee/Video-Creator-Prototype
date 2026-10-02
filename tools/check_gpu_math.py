"""Real-driver numerical checks, including ragged sizes and bounded row chunks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_video.opencl_linear import LinearEngine, KERNEL, plan


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "intel-gpu"])
    args = parser.parse_args()
    selected = plan(args.backend)
    if not selected["gpu_inference"]:
        raise RuntimeError("실제 GPU가 없습니다. 수학 검사를 GPU 성공으로 기록하지 않습니다.")
    rng, checks = np.random.default_rng(42), []
    for m, k, n, biased, budget in [(37, 65, 29, True, 2**20), (53, 31, 17, False, 8192), (16, 1536, 512, True, 64*2**20)]:
        engine = LinearEngine(selected["selected_device"]["id"], budget)
        try:
            a = rng.standard_normal((m, k)).astype("float32")
            w = rng.standard_normal((n, k)).astype("float32")
            bits = (w.view("uint32") >> 16).astype("uint16")
            bias = rng.standard_normal(n).astype("float32") if biased else None
            expected = a @ (bits.astype("uint32") << 16).view("float32").T
            if biased:
                expected += bias
            actual = engine.linear(a, bits, bias)
            np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0002)
            # Reuse the same host allocation after changing its contents; no
            # stale weight or bias bytes may survive a reused GPU allocation.
            bits[:] = (rng.standard_normal((n, k)).astype("float32").view("uint32") >> 16).astype("uint16")
            if biased:
                bias[:] = rng.standard_normal(n).astype("float32")
            expected = a @ (bits.astype("uint32") << 16).view("float32").T
            if biased:
                expected += bias
            actual = engine.linear(a, bits, bias)
            np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0002)
            metrics = engine.metrics()
            assert metrics["peak_explicit_gpu_buffer_bytes"] <= budget
            assert metrics["buffer_allocations"] == 4 and metrics["buffer_reuses"] == 4
            if budget == 8192:
                assert metrics["kernel_calls"] > 1
            checks.append({"shape": [m,k,n], "bias": biased, "maximum_absolute_error": float(np.abs(actual-expected).max()), **metrics})
        finally:
            engine.close()
    engine = LinearEngine(selected["selected_device"]["id"], 8192)
    try:
        for m, k, n in [(19, 17, 89), (43, 73, 13), (11, 27, 31)]:
            a = rng.standard_normal((m, k)).astype("float32")
            bits = (rng.standard_normal((n, k)).astype("float32").view("uint32") >> 16).astype("uint16")
            actual = engine.linear(a, bits)
            expected = a @ (bits.astype("uint32") << 16).view("float32").T
            np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0002)
            assert engine.metrics()["peak_explicit_gpu_buffer_bytes"] <= 8192
        checks.append({"case": "changing_shapes_with_tight_budget", **engine.metrics()})
    finally:
        engine.close()
    report = {"state": "passed_real_gpu_math", "kernel_sha256": hashlib.sha256(KERNEL.encode()).hexdigest(), "checks": checks,
              "meaning": "matrix correctness and explicit buffer limit; not video realism or Gram speed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"state": report["state"], "device": selected["selected_device"]["name"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
