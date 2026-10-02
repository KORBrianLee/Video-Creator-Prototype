"""Same-driver allocation comparison; not a video quality or Gram benchmark."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_video.opencl_linear import LinearEngine, plan


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("local_video._opencl_baseline", args.baseline)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    selected = plan("gpu")["selected_device"]
    rng, values = np.random.default_rng(42), []
    for m, k, n in [(37, 65, 29), (53, 31, 17), (16, 1536, 512), (8, 512, 1536)]:
        a = rng.standard_normal((m, k)).astype("float32")
        w = (rng.standard_normal((n, k)).astype("float32").view("uint32") >> 16).astype("uint16")
        b = rng.standard_normal(n).astype("float32")
        values.append((a, w, b))
    engines = [previous.LinearEngine(selected["id"]), LinearEngine(selected["id"])]
    counts, timings = [0, 0], [[], []]
    try:
        for index, engine in enumerate(engines):
            original = engine.dll.clCreateBuffer
            def allocate(*arguments, index=index, original=original):
                counts[index] += 1
                return original(*arguments)
            engine.dll.clCreateBuffer = allocate
        for turn in range(5):
            # Alternate order after a common warm-up to reduce order effects.
            for index in ([0, 1] if turn % 2 == 0 else [1, 0]):
                engine = engines[index]
                start = time.monotonic()
                for _ in range(10):
                    for a, w, b in values:
                        actual = engine.linear(a, w, b)
                        if turn == 0:
                            expected = a @ (w.astype("uint32") << 16).view("float32").T + b
                            np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.0002)
                if turn:
                    timings[index].append(time.monotonic() - start)
        metrics = [engine.metrics() for engine in engines]
        assert metrics[0]["uploaded_bytes"] == metrics[1]["uploaded_bytes"]
        assert counts[1] < counts[0]
        assert all(item["peak_explicit_gpu_buffer_bytes"] <= 64 * 2**20 for item in metrics)
        report = {"state": "passed_allocation_comparison", "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  "device": selected, "linear_calls_each": 200, "baseline_allocations": counts[0], "optimized_allocations": counts[1],
                  "allocation_reduction_percent": round(100 * (1 - counts[1] / counts[0]), 2),
                  "baseline_median_seconds": round(statistics.median(timings[0]), 4),
                  "optimized_median_seconds": round(statistics.median(timings[1]), 4),
                  "timings": timings, "metrics": metrics, "target_iris_generation_verified": False,
                  "meaning": "same synthetic matrices and fixed cap; timing is this host only, not end-to-end video speed"}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("state", "baseline_allocations", "optimized_allocations", "allocation_reduction_percent")}))
    finally:
        for engine in engines:
            engine.close()


if __name__ == "__main__":
    main()
