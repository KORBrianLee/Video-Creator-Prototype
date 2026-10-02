"""Micro-benchmark of the OpenCL linear engine against PyTorch CPU on model-sized matrices."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from local_video import opencl_linear

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--shapes", default="1536x1536,1536x9216,9216x1536")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tile", default="4x4", help="outputs per work-item, ROWSxCOLS (multiples of 4)")
    parser.add_argument("--no-cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    plan = opencl_linear.plan()
    tile = tuple(map(int, args.tile.split("x")))
    engine = opencl_linear.LinearEngine(plan["selected_device"]["id"], tile=tile)
    rng = np.random.default_rng(1)
    report = []
    for shape in args.shapes.split(","):
        k, n = map(int, shape.split("x"))
        activation = rng.standard_normal((args.tokens, k), dtype=np.float32)
        weight = torch.from_numpy(rng.standard_normal((n, k), dtype=np.float32)).to(torch.bfloat16)
        bits = weight.view(torch.uint16).numpy()
        reference = torch.nn.functional.linear(torch.from_numpy(activation), weight.float()).numpy()
        engine.linear(activation, bits)
        started = time.monotonic()
        for _ in range(args.repeat):
            result = engine.linear(activation, bits)
        gpu = (time.monotonic() - started) / args.repeat
        started = time.monotonic()
        for _ in range(0 if args.no_cpu else args.repeat):
            torch.nn.functional.linear(torch.from_numpy(activation), weight.float())
        cpu_fp32 = (time.monotonic() - started) / args.repeat or 1e9
        flops = 2 * args.tokens * k * n
        report.append({"shape": shape, "tokens": args.tokens, "gpu_s": round(gpu, 4), "cpu_fp32_s": round(cpu_fp32, 4),
                       "gpu_gflops": round(flops / gpu / 1e9, 1), "cpu_gflops": round(flops / cpu_fp32 / 1e9, 1),
                       "max_abs_error": float(np.abs(result - reference).max()),
                       "relative_error": float(np.abs(result - reference).max() / np.abs(reference).max())})
    print(json.dumps({"device": engine.device["name"], "results": report, "engine": engine.metrics() | {"device": None}}, indent=1))
    engine.close()


if __name__ == "__main__":
    main()
