"""Run one stage on the XPU and print GPU allocator and system memory every few seconds."""
import sys
import threading
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import accelerator, resources  # noqa: E402

root, output, stage_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
gpu_backend = sys.argv[5] if len(sys.argv) > 5 else "xpu"
accelerator.enable(root, gpu_backend)
import torch  # noqa: E402
gpu = getattr(torch, gpu_backend)

started = time.monotonic()


def report():
    import os
    from local_video import backend
    gib = 2**30
    while True:
        time.sleep(interval)
        working, free = backend._memory(os.getpid())
        reclaimable = backend._reclaimable(os.getpid(), working)
        print(f"t={time.monotonic() - started:6.1f}s gpu_alloc={gpu.memory_allocated() / gib:5.2f} "
              f"gpu_reserved={gpu.memory_reserved() / gib:5.2f} working_set={working / gib:5.2f} "
              f"reclaimable_mapped={reclaimable / gib:5.2f} private={(working - reclaimable) / gib:5.2f} "
              f"system_free={free / gib:5.2f}", flush=True)


threading.Thread(target=report, daemon=True).start()
from local_video import neodragon_stage  # noqa: E402

neodragon_stage.stage(root, output, stage_name)
