"""Per-computer resource limits, recomputed before every stage.

Values marked "auto" in video.config.json follow the current computer: RAM
limits scale with total and currently free memory, GPU buffers with shared RAM
or dedicated VRAM, CPU threads with physical cores. A number in the config is
used as a fixed manual limit. Only machine-stable values (threads) may enter
cache keys; momentary limits never change numerical results.
"""
from __future__ import annotations
import ctypes
import math
import os

GIB, MIB = 2**30, 2**20
# Largest Neodragon linear weight is 27 MiB (9216x1536 BF16) plus one activation row.
GPU_FLOOR = 32 * MIB
GPU_SHARED_CEILING = 256 * MIB
GPU_DISCRETE_CEILING = 1024 * MIB
GPU_STEP = 16 * MIB
PROFILE_FLOOR_GIB = {"neodragon": 3.0, "lightning": 4.5, "wan": 4.5}
# Smallest working-set cap that completed a Neodragon scene on the 16 GB target.
WORKING_SET_FLOOR_GIB = 5.0
RESOURCE_KEYS = ("minimum_free_ram_gib", "reserve_ram_gib", "maximum_working_set_gib", "gpu_buffer_mib")


def is_auto(value):
    return value is None or value == "auto"


def check_value(key, value):
    if is_auto(value):
        return "auto"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key}는 "auto" 또는 유한한 양수여야 합니다.')
    if key == "gpu_buffer_mib" and not GPU_FLOOR // MIB <= value <= GPU_DISCRETE_CEILING // MIB:
        raise ValueError(f"gpu_buffer_mib는 {GPU_FLOOR // MIB}~{GPU_DISCRETE_CEILING // MIB} 사이여야 합니다.")
    return value


def memory_bytes():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(status)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("메모리 정보를 읽을 수 없습니다.")
    return status.total, status.available


def core_classes():
    """Efficiency class of each physical core (higher is faster on hybrid CPUs)."""
    if os.name != "nt":
        return []
    kernel = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    if not size.value:
        return []
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        return []
    classes, offset, raw = [], 0, buffer.raw
    while offset + 10 <= size.value:
        classes.append(raw[offset + 9])
        offset += int.from_bytes(raw[offset+4:offset+8], "little") or size.value
    return classes


def physical_cores():
    classes = core_classes()
    if classes:
        return len(classes)
    logical = os.cpu_count() or 4
    return max(1, logical // 2 if logical >= 4 else logical)


def performance_cores():
    classes = core_classes()
    return classes.count(max(classes)) if classes else physical_cores()


def auto_threads():
    # PyTorch splits work evenly, so slower efficiency cores would hold back every step.
    return max(1, min(8, performance_cores()))


def reserve_gib(total_bytes):
    return round(min(4.0, max(1.0, total_bytes / GIB * 0.09)), 2)


def gpu_buffer_bytes(device, total_bytes, available_bytes, override=None):
    """Explicit OpenCL buffer budget for one matrix at a time."""
    if not device:
        return None
    if not is_auto(override):
        return int(override * MIB)
    if device.get("shared_memory", True):
        spare = max(0, available_bytes - reserve_gib(total_bytes) * GIB)
        budget = min(GPU_SHARED_CEILING, max(GPU_FLOOR, spare // 24))
    else:
        budget = min(GPU_DISCRETE_CEILING, max(GPU_FLOOR * 2, int(device.get("global_memory_bytes") or 0) // 4))
    return max(GPU_FLOOR, int(budget) // GPU_STEP * GPU_STEP)


def stage_limits(request, profile=None, device_plan=None):
    """RAM limits for the next stage from the current state of this computer."""
    profile = profile or request.get("model_profile", "neodragon")
    total, available = memory_bytes()
    reserve = request.get("reserve_ram_gib")
    reserve = reserve_gib(total) if is_auto(reserve) else float(reserve)
    floor = max(PROFILE_FLOOR_GIB.get(profile, 3.0), float((device_plan or request.get("device_plan") or {}).get("minimum_free_ram_gib") or 0))
    minimum = request.get("minimum_free_ram_gib")
    minimum = max(floor, reserve + 1.0) if is_auto(minimum) else max(floor, float(minimum))
    cap = request.get("maximum_working_set_gib")
    if is_auto(cap):
        cap = max(WORKING_SET_FLOOR_GIB, min(total / GIB * 0.5, available / GIB - reserve))
    cap = float(cap)
    if cap <= reserve:
        raise ValueError("최대 작업 메모리는 시스템 보호 메모리보다 커야 합니다.")
    return {"minimum_free_ram_gib": round(minimum, 2), "reserve_ram_gib": round(reserve, 2),
            "maximum_working_set_gib": round(cap, 2), "total_ram_gib": round(total / GIB, 2),
            "free_ram_gib": round(available / GIB, 2)}


def _profile_path(root):
    from pathlib import Path
    return Path(root) / "cache" / "resource-profile.json"


def learned_drops(root):
    import json
    try:
        data = json.loads(_profile_path(root).read_text(encoding="utf-8"))
        return {k: float(v) for k, v in data.get("stage_free_ram_drop_gib", {}).items() if isinstance(v, (int, float)) and v > 0}
    except (OSError, ValueError, AttributeError):
        return {}


def record_drop(root, stage, start_free_bytes, lowest_free_bytes):
    """Remember how far a stage pulled system free RAM down on this computer.

    The drop, not the child's working set, is recorded: mapped model pages are
    counted in the working set yet stay reclaimable. Older highs decay slowly.
    """
    import json
    drop = start_free_bytes - lowest_free_bytes
    if not root or drop <= 0:
        return
    path = _profile_path(root)
    drops = learned_drops(root)
    drops[stage] = round(max(drop / GIB, drops.get(stage, 0) * 0.9), 3)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(f".{os.getpid()}.tmp")
        pending.write_text(json.dumps({"schema": 1, "stage_free_ram_drop_gib": drops}, indent=2), encoding="utf-8")
        pending.replace(path)
    except OSError:
        pass


def start_requirement_gib(root, stage, limits):
    """Free RAM needed so the learned drop of a stage (or the largest, if None) stays above the stop floor."""
    drops = learned_drops(root) if root else {}
    drop = drops.get(stage) if stage else max(drops.values(), default=None)
    if not drop:
        return limits["minimum_free_ram_gib"]
    return round(max(limits["minimum_free_ram_gib"], drop * 1.1 + limits["reserve_ram_gib"] * 0.75), 2)


class PressureGuard:
    """Tolerate short RAM dips; stop on sustained shortage or before Windows pages heavily."""
    def __init__(self, reserve_bytes, grace_seconds=15.0):
        self.reserve, self.floor, self.grace = reserve_bytes, reserve_bytes * 0.75, grace_seconds
        self.since, self.events = None, 0

    def check(self, available, now):
        if available < self.floor:
            raise MemoryError(f"시스템 여유 RAM {available/GIB:.2f}GiB가 즉시 중지 기준 {self.floor/GIB:.2f}GiB보다 낮습니다.")
        if available >= self.reserve:
            self.since = None
            return False
        if self.since is None:
            self.since, self.events = now, self.events + 1
        if now - self.since > self.grace:
            raise MemoryError(f"시스템 여유 RAM이 보호 기준 {self.reserve/GIB:.2f}GiB 미만으로 {self.grace:.0f}초 넘게 유지됐습니다.")
        return True


def wait_for_memory(minimum_gib, is_cancelled, progress, timeout=300.0, poll=2.0):
    import time
    started = time.monotonic()
    while True:
        if is_cancelled():
            raise InterruptedError("RAM 회복 대기 중 취소됐습니다.")
        free = memory_bytes()[1] / GIB
        if free >= minimum_gib:
            return free
        if time.monotonic() - started > timeout:
            raise MemoryError(f"{timeout:.0f}초 동안 여유 RAM이 {minimum_gib}GiB로 회복되지 않았습니다. 다른 앱을 닫고 다시 요청하세요.")
        progress({"stage": "waiting_for_memory", "free_ram_gib": round(free, 2), "needed_free_ram_gib": minimum_gib})
        time.sleep(poll)


def describe(cfg, profile, device_plan, root=None):
    limits = stage_limits(cfg, profile, device_plan)
    if root is not None:
        limits["learned_stage_free_ram_drop_gib"] = learned_drops(root)
        limits["minimum_free_ram_gib"] = start_requirement_gib(root, None, limits)
    device = (device_plan or {}).get("selected_device")
    total, available = limits["total_ram_gib"] * GIB, limits["free_ram_gib"] * GIB
    budget = gpu_buffer_bytes(device, total, available, cfg.get("gpu_buffer_mib")) if device_plan and device_plan.get("gpu_inference") and profile == "neodragon" else None
    return {**limits, "threads": cfg["threads"], "physical_cores": physical_cores(), "performance_cores": performance_cores(),
            "logical_processors": os.cpu_count(),
            "gpu_buffer_mib": budget // MIB if budget else None,
            "gpu_buffer_range_mib": [GPU_FLOOR // MIB, (GPU_SHARED_CEILING if (device or {}).get("shared_memory", True) else GPU_DISCRETE_CEILING) // MIB] if budget else None,
            "mode": {key: "auto" if is_auto(cfg.get(key)) else "manual" for key in RESOURCE_KEYS} | {"threads": "auto" if is_auto(cfg.get("threads_setting")) else "manual"},
            "policy": "recomputed_before_each_stage_gpu_buffer_shrinks_under_ram_pressure"}
