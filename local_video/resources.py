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


def length_bucket(frames):
    """Clip-length class for learned RAM needs: an 8s decode must not set the bar for a 2s clip."""
    if frames is None:
        return None
    return "2s" if frames <= 49 else "4s" if frames <= 97 else "8s" if frames <= 193 else "long"


def drop_key(stage, frames=None):
    bucket = length_bucket(frames)
    return stage if bucket is None else f"{stage}@{bucket}"


def record_drop(root, stage, start_free_bytes, lowest_free_bytes, own_peak_bytes=None, frames=None):
    """Remember how far a stage pulled system free RAM down on this computer, per clip-length class.

    The drop, not the child's working set, is recorded: mapped model pages are
    counted in the working set yet stay reclaimable. It is capped by the child's
    own non-reclaimable peak so other programs growing meanwhile are not learned
    as the stage's need. Older highs decay slowly.
    """
    import json
    drop = start_free_bytes - lowest_free_bytes
    if own_peak_bytes is not None:
        drop = min(drop, own_peak_bytes)
    if not root or drop <= 0:
        return
    path = _profile_path(root)
    drops = learned_drops(root)
    key = drop_key(stage, frames)
    drops[key] = round(max(drop / GIB, drops.get(key, 0) * 0.9), 3)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(f".{os.getpid()}.tmp")
        pending.write_text(json.dumps({"schema": 1, "stage_free_ram_drop_gib": drops}, indent=2), encoding="utf-8")
        pending.replace(path)
    except OSError:
        pass


def learned_need_bytes(root, stage, frames=None):
    """Learned non-reclaimable need of a stage for this clip length (0 when never measured)."""
    drops = learned_drops(root) if root else {}
    return int((drops.get(drop_key(stage, frames)) or drops.get(stage) or 0) * GIB)


def start_requirement_gib(root, stage, limits, frames=None):
    """Free RAM at which a stage runs without paging (a recommendation for pacing and reports, not a gate).

    Needs learned for the same clip-length class are used first; records without a class are the fallback.
    """
    drops = learned_drops(root) if root else {}
    bucket = length_bucket(frames)
    if stage:
        drop = drops.get(drop_key(stage, frames)) or drops.get(stage)
    elif bucket:
        drop = max((v for k, v in drops.items() if k.endswith("@" + bucket)), default=None)
    else:
        drop = max((v for k, v in drops.items() if "@" not in k), default=None)
    if not drop:
        return limits["minimum_free_ram_gib"]
    return round(max(limits["minimum_free_ram_gib"], drop * 1.1 + limits["reserve_ram_gib"] * 0.75), 2)


def commit_bytes():
    """(commit limit, commit available): RAM plus page file that Windows can still promise.

    Running out of commit is what makes allocations fail. Low physical RAM alone only means paging:
    read-only mapped weights are dropped and re-read, so the work slows down instead of failing.
    """
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(status)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("commit 정보를 읽을 수 없습니다.")
    return status.page, status.page_available


def commit_floor(commit_limit):
    """Commit that must stay free for Windows and other programs: 4% of the limit, at least 0.75 GiB."""
    return max(int(0.75 * GIB), int(commit_limit * 0.04))


class ElasticGuard:
    """Use whatever memory is available instead of a fixed reservation.

    - Physical RAM is plentiful: run normally (`check` returns False).
    - Physical RAM is tight: keep going in low-memory mode (`check` returns True); the caller trims the
      stage's working set so mapped weights page out first. The stage slows down; it is not stopped.
    - Commit is about to run out (allocations would fail for every program): stop after a short grace.
    """
    def __init__(self, reserve_bytes, commit_floor_bytes, grace_seconds=15.0):
        self.reserve, self.commit_floor, self.grace = reserve_bytes, commit_floor_bytes, grace_seconds
        self.since, self.tight_events = None, 0

    def check(self, available, commit_available, now):
        if commit_available < self.commit_floor:
            if self.since is None:
                self.since = now
            if now - self.since > self.grace or commit_available < self.commit_floor / 2:
                raise MemoryError(f"commit 여유 {commit_available/GIB:.2f}GiB가 하한 {self.commit_floor/GIB:.2f}GiB보다 낮아 "
                                  "메모리 할당 실패 위험이 있습니다. 이 작업만 중지했습니다.")
        else:
            self.since = None
        tight = available < self.reserve
        self.tight_events += tight
        return tight


def wait_for_commit(needed_bytes, is_cancelled, progress, timeout=300.0, poll=2.0):
    """Wait only when the work cannot be promised memory at all; low physical RAM never blocks a start."""
    import time
    started = time.monotonic()
    while True:
        limit, available = commit_bytes()
        if available - needed_bytes >= commit_floor(limit):
            return available
        if is_cancelled():
            raise InterruptedError("메모리 대기 중 취소됐습니다.")
        if time.monotonic() - started > timeout:
            raise MemoryError(f"{timeout:.0f}초 동안 commit 여유가 {needed_bytes/GIB:.2f}GiB만큼 생기지 않았습니다. "
                              "다른 큰 프로그램을 닫거나 페이지 파일을 늘리세요.")
        progress({"stage": "waiting_for_memory", "commit_available_gib": round(available / GIB, 2),
                  "needed_commit_gib": round(needed_bytes / GIB, 2)})
        time.sleep(poll)


def trim_working_set(process_id):
    """Ask Windows to move a stage's pages to standby: mapped weights are released first and re-read later."""
    if os.name != "nt":
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x0400 | 0x0100, False, process_id)
    if not handle:
        return False
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        return bool(psapi.EmptyWorkingSet(handle))
    finally:
        kernel.CloseHandle(handle)


class LowMemoryMode:
    """Trim a stage at most every `interval` seconds while RAM is tight, and count how often that happened."""
    def __init__(self, process_id, interval=10.0, trim=None):
        self.process_id, self.interval, self.trim = process_id, interval, trim
        self.last, self.trims = None, 0

    def update(self, tight, now):
        if tight and (self.last is None or now - self.last >= self.interval):
            trim = self.trim or trim_working_set
            self.last, self.trims = now, self.trims + bool(trim(self.process_id))
        return tight


def run_unthrottled(handle=None):
    """Opt a process out of Windows power throttling (EcoQoS).

    Windowless below-normal processes are otherwise eligible for efficiency
    cores and low clocks, which slows every step without saving total energy.
    Priority stays below normal so the desktop remains responsive.
    """
    if os.name != "nt":
        return False
    class State(ctypes.Structure):
        _fields_ = [("version", ctypes.c_ulong), ("control", ctypes.c_ulong), ("state", ctypes.c_ulong)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    # PROCESS_POWER_THROTTLING_EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, state 0 = never throttle.
    state = State(1, 0x1 | 0x4, 0)
    target = ctypes.c_void_p(int(handle)) if handle else kernel.GetCurrentProcess()
    return bool(kernel.SetProcessInformation(target, 4, ctypes.byref(state), ctypes.sizeof(state)))


class CpuLimitMonitor:
    """Firmware clock limit from '% Performance Limit' (100 means clocks are not held back)."""
    COUNTERS = {"limit_pct": r"\Processor Information(_Total)\% Performance Limit",
                "performance_pct": r"\Processor Information(_Total)\% Processor Performance"}

    def __init__(self):
        self.query, self.counters, self.lowest, self.limited_seconds, self.last = None, {}, None, 0.0, None
        if os.name != "nt":
            return
        try:
            pdh = self.pdh = ctypes.WinDLL("pdh")
            query = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)):
                return
            self.query = query
            for name, path in self.COUNTERS.items():
                counter = ctypes.c_void_p()
                if not pdh.PdhAddEnglishCounterW(query, ctypes.c_wchar_p(path), None, ctypes.byref(counter)):
                    self.counters[name] = counter
            pdh.PdhCollectQueryData(query)
        except OSError:
            self.query = None

    def sample(self, now=None):
        if not self.query or not self.counters:
            return None
        class Value(ctypes.Structure):
            _fields_ = [("status", ctypes.c_ulong), ("pad", ctypes.c_ulong), ("value", ctypes.c_double)]
        if self.pdh.PdhCollectQueryData(self.query):
            return None
        result = {}
        for name, counter in self.counters.items():
            value = Value()
            if not self.pdh.PdhGetFormattedCounterValue(counter, 0x200, None, ctypes.byref(value)):
                result[name] = round(value.value, 1)
        limit = result.get("limit_pct")
        if limit is not None:
            self.lowest = limit if self.lowest is None else min(self.lowest, limit)
            if now is not None and self.last is not None and limit < 99:
                self.limited_seconds += now - self.last
        self.last = now
        return result or None

    def summary(self):
        return {"cpu_limit_lowest_pct": self.lowest, "cpu_limited_seconds": round(self.limited_seconds, 1)}

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def wait_for_cooling(monitor, is_cancelled, progress, threshold=60.0, timeout=90.0, poll=3.0):
    """Hold a new stage while firmware caps clocks hard, so heat does not compound into a thermal trip."""
    import time
    started = time.monotonic()
    while True:
        sample = monitor.sample(time.monotonic()) if monitor else None
        limit = (sample or {}).get("limit_pct")
        if limit is None or limit >= threshold or time.monotonic() - started > timeout:
            return limit
        if is_cancelled():
            raise InterruptedError("냉각 대기 중 취소됐습니다.")
        progress({"stage": "cooling_down", "cpu_limit_pct": limit, "resume_at_pct": threshold})
        time.sleep(poll)


def power_state():
    if os.name != "nt":
        return None
    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {"on_ac_power": status.ac == 1, "battery_saver": bool(status.saver & 1),
            "battery_percent": None if status.percent == 255 else status.percent}


def describe(cfg, profile, device_plan, root=None):
    limits = stage_limits(cfg, profile, device_plan)
    if root is not None:
        # Readiness only needs the profile floor; each stage checks its own learned need for the clip's
        # length right before it starts. Using the largest need of any stage and length here made a 2s
        # clip wait for RAM that only an 8s decode needs.
        limits["learned_stage_free_ram_drop_gib"] = learned_drops(root)
        limits["largest_learned_need_gib"] = start_requirement_gib(root, None, limits)
    device = (device_plan or {}).get("selected_device")
    total, available = limits["total_ram_gib"] * GIB, limits["free_ram_gib"] * GIB
    budget = gpu_buffer_bytes(device, total, available, cfg.get("gpu_buffer_mib")) if device_plan and device_plan.get("gpu_inference") and profile == "neodragon" else None
    return {**limits, "threads": cfg["threads"], "physical_cores": physical_cores(), "performance_cores": performance_cores(),
            "logical_processors": os.cpu_count(),
            "gpu_buffer_mib": budget // MIB if budget else None,
            "gpu_buffer_range_mib": [GPU_FLOOR // MIB, (GPU_SHARED_CEILING if (device or {}).get("shared_memory", True) else GPU_DISCRETE_CEILING) // MIB] if budget else None,
            "mode": {key: "auto" if is_auto(cfg.get(key)) else "manual" for key in RESOURCE_KEYS} | {"threads": "auto" if is_auto(cfg.get("threads_setting")) else "manual"},
            "power": power_state(), "power_throttling": "disabled_for_engine_processes",
            "policy": "recomputed_before_each_stage_gpu_buffer_shrinks_under_ram_pressure"}
