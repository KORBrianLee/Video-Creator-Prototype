"""Headless stable-diffusion.cpp video inference and verified clip caching.

All frames come from temporal diffusion. No frame interpolation, image pan,
or still-image fallback is implemented. The process is deliberately isolated
so cancellation and memory protection stop only this application's engine.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
from typing import Callable

ENGINE_VERSION = "local-video-cpp-v3-gpu"
# One continuous generation per request; longer clips are never made by joining shorter ones.
DURATIONS = (2, 4, 8, 10, 15)
PRESETS = {
    "lightning": {
        "smoke": {"width": 256, "height": 256, "frames": 8, "steps": 4, "fps": 8, "vae_tile_size": "256x256"},
        "preview": {"width": 256, "height": 256, "frames": 16, "steps": 4, "fps": 8, "vae_tile_size": "256x256"},
        "quality": {"width": 384, "height": 384, "frames": 16, "steps": 4, "fps": 8, "vae_tile_size": "256x256"},
    },
    "wan": {
        "smoke": {"width": 320, "height": 192, "frames": 9, "steps": 20, "fps": 16, "vae_tile_size": "128x128"},
        "preview": {"width": 512, "height": 288, "frames": 33, "steps": 30, "fps": 16, "vae_tile_size": "128x128"},
        "quality": {"width": 832, "height": 480, "frames": 33, "steps": 50, "fps": 16, "vae_tile_size": "256x256"},
    },
}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _cancel(is_cancelled: Callable[[], bool]) -> None:
    if is_cancelled():
        raise InterruptedError("영상 생성이 취소됐습니다. 완료된 장면 캐시는 유지됩니다.")


def _path_in_runtime(path: Path, runtime: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(runtime):
        raise ValueError("모델·캐시·출력 경로는 선택한 실행 폴더 안이어야 합니다.")
    return path


def _memory(process_id: int) -> tuple[int, int]:
    """Return owned process working set and system available physical bytes."""
    if os.name != "nt":
        return 0, 2**63 - 1
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("page", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                    ("virtual", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t),
                    ("peak_page", ctypes.c_size_t), ("page", ctypes.c_size_t),
                    ("peak_nonpage", ctypes.c_size_t), ("nonpage", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("시스템 여유 메모리를 읽을 수 없습니다.")
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, process_id)
    if not handle:
        # The process may have finished between poll() and this query.
        return 0, status.available
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError("추론 프로세스 메모리를 읽을 수 없습니다.")
        return counters.working, status.available
    finally:
        kernel.CloseHandle(handle)


def _commit(process_id: int) -> int:
    """Private commit charge (what a Windows Job memory limit counts), 0 when unavailable."""
    if os.name != "nt":
        return 0
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t),
                    ("peak_page", ctypes.c_size_t), ("page", ctypes.c_size_t),
                    ("peak_nonpage", ctypes.c_size_t), ("nonpage", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, process_id)
    if not handle:
        return 0
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        return counters.pagefile if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else 0
    finally:
        kernel.CloseHandle(handle)


def _reclaimable(process_id: int, working: int) -> int:
    """Part of the working set Windows can drop without pagefile writes.

    Memory-mapped model pages are file-backed: under pressure they are discarded
    and re-read, unlike private memory, which would be written to the pagefile.
    The private working set (shared pages excluded) is the exact measure; commit
    charge is wrong here because copy-on-write mappings are charged in full.
    """
    if os.name != "nt" or working <= 0:
        return 0
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t),
                    ("peak_page", ctypes.c_size_t), ("page", ctypes.c_size_t),
                    ("peak_nonpage", ctypes.c_size_t), ("nonpage", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
                    ("private_usage", ctypes.c_size_t), ("private_working", ctypes.c_ulonglong),
                    ("shared_commit", ctypes.c_ulonglong)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, process_id)
    if not handle:
        return 0
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) and counters.private_working:
            return max(0, working - counters.private_working)
        legacy = Counters()
        legacy.cb = Counters.pagefile.offset + Counters.pagefile.size + ctypes.sizeof(ctypes.c_size_t)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(legacy), legacy.cb):
            return 0
        return max(0, working - legacy.pagefile)
    finally:
        kernel.CloseHandle(handle)


def _stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _engine(request: dict, runtime: Path) -> tuple[Path, str, str, str]:
    backend = request.get("backend", "cpu")
    from .devices import plan
    saved = request.get("device_plan", {})
    selected_id = (saved.get("selected_device") or {}).get("id")
    selected = plan(runtime, backend, selected_id, saved.get("maximum_gpu_budget_gib"))
    if saved and selected["backend_assignment"] != saved.get("backend_assignment"):
        raise RuntimeError("요청 이후 사용 가능한 GPU가 바뀌었습니다. 새 작업을 요청하세요.")
    request["device_plan"] = selected
    binary = _path_in_runtime(runtime / "engines" / ("vulkan" if selected["gpu_inference"] else "cpu") / "sd-cli.exe", runtime)
    if not binary.is_file():
        raise FileNotFoundError("로컬 영상 엔진이 없습니다. 설치 상태를 확인하세요.")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    help_run = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30, creationflags=flags)
    if help_run.returncode:
        raise RuntimeError("로컬 영상 엔진의 지원 옵션을 확인할 수 없습니다.")
    help_text = help_run.stdout + help_run.stderr
    assignment = selected["backend_assignment"]
    fingerprint = hashlib.sha256()
    for candidate in [binary, binary.with_name("stable-diffusion.dll")]:
        file = _path_in_runtime(candidate, runtime)
        if file.is_file():
            with file.open("rb") as stream:
                fingerprint.update(hashlib.file_digest(stream, "sha256").digest())
    return binary, assignment, help_text, fingerprint.hexdigest()


def _model_paths(spec: dict, model_dir: Path, profile: str,
                 is_cancelled: Callable = lambda: False, progress: Callable = lambda update: None) -> dict[str, Path]:
    model_dir = Path(model_dir).resolve()
    required = {"base", "motion"} if profile == "lightning" else {"diffusion", "text_encoder", "vae"}
    paths = {}
    items = list(spec.get("files", []))
    prepared = spec.get("prepared_base") if profile == "lightning" else None
    if prepared is not None:
        if not isinstance(prepared, dict):
            raise ValueError("파생 Lightning 모델 잠금 자료가 올바르지 않습니다.")
        original_base = next((item for item in items if item.get("role") == "base"), {})
        if prepared.get("input_sha256") != original_base.get("sha256") or prepared.get("preparation_version") != "lightning-linear-alphas-gguf-v1":
            raise ValueError("파생 Lightning 모델이 현재 원본·준비 버전과 일치하지 않습니다.")
        # Preparation independently proves unchanged learned tensor payload.
        # A Gram can carry just the verified derivative and motion module.
        items = [{**item, **prepared, "role": "base"} if item.get("role") == "base" else item for item in items]
    for item in items:
        role, name = item.get("role"), item.get("filename", "")
        if role not in required:
            continue
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() not in {".gguf", ".safetensors"}:
            raise ValueError("허용되지 않는 모델 파일 경로입니다.")
        file = (model_dir / name).resolve()
        if not file.is_relative_to(model_dir) or not file.is_file():
            raise FileNotFoundError(f"모델 파일이 없습니다: {name}")
        info = file.stat()
        if item.get("size") is not None and info.st_size != item["size"]:
            raise ValueError(f"모델 파일 크기가 잠금 자료와 다릅니다: {name}")
        expected_hash = item.get("sha256", "")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise ValueError(f"모델 파일 SHA256 잠금 값이 없습니다: {name}")
        marker_path = model_dir / (name + ".verified.json")
        if not marker_path.resolve().is_relative_to(model_dir):
            raise ValueError("모델 검증 기록 경로가 실행 폴더 밖으로 연결됩니다.")
        fingerprint = {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
                       "ctime_ns": info.st_ctime_ns, "file_id": info.st_ino, "expected_sha256": expected_hash}
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        if marker.get("fingerprint") != fingerprint or marker.get("verified_sha256") != expected_hash:
            progress({"stage": "verifying_model", "model_role": role, "model_filename": name})
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                while data := stream.read(4*1024*1024):
                    _cancel(is_cancelled)
                    digest.update(data)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"모델 SHA256 검증 실패: {name}. 재다운로드가 필요합니다.")
            # A file changed while hashing must never receive a verified marker.
            after = file.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino):
                raise ValueError(f"검증 중 모델 파일이 변경됐습니다: {name}")
            _write_json(marker_path, {"fingerprint": fingerprint, "verified_sha256": expected_hash,
                                      "verification": "sha256", "verified_at": time.time()})
        _cancel(is_cancelled)
        paths[role] = file
    if paths.keys() != required:
        raise ValueError("선택한 영상 모델의 구성 파일이 불완전합니다.")
    return paths


def build_command(binary: Path, assignment: str, profile: str, paths: dict,
                  settings: dict, scene: dict, work: Path, threads: int, sampling: dict) -> list[str]:
    """An argv list only; user prompts never become shell commands."""
    command = [str(binary), "--mode", "vid_gen", "--backend", assignment,
               "--params-backend", "disk", "--mmap", "--disable-prefetch",
               "--conditioning-cache-size", "0", "--threads", str(threads),
               "--prompt-file", str(work / "prompt.txt"),
               "--negative-prompt-file", str(work / "negative.txt"),
               "--width", str(settings["width"]), "--height", str(settings["height"]),
               "--video-frames", str(settings["frames"]), "--fps", str(settings["fps"]),
               "--steps", str(settings["steps"]), "--seed", str(scene["seed"]),
               "--vae-tiling", "--vae-tile-size", settings.get("vae_tile_size", "256x256" if profile == "lightning" else "128x128"), "--diffusion-fa",
               "--output", str(work / "frame_%03d.png"), "--output-begin-idx", "0"]
    if profile == "lightning":
        sigmas = sampling.get("sigmas")
        if not isinstance(sigmas, list) or len(sigmas) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in sigmas):
            raise ValueError("Lightning 4단계 sampler sigma 설정이 없습니다.")
        if not all(a > b for a, b in zip(sigmas, sigmas[1:])) or sigmas[-1] != 0:
            raise ValueError("Lightning sigma는 내림차순이며 마지막 값이 0이어야 합니다.")
        command += ["--model", str(paths["base"]), "--motion-module", str(paths["motion"]),
                    "--cfg-scale", "1", "--sampling-method", "euler", "--sigmas", ",".join(format(v, ".15g") for v in sigmas)]
    else:
        command += ["--diffusion-model", str(paths["diffusion"]), "--t5xxl", str(paths["text_encoder"]),
                    "--vae", str(paths["vae"]), "--cfg-scale", "6", "--sampling-method", "euler", "--flow-shift", "3.0",
                    "--temporal-tiling", "--extra-tiling-args", "temporal_tile_frames=4,temporal_tile_overlap=1"]
    return command


def _validate_flags(command: list[str], help_text: str) -> None:
    supported = set(re.findall(r"(?<!\w)--[a-z][a-z0-9_-]*", help_text))
    missing = [flag for flag in command[1:] if flag.startswith("--") and flag not in supported]
    if missing:
        raise RuntimeError("설치된 엔진에 필수 옵션이 없습니다: " + ", ".join(missing))


def _infer(command: list[str], work: Path, request: dict, progress: Callable,
           is_cancelled: Callable, scene_index: int, scene_count: int, steps: int,
           stage_name: str = "inference") -> dict:
    _cancel(is_cancelled)
    env = os.environ.copy()
    runtime = Path(request["_runtime_dir"])
    temporary_root = _path_in_runtime(runtime / "tmp", runtime.resolve())
    env.update(TEMP=str(temporary_root), TMP=str(temporary_root), TMPDIR=str(temporary_root))
    temporary_root.mkdir(parents=True, exist_ok=True)
    from .resources import (CpuLimitMonitor, ElasticGuard, LowMemoryMode, commit_bytes, commit_floor, is_auto,
                            learned_need_bytes, record_drop, run_unthrottled, stage_limits, wait_for_commit,
                            wait_for_cooling)
    limits = stage_limits(request)
    manual_cap = request.get("maximum_working_set_gib")
    cap = None if is_auto(manual_cap) else float(manual_cap) * 2**30
    reserve = limits["reserve_ram_gib"] * 2**30
    # Elastic memory: no free-RAM gate. Commit must cover the learned need; tight RAM only slows the engine.
    wait_for_commit(learned_need_bytes(runtime, stage_name), is_cancelled, progress)
    guard = ElasticGuard(reserve, commit_floor(commit_bytes()[0]))
    _, start_free = _memory(os.getpid())
    lowest_free, own_peak = start_free, 0
    timeout = float(request.get("maximum_scene_seconds", 7200 if request.get("model_profile") == "wan" else 3600))
    if not all(math.isfinite(v) and v > 0 for v in (reserve, timeout)) or (cap is not None and not cap > 0):
        raise ValueError("메모리와 시간 보호 기준은 유한한 양수여야 합니다.")
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
    monitor = CpuLimitMonitor()
    wait_for_cooling(monitor, is_cancelled, progress)
    started, peak, last_progress, position, stage = time.monotonic(), 0, 0.0, 0, "loading"
    cpu = None
    log_path = work / "inference.log"
    with log_path.open("wb", buffering=0) as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                   cwd=work, env=env, shell=False, creationflags=flags)
        run_unthrottled(getattr(process, "_handle", None))
        low_memory = LowMemoryMode(process.pid)
        try:
            while process.poll() is None:
                _cancel(is_cancelled)
                working, available = _memory(process.pid)
                reclaimable = _reclaimable(process.pid, working)
                peak = max(peak, working)
                own_peak = max(own_peak, working - reclaimable)
                lowest_free = min(lowest_free, available + reclaimable)
                if cap is not None and working - reclaimable > cap:
                    raise MemoryError(f"추론 메모리 {working/2**30:.2f}GB가 사용자 지정 상한 {cap/2**30:.2f}GB를 초과했습니다.")
                now = time.monotonic()
                pressured = low_memory.update(guard.check(available + reclaimable, commit_bytes()[1], now), now)
                if now - started > timeout:
                    raise TimeoutError("장면 생성 시간 보호 기준을 넘었습니다. 완료된 장면 캐시는 유지됩니다.")
                if now - last_progress >= 2:
                    cpu = monitor.sample(now) or cpu
                    with log_path.open("rb") as read_log:
                        read_log.seek(position)
                        tail = read_log.read(65536).decode("utf-8", errors="replace")
                        position = read_log.tell()
                    matches = re.findall(r"(?:^|[\r\n\s])(\d+)\s*/\s*(\d+)\s*[-|\[]", tail)
                    if matches:
                        stage = "sampling"
                    if re.search(r"decoding\s+\d+\s+latents|decode_first_stage|latent\s+\d+\s+decoded|VAE\s+decoding", tail, re.I):
                        stage = "decoding"
                    update = {"stage": stage, "scene_index": scene_index, "scene_count": scene_count,
                              "scene_elapsed_seconds": round(now-started, 1), "engine_pid": process.pid,
                              "working_set_gib": round(working/2**30, 3), "peak_working_set_gib": round(peak/2**30, 3),
                              "free_ram_gib": round(available/2**30, 3), "reclaimable_mapped_gib": round(reclaimable/2**30, 3),
                              "memory_pressure": pressured, "cpu_limit_pct": (cpu or {}).get("limit_pct"),
                              "inference_log_path": str(log_path)}
                    if matches and int(matches[-1][1]) == steps:
                        update.update(step=int(matches[-1][0]), steps=steps)
                    progress(update)
                    last_progress = now
                time.sleep(0.25)
            if process.returncode:
                tail = log_path.read_bytes()[-6000:].decode("utf-8", errors="replace")
                raise RuntimeError(f"로컬 영상 엔진 종료 코드 {process.returncode}. 기록: {log_path}\n{tail}")
            _cancel(is_cancelled)
        finally:
            _stop_owned(process)
            record_drop(runtime, stage_name, start_free, lowest_free, own_peak)
            monitor.close()
    return {"inference_seconds": round(time.monotonic()-started, 3),
            "peak_working_set_gib": round(peak/2**30, 3), "inference_log_path": str(log_path), **monitor.summary()}


# Model licenses (LTX-Video and Open RAIL-M Attachment A (e)) require disclosing that shared output is machine generated.
AI_NOTICE = "AI-generated (machine generated) video. Not real footage."


def _writer(path: Path, settings: dict):
    import imageio_ffmpeg
    return imageio_ffmpeg.write_frames(str(path), (settings["width"], settings["height"]),
        fps=settings["fps"], codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=1, ffmpeg_log_level="error", ffmpeg_timeout=5,
        output_params=["-threads", "2", "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart",
                       "-metadata", f"comment={AI_NOTICE}", "-metadata", "description=" + AI_NOTICE])


def _encode_frames(work: Path, video: Path, settings: dict, is_cancelled: Callable) -> None:
    from PIL import Image
    frames = sorted(work.glob("frame_*.png"))
    if len(frames) != settings["frames"]:
        raise RuntimeError(f"모델 출력 프레임 수가 다릅니다: {len(frames)} / {settings['frames']}")
    writer = _writer(video, settings)
    writer.send(None)
    try:
        for index, file in enumerate(frames):
            _cancel(is_cancelled)
            if file.name != f"frame_{index:03d}.png":
                raise RuntimeError("모델 출력 프레임 번호가 연속하지 않습니다.")
            with Image.open(file) as frame:
                if frame.size != (settings["width"], settings["height"]):
                    raise RuntimeError("모델 출력 프레임 크기가 요청과 다릅니다.")
                writer.send(frame.convert("RGB").tobytes())
    finally:
        writer.close()


def verify_video(video: Path, settings: dict, expected_frames: int | None = None,
                 is_cancelled: Callable = lambda: False) -> dict:
    """Full decoding detects corruption and static output; it cannot judge realism."""
    import imageio_ffmpeg
    from PIL import Image
    expected = expected_frames or settings["frames"]
    _cancel(is_cancelled)
    # Validate the container before imageio opens streaming pipes. Its reader
    # leaves already-exited pipes open when a broken container has no header.
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    preflight = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
                               "-i", str(video), "-map", "0:v:0", "-frames:v", "0", "-f", "null", "-"],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=30, creationflags=flags)
    if preflight.returncode:
        raise RuntimeError("영상 컨테이너가 손상됐거나 해독할 수 없습니다.")
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", input_params=["-threads", "2"])
    try:
        metadata = next(reader)
        size = tuple(metadata["size"])
        if size != (settings["width"], settings["height"]):
            raise RuntimeError("검증 영상의 해상도가 요청과 다릅니다.")
        fps = float(metadata["fps"])
        if abs(fps-settings["fps"]) > 0.01:
            raise RuntimeError("검증 영상의 FPS가 요청과 다릅니다.")
        count, previous, total_delta, maximum_delta, changed, minimum, maximum = 0, None, 0.0, 0.0, 0, 255, 0
        deltas, brightness = [], []
        for raw in reader:
            _cancel(is_cancelled)
            if len(raw) != size[0]*size[1]*3:
                raise RuntimeError("영상 프레임을 완전히 해독하지 못했습니다.")
            thumb = Image.frombytes("RGB", size, raw).convert("L").resize((48, 48)).tobytes()
            brightness.append(sum(thumb)/len(thumb))
            minimum, maximum = min(minimum, min(thumb)), max(maximum, max(thumb))
            if previous is not None:
                delta = sum(abs(a-b) for a, b in zip(previous, thumb))/len(thumb)
                total_delta += delta
                maximum_delta = max(maximum_delta, delta)
                changed += delta > 0.02
                deltas.append(delta)
            previous = thumb
            count += 1
        if count != expected:
            raise RuntimeError(f"영상 해독 프레임 수가 요청과 다릅니다: {count} / {expected}")
        if count < 2 or changed == 0:
            raise RuntimeError("생성 영상에 측정 가능한 프레임 변화가 없습니다. 성공 결과로 보관하지 않습니다.")
        from .temporal_quality import analyze
        return {"decoded": True, "frame_count": count, "fps": fps, "width": size[0], "height": size[1],
                "continuity": analyze(deltas, brightness),
                "duration_seconds": round(count/fps, 4), "nonstatic": True,
                "mean_adjacent_frame_luma_difference": round(total_delta/max(count-1, 1), 5),
                "maximum_adjacent_frame_luma_difference": round(maximum_delta, 5),
                "changed_frame_pairs": changed, "luma_range": [minimum, maximum],
                "quality_judgment": "human_review_required_motion_signal_is_not_realism_or_quality_proof"}
    finally:
        reader.close()


def _cache_key(request: dict, scene: dict, settings: dict, engine_identity: str, assignment: str) -> str:
    value = {"schema": ENGINE_VERSION, "model_revision": request["model_revision"],
             "profile": request["model_profile"], "engine_identity": engine_identity, "backend": assignment,
             "prompt": scene["prompt"], "negative_prompt": scene.get("negative_prompt", ""),
             "seed": scene["seed"], "preset": request["preset"], "settings": settings,
             "sampling": request["_model_spec"].get("sampling", {})}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cached(cache: Path, key: str, settings: dict, is_cancelled: Callable) -> dict | None:
    try:
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        video = cache / "clip.mp4"
        if metadata.get("cache_key") != key or metadata.get("status") != "complete":
            return None
        with video.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != metadata.get("video_sha256"):
                return None
        metadata["verification"] = verify_video(video, settings, is_cancelled=is_cancelled)
        return metadata
    except InterruptedError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def _assemble(shots: list[dict], destination: Path, sheet_path: Path, settings: dict, is_cancelled: Callable) -> None:
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    width = 256
    thumb_height = max(1, round(settings["height"]*width/settings["width"]))
    tile_height = thumb_height + 24
    sheet = Image.new("RGB", (width*4, tile_height*len(shots)), "#171a21")
    draw = ImageDraw.Draw(sheet)
    _cancel(is_cancelled)
    # A single completed shot is already encoded at the requested settings.
    # Preserve its bytes instead of adding a second lossy compression pass.
    single = len(shots) == 1
    if single:
        shutil.copy2(shots[0]["video_path"], destination)
        writer = None
    else:
        writer = _writer(destination, settings)
        writer.send(None)
    try:
        for row, shot in enumerate(shots):
            reader = imageio_ffmpeg.read_frames(shot["video_path"], pix_fmt="rgb24", input_params=["-threads", "2"])
            try:
                metadata = next(reader)
                size = tuple(metadata["size"])
                selection = [round((settings["frames"]-1)*i/3) for i in range(4)]
                for index, raw in enumerate(reader):
                    _cancel(is_cancelled)
                    if writer is not None:
                        writer.send(raw)
                    if index in selection:
                        frame = Image.frombytes("RGB", size, raw).resize((width, thumb_height))
                        column = selection.index(index)
                        sheet.paste(frame, (column*width, row*tile_height))
                        draw.text((column*width+6, row*tile_height+thumb_height+4), f"{shot['id']} frame {index} seed {shot['seed']}", fill="white")
            finally:
                reader.close()
    finally:
        if writer is not None:
            writer.close()
    sheet.save(sheet_path, format="JPEG", quality=90)


def generate_project(request: dict, output_dir: Path, cache_dir: Path, model_dir: Path,
                     progress: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> dict:
    started = time.monotonic()
    _cancel(is_cancelled)
    runtime = Path(request["_runtime_dir"]).resolve()
    if os.name == "nt":
        from .storage import runtime_root
        runtime = runtime_root(runtime)
    output_dir, cache_dir, model_dir = [_path_in_runtime(Path(p), runtime) for p in (output_dir, cache_dir, model_dir)]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile = request.get("model_profile", "lightning")
    settings = dict(PRESETS[profile][request["preset"]])
    if profile == "wan" and request.get("continuous"):
        settings["frames"] = int(request["duration_seconds"]) * settings["fps"] + 1
    scenes = request["scenes"]
    if not scenes:
        raise ValueError("생성할 장면이 없습니다.")
    binary, assignment, help_text, engine_identity = _engine(request, runtime)
    paths = _model_paths(request["_model_spec"], model_dir, profile, is_cancelled, progress)
    shots = []
    for index, original in enumerate(scenes):
        _cancel(is_cancelled)
        scene = dict(original)
        scene.setdefault("seed", (request.get("seed", 42)+index) % 2**31)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", scene["id"]):
            raise ValueError("잘못된 장면 ID입니다.")
        key = _cache_key(request, scene, settings, engine_identity, assignment)
        cache = _path_in_runtime(cache_dir / key, runtime)
        for candidate in [cache / "metadata.json", cache / "clip.mp4", cache / "inference.log", cache / "command.json"]:
            _path_in_runtime(candidate, runtime)
        hit = _cached(cache, key, settings, is_cancelled)
        progress({"stage": "cache_reuse" if hit else "scene_start", "scene_index": index+1,
                  "scene_count": len(scenes), "scene_id": scene["id"], "steps": settings["steps"], "step": 0})
        if hit is None:
            # Unsuccessful attempts remain separate and never invalidate a prior success.
            work = _path_in_runtime(runtime / "tmp" / ("scene-" + key[:16] + "-" + uuid.uuid4().hex[:8]), runtime)
            work.mkdir(parents=True)
            (work / "prompt.txt").write_text(scene["prompt"], encoding="utf-8")
            (work / "negative.txt").write_text(scene.get("negative_prompt", ""), encoding="utf-8")
            command = build_command(binary, assignment, profile, paths, settings, scene, work,
                                    max(1, min(int(request.get("threads", 4)), 8)), request["_model_spec"].get("sampling", {}))
            selected = request["device_plan"]
            if selected["gpu_inference"]:
                if selected["policy"] == "discrete_gpu_diffusion_ssd_auxiliary":
                    position = command.index("--params-backend")
                    device = selected["selected_device"]["id"]
                    command[position+1] = f"diffusion={device},te=disk,vae=disk"
                command += ["--max-vram", str(selected["maximum_gpu_budget_gib"])]
            _validate_flags(command, help_text)
            _write_json(work / "command.json", {"argv": command, "engine_identity": engine_identity})
            infer_result = _infer(command, work, request, progress, is_cancelled, index+1, len(scenes), settings["steps"])
            progress({"stage": "encoding", "scene_index": index+1, "scene_count": len(scenes)})
            video = work / "clip.mp4"
            _encode_frames(work, video, settings, is_cancelled)
            verification = verify_video(video, settings, is_cancelled=is_cancelled)
            _cancel(is_cancelled)
            with video.open("rb") as stream:
                video_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / "inference.log", cache / "inference.log")
            shutil.copy2(work / "command.json", cache / "command.json")
            os.replace(video, cache / "clip.mp4")
            hit = {"status": "complete", "cache_key": key, "video_sha256": video_hash,
                   "prompt": scene["prompt"], "negative_prompt": scene.get("negative_prompt", ""), "seed": scene["seed"],
                   "model_revision": request["model_revision"], "model_profile": profile, "settings": settings,
                   "engine_identity": engine_identity, "backend_assignment": assignment,
                   "negative_prompt_effective": profile != "lightning", "verification": verification,
                   **infer_result, "inference_log_path": str(cache / "inference.log")}
            _write_json(cache / "metadata.json", hit)
            # Delete only this attempt's generated frames, after a verified cache commit.
            shutil.rmtree(work)
            reused = False
        else:
            reused = True
        shot = {**hit, "id": scene["id"], "video_path": str((cache / "clip.mp4").resolve()), "cache_hit": reused}
        shots.append(shot)
        progress({"stage": "scene_complete", "scene_index": index+1, "scene_count": len(scenes), "scene_id": scene["id"], "cache_hit": reused})
    _cancel(is_cancelled)
    progress({"stage": "assembling", "scene_index": len(scenes), "scene_count": len(scenes)})
    video = _path_in_runtime(output_dir / "video.mp4", runtime)
    sheet = _path_in_runtime(output_dir / "contact-sheet.jpg", runtime)
    pending = _path_in_runtime(output_dir / ("video-" + uuid.uuid4().hex[:8] + ".mp4"), runtime)
    _assemble(shots, pending, sheet, settings, is_cancelled)
    verification = verify_video(pending, settings, expected_frames=len(shots)*settings["frames"], is_cancelled=is_cancelled)
    _cancel(is_cancelled)
    os.replace(pending, video)
    with video.open("rb") as stream:
        video_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"video_path": str(video.resolve()), "video_sha256": video_hash,
              "contact_sheet_path": str(sheet.resolve()), "shots": shots,
              "verification": verification, "elapsed_seconds": round(time.monotonic()-started, 3),
              "settings": settings, "model_profile": profile, "model_revision": request["model_revision"],
              "engine_version": ENGINE_VERSION, "engine_identity": engine_identity, "backend_assignment": assignment,
              "device_plan": request["device_plan"], "gpu_inference": request["device_plan"]["gpu_inference"],
              "continuous": bool(request.get("continuous")), "edit_count": max(0, len(shots)-1),
              "generation_method": "temporal_diffusion_generated_frames_no_interpolation",
              "target_iris_generation_verified": False, "visual_review": "required",
              "negative_prompt_effective": profile != "lightning",
              "negative_prompt_note": "Lightning CFG=1에서는 제외 프롬프트가 실제 생성에 영향을 주지 않습니다." if profile == "lightning" else "제외 프롬프트 사용",
              "quality_note": "프레임 해독과 변화 검증만 완료했습니다. 동작 정확성·사실감·프롬프트 준수는 직접 검토해야 합니다."}
    _write_json(_path_in_runtime(output_dir / "result.json", runtime), result)
    return result
