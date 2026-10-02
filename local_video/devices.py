"""Use enumerated Vulkan IDs; never assume adapter zero is the discrete GPU."""
from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess

BACKENDS = {"auto", "cpu", "gpu", "intel-vulkan", "intel-gpu"}


def parse_devices(output):
    unified = {}
    for index, uma in re.findall(r"^\s*(\d+)\s*=.*?\buma:\s*([01])", output, re.M):
        unified[int(index)] = uma == "1"
    devices = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(Vulkan\d+)\s+(.+?)\s*", line, re.I)
        if not match:
            continue
        device_id, name = match.groups()
        lower = name.lower()
        vendor = "intel" if "intel" in lower else "nvidia" if "nvidia" in lower else "amd" if any(x in lower for x in ["amd", "radeon"]) else "other"
        # Driver UMA metadata wins over brand-name heuristics (Arc may be integrated).
        shared = unified.get(int(re.search(r"\d+$", device_id)[0]),
                             vendor == "intel" or vendor == "other")
        devices.append({"id": device_id, "name": name, "vendor": vendor,
                        "shared_memory": shared, "kind": "integrated" if shared else "discrete"})
    return devices


def choose_device(devices, backend="auto", requested_id=None):
    if backend not in BACKENDS:
        raise ValueError("장치 선택: auto, gpu, intel-gpu, intel-vulkan, cpu")
    if backend == "cpu":
        if requested_id:
            raise ValueError("CPU 모드에는 GPU 번호를 지정하지 마세요.")
        return None
    candidates = [d for d in devices if backend not in {"intel-vulkan", "intel-gpu"} or d["vendor"] == "intel"]
    if requested_id:
        candidates = [d for d in candidates if d["id"].lower() == str(requested_id).lower()]
        if not candidates:
            raise RuntimeError("지정한 GPU가 감지되지 않았습니다. 실제 장치 목록을 확인하세요.")
    if not candidates:
        if backend == "auto":
            return None
        raise RuntimeError("요청한 GPU를 감지하지 못했습니다. GPU 강제 모드를 CPU로 바꿔 실행하지 않습니다.")
    return sorted(candidates, key=lambda d: (d["shared_memory"],
                   {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}[d["vendor"]],
                   int(re.search(r"\d+$", d["id"])[0])))[0]


def plan(runtime: Path, backend="auto", requested_id=None, budget_gib=None):
    if backend not in BACKENDS:
        raise ValueError("올바르지 않은 GPU 설정입니다.")
    devices, note = [], None
    binary = (runtime / "engines" / "vulkan" / "sd-cli.exe").resolve()
    if not binary.is_relative_to(runtime.resolve()):
        raise ValueError("GPU 엔진이 실행 폴더 밖으로 연결됩니다.")
    if backend != "cpu" and binary.is_file():
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            run = subprocess.run([str(binary), "--list-devices"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=30,
                                 creationflags=flags)
            if run.returncode:
                raise RuntimeError("Vulkan 장치 조회 실패")
            devices = parse_devices(run.stdout + "\n" + run.stderr)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            if backend != "auto":
                raise RuntimeError("GPU 엔진을 사용할 수 없습니다: " + str(exc)) from exc
            note = "GPU 조회 실패; CPU 선택: " + str(exc)
    selected = choose_device(devices, backend, requested_id)
    if selected is None:
        return {"backend_assignment": "cpu", "selected_device": None, "devices": devices,
                "gpu_inference": False, "fallback_reason": note or ("사용 가능한 Vulkan GPU가 없습니다." if backend == "auto" else None),
                "minimum_free_ram_gib": 8.0, "maximum_gpu_budget_gib": None}
    shared = selected["shared_memory"]
    budget = min(float(budget_gib or (1.5 if shared else 6.0)), 1.5 if shared else 6.0)
    return {"backend_assignment": f'diffusion={selected["id"]},te=cpu,vae=cpu',
            "selected_device": selected, "devices": devices, "gpu_inference": True,
            "fallback_reason": None, "minimum_free_ram_gib": 5.0 if shared else 6.0,
            "maximum_gpu_budget_gib": budget,
            "budget_scope": "managed_weights_and_runner_buffers_not_total_driver_memory",
            "policy": "shared_gpu_stream_weights_from_ssd" if shared else "discrete_gpu_diffusion_ssd_auxiliary"}
