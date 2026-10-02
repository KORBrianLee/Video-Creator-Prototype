"""Portable Neo CPU video backend with one model phase resident at a time.

This controller uses stdlib only. Torch runs in owned, cancellable subprocesses
and generated motion comes from the vendor's autoregressive video transformer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from . import backend as media

STAGES = ("video_text", "video_encode",
          "video_pack", "video_infer", "video_decode", "safety")
PRESETS = {
    "smoke": {"width": 320, "height": 192, "frames": 9, "fps": 24, "steps": 3},
    "preview": {"width": 384, "height": 256, "frames": 49, "fps": 24, "steps": 3},
    "quality": {"width": 512, "height": 320, "frames": 49, "fps": 24, "steps": 3},
}
ENGINE_VERSION = "neodragon-staged-v2-opencl-stream"


def runtime_errors(app, root):
    """Inspect package metadata without importing Torch into Cursor's server."""
    site = root / "runtime" / "neodragon-python" / "Lib" / "site-packages"
    lock = json.loads((app / "neodragon-runtime.lock.json").read_text(encoding="utf-8"))
    normal = lambda name: name.lower().replace("_", "-").replace(".", "-")
    installed = {}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name = dist.metadata.get("Name")
        if isinstance(name, str) and name:
            installed[normal(name)] = dist.version
    return [f'CPU 라이브러리 미준비: {item["name"]}=={item["version"]}'
            for item in lock["packages"] if installed.get(normal(item["name"])) != item["version"]]


def pipeline_revision(app):
    digest = hashlib.sha256()
    for relative in ("local_video/neodragon.py", "local_video/neodragon_stage.py",
                     "local_video/neodragon_quant.py", "local_video/backend.py",
                     "local_video/conditioning.py", "local_video/prepared.py",
                     "local_video/temporal_quality.py", "local_video/opencl_linear.py",
                     "local_video/devices.py", "local_video/storage.py"):
        path = app / relative
        if not path.is_file():
            return "missing_backend"
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(root, threads):
    return {**os.environ, "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
            "HF_HOME": str(root / "cache" / "huggingface"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TORCH_HOME": str(root / "cache" / "torch"),
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "false"}


def first_frame_key(scene, settings, identity, model_sha256):
    value = {"version": "sd15-q4-euler20-fa-v2", "engine": identity, "model_sha256": model_sha256,
             "prompt": scene.get("first_frame_prompt", scene["prompt"]), "negative_prompt": scene.get("negative_prompt", ""),
             "seed": scene["seed"], "width": settings["width"], "height": settings["height"],
             "steps": 20, "cfg": 7, "sampler": "euler"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def commit_first_frame(root, cache, source, settings, record):
    from PIL import Image
    source = media._path_in_runtime(source, root)
    cache = media._path_in_runtime(cache, root)
    cache.mkdir(parents=True, exist_ok=True)
    destination = media._path_in_runtime(cache / "image.png", root)
    with Image.open(source) as image:
        image.load()
        if image.size != (settings["width"], settings["height"]):
            raise RuntimeError("첫 장면 해상도가 요청과 다릅니다.")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    shutil.copy2(source, destination)
    media._write_json(cache / "metadata.json", {**record, "image_sha256": digest})


def first_frame(root, work, scene, settings, request, progress, is_cancelled):
    """Generate one locally conditioned input image, never a still-video fallback."""
    from PIL import Image
    if scene.get("image_path"):
        source = media._path_in_runtime(Path(scene["image_path"]), root)
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != scene["image_sha256"]:
                raise RuntimeError("참고 이미지가 요청 후 변경됐습니다.")
        with Image.open(source) as image:
            image.convert("RGB").resize((settings["width"], settings["height"])).save(work / "first-frame.png")
        return {"name": "first_frame", "source": "provided_image", "elapsed_seconds": 0,
                "state": "completed", "peak_working_set_bytes": 0}
    if (root / "engines" / "vulkan" / "sd-cli.exe").is_file() and request.get("backend", "auto") != "cpu":
        try:
            return _first_frame_attempt(root, work, scene, settings, request, progress, is_cancelled, True)
        except (MemoryError, InterruptedError, TimeoutError):
            raise
        except Exception as exc:
            progress({"stage": "first_frame_gpu_unavailable", "reason": str(exc)[:300]})
    return _first_frame_attempt(root, work, scene, settings, request, progress, is_cancelled, False)


def _first_frame_attempt(root, work, scene, settings, request, progress, is_cancelled, gpu):
    """Vulkan GPU first frame when the GPU engine works, otherwise the CPU engine with the same model."""
    from PIL import Image
    binary, assignment, help_text, identity = media._engine({**request, "backend": "auto" if gpu else "cpu", "device_plan": {}}, root)
    if gpu != (assignment != "cpu"):
        raise RuntimeError("Vulkan GPU를 사용할 수 없어 CPU 첫 장면 엔진으로 전환합니다." if gpu else "첫 장면 CPU 엔진 선택이 올바르지 않습니다.")
    model = media._path_in_runtime(root / "models" / "stable-diffusion-v1-5-Q4_0.gguf", root)
    item = next(f for f in request["_model_spec"]["files"] if f["filename"] == model.name)
    key = first_frame_key(scene, settings, identity, item["sha256"])
    cache = media._path_in_runtime(root / "cache" / "first-frames" / key, root)
    try:
        image = media._path_in_runtime(cache / "image.png", root)
        metadata_path = media._path_in_runtime(cache / "metadata.json", root)
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        with image.open("rb") as stream:
            valid = record.get("cache_key") == key and hashlib.file_digest(stream, "sha256").hexdigest() == record.get("image_sha256")
        if valid:
            with Image.open(image) as frame:
                frame.load()
                valid = frame.size == (settings["width"], settings["height"])
        if valid:
            shutil.copy2(image, work / "first-frame.png")
            progress({"stage": "first_frame_cache_reuse"})
            return {"name": "first_frame", "source": "local_sd15_q4", "state": "completed",
                    "elapsed_seconds": 0, "peak_working_set_bytes": 0, "engine_identity": identity,
                    "cache_hit": True, "cache_key": key}
    except (OSError, ValueError, KeyError):
        pass
    (work / "prompt.txt").write_text(scene.get("first_frame_prompt", scene["prompt"]), encoding="utf-8")
    (work / "negative.txt").write_text(scene.get("negative_prompt", ""), encoding="utf-8")
    command = [str(binary), "--mode", "img_gen", "--backend", "vulkan" if gpu else "cpu", "--params-backend", "disk",
               "--mmap", "--disable-prefetch", "--conditioning-cache-size", "0", "--model", str(model),
               "--prompt-file", str(work / "prompt.txt"), "--negative-prompt-file", str(work / "negative.txt"),
               "--width", str(settings["width"]), "--height", str(settings["height"]),
               "--steps", "20", "--cfg-scale", "7", "--sampling-method", "euler", "--seed", str(scene["seed"]),
               "--threads", str(request["threads"]), "--output", str(work / "first-frame.png"), "--diffusion-fa"]
    if not gpu:
        command += ["--clip-on-cpu", "--vae-on-cpu", "--vae-tiling", "--vae-tile-size", "256x256"]
    media._validate_flags(command, help_text)
    media._write_json(work / "first-frame-command.json", {"argv": command, "engine_identity": identity})
    result = media._infer(command, work, request, progress, is_cancelled, 1, 1, 20,
                          stage_name="first_frame_gpu" if gpu else "first_frame")
    with Image.open(work / "first-frame.png") as image:
        if image.size != (settings["width"], settings["height"]):
            raise RuntimeError("첫 장면 해상도가 요청과 다릅니다.")
    media._cancel(is_cancelled)
    commit_first_frame(root, cache, work / "first-frame.png", settings,
                       {"cache_key": key, "engine_identity": identity, "source": "local_sd15_q4"})
    return {"name": "first_frame", "source": "local_sd15_q4", "state": "completed",
            "elapsed_seconds": result["inference_seconds"],
            "peak_working_set_bytes": round(result["peak_working_set_gib"]*2**30), "engine_identity": identity}


def run_stage(root, work, name, request, progress, is_cancelled):
    media._cancel(is_cancelled)
    app = Path(__file__).resolve().parents[1]
    if name == "video_pack" and request.get("cpu_precision") == "bf16_stream":
        return {"name": name, "state": "completed", "elapsed_seconds": 0,
                "peak_working_set_bytes": 0, "not_required_original_mapped_weights": True}
    if name == "video_pack":
        from .prepared import verify
        started = time.monotonic()
        prepared = verify(root, app, is_cancelled)
        if prepared is not None:
            progress({"stage": "video_pack_reuse"})
            record = {"name": name, "state": "completed", "reused_derived_weights": True,
                      "elapsed_seconds": round(time.monotonic()-started, 3),
                      "peak_working_set_bytes": 0, "one_time_conversion": False,
                      "verification": prepared}
            media._write_json(work / (name + ".metrics.json"), record)
            return record
    _, free = media._memory(os.getpid())
    from .resources import (CpuLimitMonitor, PressureGuard, record_drop, run_unthrottled, stage_limits,
                            start_requirement_gib, wait_for_cooling)
    limits = stage_limits(request, "neodragon")
    minimum = start_requirement_gib(root, name, limits)
    start_free = free
    if free < minimum * 2**30:
        raise MemoryError(f"단계 {name}: 여유 RAM {free/2**30:.2f}GiB, 시작 기준 {minimum}GiB 미달입니다.")
    conversion = name == "video_pack"
    cap = max(6.5, limits["maximum_working_set_gib"]) if conversion else limits["maximum_working_set_gib"]
    reserve = limits["reserve_ram_gib"]
    guard = PressureGuard(reserve * 2**30)
    if conversion:
        packed = root / "models" / "experimental-neodragon" / "derived" / "transformer-cpu-int8-v2.pt"
        if shutil.disk_usage(root).free < 3 * 2**30:
            raise RuntimeError("D 공간이 부족해 최초 INT8 변환을 시작할 수 없습니다.")
        if free < 7.5 * 2**30:
            raise RuntimeError("최초 INT8 변환 또는 재준비는 여유 RAM 7.5GiB가 필요합니다. 준비된 D 실행 폴더를 복사하면 다시 변환하지 않습니다.")
    python = media._path_in_runtime(root / "runtime" / "neodragon-python" / "python.exe", root)
    stage_file = app / "local_video" / "neodragon_stage.py"
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
    record = {"name": name, "state": "running", "peak_working_set_bytes": 0, "peak_private_resident_bytes": 0,
              "minimum_system_free_bytes": free, "working_set_limit_gib": cap,
              "reserve_ram_gib": reserve, "one_time_conversion": conversion}
    monitor = CpuLimitMonitor()
    wait_for_cooling(monitor, is_cancelled, progress)
    started = heartbeat = time.monotonic()
    process = cpu = None
    try:
        with (work / (name + ".log")).open("wb") as log:
            process = subprocess.Popen([str(python), str(stage_file), "--runtime-dir", str(root),
                                        "--stage", name, "--output", str(work)],
                                       cwd=work, env=environment(root, request["threads"]),
                                       stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=flags)
            record["process_id"] = process.pid
            run_unthrottled(getattr(process, "_handle", None))
            progress({"stage": name, "phase_process_id": process.pid})
            while process.poll() is None:
                media._cancel(is_cancelled)
                working, free = media._memory(process.pid)
                reclaimable = media._reclaimable(process.pid, working)
                record["peak_working_set_bytes"] = max(record["peak_working_set_bytes"], working)
                record["peak_private_resident_bytes"] = max(record["peak_private_resident_bytes"], working - reclaimable)
                record["minimum_system_free_bytes"] = min(record["minimum_system_free_bytes"], free + reclaimable)
                if working - reclaimable > cap * 2**30:
                    raise MemoryError(f"단계 {name}의 작업 메모리가 상한 {cap}GiB를 넘었습니다. 이 작업만 중지했습니다.")
                guard.check(free + reclaimable, time.monotonic())
                if time.monotonic() - started > request.get("maximum_scene_seconds", 1800):
                    raise RuntimeError(f"단계 {name}가 실행 제한 시간을 넘었습니다.")
                if time.monotonic() - heartbeat >= 15:
                    cpu = monitor.sample(time.monotonic()) or cpu
                    progress({"stage": name, "working_set_gib": round(working/2**30, 3),
                              "cpu_limit_pct": (cpu or {}).get("limit_pct"),
                              "peak_working_set_gib": round(record["peak_working_set_bytes"]/2**30, 3),
                              "free_ram_gib": round(free/2**30, 3), "reclaimable_mapped_gib": round(reclaimable/2**30, 3),
                              "scene_elapsed_seconds": round(time.monotonic()-started, 1)})
                    heartbeat = time.monotonic()
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f"단계 {name} 실패. 자세한 기록: {work / (name + '.log')}")
        record["state"] = "completed"
        if name == "video_infer" and request.get("device_plan", {}).get("gpu_inference"):
            ran_on, device_record = None, {}
            if (work / "torch-device.json").is_file():
                device_record = json.loads((work / "torch-device.json").read_text(encoding="utf-8"))
                ran_on = device_record.get("device")
            if ran_on in {"xpu", "cuda"}:
                record.update(gpu_inference=True, gpu={"backend": f"torch_{ran_on}", "scope": "all_video_transformer_ops",
                                                       "weights": device_record.get("weights"),
                                                       "peak_gpu_allocated_bytes": device_record.get("peak_gpu_allocated_bytes")})
            else:
                gpu = json.loads((work / "opencl-metrics.json").read_text(encoding="utf-8"))
                from .resources import GPU_DISCRETE_CEILING
                ceiling = min(gpu.get("buffer_ceiling_bytes", 0), GPU_DISCRETE_CEILING)
                if gpu.get("kernel_calls", 0) < 1 or gpu.get("peak_explicit_gpu_buffer_bytes", 2**63) > ceiling:
                    raise RuntimeError("GPU 실행 또는 저메모리 버퍼 상한 증거를 확인하지 못했습니다.")
                record.update(gpu_inference=True, gpu=gpu)
        return record
    except Exception as exc:
        record.update(state="cancelled" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
        raise
    finally:
        if process is not None:
            media._stop_owned(process)
            record_drop(root, name, start_free, record["minimum_system_free_bytes"], record["peak_private_resident_bytes"])
        record.update(monitor.summary())
        monitor.close()
        record["elapsed_seconds"] = round(time.monotonic()-started, 3)
        media._write_json(work / (name + ".metrics.json"), record)


def encode_frames(work, destination, settings, is_cancelled):
    from PIL import Image
    paths = sorted(work.glob("frame-*.png"))
    if len(paths) != settings["frames"]:
        raise RuntimeError("생성 모델의 프레임 수가 요청과 다릅니다.")
    writer = media._writer(destination, settings)
    writer.send(None)
    try:
        for index, path in enumerate(paths):
            media._cancel(is_cancelled)
            if path.name != f"frame-{index:03d}.png":
                raise RuntimeError("출력 프레임 번호가 연속하지 않습니다.")
            with Image.open(path) as frame:
                if frame.size != (settings["width"], settings["height"]):
                    raise RuntimeError("출력 프레임 해상도가 요청과 다릅니다.")
                writer.send(frame.convert("RGB").tobytes())
    finally:
        writer.close()


def cached(cache, key, settings, is_cancelled):
    try:
        record = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        safety = record.get("safety", {})
        if record.get("status") != "complete" or record.get("cache_key") != key or safety.get("unsafe") is not False:
            return None
        with (cache / "clip.mp4").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record.get("video_sha256"):
                return None
        record["verification"] = media.verify_video(cache / "clip.mp4", settings, is_cancelled=is_cancelled)
        return record
    except InterruptedError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def run_stage_adaptive(root, work, name, request, progress, is_cancelled, runner=None):
    """Run a stage; if its PyTorch GPU backend fails, rerun that stage on the CPU/OpenCL path.

    Memory shortages, cancellation and time limits are not hardware failures and are never retried here.
    """
    from . import accelerator
    runner = runner or run_stage
    plan = request.get("device_plan", {})
    backend = plan.get("torch_device")
    if name not in accelerator.GPU_STAGES or backend not in accelerator.BACKENDS:
        return runner(root, work, name, request, progress, is_cancelled)
    try:
        return runner(root, work, name, request, progress, is_cancelled)
    except (MemoryError, InterruptedError, TimeoutError):
        raise
    except Exception as exc:
        accelerator.mark_failed(root, backend, f"{name}: {exc}")
        plan["torch_device"], plan["torch_fallback_reason"] = "cpu", f"{backend} failed in {name}: {str(exc)[:200]}"
        stage_request = json.loads((work / "request.json").read_text(encoding="utf-8"))
        stage_request["device_plan"] = plan
        media._write_json(work / "request.json", stage_request)
        progress({"stage": "gpu_backend_failed_using_cpu_path", "backend": backend, "step_name": name})
        return runner(root, work, name, request, progress, is_cancelled)


def settings_for(request):
    settings = dict(PRESETS[request["preset"]])
    if request.get("continuous"):
        duration = request.get("duration_seconds", 4)
        if len(request["scenes"]) != 1 or request["preset"] == "smoke" or type(duration) is not int or duration not in {2, 4, 8}:
            raise ValueError("연속 영상은 한 장면과 2/4/8초 설정을 사용해야 합니다.")
        # Neo's causal VAE requires 8*n+1 frames; all are model-generated.
        settings["frames"] = duration * settings["fps"] + 1
    return settings


def generate_project(request, output_dir, cache_dir, model_dir, progress, is_cancelled):
    started = time.monotonic()
    root = Path(request["_runtime_dir"]).resolve()
    from .storage import runtime_root
    root = runtime_root(root)
    from .opencl_linear import plan
    saved = request.get("device_plan", {})
    selected = plan(request["backend"], (saved.get("selected_device") or {}).get("id"))
    if saved and (selected["backend_assignment"] != saved.get("backend_assignment")
                  or selected.get("selected_device") != saved.get("selected_device")):
        raise RuntimeError("요청 이후 GPU가 바뀌었습니다. 새 작업을 요청하세요.")
    from . import accelerator
    choice = accelerator.choose(root, selected.get("selected_device"), request.get("backend", "auto")) if selected.get("gpu_inference") else {}
    selected["torch_device"] = choice.get("torch_device", "cpu")
    selected["torch_discrete"] = bool(choice.get("discrete"))
    request["device_plan"] = selected
    output_dir, cache_dir, model_dir = [media._path_in_runtime(Path(p), root) for p in (output_dir, cache_dir, model_dir)]
    for path in (output_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    settings = settings_for(request)
    shots = []
    for index, scene in enumerate(request["scenes"]):
        media._cancel(is_cancelled)
        key_value = {"engine": ENGINE_VERSION, "pipeline": request["pipeline_revision"],
                     "models": request["model_revision"], "prompt": scene["prompt"],
                     "seed": scene["seed"], "settings": settings,
                     "cpu_precision": request.get("cpu_precision", "int8"),
                     "device_plan": selected,
                     "prompt_modifier": scene.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours"),
                     "first_image_sha256": scene.get("image_sha256"),
                     "first_frame_prompt": scene.get("first_frame_prompt", scene["prompt"]) if not scene.get("image_path") else None,
                     "first_image_negative_prompt": scene.get("negative_prompt", "") if not scene.get("image_path") else ""}
        key = hashlib.sha256(json.dumps(key_value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache = media._path_in_runtime(cache_dir / key, root)
        for name in ("metadata.json", "clip.mp4"):
            media._path_in_runtime(cache / name, root)
        hit = cached(cache, key, settings, is_cancelled)
        reused = hit is not None
        progress({"stage": "cache_reuse" if hit else "scene_start", "scene_index": index+1,
                  "scene_count": len(request["scenes"]), "scene_id": scene["id"],
                  "steps": len(STAGES)+1, "step": 0})
        if hit is None:
            work = media._path_in_runtime(root / "tmp" / ("neo-" + key[:12] + "-" + uuid.uuid4().hex[:8]), root)
            work.mkdir(parents=True)
            media._write_json(work / "request.json", {"prompt": scene["prompt"], "seed": scene["seed"],
                                                     "threads": request["threads"], "device_plan": selected,
                                                     "gpu_buffer_mib": request.get("gpu_buffer_mib", "auto"),
                                                     "reserve_ram_gib": request.get("reserve_ram_gib", "auto"),
                                                     "cpu_precision": request.get("cpu_precision", "int8"), **settings})
            if "prompt_modifier" in scene:
                stage_request = json.loads((work / "request.json").read_text(encoding="utf-8"))
                stage_request["prompt_modifier"] = scene["prompt_modifier"]
                media._write_json(work / "request.json", stage_request)
            progress({"stage": "first_frame"})
            phases = [first_frame(root, work, scene, settings, request, progress, is_cancelled)]
            for number, name in enumerate(STAGES):
                progress({"step": number+2, "steps": len(STAGES)+1})
                from .conditioning import run_cached
                phases.append(run_cached(root, work, name, request, progress, is_cancelled, run_stage_adaptive))
            safety = json.loads((work / "safety.json").read_text(encoding="utf-8"))
            if safety.get("unsafe") is not False:
                raise RuntimeError("필수 안전 검사에서 결과를 거부했습니다.")
            progress({"stage": "encoding"})
            video = work / "clip.mp4"
            encode_frames(work, video, settings, is_cancelled)
            verification = media.verify_video(video, settings, is_cancelled=is_cancelled)
            media._cancel(is_cancelled)
            with video.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            cache.mkdir(parents=True, exist_ok=True)
            # Keep phase logs, measurements and the initial frame for review.
            for path in work.iterdir():
                if path.suffix in {".log", ".json"} or path.name == "first-frame.png":
                    destination = media._path_in_runtime(cache / path.name, root)
                    shutil.copy2(path, destination)
            os.replace(video, cache / "clip.mp4")
            hit = {"status": "complete", "cache_key": key, "video_sha256": digest,
                   "prompt": scene["prompt"], "seed": scene["seed"], "safety": safety,
                   "first_frame_prompt": scene.get("first_frame_prompt", scene["prompt"]),
                   "verification": verification, "phases": phases, "settings": settings,
                   "model_profile": "neodragon", "pipeline_revision": request["pipeline_revision"],
                   "model_revision": request["model_revision"], "backend_assignment": selected["backend_assignment"],
                   "device_plan": selected, "gpu_inference": selected["gpu_inference"],
                   "cpu_precision": request.get("cpu_precision", "int8"),
                   "prompt_modifier": scene.get("prompt_modifier", ", cinematic, realistic textures, high detail, natural colours"),
                   "negative_prompt_effective": "first_frame_only" if not scene.get("image_path") else False,
                   "elapsed_seconds": round(sum(p["elapsed_seconds"] for p in phases), 3)}
            media._write_json(cache / "metadata.json", hit)
            # Every removed path is this attempt's verified D subfolder.
            if not work.resolve().is_relative_to((root / "tmp").resolve()):
                raise ValueError("임시 폴더가 허용된 D 경로 밖입니다.")
            shutil.rmtree(work)
        shots.append({**hit, "id": scene["id"], "cache_hit": reused,
                      "reference_attribution": scene.get("reference_attribution"),
                      "video_path": str((cache / "clip.mp4").resolve())})
        progress({"stage": "scene_complete", "scene_index": index+1, "scene_count": len(request["scenes"])})
    media._cancel(is_cancelled)
    video = media._path_in_runtime(output_dir / "video.mp4", root)
    sheet = media._path_in_runtime(output_dir / "contact-sheet.jpg", root)
    pending = media._path_in_runtime(output_dir / ("video-" + uuid.uuid4().hex[:8] + ".mp4"), root)
    progress({"stage": "assembling"})
    media._assemble(shots, pending, sheet, settings, is_cancelled)
    verification = media.verify_video(pending, settings, len(shots)*settings["frames"], is_cancelled)
    media._cancel(is_cancelled)
    os.replace(pending, video)
    with video.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"video_path": str(video), "video_sha256": digest, "contact_sheet_path": str(sheet),
              "model_profile": "neodragon", "model_revision": request["model_revision"], "settings": settings,
              "shots": shots, "verification": verification, "elapsed_seconds": round(time.monotonic()-started, 3),
              "engine_version": ENGINE_VERSION, "backend_assignment": selected["backend_assignment"], "visual_review": "required",
              "device_plan": selected, "gpu_inference": selected["gpu_inference"],
              "generation_method": "autoregressive_temporal_diffusion_no_frame_interpolation",
              "continuous": request.get("continuous", False), "edit_count": max(0, len(shots)-1),
              "cpu_precision": request.get("cpu_precision", "int8"),
              "site_template": request.get("site_template"), "weights_finetuned": False,
              "continuity": verification.get("continuity"),
              "reference_attributions": [scene["reference_attribution"] for scene in request["scenes"] if scene.get("reference_attribution")],
              "negative_prompt_effective": "first_frame_only" if any(not scene.get("image_path") for scene in request["scenes"]) else False,
              "negative_prompt_note": "제외 프롬프트는 SD 첫 이미지 생성에만 적용됩니다. 제공된 참고 화면과 영상 증류 모델(CFG=0)에는 적용되지 않습니다.",
              "target_iris_generation_verified": False,
              "quality_note": "각 장면의 움직임은 로컬 영상 모델이 생성합니다. 새로운 장면의 동작·사실감은 결과를 보고 확인하세요."}
    media._write_json(output_dir / "result.json", result)
    if result["reference_attributions"]:
        media._write_json(media._path_in_runtime(output_dir / "ATTRIBUTION.json", root), result["reference_attributions"])
        attribution_path = media._path_in_runtime(output_dir / "ATTRIBUTION.md", root)
        attribution_path.write_text("\n\n".join(
            f"Author: {item['author']}\nSource: {item['source_url']}\nLicense: {item['license']} ({item['license_url']})\nChanges: {item['changes']}"
            for item in result["reference_attributions"]), encoding="utf-8")
        result["attribution_path"] = str(attribution_path)
        media._write_json(output_dir / "result.json", result)
    return result
