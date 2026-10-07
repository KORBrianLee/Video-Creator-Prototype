"""Small, dependency-free job controller. Heavy inference lives in a child process."""
from __future__ import annotations
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import time
import uuid

from .resources import RESOURCE_KEYS

MEMORY_RETRIES = 2

APP = Path(__file__).resolve().parent.parent
TERMINAL = {"completed", "failed", "cancelled"}
PRESETS = {"smoke", "preview", "quality"}
PROFILES = {"neodragon", "lightning", "wan"}
DEFAULT_NEGATIVE = "static image, slideshow, blurry, deformed, inconsistent motion, distorted limbs, text, subtitles, watermark"

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    # Windows refuses the replace while a reader (status poll, antivirus scan) holds the file open.
    for attempt in range(60):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.25))
    temporary.unlink(missing_ok=True)
    # Rename needs delete sharing, an in-place write does not; readers retry a torn read (read_json).
    for attempt in range(20):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.25)
    raise OSError("작업 상태를 저장할 수 없습니다.")

def read_json(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            if attempt == 9:
                raise
            time.sleep(0.1)

def load_config():
    from .storage import default_runtime, runtime_root
    from .devices import BACKENDS
    from .resources import auto_threads, check_value, is_auto
    config = {"runtime_dir": default_runtime(), "backend": "auto", "model_profile": "neodragon", "threads": "auto",
              **{key: "auto" for key in RESOURCE_KEYS}}
    if (APP / "video.config.json").is_file():
        config.update(read_json(APP / "video.config.json"))
    if os.environ.get("CVL_RUNTIME_DIR"):
        config["runtime_dir"] = os.environ["CVL_RUNTIME_DIR"]
    root = runtime_root(config["runtime_dir"])
    if config["backend"] not in BACKENDS:
        raise ValueError("지원 장치 선택: auto, gpu, intel-gpu, intel-vulkan, cpu")
    if config.get("gpu_device") is not None and not re.fullmatch(r"(?:Vulkan|OpenCL)\d+", str(config["gpu_device"]), re.I):
        raise ValueError("gpu_device는 실제 감지한 GPU 번호여야 합니다.")
    budget = config.get("gpu_memory_budget_gib")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0):
        raise ValueError("GPU 예산은 유한한 양수여야 합니다.")
    if config["model_profile"] not in PROFILES:
        raise ValueError("지원 모델 선택: neodragon, lightning 또는 wan")
    for key in RESOURCE_KEYS:
        config[key] = check_value(key, config.get(key))
    if not is_auto(config["maximum_working_set_gib"]) and not is_auto(config["reserve_ram_gib"]) and config["maximum_working_set_gib"] <= config["reserve_ram_gib"]:
        raise ValueError("최대 작업 메모리는 시스템 보호 메모리보다 커야 합니다.")
    config["threads_setting"] = config["threads"]
    if is_auto(config["threads"]):
        config["threads"] = auto_threads()
    elif isinstance(config["threads"], bool) or not isinstance(config["threads"], int):
        raise ValueError('threads는 "auto" 또는 정수여야 합니다.')
    config["threads"] = max(1, min(int(config["threads"]), os.cpu_count() or 4, 8))
    return config, root

def memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("page", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                    ("virtual", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]
    if os.name != "nt":
        raise RuntimeError("이 포터블 배포는 Windows용입니다.")
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("메모리 정보를 읽을 수 없습니다.")
    return {"total_ram_gib": round(status.total / 2**30, 2), "free_ram_gib": round(status.available / 2**30, 2)}

def manifest(profile=None):
    if profile is None:
        profile = load_config()[0]["model_profile"]
    name = {"neodragon": "neodragon.lock.json", "lightning": "lightning.lock.json", "wan": "models.lock.json"}[profile]
    return read_json(APP / name)

def cpp_revision(app):
    digest = hashlib.sha256()
    for relative in ("local_video/backend.py", "local_video/devices.py", "local_video/storage.py"):
        path = app / relative
        if not path.is_file():
            return "missing_backend"
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def quality_validation(profile, root):
    if profile == "wan":
        try:
            from .devices import plan
            cfg, _ = load_config()
            selected = plan(root, cfg["backend"], cfg.get("gpu_device"), cfg.get("gpu_memory_budget_gib"))
            record = read_json(APP / "docs" / "quality" / "wan-vulkan.json")
            if record.get("state") != "passed_gpu_visual_review" or record.get("pipeline_revision") != cpp_revision(APP):
                return "pending_gpu_visual_validation", False
            if not selected["gpu_inference"] or record.get("validated_device_name") != selected["selected_device"]["name"]:
                return "target_gpu_visual_validation_required", False
            if record.get("model_revision") != hashlib.sha256(json.dumps(manifest(profile), sort_keys=True).encode()).hexdigest():
                return "validation_model_changed", False
            video = runtime_path(root, "app", *Path(record["video_relative_path"]).parts)
            with video.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["video_sha256"]:
                    return "validation_artifact_changed", False
            return "passed_gpu_visual_review_target_gram_unverified", True
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            return "pending_gpu_visual_validation", False
    if profile != "neodragon":
        return "failed_visual_validation", False
    from .neodragon import pipeline_revision
    try:
        from .opencl_linear import plan
        cfg, _ = load_config()
        selected = plan(cfg["backend"], cfg.get("gpu_device"))
        gpu = selected["gpu_inference"]
        path = APP / "docs" / "quality" / ("neodragon-opencl.json" if gpu else "neodragon-cpu.json")
        record = read_json(path)
        if record.get("state") != ("passed_gpu_visual_review" if gpu else "passed_cpu_visual_review") or record.get("pipeline_revision") != pipeline_revision(APP):
            return "pending_visual_validation", False
        if gpu and record.get("validated_device_name") != selected["selected_device"]["name"]:
            return "target_gpu_visual_validation_required", False
        if gpu and record.get("validated_driver") != selected["selected_device"].get("driver"):
            return "target_gpu_driver_validation_required", False
        expected = record.get("video_sha256")
        if record.get("video_relative_path"):
            video = runtime_path(root, "app", *Path(record["video_relative_path"]).parts)
        else:
            video = Path(record["video_path"]).resolve()
        if not video.is_relative_to(root) or not video.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
            return "validation_artifact_missing", False
        with video.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                return "validation_artifact_changed", False
        if record.get("model_revision") != hashlib.sha256(json.dumps(manifest(profile), sort_keys=True).encode()).hexdigest():
            return "validation_model_changed", False
        return "passed_opencl_visual_review_target_gram_unverified" if gpu else "passed_cpu_visual_review_target_gram_unverified", True
    except (OSError, ValueError, KeyError, RuntimeError):
        return "pending_visual_validation", False

def runtime_path(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("실제 데이터 경로가 선택한 실행 폴더 밖으로 연결됩니다.")
    return path


def construction_validation(root):
    """A beach baseline must never certify machinery/worker realism."""
    from .neodragon import pipeline_revision
    from .site_profiles import revision
    report = APP / "docs" / "quality" / "construction-cpu.json"
    state = "pending_construction_visual_review"
    ready = False
    try:
        record = read_json(report)
        state = record.get("state", state)
        if (state == "passed_construction_visual_review" and record.get("production_ready") is True
                and record.get("pipeline_revision") == pipeline_revision(APP)
                and record.get("profile_revision") == revision()
                and record.get("model_revision") == hashlib.sha256(json.dumps(manifest("neodragon"), sort_keys=True).encode()).hexdigest()):
            path = runtime_path(root, "app", *Path(record["video_relative_path"]).parts)
            with path.open("rb") as stream:
                ready = hashlib.file_digest(stream, "sha256").hexdigest() == record["video_sha256"]
            if not ready:
                state = "construction_validation_artifact_changed"
        elif state == "passed_construction_visual_review":
            state = "pending_construction_visual_review"
    except (OSError, ValueError, KeyError, TypeError):
        state = "pending_construction_visual_review"
    return {"state": state, "production_ready": ready, "weights_finetuned": False,
            "sora2_quality_verified": False, "report": str(report), "draft_only": not ready}


def check_domain_request(request, root):
    if request.get("domain") == "construction" and not request.get("diagnostic", False):
        if not construction_validation(root)["production_ready"]:
            raise RuntimeError("중장비·작업자 현장 영상은 실제 동작·형태 유지 품질 검증을 통과하지 못했습니다. 일반 제작 성공으로 표시하지 않습니다. diagnostic=true 초안 시험만 가능합니다. docs/SITE_ADAPTATION.md를 확인하세요.")

def completed_result_intact(state, root):
    result = state.get("result", {})
    expected = result.get("video_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return False
    try:
        path = Path(result["video_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected
    except (OSError, KeyError):
        return False

def check_model_files(root, profile, lock, verify=False):
    """Check this runtime; removed optional weights never trigger a new job."""
    files = []
    for original in lock["files"]:
        item = {**original, **lock["prepared_base"]} if original["role"] == "base" and profile == "lightning" and lock.get("prepared_base") else original
        file = runtime_path(root, "models", item["filename"])
        integrity, ok = "size_only", False
        try:
            ok = file.is_file() and file.stat().st_size == item["size"]
            if ok and verify:
                with file.open("rb") as stream:
                    ok = hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
                integrity = "sha256_verified" if ok else "sha256_mismatch"
        except OSError:
            integrity, ok = "unavailable", False
        files.append({"role": item["role"], "filename": item["filename"], "ready": ok, "integrity": integrity})
    return files


def profile_availability(root, profile, files=None):
    if files is None:
        try:
            files = check_model_files(root, profile, manifest(profile))
        except (OSError, ValueError, KeyError):
            return {"model_profile": profile, "state": "manifest_unavailable", "weights_present": False}
    missing = sum(not item["ready"] for item in files)
    result = {"model_profile": profile, "state": "weights_present" if not missing else "weights_incomplete",
              "weights_present": not missing, "required_file_count": len(files), "unready_file_count": missing,
              "meaning": "weight_presence_only_not_engine_or_video_quality_validation"}
    if missing:
        lock_name = {"neodragon": "neodragon.lock.json", "lightning": "lightning.lock.json", "wan": "models.lock.json"}[profile]
        result["recovery"] = {"setup_arguments": ["-RuntimeDir", str(root), "-Profile", profile],
                              "source_lock": str(APP / lock_name), "automatic_download": False}
    return result


def accelerator_view(root, selected, backend_setting):
    from . import accelerator
    try:
        return accelerator.describe(root, selected.get("selected_device") if selected.get("gpu_inference") else None, backend_setting)
    except Exception as exc:  # the doctor must report problems, never fail because of one
        return {"torch_device": "cpu", "error": str(exc)[:200]}


def doctor(verify=False, profile=None):
    cfg, root = load_config()
    profile = profile or cfg["model_profile"]
    if profile not in PROFILES:
        raise ValueError("올바르지 않은 모델입니다.")
    errors = []
    mem = memory()
    if not root.is_dir():
        errors.append("선택한 SSD의 실행 환경이 준비되지 않았습니다.")
    lock = manifest(profile)
    model_files = check_model_files(root, profile, lock, verify)
    for item in model_files:
        if not item["ready"]:
            errors.append(f'모델 파일 미준비 또는 손상: {item["filename"]}')
    for module in ["PIL", "imageio_ffmpeg"]:
        if not importlib.util.find_spec(module):
            errors.append(f"실행 라이브러리 미설치: {module}")
    from .devices import plan
    try:
        if profile == "neodragon":
            from .opencl_linear import plan as neo_plan
            selected = neo_plan(cfg["backend"], cfg.get("gpu_device"))
        else:
            selected = plan(root, cfg["backend"], cfg.get("gpu_device"), cfg.get("gpu_memory_budget_gib"))
    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))
        selected = {"gpu_inference": False, "minimum_free_ram_gib": 8.0, "selected_device": None}
    engine = runtime_path(root, "engines", "vulkan" if selected["gpu_inference"] and profile != "neodragon" else "cpu", "sd-cli.exe")
    if not engine.is_file():
        errors.append("선택한 로컬 추론 엔진이 설치되지 않았습니다.")
    if profile == "neodragon":
        runtime_python = runtime_path(root, "runtime", "neodragon-python", "python.exe")
        if not runtime_python.is_file():
            errors.append("Neodragon의 CPU 전용 실행 환경이 없습니다.")
        else:
            from .neodragon import runtime_errors
            errors.extend(runtime_errors(APP, root))
        for item in lock["engine"]["files"]:
            source = runtime_path(root, "engines", "experimental-neodragon", *item["path"].split("/"))
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                errors.append("Neodragon 고정 소스 미준비 또는 손상: " + item["path"])
        packed = runtime_path(root, "models", "experimental-neodragon", "derived", "transformer-cpu-int8-v2.pt")
        if not selected["gpu_inference"] and (not packed.is_file() or not packed.with_suffix(".provenance.json").is_file()):
            errors.append("최초 INT8 준비가 필요합니다. setup.cmd를 실행하거나 준비된 D 실행 폴더를 복사하세요.")
        elif verify and packed.is_file() and packed.with_suffix(".provenance.json").is_file():
            provenance = read_json(packed.with_suffix(".provenance.json"))
            with packed.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != provenance.get("sha256"):
                    errors.append("INT8 변환 파일의 SHA256이 다릅니다.")
    from .resources import describe
    resource_plan = describe(cfg, profile, selected if profile == "wan" or selected.get("gpu_inference") else None, root)
    minimum = resource_plan["minimum_free_ram_gib"]
    # Elastic memory: little free RAM only means slower, paging-friendly execution (low-memory mode).
    # Only a near-empty commit (RAM + page file Windows can still promise) blocks a start.
    from .resources import commit_bytes, commit_floor
    commit_limit, commit_available = commit_bytes()
    resource_plan["low_memory_mode"] = mem["free_ram_gib"] < minimum
    resource_plan["commit_available_gib"] = round(commit_available / 2**30, 2)
    commit_short = commit_available < commit_floor(commit_limit) + 2**30
    ram_only = commit_short and not errors
    if commit_short:
        errors.append(f"commit 여유 {commit_available/2**30:.2f}GiB로 메모리 할당 실패 위험이 있습니다. 큰 프로그램을 닫거나 페이지 파일을 늘리세요.")
    free_disk = round(shutil.disk_usage(root if root.exists() else Path(root.anchor)).free / 2**30, 2)
    if free_disk < 1:
        errors.append("선택한 SSD의 여유 공간이 1GB 미만입니다.")
        ram_only = False
    quality_status, quality_ready = quality_validation(profile, root)
    return {"ready": not errors, "ready_for_generation": not errors and quality_ready, "quality_ready": quality_ready,
            "quality_status": quality_status,
            "quality_report": str(APP / "docs" / "VALIDATION.md"),
            "model_selection_report": str(APP / "docs" / "MODEL_SELECTION.md"),
            "backend": cfg["backend"], "device_plan": selected, "model_profile": profile, "model_license": lock["model_license"], "threads": cfg["threads"], "runtime_dir": str(root),
            **mem, "free_disk_gib": free_disk, "model_download_gib": round(sum(x["size"] for x in lock["files"]) / 2**30, 2),
            "minimum_free_ram_gib": minimum, "memory_threshold_is": "adaptive_project_policy_not_official_minimum",
            "resource_plan": resource_plan, "ram_shortfall_only": ram_only,
            "accelerator": accelerator_view(root, selected, cfg["backend"]) if profile == "neodragon" else None,
            "model_files": model_files, "errors": errors, "target_iris_generation_verified": False,
            "model_profiles": [profile_availability(root, name, model_files if name == profile else None) for name in sorted(PROFILES)],
            "construction": construction_validation(root) if profile == "neodragon" else None,
            "next": "video_generate" if not errors and quality_ready else "resolve_resources" if errors else "review_model_selection_and_quality_report"}

def normalize(value):
    if not isinstance(value, dict):
        raise ValueError("영상 요청은 JSON 객체여야 합니다.")
    from .site_profiles import expand
    initial_config, initial_root = load_config()
    value = expand({**value, "model_profile": value.get("model_profile", initial_config["model_profile"])}, initial_root)
    preset = value.get("preset", "preview")
    if preset not in PRESETS:
        raise ValueError("preset: preview, quality, smoke 중 하나를 선택하세요.")
    seed = value.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31-1:
        raise ValueError("seed는 0~2147483647 정수여야 합니다.")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= 12:
        raise ValueError("장면은 한 작업에 1~12개입니다. 먼저 한 장면의 초안을 검토하세요.")
    continuous = value.get("continuous", False)
    if not isinstance(continuous, bool):
        raise ValueError("continuous는 true 또는 false여야 합니다.")
    if continuous and (len(scenes) != 1 or preset == "smoke"):
        raise ValueError("연속 촬영은 preview/quality의 단일 장면만 허용합니다. 여러 클립을 이어 붙이지 않습니다.")
    duration = value.get("duration_seconds", 4 if continuous else 2)
    from .backend import DURATIONS
    if isinstance(duration, bool) or not isinstance(duration, int) or duration not in DURATIONS:
        raise ValueError(f"duration_seconds는 {', '.join(map(str, DURATIONS))} 중 하나입니다.")
    if not continuous and duration != 2:
        raise ValueError("2초보다 긴 영상은 continuous=true로 요청하세요. 한 번의 생성으로 이어서 만듭니다.")
    normalized = []
    seen = set()
    for index, item in enumerate(scenes):
        if not isinstance(item, dict):
            raise ValueError("장면은 객체여야 합니다.")
        sid = item.get("id", f"s{index+1:02d}")
        if not isinstance(sid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", sid) or sid in seen:
            raise ValueError("장면 ID는 서로 다른 짧은 영문·숫자 이름이어야 합니다.")
        seen.add(sid)
        prompt = item.get("prompt", "")
        negative = item.get("negative_prompt", DEFAULT_NEGATIVE)
        scene_seed = item.get("seed", (seed + index) % 2**31)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 3000:
            raise ValueError("장면 프롬프트는 1~3000자입니다.")
        if not isinstance(negative, str) or len(negative) > 1500:
            raise ValueError("제외 프롬프트는 1500자 이하여야 합니다.")
        if not isinstance(scene_seed, int) or isinstance(scene_seed, bool) or not 0 <= scene_seed < 2**31:
            raise ValueError("장면 seed는 0~2147483647 정수여야 합니다.")
        scene = {"id": sid, "prompt": prompt.strip(), "negative_prompt": negative.strip(), "seed": scene_seed}
        if "first_frame_prompt" in item:
            first_prompt = item["first_frame_prompt"]
            if not isinstance(first_prompt, str) or not first_prompt.strip() or len(first_prompt) > 3000:
                raise ValueError("첫 화면 프롬프트는 1~3000자입니다.")
            scene["first_frame_prompt"] = first_prompt.strip()
        if "prompt_modifier" in item:
            modifier = item["prompt_modifier"]
            if not isinstance(modifier, str) or len(modifier) > 200:
                raise ValueError("prompt_modifier는 200자 이하 문자열입니다. 빈 문자열로 영화 효과 보정을 제외할 수 있습니다.")
            scene["prompt_modifier"] = modifier
        if item.get("image_path"):
            _, image_root = load_config()
            image = Path(item["image_path"])
            if not image.is_absolute() or not image.resolve().is_relative_to(image_root):
                raise ValueError("참고 이미지는 선택한 실행 폴더 안의 절대 경로여야 합니다.")
            image = image.resolve(strict=True)
            if image.stat().st_size > 20 * 2**20:
                raise ValueError("참고 이미지는 20MiB 이하여야 합니다.")
            with image.open("rb") as stream:
                scene.update(image_path=str(image), image_sha256=hashlib.file_digest(stream, "sha256").hexdigest())
        if item.get("reference_attribution"):
            scene["reference_attribution"] = item["reference_attribution"]
        normalized.append(scene)
    title, script = value.get("title", "로컬 영상"), value.get("script", "")
    if not isinstance(title, str) or len(title) > 200 or not isinstance(script, str) or len(script) > 16000:
        raise ValueError("제목은 200자, 원문 스크립트는 16000자 이하여야 합니다.")
    cfg, root = load_config()
    profile = value.get("model_profile", cfg["model_profile"])
    if profile not in PROFILES:
        raise ValueError("model_profile: neodragon, lightning 또는 wan")
    if (continuous or value.get("site_template")) and profile == "lightning":
        raise ValueError("연속 생성은 neodragon 또는 wan 경로를 사용하세요.")
    if profile == "wan" and any(s.get("image_path") or s.get("first_frame_prompt") for s in normalized):
        raise ValueError("Wan 1.3B는 텍스트 영상 모델입니다. 참고 이미지나 첫 화면 프롬프트를 적용했다고 표시할 수 없습니다.")
    precision = value.get("cpu_precision", "bf16_stream" if preset == "quality" and profile == "neodragon" else "int8")
    if precision not in {"int8", "bf16_stream"} or (profile != "neodragon" and precision != "int8"):
        raise ValueError("cpu_precision은 neodragon에서 int8 또는 bf16_stream을 사용하세요.")
    domain = value.get("domain", "construction" if value.get("site_template") else "general")
    if domain not in {"general", "construction"} or (value.get("site_template") and domain != "construction"):
        raise ValueError("domain은 general 또는 construction입니다. 현장 템플릿은 construction을 사용합니다.")
    spec = manifest(profile)
    pipeline_file = APP / "local_video" / "backend.py"
    pipeline_revision = cpp_revision(APP)
    if profile == "neodragon":
        from .neodragon import pipeline_revision as neo_revision
        pipeline_revision = neo_revision(APP)
    video_model = value.get("video_model", "auto")
    if video_model not in {"auto", "neo", "ltx", "skyreels"} or (video_model != "auto" and profile != "neodragon"):
        raise ValueError("video_model은 neodragon 경로에서 auto, neo, ltx, skyreels 중 하나입니다.")
    anchor_end = value.get("anchor_end")
    if anchor_end is not None and (isinstance(anchor_end, bool) or not isinstance(anchor_end, (int, float)) or not 0 < anchor_end <= 1):
        raise ValueError("anchor_end는 0보다 크고 1 이하인 수입니다(마지막 프레임을 첫 화면 구도에 묶는 강도).")
    cond_noise = value.get("cond_noise")
    if cond_noise is not None and (isinstance(cond_noise, bool) or not isinstance(cond_noise, (int, float)) or not 0 <= cond_noise <= 1):
        raise ValueError("cond_noise는 0~1 사이 수입니다(키프레임 조건에 섞는 노이즈, 기본 0.15).")
    anchor_end_image = None
    if value.get("anchor_end_image"):
        _, image_root = load_config()
        end_image = Path(value["anchor_end_image"])
        if anchor_end is None or not end_image.is_absolute() or not end_image.resolve().is_relative_to(image_root):
            raise ValueError("anchor_end_image는 anchor_end와 함께, 실행 폴더 안의 절대 경로로 지정합니다.")
        end_image = end_image.resolve(strict=True)
        with end_image.open("rb") as stream:
            anchor_end_image = {"path": str(end_image), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    diagnostic = value.get("diagnostic", False)
    if not isinstance(diagnostic, bool):
        raise ValueError("diagnostic은 true 또는 false여야 합니다.")
    from .devices import plan
    if profile == "neodragon":
        from .opencl_linear import plan as neo_plan
        selected = neo_plan(cfg["backend"], cfg.get("gpu_device"))
        if selected["gpu_inference"]:
            if "cpu_precision" in value and precision != "bf16_stream":
                raise ValueError("GPU 스트리밍에는 원본 BF16 행렬을 사용합니다. cpu_precision을 생략하거나 bf16_stream으로 지정하세요.")
            precision = "bf16_stream"
    else:
        selected = plan(root, cfg["backend"], cfg.get("gpu_device"), cfg.get("gpu_memory_budget_gib"))
    backend = cfg["backend"]
    return {"title": title, "script": script, "preset": preset, "seed": seed, "scenes": normalized,
            "continuous": continuous, "duration_seconds": duration,
            "site_template": value.get("site_template"),
            "cpu_precision": precision,
            "domain": domain,
            "pipeline_revision": pipeline_revision,
            "diagnostic": diagnostic,
            "video_model": video_model,
            **({"anchor_end": float(anchor_end)} if anchor_end is not None else {}),
            **({"anchor_end_image": anchor_end_image} if anchor_end_image else {}),
            **({"cond_noise": float(cond_noise)} if cond_noise is not None else {}),
            "model_profile": profile, "_model_spec": spec,
            "model_revision": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "backend": backend, "device_plan": selected, "threads": cfg["threads"],
            **{key: cfg.get(key, "auto") for key in RESOURCE_KEYS},
            "minimum_free_ram_gib": "auto" if cfg.get("minimum_free_ram_gib", "auto") == "auto" else max(cfg["minimum_free_ram_gib"], selected.get("minimum_free_ram_gib", 3.0)),
            "_runtime_dir": str(root)}

def job_path(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise ValueError("올바르지 않은 작업 ID입니다.")
    _, root = load_config()
    return runtime_path(root, "jobs", job_id)

def status(job_id):
    job = job_path(job_id)
    if not (job / "status.json").is_file():
        raise ValueError("작업을 찾을 수 없습니다.")
    state = read_json(job / "status.json")
    if state.get("state") not in TERMINAL and state.get("worker_pid") and os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        handle = kernel.OpenProcess(0x1000, False, state["worker_pid"])
        alive = True
        if handle:
            exit_code = ctypes.c_ulong()
            if kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                alive = exit_code.value == 259
            # Windows reuses PIDs: a process that started after the job was created is not its worker.
            times = [ctypes.c_ulonglong() for _ in range(4)]
            if alive and kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                started = times[0].value / 1e7 - 11644473600
                alive = started <= state.get("created_at", started) + 30
            kernel.CloseHandle(handle)
        elif ctypes.get_last_error() == 87:
            alive = False
        if not alive:
            state = read_json(job / "status.json")
            if state.get("state") not in TERMINAL:
                state.update(state="failed", stage="worker_exited", error="작업자가 비정상 종료됐습니다. 성공한 장면 캐시는 유지됩니다.", completed_at=time.time())
                write_json(job / "status.json", state)
    state["request_path"] = str(job / "request.json")
    state["log_path"] = str(job / "worker.log")
    return state


def _reusable_status(path, digest, root):
    """An index is only a hint; the real job and output remain authoritative."""
    try:
        path = runtime_path(root, *path.relative_to(root).parts)
        jid = path.parent.name
        if not re.fullmatch(r"[0-9a-f]{12}", jid):
            return None
        state = read_json(path)
        if not isinstance(state, dict) or state.get("request_hash") != digest:
            return None
        if state.get("job_id") != jid or state.get("state") not in {"queued", "running", "completed"}:
            return None
        state = status(jid)
        if state.get("request_hash") != digest or state.get("state") not in {"queued", "running", "completed"}:
            return None
        if state["state"] == "completed" and not completed_result_intact(state, root):
            return None
        return {**state, "reused_existing_job": True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _remember_request(root, digest, jid):
    try:
        path = runtime_path(root, "cache", "request-index", digest + ".json")
        write_json(path, {"request_hash": digest, "job_id": jid})
    except (OSError, ValueError):
        # Optional acceleration must never change job state or write outside D.
        pass


def _indexed_request(root, digest):
    try:
        path = runtime_path(root, "cache", "request-index", digest + ".json")
        hint = read_json(path)
        if not isinstance(hint, dict) or hint.get("request_hash") != digest:
            return None
        jid = hint.get("job_id")
        if not isinstance(jid, str) or not re.fullmatch(r"[0-9a-f]{12}", jid):
            return None
        return _reusable_status(runtime_path(root, "jobs", jid, "status.json"), digest, root)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _require_startable(check, diagnostic):
    # A job blocked only by RAM is queued; the worker waits for this computer's RAM to recover.
    if not check["ready"] and not check.get("ram_shortfall_only"):
        raise RuntimeError("생성 준비 미완료: " + "; ".join(check["errors"]))
    if not diagnostic and not check.get("quality_ready", check.get("ready_for_generation", False)):
        raise RuntimeError("영상 품질 검증이 실패했거나 아직 완료되지 않았습니다. 제작 작업을 시작하지 않았습니다.")

def _require_supported_duration(request, check):
    """Long clips need the long-video model; on Neo-only computers they are accepted only as diagnostics."""
    view = check.get("accelerator") or {}
    maximum = view.get("maximum_duration_seconds", 8)
    duration = request.get("duration_seconds", 2) if request.get("continuous") else 2
    if duration > maximum and not request.get("diagnostic"):
        raise ValueError(f"이 컴퓨터의 정식 최대 길이는 {maximum}초입니다({view.get('video_model', 'Neo')}). "
                         f"Neo는 약 7초 이후 화면이 무너져 {duration}초는 diagnostic=true 시험으로만 요청할 수 있습니다. "
                         f"사유: {view.get('model_tier_reason') or view.get('reason') or 'NVIDIA 장시간 모델 없음'}")


def submit(value):
    request = normalize(value)
    _, root = load_config()
    check_domain_request(request, root)
    check = None
    if not request["diagnostic"]:
        # Production readiness is checked before reusing an earlier job too.
        check = doctor(profile=request["model_profile"])
        _require_startable(check, False)
        _require_supported_duration(request, check)
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    jobs = runtime_path(root, "jobs")
    reused = _indexed_request(root, digest)
    if reused is not None:
        return reused
    if jobs.is_dir():
        for path in jobs.glob("*/status.json"):
            reused = _reusable_status(path, digest, root)
            if reused is not None:
                _remember_request(root, digest, reused["job_id"])
                return reused
    check = check or doctor(profile=request["model_profile"])
    _require_startable(check, request["diagnostic"])
    jobs.mkdir(parents=True, exist_ok=True)
    pending = [p for p in jobs.glob("*/status.json") if read_json(p).get("state") in {"queued", "running"}]
    if len(pending) >= 12:
        raise RuntimeError("대기 작업이 12개입니다. 기존 작업을 확인하거나 취소하세요.")
    jid = uuid.uuid4().hex[:12]
    job = job_path(jid)
    job.mkdir()
    write_json(job / "request.json", request)
    initial = {"job_id": jid, "state": "queued", "stage": "waiting_for_memory" if check.get("ram_shortfall_only") else "waiting_for_local_engine",
               "request_hash": digest, "created_at": time.time()}
    if check.get("ram_shortfall_only"):
        initial.update(free_ram_gib=check["free_ram_gib"], needed_free_ram_gib=check["minimum_free_ram_gib"])
    write_json(job / "status.json", initial)
    env = os.environ.copy()
    temporary = runtime_path(root, "tmp")
    from .storage import cache_environment
    env.update(cache_environment(root))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CVL_RUNTIME_DIR=str(root), TEMP=str(temporary), TMP=str(temporary))
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
    try:
        with (job / "worker.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(APP / "video_agent.py"), "_worker", jid], stdout=log, stderr=log, stdin=subprocess.DEVNULL, cwd=root, env=env, creationflags=flags)
        _remember_request(root, digest, jid)
        # The worker owns subsequent state writes; the parent never races it.
        return {**initial, "worker_pid": process.pid, "log_path": str(job / "worker.log")}
    except Exception as exc:
        write_json(job / "status.json", {**initial, "state": "failed", "error": str(exc)})
        raise

class EngineLock:
    def __init__(self, path):
        self.path, self.stream = path, None
    def acquire(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            self.stream.close()
            self.stream = None
            return False
    def release(self):
        if self.stream:
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()
            self.stream = None

def worker(job_id):
    from .resources import run_unthrottled
    run_unthrottled()
    job = job_path(job_id)
    cfg, root = load_config()
    initial = read_json(job / "status.json")
    cancelled = lambda: (job / "cancel.request").exists()
    lock = EngineLock(runtime_path(root, "engine.lock"))
    state = dict(initial)
    def progress(update):
        state.update(update)
        state["updated_at"] = time.time()
        write_json(job / "status.json", state)
    try:
        progress({"worker_pid": os.getpid()})
        last_heartbeat = time.monotonic()
        while not lock.acquire():
            if cancelled():
                raise InterruptedError("대기 중 취소됐습니다.")
            time.sleep(1)
            if time.monotonic() - last_heartbeat > 15:
                progress({"stage": "waiting_for_local_engine"})
                last_heartbeat = time.monotonic()
        if cancelled():
            raise InterruptedError("취소됐습니다.")
        request = read_json(job / "request.json")
        check = doctor(profile=request["model_profile"])
        if not check["ready"] and check.get("ram_shortfall_only"):
            from local_video.resources import wait_for_commit
            wait_for_commit(2**30, cancelled, progress)
            check = doctor(profile=request["model_profile"])
        if not check["ready"]:
            raise RuntimeError("실행 직전 자원 확인 실패: " + "; ".join(check["errors"]))
        if not request.get("diagnostic", False) and not check.get("ready_for_generation", False):
            raise RuntimeError("실행 직전 영상 품질 검증 기록을 확인할 수 없습니다.")
        check_domain_request(request, root)
        progress({"state": "running", "stage": "loading", "started_at": time.time()})
        if request["model_profile"] == "neodragon":
            from local_video.integrity import verify
            verify(request, root, progress, cancelled)
            from local_video.neodragon import generate_project
        else:
            from local_video.backend import generate_project
        from local_video.resources import learned_need_bytes, wait_for_commit
        for attempt in range(MEMORY_RETRIES + 1):
            try:
                result = generate_project(request, runtime_path(root, "jobs", job_id, "output"), runtime_path(root, "cache", "shots"), runtime_path(root, "models"), progress, cancelled)
                break
            except MemoryError as exc:
                # Finished scenes are cached, so a retry resumes at the scene that ran short.
                if attempt == MEMORY_RETRIES or cancelled():
                    raise
                progress({"stage": "waiting_for_memory", "memory_retry": attempt + 1, "memory_error": str(exc)[:300]})
                frames = int(request.get("duration_seconds", 2)) * 24 + 1 if request.get("continuous") else 49
                # Retry once commit can cover the stage again (it ran short); free RAM is not required.
                wait_for_commit(max(learned_need_bytes(root, "video_infer", frames), 2**30), cancelled, progress)
                progress({"stage": "loading", "memory_retry": attempt + 1})
        if cancelled():
            raise InterruptedError("취소됐습니다.")
        result.update(domain=request.get("domain", "general"), production_realism_verified=False)
        if request.get("domain") == "construction":
            result["quality_note"] = "현장 모드는 실험 초안입니다. 가중치 재학습과 작업자·장비의 자연스러운 형태·동작 품질은 확보되지 않았습니다."
        write_json(runtime_path(root, "jobs", job_id, "output", "result.json"), result)
        progress({"state": "completed", "stage": "complete", "result": result, "completed_at": time.time(), "visual_review": "required_not_automatically_verified"})
    except Exception as exc:
        progress({"state": "cancelled" if cancelled() or isinstance(exc, InterruptedError) else "failed", "stage": "stopped", "error": str(exc)[:2000], "completed_at": time.time()})
        print(repr(exc), file=sys.stderr, flush=True)
    finally:
        lock.release()

def cancel(job_id):
    job = job_path(job_id)
    current = status(job_id)
    if current["state"] in TERMINAL:
        return current
    (job / "cancel.request").write_text("cancel", encoding="ascii")
    return {"job_id": job_id, "state": current["state"], "cancel_requested": True}

def wait(job_id, seconds=45):
    deadline = time.monotonic() + min(max(seconds, 0), 60)
    while True:
        current = status(job_id)
        if current["state"] in TERMINAL or time.monotonic() >= deadline:
            return current
        time.sleep(min(1, max(0, deadline-time.monotonic())))

def list_jobs(limit=10):
    _, root = load_config()
    files = sorted((root / "jobs").glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [status(p.parent.name) for p in files[:max(1, min(limit, 30))]]
