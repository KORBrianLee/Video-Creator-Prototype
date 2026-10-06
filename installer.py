"""Explicit one-time downloads; generation itself never accesses the network."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
sys.stdout.reconfigure(encoding="utf-8")

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def d_root(value):
    # Retain the installer function name for older preparation scripts.
    from local_video.storage import runtime_root
    return runtime_root(value)

def download(url, dest, expected_sha=None, size=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (size is None or dest.stat().st_size == size) and (expected_sha is None or sha(dest) == expected_sha):
        print(json.dumps({"file": dest.name, "state": "already_verified"}), flush=True)
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "CursorVideoLocal/0.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                resumed = response.status == 206 and offset > 0
                with partial.open("ab" if resumed else "wb") as stream:
                    done = offset if resumed else 0
                    last_report = time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last_report > 30:
                            print(json.dumps({"file": dest.name, "downloaded_MiB": round(done / 2**20), "total_MiB": round(size / 2**20) if size else None}), flush=True)
                            last_report = time.monotonic()
            if size is not None and partial.stat().st_size != size:
                raise ValueError("Downloaded size mismatch")
            if expected_sha is not None and sha(partial) != expected_sha:
                partial.unlink()
                raise ValueError("SHA256 mismatch; file was not installed")
            partial.replace(dest)
            print(json.dumps({"file": dest.name, "state": "verified", "sha256": sha(dest)}), flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
        for item in z.infolist():
            destination = target / item.filename
            if not item.is_dir() and destination.is_file() and destination.stat().st_size == item.file_size:
                with z.open(item) as incoming, destination.open("rb") as existing:
                    unchanged = hashlib.file_digest(incoming, "sha256").digest() == hashlib.file_digest(existing, "sha256").digest()
                if unchanged:
                    continue  # Keep a matching Windows executable that may be running.
            z.extract(item, target)

def inside(root, *parts):
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("설치 경로가 선택한 실행 폴더 밖으로 연결됩니다.")
    return target

ADAPTIVE_DEFAULTS = {"threads": "auto", "minimum_free_ram_gib": "auto", "reserve_ram_gib": "auto",
                     "maximum_working_set_gib": "auto", "gpu_buffer_mib": "auto"}
FIXED_DEFAULTS = {"threads": 4, "minimum_free_ram_gib": 3.0, "reserve_ram_gib": (1.0, 1.5),
                  "maximum_working_set_gib": 5.0, "gpu_buffer_mib": "auto"}

def deploy(root, profile=None):
    target = inside(root, "app")
    target.mkdir(parents=True, exist_ok=True)
    allow = {"local_video", ".cursor", "docs", "examples", "tests", "tools"}
    for item in SOURCE.iterdir():
        if item.resolve() == (target / item.name).resolve():
            continue
        if item.name in {"video.config.json", "runtime-dir.txt"} and (target / item.name).exists():
            continue
        if item.is_file() and (item.suffix in {".py", ".json", ".md", ".txt", ".cmd", ".ps1"} or item.name == "LICENSE"):
            shutil.copy2(item, inside(root, "app", item.name))
        elif item.is_dir() and item.name in allow:
            shutil.copytree(item, inside(root, "app", item.name), dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    cfg = {"runtime_dir": str(root), "backend": "auto", "model_profile": profile or "neodragon", **ADAPTIVE_DEFAULTS}
    if not (target / "video.config.json").exists():
        (target / "video.config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        saved = json.loads((target / "video.config.json").read_text(encoding="utf-8-sig"))
        updated = {**saved, "runtime_dir": str(root)}
        # Fixed values written by releases before 0.7.3 become adaptive; user-chosen numbers stay.
        for key, old in FIXED_DEFAULTS.items():
            value = updated.get(key, "auto")
            if value in (old if isinstance(old, tuple) else (old,)):
                updated[key] = "auto"
        if updated != saved:
            (target / "video.config.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    python = root / "runtime" / "python" / "python.exe"
    cursor = target / ".cursor"
    cursor.mkdir(exist_ok=True)
    path = inside(root, "app", ".cursor", "mcp.json")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    if not isinstance(config, dict) or not isinstance(config.get("mcpServers", {}), dict):
        raise ValueError("기존 Cursor 연결 설정 형식이 올바르지 않습니다. 파일을 덮어쓰지 않았습니다.")
    config.setdefault("mcpServers", {})["local-video"] = {"command": str(python), "args": [str(target / "mcp_server.py")], "env": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "CVL_RUNTIME_DIR": str(root), "TEMP": str(root / "tmp"), "TMP": str(root / "tmp")}}
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)
    print(json.dumps({"project": str(target), "cursor_config": str(cursor / "mcp.json")}, ensure_ascii=False), flush=True)

PROFILE_LOCKS = {"neodragon": "neodragon.lock.json", "lightning": "lightning.lock.json", "wan": "models.lock.json"}
SITE_LOCK = "site-data.lock.json"

def install_locks(profile):
    names = ["runtime.lock.json", "models.lock.json", PROFILE_LOCKS[profile]]
    if profile == "neodragon":
        names += ["neodragon-research.lock.json", "neodragon-runtime.lock.json", "neodragon-safety.lock.json"]
    return names

def lock_digest(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def installed_digests(root):
    record = root / "audits" / "installed-locks.json"
    if record.is_file():
        return json.loads(record.read_text(encoding="utf-8"))
    app = root / "app"
    if app.resolve() == SOURCE or not app.is_dir():
        return {}
    # Installs made before this record existed: the previous deploy copied its locks into app.
    return {path.name: lock_digest(path) for path in app.glob("*.lock.json")}

def record_digests(root, names):
    path = root / "audits" / "installed-locks.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    saved.update({name: lock_digest(SOURCE / name) for name in names})
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

def prepare_site(root, baseline):
    manifest = root / "datasets" / "construction-v1" / "manifest.json"
    if manifest.is_file() and baseline.get(SITE_LOCK) == lock_digest(SOURCE / SITE_LOCK):
        return
    result = subprocess.run([sys.executable, "-B", str(SOURCE / "tools" / "prepare_site_data.py"), "--runtime-dir", str(root)])
    if result.returncode == 0 and json.loads(manifest.read_text(encoding="utf-8")).get("state") == "references_prepared_weights_not_trained":
        record_digests(root, [SITE_LOCK])
    else:
        print(json.dumps({"site_data": "not_prepared", "retry": "update.cmd"}, ensure_ascii=False), flush=True)

TORCH_LOCKS = {"xpu": "neodragon-xpu.lock.json", "cuda128": "neodragon-cuda128.lock.json", "cuda126": "neodragon-cuda126.lock.json"}


def torch_runtime_for(vendor, compute_capability=None):
    """(backend, lock) for the computer's main GPU, or (None, None) when no PyTorch GPU build applies.

    CUDA 12.8 builds cover Turing and newer (including RTX 50); CUDA 12.6 builds keep Maxwell to Volta working.
    AMD and unknown GPUs keep the Vulkan and OpenCL paths instead of a download that cannot be used.
    """
    if vendor == "intel":
        return "xpu", "xpu"
    if vendor == "nvidia":
        return "cuda", "cuda128" if compute_capability is None or compute_capability >= 7.5 else "cuda126"
    return None, None


def nvidia_compute_capability():
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        values = [float(line) for line in run.stdout.split() if line.replace(".", "", 1).isdigit()]
        return max(values) if run.returncode == 0 and values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def detect_torch_runtime(root):
    """Pick the PyTorch GPU runtime from the GPU the planner would choose on this computer."""
    from local_video import devices
    try:
        selected = devices.plan(Path(root), "auto").get("selected_device")
    except (OSError, RuntimeError, ValueError):
        selected = None
    if not selected:
        return None, None
    vendor = selected.get("vendor")
    return torch_runtime_for(vendor, nvidia_compute_capability() if vendor == "nvidia" else None)


def torch_runtime_needed(root, profile, no_gpu_torch, detect=detect_torch_runtime):
    """Runtime folders still to install for this computer's GPU; empty for CPU-only, AMD and opt-out installs."""
    if profile != "neodragon" or no_gpu_torch or os.name != "nt":
        return []
    backend, lock_key = detect(root)
    if backend is None:
        return []
    from local_video import accelerator
    record = accelerator.site(root, backend) / "install-record.json"
    try:
        current = json.loads(record.read_text(encoding="utf-8")).get("lock") == lock_digest(SOURCE / TORCH_LOCKS[lock_key])
    except (OSError, ValueError):
        current = False
    return [] if current else [f"runtime/{accelerator.BACKENDS[backend]['site']}"]


def skyreels_needed(root, profile, no_gpu_torch, detect=detect_torch_runtime):
    """The long-video model is only downloaded for computers whose main GPU is NVIDIA (CUDA tier)."""
    if profile != "neodragon" or no_gpu_torch or os.name != "nt":
        return []
    backend, _ = detect(root)
    if backend != "cuda":
        return []
    from local_video import skyreels
    return [] if skyreels.installed(root, SOURCE) else ["models/skyreels-v2-df-1.3b"]


def ltx_needed(root, profile, no_gpu_torch, detect=detect_torch_runtime):
    """LTX-Video (long clips on Intel GPUs) is downloaded only when the main GPU uses the XPU runtime."""
    if profile != "neodragon" or no_gpu_torch or os.name != "nt":
        return []
    backend, _ = detect(root)
    if backend != "xpu":
        return []
    from local_video import ltx
    return [] if ltx.installed(root, SOURCE) else ["models/ltx-video-2b-0.9.8"]


def install_skyreels(root, convert=None):
    from local_video import skyreels
    return install_locked_model(root, skyreels.LOCK, convert)


def install_locked_model(root, lock_name, convert=None):
    """Download each locked file, verify it, and rewrite FP32 shards as BF16 right away (resumable).

    Peak extra disk is one FP32 shard (about 5 GB). Files may come from different repositories
    (`repo`/`revision`/`source` per file); otherwise the lock's repo and the file path are used.
    """
    from local_video import skyreels
    lock = json.loads((SOURCE / lock_name).read_text(encoding="utf-8"))
    target = inside(root, "models", lock["folder"])
    target.mkdir(parents=True, exist_ok=True)
    record_path = target / "install-record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    if record.get("revision") != lock["revision"]:
        record = {"revision": lock["revision"], "files": {}}
    pending = [item for item in lock["files"] if item["path"] not in record["files"] or not (target / item["path"]).is_file()]
    final_bytes = sum(item["size"] // 2 if item["convert_to_bf16"] else item["size"] for item in pending)
    largest = max((item["size"] for item in pending), default=0)
    if pending and shutil.disk_usage(root).free < final_bytes + largest + 2 * 2**30:
        need = round((final_bytes + largest + 2 * 2**30) / 2**30)
        raise ValueError(f"{lock['model']} 모델을 받을 SSD 여유 공간이 부족합니다(약 {need}GB 필요).")
    if convert is None:
        python = inside(root, "runtime", "neodragon-python", "python.exe")
        def convert(path):
            result = subprocess.run([str(python), "-B", str(SOURCE / "local_video" / "skyreels.py"), "--convert", str(path)],
                                    env={**os.environ, "TMP": str(root / "tmp"), "TEMP": str(root / "tmp")})
            if result.returncode:
                raise RuntimeError(f"BF16 변환 실패: {path.name}")
    for item in pending:
        destination = inside(root, "models", lock["folder"], *item["path"].split("/"))
        url = (f'https://huggingface.co/{item.get("repo", lock.get("repo"))}/resolve/'
               f'{item.get("revision", lock["revision"])}/{item.get("source", item["path"])}')
        download(url, destination, item["sha256"], item["size"])
        if item["convert_to_bf16"]:
            convert(destination)
        record["files"][item["path"]] = sha(destination)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    converted_folders = {item["path"].rsplit("/", 1)[0] for item in lock["files"] if item["convert_to_bf16"] and "/" in item["path"]}
    for index in [item["path"] for item in lock["files"] if item["path"].endswith(".safetensors.index.json")]:
        if index.rsplit("/", 1)[0] in converted_folders and (target / index).is_file():
            skyreels.fix_index_total_size(target / index)
            record["files"][index] = sha(target / index)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"model": lock["model"], "state": "installed", "folder": str(target)}, ensure_ascii=False), flush=True)
    return True


def install_torch_runtime(root, detect=detect_torch_runtime):
    """Hash-pinned install beside the CPU runtime. Failure leaves the CPU/Vulkan/OpenCL paths working."""
    from local_video import accelerator
    backend, lock_key = detect(root)
    if backend is None:
        return False
    lock_name = TORCH_LOCKS[lock_key]
    lock = json.loads((SOURCE / lock_name).read_text(encoding="utf-8"))
    python = inside(root, "runtime", "neodragon-python", "python.exe")
    target = inside(root, "runtime", accelerator.BACKENDS[backend]["site"])
    requirements = inside(root, "tmp", f"{backend}-requirements.txt")
    requirements.write_text("".join(f'{p["name"]}=={p["version"]} --hash=sha256:{p["sha256"]}\n' for p in lock["packages"]), encoding="utf-8")
    environment = {**os.environ, "PIP_CACHE_DIR": str(root / "cache" / "pip"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp")}
    result = subprocess.run([str(python), "-m", "pip", "install", "--target", str(target), "--upgrade", "--no-deps",
                             "--require-hashes", "--index-url", lock["index_url"], "--extra-index-url", lock["extra_index_url"],
                             "--no-warn-script-location", "-r", str(requirements)], env=environment)
    if result.returncode:
        print(json.dumps({"gpu_runtime": "not_installed", "backend": backend, "effect": "PyTorch GPU path unavailable; CPU/Vulkan/OpenCL paths are used", "retry": "update.cmd"}, ensure_ascii=False), flush=True)
        return False
    (target / "install-record.json").write_text(json.dumps({"lock": lock_digest(SOURCE / lock_name), "lock_name": lock_name}), encoding="utf-8")
    accelerator._record_path(root, backend).unlink(missing_ok=True)
    print(json.dumps({"gpu_runtime": "installed", "backend": backend, "gpu_probe": accelerator.probe(root, backend)}, ensure_ascii=False), flush=True)
    return True


def wants_vulkan(profile, no_vulkan):
    return profile == "wan" or not no_vulkan


def missing_engines(root, with_vulkan):
    """Engine folders an update must download; the Vulkan GPU engine is part of every default install."""
    return ["engines/vulkan"] if with_vulkan and not (Path(root) / "engines" / "vulkan" / "sd-cli.exe").is_file() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "models", "deploy", "all", "update"])
    from local_video.storage import default_runtime
    parser.add_argument("--runtime-dir", default=default_runtime())
    parser.add_argument("--with-vulkan", action="store_true", help="kept for old scripts; Vulkan is installed by default")
    parser.add_argument("--no-vulkan", action="store_true", help="skip the Vulkan GPU engine (CPU only)")
    parser.add_argument("--no-gpu-torch", "--no-xpu", dest="no_gpu_torch", action="store_true",
                        help="skip the PyTorch GPU runtime (CUDA or XPU); the video model stays on the CPU/OpenCL path")
    parser.add_argument("--backend", choices=["auto", "gpu", "intel-gpu", "intel-vulkan", "cpu"])
    parser.add_argument("--profile", choices=["neodragon", "lightning", "wan"])
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    if args.profile is None:
        deployed = root / "app" / "video.config.json"
        saved_profile = json.loads(deployed.read_text(encoding="utf-8-sig")).get("model_profile") if deployed.is_file() else None
        args.profile = saved_profile if saved_profile in PROFILE_LOCKS else "neodragon"
    args.with_vulkan = wants_vulkan(args.profile, args.no_vulkan)
    baseline = installed_digests(root)
    if args.action == "update":
        stale = [name for name in install_locks(args.profile) if baseline.get(name) != lock_digest(SOURCE / name)]
        stale += (missing_engines(root, args.with_vulkan) + torch_runtime_needed(root, args.profile, args.no_gpu_torch)
                  + skyreels_needed(root, args.profile, args.no_gpu_torch) + ltx_needed(root, args.profile, args.no_gpu_torch))
        print(json.dumps({"update": "full_verify" if stale else "code_only", "changed_locks": stale}), flush=True)
        args.action = "all" if stale else "deploy"
        verified = not stale
    else:
        verified = False
    root.mkdir(parents=True, exist_ok=True)
    for name in ["tmp", "downloads", "models", "cache", "jobs", "audits"]:
        inside(root, name).mkdir(exist_ok=True)
    os.environ.update(TMP=str(root / "tmp"), TEMP=str(root / "tmp"), PIP_CACHE_DIR=str(root / "cache" / "pip"), PYTHONDONTWRITEBYTECODE="1")
    lock = json.loads((SOURCE / "models.lock.json").read_text(encoding="utf-8"))
    lock_file = PROFILE_LOCKS[args.profile]
    model_lock = json.loads((SOURCE / lock_file).read_text(encoding="utf-8"))
    if args.action in {"prepare", "all"}:
        archive = root / "downloads" / "python-3.12.10-embed-amd64.zip"
        runtime_lock = json.loads((SOURCE / "runtime.lock.json").read_text(encoding="utf-8"))
        download(runtime_lock["python"]["url"], archive, runtime_lock["python"]["sha256"])
        pyroot = inside(root, "runtime", "python")
        extract(archive, pyroot)
        (pyroot / "python312._pth").write_text("python312.zip\n.\nLib/site-packages\n../../app\nimport site\n", encoding="utf-8")
        # Pinned Windows wheels are zip archives. No system Python or pip needed.
        for item in runtime_lock["wheels"]:
            wheel = root / "downloads" / item["filename"]
            download(item["url"], wheel, item["sha256"], item["size"])
            extract(wheel, inside(root, "runtime", "python", "Lib", "site-packages"))
        for backend in (["cpu", "vulkan"] if args.with_vulkan else ["cpu"]):
            item = lock["engine"][backend]
            archive = root / "downloads" / item["url"].split("/")[-1]
            download(item["url"], archive, item["sha256"], item["size"])
            extract(archive, inside(root, "engines", backend))
        audit = {"python_source": "https://www.python.org/ftp/python/3.12.10/", "python_archive_sha256": sha(root / "downloads" / "python-3.12.10-embed-amd64.zip"), "note": "Python archive downloaded from python.org over TLS; digest recorded after acquisition."}
        (root / "audits" / "runtime-source.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if args.profile == "neodragon":
            from tools.setup_neodragon_probe import prepare
            from local_video.neodragon import runtime_errors
            if runtime_errors(SOURCE, root) or not inside(root, "engines", "experimental-neodragon", "source-provenance.json").is_file():
                prepare(root, json.loads((SOURCE / "neodragon-research.lock.json").read_text(encoding="utf-8")))
            if torch_runtime_needed(root, args.profile, args.no_gpu_torch):
                install_torch_runtime(root)
            for needed, lock_name in ((skyreels_needed, "skyreels.lock.json"), (ltx_needed, "ltx.lock.json")):
                if needed(root, args.profile, args.no_gpu_torch):
                    try:
                        install_locked_model(root, lock_name)
                    except (OSError, ValueError, RuntimeError) as exc:
                        print(json.dumps({"long_video_model": "not_installed", "lock": lock_name, "reason": str(exc)[:200],
                                          "effect": "Neo is used, up to 8 seconds", "retry": "update.cmd"}, ensure_ascii=False), flush=True)
    if args.action in {"models", "all"}:
        missing_bytes = sum(x["size"] for x in model_lock["files"] if not (root / "models" / x["filename"]).is_file())
        if shutil.disk_usage(root).free < missing_bytes + 2 * 2**30:
            raise ValueError("선택한 SSD의 여유 공간이 부족합니다.")
        for item in model_lock["files"]:
            source_file = item.get("source_file", item.get("path"))
            url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{source_file}'
            download(url, inside(root, "models", *item["filename"].split("/")), item["sha256"], item["size"])
        if args.profile == "lightning" and model_lock.get("prepared_base"):
            from prepare_lightning import prepare
            prepared = model_lock["prepared_base"]
            base = next(item for item in model_lock["files"] if item["role"] == "base")
            prepare(root / "models" / base["filename"], root / "models" / prepared["filename"], base["sha256"])
            if sha(root / "models" / prepared["filename"]) != prepared["sha256"]:
                raise ValueError("Lightning 호환 사본 SHA256이 고정된 값과 다릅니다.")
        (root / "models" / lock_file).write_text(json.dumps(model_lock, indent=2), encoding="utf-8")
    deploy(root, args.profile)
    if args.backend is not None:
        config_path = inside(root, "app", "video.config.json")
        saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
        saved["backend"] = args.backend
        config_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.profile == "neodragon" and args.action in {"models", "all"}:
        config_path = inside(root, "app", "video.config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(model_profile="neodragon")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.profile == "neodragon" and args.action in {"models", "all"}:
        from local_video.opencl_linear import plan
        config = json.loads(inside(root, "app", "video.config.json").read_text(encoding="utf-8"))
        selected = plan(config.get("backend", "auto"), config.get("gpu_device"))
        if selected["gpu_inference"]:
            print(json.dumps({"int8_conversion": "not_required", "reason": "GPU streams original BF16 matrices without creating a full INT8 copy"}), flush=True)
        else:
            from local_video.neodragon import run_stage
            work = inside(root, "tmp", "neodragon-prepare-int8")
            work.mkdir(parents=True, exist_ok=True)
            (work / "request.json").write_text(json.dumps({"prompt": "", "width": 512, "height": 320, "frames": 49, "seed": 42, "threads": 4}), encoding="utf-8")
            request = {"threads": 4}
            result = run_stage(root, work, "video_pack", request, lambda update: print(json.dumps(update), flush=True), lambda: False)
            (root / "audits" / "neodragon-int8-setup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.action == "all" or verified:
        record_digests(root, install_locks(args.profile))
        prepare_site(root, baseline)

if __name__ == "__main__":
    main()
