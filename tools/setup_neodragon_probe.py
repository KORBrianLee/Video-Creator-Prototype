"""Prepare a separate D-only CPU research environment, never a production engine."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, download, extract, inside, sha


def read(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CursorVideoLocal/0.3-research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Metadata size limit exceeded")
    return raw


def prepare(root, lock):
    pyroot = inside(root, "runtime", "neodragon-python")
    pyroot.mkdir(parents=True, exist_ok=True)
    runtime_lock = json.loads((SOURCE / "runtime.lock.json").read_text(encoding="utf-8"))
    spec = runtime_lock["python"]
    archive = inside(root, "downloads", spec["filename"])
    download(spec["url"], archive, spec["sha256"])
    extract(archive, pyroot)
    site = pyroot / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (pyroot / "python312._pth").write_text("python312.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
    pip_info = json.loads(read("https://pypi.org/pypi/pip/25.2/json"))
    pip_wheel = next(x for x in pip_info["urls"] if x["filename"].endswith("py3-none-any.whl"))
    wheel = inside(root, "downloads", pip_wheel["filename"])
    download(pip_wheel["url"], wheel, pip_wheel["digests"]["sha256"], pip_wheel["size"])
    extract(wheel, site)
    environment = {**os.environ, "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
                   "PIP_CACHE_DIR": str(root / "cache" / "pip"), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONIOENCODING": "utf-8", "HF_HOME": str(root / "cache" / "huggingface"),
                   "TORCH_HOME": str(root / "cache" / "torch")}
    dependency_lock = SOURCE / "neodragon-runtime.lock.json"
    pinned = json.loads(dependency_lock.read_text(encoding="utf-8"))
    requirements = [x["name"] + " @ " + x["url"].split("#", 1)[0] + "#sha256=" + x["sha256"]
                    for x in pinned["packages"]]
    report = inside(root, "audits", "neodragon-cpu-dependencies.json")
    log_path = inside(root, "audits", "neodragon-cpu-setup.log")
    command = [str(pyroot / "python.exe"), "-m", "pip", "install", "--target", str(site),
               "--only-binary=:all:", "--no-input", "--no-cache-dir", "--no-compile",
               "--disable-pip-version-check", "--no-deps", "--report", str(report), *requirements]
    print(json.dumps({"stage": "installing_isolated_cpu_dependencies", "log": str(log_path)}), flush=True)
    with log_path.open("wb") as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    code_root = inside(root, "engines", "experimental-neodragon")
    code_root.mkdir(parents=True, exist_ok=True)
    code_repo, revision = lock["code_repo"], lock["code_revision"]
    tree = json.loads(read(f"https://api.github.com/repos/{code_repo}/git/trees/{revision}?recursive=1"))
    acquired = []
    for item in tree["tree"]:
        name = item["path"]
        if item["type"] != "blob" or not (name == "LICENSE.txt" or (name.startswith("neodragon/") and name.endswith(".py"))):
            continue
        payload = read(f"https://raw.githubusercontent.com/{code_repo}/{revision}/{name}")
        digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if digest != item["sha"]:
            raise ValueError("Pinned Git source content mismatch")
        target = inside(root, "engines", "experimental-neodragon", *name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        acquired.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest()})
    (code_root / "source-provenance.json").write_text(json.dumps({"repo": code_repo, "revision": revision, "files": acquired}, indent=2), encoding="utf-8")
    rail = read("https://www.qualcomm.com/site/responsible-ai-license")
    (code_root / "Qualcomm-Responsible-AI-License-source.html").write_bytes(rail)
    print(json.dumps({"stage": "cpu_environment_prepared", "python": str(pyroot / "python.exe"), "code_revision": revision}), flush=True)


def models(root, lock):
    missing = sum(f["size"] for f in lock["files"] if not inside(root, "models", "experimental-neodragon", *f["path"].split("/")).is_file())
    import shutil
    if shutil.disk_usage(root).free < missing + 3 * 2**30:
        raise ValueError("Not enough D space for research weights and reserve")
    model_root = inside(root, "models", "experimental-neodragon")
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in lock["files"]:
        target = inside(root, "models", "experimental-neodragon", *spec["path"].split("/"))
        url = f'https://huggingface.co/{lock["repo"]}/resolve/{lock["revision"]}/' + urllib.parse.quote(spec["path"], safe="/")
        download(url, target, spec.get("sha256"), spec["size"])
        if not spec.get("sha256") and spec.get("git_blob"):
            payload = target.read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if digest != spec["git_blob"]:
                raise ValueError("Pinned model configuration content mismatch")
        manifest.append({**spec, "acquired_sha256": sha(target)})
    (model_root / "model-provenance.json").write_text(json.dumps({**lock, "files": manifest}, indent=2), encoding="utf-8")
    safety_root = inside(root, "models", "experimental-safety-checker")
    safety_root.mkdir(parents=True, exist_ok=True)
    metadata_path = safety_root / "metadata.json"
    if metadata_path.exists():
        safety = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        safety = json.loads(read("https://huggingface.co/api/models/CompVis/stable-diffusion-safety-checker?blobs=true"))
        metadata_path.write_text(json.dumps(safety, indent=2), encoding="utf-8")
    safety_files = []
    for item in safety["siblings"]:
        name = item["rfilename"]
        if name not in {"README.md", "config.json", "preprocessor_config.json", "pytorch_model.bin"}:
            continue
        target = inside(root, "models", "experimental-safety-checker", name)
        url = f'https://huggingface.co/{safety["id"]}/resolve/{safety["sha"]}/{name}'
        download(url, target, item.get("lfs", {}).get("sha256"), item["size"])
        safety_files.append({"path": name, "sha256": sha(target), "size": item["size"]})
    (safety_root / "model-provenance.json").write_text(json.dumps({"repo": safety["id"], "revision": safety["sha"], "files": safety_files}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "models", "all"])
    parser.add_argument("--runtime-dir", default=__import__("local_video.storage", fromlist=["default_runtime"]).default_runtime())
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    for name in ["tmp", "downloads", "cache", "audits"]:
        inside(root, name).mkdir(parents=True, exist_ok=True)
    lock = json.loads((SOURCE / "neodragon-research.lock.json").read_text(encoding="utf-8"))
    if args.action in {"prepare", "all"}:
        prepare(root, lock)
    if args.action in {"models", "all"}:
        models(root, lock)


if __name__ == "__main__":
    main()
