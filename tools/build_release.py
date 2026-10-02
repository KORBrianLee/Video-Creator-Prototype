"""Reproducible app-source rebuild; runtime binaries and weights stay separate."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside
from local_video.control import quality_validation
from mcp_server import SERVER_INFO


def files(root):
    selected = {}
    folders = {"local_video", "tools", "tests", "docs", "examples"}
    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(SOURCE)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".tmp"}:
            continue
        if relative.as_posix() in {"video.config.json", ".cursor/mcp.json", "BUILD_MANIFEST.json"}:
            continue
        top = relative.parts[0]
        permitted = top in folders or (top == ".cursor" and len(relative.parts) > 2
                                       and relative.parts[1] in {"rules", "skills"})
        permitted |= len(relative.parts) == 1 and (path.suffix in {".py", ".json", ".md", ".txt", ".cmd", ".ps1"}
                                                   or path.name == "LICENSE")
        if permitted:
            if not path.resolve().is_relative_to(SOURCE.resolve()):
                raise ValueError("Release source leaves the app directory: " + str(relative))
            data = path.read_bytes()
            if path.suffix == ".py":
                ast.parse(data.decode("utf-8-sig"), filename=str(relative))
            selected[relative.as_posix()] = data
    # Carry the already-reviewed baseline with the wrapper, so copying the
    # rebuilt app beside the prepared D runtime does not lose its quality gate.
    for relative in ("examples/validated-beach.mp4", "examples/validated-beach.jpg"):
        selected[relative] = inside(root, "app", *relative.split("/")).read_bytes()
    for relative in ("examples/site-draft.mp4", "examples/site-draft.jpg", "examples/site-draft-ATTRIBUTION.md"):
        path = inside(root, "app", *relative.split("/"))
        if path.is_file():
            selected[relative] = path.read_bytes()
    return dict(sorted(selected.items()))


def verify(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release ZIP CRC check failed")
        manifest = json.loads(archive.read("BUILD_MANIFEST.json"))
        expected = {item["path"] for item in manifest["files"]}
        if set(archive.namelist()) != expected | {"BUILD_MANIFEST.json"} or len(archive.namelist()) != len(expected)+1:
            raise ValueError("Release file inventory mismatch")
        for item in manifest["files"]:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Release hash mismatch: " + item["path"])
            if item["path"].endswith(".py"):
                ast.parse(data.decode("utf-8-sig"), filename=item["path"])
        quality = json.loads(archive.read("docs/quality/neodragon-cpu.json"))
        if hashlib.sha256(archive.read("examples/validated-beach.mp4")).hexdigest() != quality["video_sha256"]:
            raise ValueError("Release baseline video does not match the quality record")
        if "examples/site-draft.mp4" in expected:
            domain = json.loads(archive.read("docs/quality/construction-cpu.json"))
            if hashlib.sha256(archive.read("examples/site-draft.mp4")).hexdigest() != domain["video_sha256"]:
                raise ValueError("Release construction draft does not match its review record")
            if "examples/site-draft-ATTRIBUTION.md" not in expected:
                raise ValueError("The licensed construction draft needs its attribution")
        if "examples/validated-gpu.mp4" in expected:
            gpu = json.loads(archive.read("docs/quality/neodragon-opencl.json"))
            if hashlib.sha256(archive.read("examples/validated-gpu.mp4")).hexdigest() != gpu["video_sha256"]:
                raise ValueError("Release GPU sample does not match its review record")
        return manifest


def main():
    parser = argparse.ArgumentParser()
    from local_video.storage import default_runtime
    parser.add_argument("--runtime-dir", default=default_runtime())
    parser.add_argument("--verify")
    args = parser.parse_args()
    root = d_root(args.runtime_dir)
    if args.verify:
        path = Path(args.verify).resolve(strict=True)
        if not path.is_relative_to((root / "releases").resolve()):
            raise ValueError("Release verification is limited to the selected SSD release directory")
        manifest = verify(path)
        print(json.dumps({"verified": True, "version": manifest["version"], "files": len(manifest["files"])}))
        return
    state, ready = quality_validation("neodragon", root)
    if not ready:
        # A source rebuild can preserve an unverified/failed quality state.
        # It must not relabel the historical CPU beach as current GPU proof.
        print(json.dumps({"quality_status": state, "production_quality_certified": False}), file=sys.stderr)
    payload = files(root)
    manifest = {"schema": 1, "version": SERVER_INFO["version"],
        "artifact_scope": "app_source_and_reviewed_sample_no_runtime_binaries_or_model_weights",
        "quality_status": state, "gram_verified": False, "construction_realism_verified": False,
        "files": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for path, data in payload.items()]}
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    identity = hashlib.sha256(encoded).hexdigest()[:12]
    directory = inside(root, "releases")
    directory.mkdir(exist_ok=True)
    path = inside(root, "releases", f"cursor-video-local-{manifest['version']}-{identity}.zip")
    pending = path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in {**payload, "BUILD_MANIFEST.json": encoded}.items():
                item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                archive.writestr(item, data)
        verify(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_bytes(encoded)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    latest = {"version": manifest["version"], "archive": str(path), "sha256": digest,
              "manifest": str(sidecar), "file_count": len(payload), "bytes": path.stat().st_size}
    inside(root, "releases", "LATEST.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(latest, ensure_ascii=False))


if __name__ == "__main__":
    main()
