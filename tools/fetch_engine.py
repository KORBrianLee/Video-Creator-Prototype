"""Fetch the pinned upstream headless CPU engine to the selected SSD runtime."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import urllib.request
import zipfile

ROOT = Path(__import__("local_video.storage", fromlist=["default_runtime"]).default_runtime())
NAME = "sd-master-3f8527a-bin-win-cpu-x64.zip"
SHA = "5e7caca2080321b25a12c1fa4175cb7d953f2b182309f8f73bfc9c725231d26c"
URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-929-3f8527a/" + NAME
ROOT.joinpath("downloads").mkdir(parents=True, exist_ok=True)
dest = ROOT / "downloads" / NAME
if not dest.exists() or hashlib.file_digest(dest.open("rb"), "sha256").hexdigest() != SHA:
    urllib.request.urlretrieve(URL, dest)
with dest.open("rb") as f:
    assert hashlib.file_digest(f, "sha256").hexdigest() == SHA, "Engine SHA256 mismatch"
target = ROOT / "engines" / "cpu"
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(dest) as z:
    for item in z.infolist():
        resolved = (target / item.filename).resolve()
        if not resolved.is_relative_to(target.resolve()):
            raise ValueError("Unsafe archive path")
    z.extractall(target)
print(str(target))
