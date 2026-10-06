"""Explicit SSD selection with the same containment rules on C and D."""
from pathlib import Path, PureWindowsPath

PREFERRED_RUNTIME = r"D:\VideoCreator\CursorVideoRuntime"
FALLBACK_RUNTIME = r"C:\CursorVideoRuntime"


def runtime_root(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("실행 폴더는 절대 경로여야 합니다.")
    windows = PureWindowsPath(value)
    if (windows.drive.upper() not in {"C:", "D:"} or not windows.is_absolute()
            or len(windows.parts) < 2 or ".." in windows.parts):
        raise ValueError("C 또는 D SSD의 전용 하위 폴더를 지정하세요. 드라이브 루트·상대 경로·네트워크 경로는 허용하지 않습니다.")
    volume = Path(windows.anchor)
    if not volume.is_dir():
        raise ValueError("선택한 드라이브가 없습니다. 저장 위치를 자동 변경하지 않습니다.")
    root = Path(str(windows)).resolve()
    if root.drive.upper() != windows.drive.upper() or not root.is_relative_to(volume.resolve()):
        raise ValueError("실제 실행 폴더가 선택한 SSD 밖으로 연결됩니다.")
    return root


def default_runtime():
    """D holds models, caches and compute scratch; C is only used when the PC has no D volume."""
    return PREFERRED_RUNTIME if Path(PureWindowsPath(PREFERRED_RUNTIME).anchor).is_dir() else FALLBACK_RUNTIME


def cache_environment(root):
    """Every cache a library or GPU driver would otherwise write under the user profile on C."""
    root = Path(root)
    cache = root / "cache"
    temporary = str(root / "tmp")
    return {"TEMP": temporary, "TMP": temporary,
            "HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"), "PIP_CACHE_DIR": str(cache / "pip"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
            # SYCL_CACHE_DIR / NEO_CACHE_DIR are left alone: redirecting the Intel kernel cache hung Iris Xe kernels.
            "CUDA_CACHE_PATH": str(cache / "cuda"), "MPLCONFIGDIR": str(cache / "matplotlib")}
