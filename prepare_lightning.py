"""Add Lightning's linear beta schedule to a *copy* of the SD1.5 GGUF.

Only the new GGUF tensor table and appended F32 alphas_cumprod are changed.
The original tensor payload is copied byte for byte, and original downloads
are never replaced. Standard-library streaming keeps temporary RAM bounded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import uuid

PREPARATION_VERSION = "lightning-linear-alphas-gguf-v1"
TENSOR_NAME = "alphas_cumprod"
FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
         4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
         10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def _read(stream, count):
    if count < 0 or count > 64*1024*1024:
        raise ValueError("GGUF header 크기가 허용 범위를 넘습니다.")
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("GGUF header가 잘렸습니다.")
    return data


def _integer(stream, code):
    return struct.unpack(code, _read(stream, struct.calcsize(code)))[0]


def _string(stream):
    return _read(stream, _integer(stream, "<Q")).decode("utf-8")


def _metadata_value(stream, dtype, depth=0):
    if depth > 4:
        raise ValueError("허용되지 않는 GGUF metadata 중첩입니다.")
    if dtype in FIXED:
        code, length = FIXED[dtype]
        return struct.unpack(code, _read(stream, length))[0]
    if dtype == 8:
        return _string(stream)
    if dtype == 9:
        inner, count = _integer(stream, "<I"), _integer(stream, "<Q")
        if count > 10_000_000:
            raise ValueError("GGUF metadata 배열이 너무 큽니다.")
        if inner in FIXED:
            _read(stream, count*FIXED[inner][1])
        else:
            for _ in range(count):
                _metadata_value(stream, inner, depth+1)
        return None
    raise ValueError(f"지원하지 않는 GGUF metadata 자료형: {dtype}")


def inspect_gguf(source: Path) -> dict:
    with source.open("rb") as stream:
        if _read(stream, 4) != b"GGUF":
            raise ValueError("입력은 GGUF 파일이어야 합니다.")
        version = _integer(stream, "<I")
        if version not in {2, 3}:
            raise ValueError("지원하지 않는 GGUF version입니다.")
        tensor_count, metadata_count = _integer(stream, "<Q"), _integer(stream, "<Q")
        if not 0 < tensor_count < 100_000 or metadata_count > 100_000:
            raise ValueError("GGUF 항목 수가 허용 범위를 넘습니다.")
        alignment = 32
        for _ in range(metadata_count):
            key = _string(stream)
            value = _metadata_value(stream, _integer(stream, "<I"))
            if key == "general.alignment":
                alignment = value
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment-1):
            raise ValueError("유효하지 않은 GGUF data 정렬입니다.")
        metadata_end = stream.tell()
        entries, names = [], set()
        for _ in range(tensor_count):
            start = stream.tell()
            name = _string(stream)
            if name in names:
                raise ValueError("중복 GGUF tensor 이름입니다.")
            names.add(name)
            dimensions = _integer(stream, "<I")
            if not 1 <= dimensions <= 4:
                raise ValueError("지원하지 않는 GGUF tensor 차원입니다.")
            shape = [_integer(stream, "<Q") for _ in range(dimensions)]
            dtype, offset = _integer(stream, "<I"), _integer(stream, "<Q")
            end = stream.tell()
            stream.seek(start)
            raw = _read(stream, end-start)
            entries.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset, "raw": raw})
        header_end = stream.tell()
        data_start = (header_end+alignment-1)//alignment*alignment
        if data_start >= source.stat().st_size or any(e["offset"] >= source.stat().st_size-data_start for e in entries):
            raise ValueError("GGUF tensor payload 위치가 파일 범위를 벗어납니다.")
        stream.seek(0)
        metadata_header = _read(stream, metadata_end)
    return {"version": version, "tensor_count": tensor_count, "metadata_count": metadata_count,
            "alignment": alignment, "metadata_header": metadata_header,
            "entries": entries, "data_start": data_start}


def linear_alphas() -> list[float]:
    """1000-step linear beta .00085 -> .012; stored as little-endian F32."""
    # Float32 beta/alpha operations match the scheduler's tensor precision;
    # accumulate in double, then round the stored cumulative values to F32.
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    start, end = f32(0.00085), f32(0.012)
    result, product = [], 1.0
    for step in range(1000):
        beta = f32(start+(end-start)*step/999)
        alpha = f32(1.0-beta)
        product *= alpha
        result.append(f32(product))
    return result


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_record(path: Path, value: dict):
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def prepare(source: Path, destination: Path, expected_sha256: str) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.parent != source.parent:
        raise ValueError("원본과 별도의 같은 모델 폴더에 파생 파일을 저장해야 합니다.")
    if source.suffix.lower() != ".gguf" or destination.name != source.stem+"-lightning-linear.gguf":
        raise ValueError("파생 파일 이름은 원본 이름 뒤에 -lightning-linear.gguf를 붙여야 합니다.")
    if os.name == "nt" and (source.drive.upper() not in {"C:", "D:"} or destination.drive.upper() != source.drive.upper()):
        raise ValueError("모델 준비는 D 드라이브에서만 가능합니다.")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("원본 SHA256 잠금 값이 필요합니다.")
    before = source.stat()
    source_hash = _hash(source)
    if source_hash != expected_sha256:
        raise ValueError("원본 모델 SHA256 검증에 실패했습니다. 원본을 보존하고 중단합니다.")
    record_path = destination.with_suffix(".provenance.json")
    if not record_path.resolve().is_relative_to(source.parent):
        raise ValueError("검증 기록 경로가 모델 폴더 밖으로 연결됩니다.")
    try:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if destination.is_file() and existing.get("preparation_version") == PREPARATION_VERSION and existing.get("input_sha256") == source_hash:
        if _hash(destination) == existing.get("sha256"):
            return {**existing, "reused": True}
    if destination.exists() and not existing.get("preparation_version"):
        raise FileExistsError("출처 기록이 없는 기존 파생 경로를 덮어쓰지 않습니다.")
    original = inspect_gguf(source)
    alpha_values = linear_alphas()
    alpha_bytes = struct.pack("<1000f", *alpha_values)
    original_payload_size = before.st_size-original["data_start"]
    alpha_offset = (original_payload_size+original["alignment"]-1)//original["alignment"]*original["alignment"]
    entries = [e for e in original["entries"] if e["name"] != TENSOR_NAME]
    tensor_header = b"".join(e["raw"] for e in entries)
    encoded_name = TENSOR_NAME.encode()
    tensor_header += (struct.pack("<Q", len(encoded_name))+encoded_name+
                      struct.pack("<IQIQ", 1, 1000, 0, alpha_offset))
    metadata_header = bytearray(original["metadata_header"])
    struct.pack_into("<Q", metadata_header, 8, len(entries)+1)
    header = bytes(metadata_header)+tensor_header
    new_data_start = (len(header)+original["alignment"]-1)//original["alignment"]*original["alignment"]
    temporary = destination.with_name(destination.name+"."+uuid.uuid4().hex+".tmp")
    original_payload_hash = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            output_stream.write(header)
            output_stream.write(b"\0"*(new_data_start-len(header)))
            input_stream.seek(original["data_start"])
            copied = 0
            while chunk := input_stream.read(4*1024*1024):
                original_payload_hash.update(chunk)
                output_stream.write(chunk)
                copied += len(chunk)
            if copied != original_payload_size:
                raise ValueError("모델 복사 중 원본 payload 크기가 변했습니다.")
            output_stream.write(b"\0"*(alpha_offset-original_payload_size))
            output_stream.write(alpha_bytes)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("모델 준비 중 원본이 변경됐습니다.")
        prepared = inspect_gguf(temporary)
        if len(prepared["entries"]) != len(entries)+1 or prepared["entries"][-1]["name"] != TENSOR_NAME:
            raise ValueError("파생 GGUF tensor 구조 검증 실패입니다.")
        # Re-read the generated payload, independently of the write operation.
        copied_hash = hashlib.sha256()
        with temporary.open("rb") as stream:
            stream.seek(new_data_start)
            remaining = original_payload_size
            while remaining:
                chunk = stream.read(min(4*1024*1024, remaining))
                if not chunk:
                    raise ValueError("파생 모델의 원본 payload가 잘렸습니다.")
                copied_hash.update(chunk)
                remaining -= len(chunk)
            stream.seek(new_data_start+alpha_offset)
            if stream.read(4000) != alpha_bytes:
                raise ValueError("파생 alpha tensor 검증 실패입니다.")
        if copied_hash.digest() != original_payload_hash.digest():
            raise ValueError("파생 모델의 기존 tensor payload가 원본과 다릅니다.")
        output_hash = _hash(temporary)
        output_size = temporary.stat().st_size
        os.replace(temporary, destination)
        result = {"filename": destination.name, "path": str(destination), "size": output_size, "sha256": output_hash,
                  "input_filename": source.name, "input_sha256": source_hash, "input_size": before.st_size,
                  "preparation_version": PREPARATION_VERSION, "prepared_at": time.time(),
                  "added_or_replaced_tensor": {"name": TENSOR_NAME, "dtype": "F32", "shape": [1000],
                       "schedule": "linear", "beta_start": 0.00085, "beta_end": 0.012,
                       "alpha_tensor_sha256": hashlib.sha256(alpha_bytes).hexdigest()},
                  "preservation": {"original_file_unchanged": True, "original_tensor_payload_byte_identical": True,
                       "original_tensor_payload_sha256": original_payload_hash.hexdigest(), "copied_tensor_count": len(entries)},
                  "license": "CreativeML-OpenRAIL-M", "modification_notice": "SD1.5 base copy with AnimateDiff-Lightning linear alphas_cumprod scheduler tensor. Original learned weights are unchanged.",
                  "sources": ["https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/README.md",
                     "https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/src/pipeline/diffusion_engine.cpp",
                     "https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/src/model_loader.cpp"],
                  "engine_load_and_visual_quality_verified": False, "reused": False}
        _write_record(record_path, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="원본을 보존하여 Lightning linear schedule GGUF를 D 드라이브에 준비합니다.")
    parser.add_argument("--runtime-dir", default=r"D:\CursorVideoLocal")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent/"lightning.lock.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    item = next(f for f in manifest["files"] if f["role"] == "base")
    filename = item["filename"]
    if Path(filename).name != filename:
        raise ValueError("허용되지 않는 원본 모델 경로입니다.")
    model_dir = (Path(args.runtime_dir).resolve()/"models").resolve()
    if not model_dir.is_relative_to(Path(args.runtime_dir).resolve()):
        raise ValueError("모델 폴더가 실행 경로 밖으로 연결됩니다.")
    source = model_dir/filename
    if source.stat().st_size != item["size"]:
        raise ValueError("원본 모델 파일 크기가 잠금 값과 다릅니다.")
    destination = model_dir/(source.stem+"-lightning-linear.gguf")
    result = prepare(source, destination, item["sha256"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
