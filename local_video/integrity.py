"""Cancellable pinned-model verification, reusing unchanged file fingerprints."""
import hashlib
import json


def verify(request, root, progress, is_cancelled):
    from .control import runtime_path, write_json
    record_path = runtime_path(root, "cache", "model-integrity", "neodragon.json")
    try:
        records = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        records = {}
    progress({"stage": "verifying_model_integrity"})
    for item in request["_model_spec"]["files"]:
        if is_cancelled():
            raise InterruptedError("모델 확인 중 취소됐습니다.")
        path = runtime_path(root, "models", *item["filename"].split("/"))
        stat = path.stat()
        fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
                       "expected_sha256": item["sha256"]}
        if stat.st_size != item["size"]:
            raise RuntimeError("모델 크기가 다릅니다: " + item["filename"])
        # Windows timestamps can coalesce rapid writes to small files. Hash
        # these each time; only large immutable weights use the metadata cache.
        if stat.st_size > 16 * 2**20 and records.get(item["filename"]) == fingerprint:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 2**20):
                if is_cancelled():
                    raise InterruptedError("모델 확인 중 취소됐습니다.")
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError("모델 SHA256이 다릅니다: " + item["filename"])
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino):
            raise RuntimeError("확인 중 모델이 변경됐습니다: " + item["filename"])
        records[item["filename"]] = fingerprint
    write_json(record_path, records)
