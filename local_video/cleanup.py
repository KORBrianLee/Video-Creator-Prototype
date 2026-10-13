"""Round-end storage cleanup inside the runtime folder. It previews by default and deletes only with apply=True.

Kept: jobs named in reports/STATE.json (adopted results and judged evidence), queued or running jobs, models,
model-integrity records, learned resource profiles and the text-encoding cache. Removed: failed or cancelled jobs,
finished jobs nobody recorded after `days`, clip caches older than `days`, scratch left in tmp, and request-index
hints whose job is gone. Nothing outside the runtime folder is touched.
"""
from __future__ import annotations
import json
import re
import shutil
import time
from pathlib import Path

JOB_ID = re.compile(r"\b[0-9a-f]{12}\b")
HOUR = 3600
DAY = 24 * HOUR
ACTIVE = {"queued", "running"}
# Per-key caches a repeated request rebuilds (depth below cache/<name>); a finished job keeps its own output.
CLIP_CACHES = {"shots": 1, "first-frames": 1, "conditioning": 2}
# Long clips write frames and may grow a page file on the runtime drive; below this the next one can fail.
LOW_FREE_GIB = 10
LARGEST = 5


def _size(path):
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _age(path, now):
    try:
        return now - path.stat().st_mtime
    except OSError:
        return 0


def recorded_jobs(root):
    """Job ids written anywhere in STATE.json, or None when no agent state is kept there."""
    try:
        return set(JOB_ID.findall((Path(root) / "reports" / "STATE.json").read_text(encoding="utf-8-sig")))
    except OSError:
        return None


def _state(folder, refresh):
    try:
        state = json.loads((folder / "status.json").read_text(encoding="utf-8-sig")).get("state")
    except (OSError, ValueError, AttributeError):
        return None
    if refresh and state in ACTIVE:
        try:
            # control.status reports (and stores) a queued or running job whose worker died as failed.
            state = refresh(folder.name).get("state", state)
        except (OSError, ValueError):
            pass
    return state


def candidates(root, days=3, now=None, refresh=None):
    """([(path, reason)] to delete, active job ids, recorded job ids kept or None without STATE.json)."""
    root = Path(root)
    now = time.time() if now is None else now
    recorded = recorded_jobs(root)
    found, active, kept = [], [], []
    jobs = root / "jobs"
    for folder in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        if not folder.is_dir() or not JOB_ID.fullmatch(folder.name):
            continue
        state = _state(folder, refresh)
        if state in ACTIVE:
            active.append(folder.name)
        elif recorded is not None and folder.name in recorded:
            kept.append(folder.name)
        elif state in {"failed", "cancelled", None}:
            if _age(folder, now) > HOUR:
                found.append((folder, state or "unreadable"))
        elif recorded is not None and _age(folder, now) > days * DAY:
            found.append((folder, "not_recorded"))
    removed = {path.name for path, _ in found}
    index = root / "cache" / "request-index"
    for hint in sorted(index.glob("*.json")) if index.is_dir() else []:
        try:
            job = json.loads(hint.read_text(encoding="utf-8-sig")).get("job_id")
        except (OSError, ValueError, AttributeError):
            job = None
        if not (isinstance(job, str) and JOB_ID.fullmatch(job) and (jobs / job).is_dir()) or job in removed:
            found.append((hint, "job_gone"))
    # A running job may still read a cache entry it found at the start, and owns the scratch it is writing.
    scratch_age = DAY if active else HOUR
    tmp = root / "tmp"
    found += [(entry, "scratch") for entry in (sorted(tmp.iterdir()) if tmp.is_dir() else [])
              if _age(entry, now) > scratch_age]
    if not active:
        for name, depth in CLIP_CACHES.items():
            entries = [root / "cache" / name]
            for _ in range(depth):
                entries = [child for entry in entries if entry.is_dir() for child in sorted(entry.iterdir())]
            found += [(entry, "old_cache") for entry in entries if _age(entry, now) > days * DAY]
    return found, active, (kept if recorded is not None else None)


def clean(root, days=3, apply=False, now=None, refresh=None):
    """One compact JSON for the agent: every job deleted (or to be), totals per reason, the largest other entries,
    what was kept on purpose and the free space left."""
    root = Path(root)
    found, active, kept = candidates(root, days, now, refresh)
    items, failed = [], []
    for path, reason in found:
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        size = _size(path)
        if apply:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as exc:
                failed.append({"path": path.relative_to(root).as_posix(), "error": str(exc)[:160]})
                continue
        items.append({"path": path.relative_to(root).as_posix(), "reason": reason, "mb": round(size / 2**20, 1)})
    reasons = {}
    for item in items:
        total = reasons.setdefault(item["reason"], {"count": 0, "mb": 0.0})
        total["count"] += 1
        total["mb"] = round(total["mb"] + item["mb"], 1)
    jobs = [item for item in items if item["path"].startswith("jobs/")]
    others = sorted((item for item in items if item not in jobs and item["mb"] >= 1), key=lambda item: -item["mb"])
    free = shutil.disk_usage(root).free / 2**30
    result = {"applied": apply, "days": days, "count": len(items), "mb": round(sum(i["mb"] for i in items), 1),
              "by_reason": reasons, "jobs": [{"job": i["path"][5:], "reason": i["reason"], "mb": i["mb"]} for i in jobs],
              "largest_other": others[:LARGEST], "active_jobs": active,
              "kept_recorded_jobs": len(kept) if kept is not None else None, "free_gib": round(free, 1)}
    if failed:
        result["failed"] = failed[:10]
    if kept is None:
        result["note"] = "reports/STATE.json이 없어 완료 작업은 지우지 않았습니다."
    if free < LOW_FREE_GIB:
        result["warnings"] = [f"low_free_space: 실행 폴더 드라이브 여유 {free:.1f}GiB(<{LOW_FREE_GIB})"]
    return result
