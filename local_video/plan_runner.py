"""Local batch runner: executes a fixed plan of generation runs without the Cursor agent watching.

The agent writes a plan once, starts this runner in the background and ends its turn. The runner
submits each run, waits locally, digests the result, enforces attempt limits and stops repeating an
identical failure, then writes one summary (<= 4 KiB) that the agent reads when it is notified.

Plan file (JSON):
{"name": "ltx-length-check", "max_attempts": 6, "stop_on": ["collapse"],
 "runs": [{"id": "ltx-2s", "request": {...video_generate request...}}, ...]}
"""
from __future__ import annotations
import json
import time
from pathlib import Path

SUMMARY_LIMIT = 4096
TERMINAL = {"completed", "failed", "cancelled"}


def failure_key(digest):
    """Same cause, same key: an identical failure is not retried a third time."""
    error = (digest.get("error") or "").split(".")[0]
    return f'{digest.get("state")}:{digest.get("stage")}:{error[:80]}'


def collapsed(digest):
    return any(w.startswith(("darkening", "colour_blowout")) or w == "near_black_frames" for w in digest.get("warnings", []))


def run_plan(plan, control, digest_job, report_dir, sleep=time.sleep, poll_seconds=20, clock=time.monotonic):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = min(int(plan.get("max_attempts", 6)), 6)
    stop_on = set(plan.get("stop_on", ["collapse"]))
    results, attempts, failures, stop_reason = [], 0, {}, None
    started = clock()
    for run in plan["runs"]:
        if attempts >= max_attempts:
            stop_reason = f"attempt limit {max_attempts} reached"
            break
        attempts += 1
        try:
            job = control.submit(run["request"])
        except Exception as exc:
            results.append({"id": run["id"], "state": "rejected", "error": str(exc)[:240]})
            key = f"rejected:{str(exc)[:80]}"
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= 2:
                stop_reason = "same rejection twice"
                break
            continue
        job_id = job["job_id"]
        while control.status(job_id).get("state") not in TERMINAL:
            sleep(poll_seconds)
        digest = digest_job(job_id)
        digest["id"] = run["id"]
        results.append(digest)
        (report_dir / f'{run["id"]}.json').write_text(json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8")
        if digest.get("state") != "completed":
            key = failure_key(digest)
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= 2:
                stop_reason = "same failure twice"
                break
        elif "collapse" in stop_on and collapsed(digest):
            stop_reason = f'{run["id"]} shows collapse; longer runs skipped'
            break
    summary = {"plan": plan.get("name"), "attempts": attempts, "minutes": round((clock() - started) / 60, 1),
               "stop_reason": stop_reason, "completed": sum(r.get("state") == "completed" for r in results),
               "runs": [{key: r.get(key) for key in ("id", "job", "state", "seconds", "model", "fallback", "gpu", "warnings", "error", "stages_s_gib")
                         if r.get(key) not in (None, [], {})} for r in results],
               "details": str(report_dir)}
    text = json.dumps(summary, ensure_ascii=False)
    while len(text) > SUMMARY_LIMIT and any("stages_s_gib" in r for r in summary["runs"]):
        next(r for r in summary["runs"] if "stages_s_gib" in r).pop("stages_s_gib")
        text = json.dumps(summary, ensure_ascii=False)
    (report_dir / "summary.json").write_text(text, encoding="utf-8")
    return summary


def main(plan_file):
    from . import control
    from .digest import job_digest
    plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    _, root = control.load_config()
    report = root / "reports" / f'{plan.get("name", "plan")}-{time.strftime("%Y%m%d-%H%M%S")}'
    return run_plan(plan, control, job_digest, report)
