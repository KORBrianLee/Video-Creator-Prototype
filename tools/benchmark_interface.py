"""Compare real completed replies and request lookup; never run video inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from local_video import control, neodragon
from mcp_server import SERVER_INFO, compact
from tools.validate_mcp_quality_gate import job_snapshot


def linear_lookup(root, digest):
    # The 0.6.0 lookup, with the same output-SHA check as indexed lookup.
    for path in (root / "jobs").glob("*/status.json"):
        state = control.read_json(path)
        if state.get("request_hash") == digest and state.get("state") in {"queued", "running", "completed"}:
            if state["state"] != "completed" or control.completed_result_intact(state, root):
                return {**control.status(path.parent.name), "reused_existing_job": True}
    return None


def wire(result, ident, compressed=False):
    options = {"ensure_ascii": False}
    if compressed:
        options["separators"] = (",", ":")
    return (json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}, **options) + "\n").encode("utf-8")


def main():
    _, root = control.load_config()
    baseline = control.read_json(control.runtime_path(root, "audits", "saved-0.6.0", "mcp-ready-validation.json"))
    current = control.read_json(control.runtime_path(root, "audits", "mcp-ready-validation.json"))
    assert baseline["passed"] and baseline["server_version"] == "0.6.0"
    assert current["passed"] and current["server_version"] == SERVER_INFO["version"]
    before = job_snapshot(root)
    comparisons = []
    for kind, key in [("general", "normal_generation_job_id"), ("construction", "construction_diagnostic_cached_job_id")]:
        jid = baseline[key]
        assert current[key] == jid, "Compare the same completed jobs, not different generation timings"
        state = control.status(jid)
        assert state["state"] == "completed" and control.completed_result_intact(state, root)
        for ident, response in baseline["responses"].items():
            if not response.get("content") or response.get("isError"):
                continue
            body = json.loads(response["content"][0]["text"])
            if body.get("job_id") == jid and body.get("state") == "completed" and "result" in body:
                break
        else:
            raise RuntimeError("No recorded completed response for " + jid)
        if body.get("reused_existing_job"):
            state["reused_existing_job"] = True
        old_size = len(wire(response, int(ident)))
        assert old_size == baseline["response_bytes"][ident]
        new = {"content": [{"type": "text", "text": json.dumps(compact(state), ensure_ascii=False, separators=(",", ":"))}]}
        new_size = len(wire(new, int(ident), compressed=True))
        assert new_size == current["response_bytes"][ident], "Computed reply must match the actual stdio result"
        brief = json.loads(new["content"][0]["text"])["result"]
        details = control.runtime_path(root, *Path(brief["details_path"]).relative_to(root).parts)
        assert control.read_json(details)["video_sha256"] == state["result"]["video_sha256"]
        if kind == "construction":
            assert brief["production_realism_verified"] is False and brief["weights_finetuned"] is False
            assert Path(brief["attribution_path"]).is_file() and "초안" in brief["quality_note"]
        comparisons.append({"kind": kind, "job_id": jid, "old_bytes": old_size, "new_bytes": new_size,
            "reduction_percent": round(100 * (old_size-new_size) / old_size, 2), "actual_stdio_sizes_matched": True})
    jid = baseline["construction_diagnostic_cached_job_id"]
    state = control.status(jid)
    digest = state["request_hash"]
    control._remember_request(root, digest, jid)
    finders = {"linear": lambda: linear_lookup(root, digest), "indexed": lambda: control._indexed_request(root, digest)}
    for find in finders.values():
        assert find()["job_id"] == jid
    durations = {name: [] for name in finders}
    for iteration in range(11):
        names = list(finders) if iteration % 2 == 0 else list(reversed(finders))
        for name in names:
            start = time.perf_counter()
            assert finders[name]()["job_id"] == jid
            durations[name].append((time.perf_counter()-start)*1000)
    counts = {}
    original = control.read_json
    for name, find in finders.items():
        reads = []
        def counting(path):
            if Path(path).is_relative_to(root / "jobs"):
                reads.append(str(path))
            return original(path)
        with patch.object(control, "read_json", new=counting):
            assert find()["job_id"] == jid
        counts[name] = len(reads)
    assert before == job_snapshot(root), "Read-only benchmark changed job files"
    assert "torch" not in sys.modules, "Interface benchmark must not load model libraries"
    report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": SERVER_INFO["version"], "pipeline_revision": neodragon.pipeline_revision(SOURCE),
        "responses": comparisons, "lookup_job_id": jid, "lookup_job_status_json_reads": counts,
        "lookup_median_ms": {k: round(statistics.median(v), 3) for k, v in durations.items()},
        "lookup_samples_ms": durations, "lookup_repetitions": 11,
        "lookup_scope": "request lookup only, warm OS caches, output SHA checked in both methods; excludes generation and resource checks",
        "video_inference_run": False, "torch_loaded": False, "job_files_unchanged": True,
        "target_gram_verified": False, "token_counts_measured": False,
        "limitations": "Response byte reduction is not an exact Cursor token saving, neural inference speedup, RAM reduction or realism improvement."}
    path = control.runtime_path(root, "audits", "interface-benchmark-" + SERVER_INFO["version"] + ".json")
    control.write_json(path, report)
    print(json.dumps({k: report[k] for k in ["passed", "responses", "lookup_job_status_json_reads", "lookup_median_ms", "video_inference_run"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
