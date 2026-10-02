"""Real stdio checks for normal generation, cache reuse and queued cancellation."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from installer import d_root, inside
from local_video import control
from local_video.backend import _memory
from mcp_server import SERVER_INFO
from tools.validate_mcp_quality_gate import job_snapshot


def main():
    root = d_root(os.environ.get("CVL_RUNTIME_DIR", __import__("local_video.storage", fromlist=["default_runtime"]).default_runtime()))
    server = json.loads((SOURCE / ".cursor" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["local-video"]
    env = {**os.environ, **server.get("env", {})}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root, env=env, creationflags=flags)
    lines = queue.Queue()
    def reader():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    responses, response_bytes, ident = {}, {}, 0
    def send(method, params, allow_error=False):
        nonlocal ident
        ident += 1
        process.stdin.write((json.dumps({"jsonrpc": "2.0", "id": ident, "method": method,
                                        "params": params}, ensure_ascii=False)+"\n").encode("utf-8"))
        process.stdin.flush()
        raw = lines.get(timeout=75)
        if raw is None:
            raise RuntimeError("MCP server ended unexpectedly")
        reply = json.loads(raw.decode("utf-8", errors="strict"))
        assert reply.get("jsonrpc") == "2.0" and reply.get("id") == ident and "error" not in reply
        result = reply["result"]
        if result.get("isError") and not allow_error:
            raise RuntimeError(str(result))
        responses[str(ident)], response_bytes[str(ident)] = result, len(raw)
        return result
    def call(name, arguments):
        return json.loads(send("tools/call", {"name": name, "arguments": arguments})["content"][0]["text"])
    lock = None
    try:
        initialized = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "portable-video-audit", "version": "1"}})
        assert initialized["serverInfo"]["version"] == SERVER_INFO["version"]
        process.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        process.stdin.flush()
        listed = send("tools/list", {})
        assert len(listed["tools"]) == 6
        modern_meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                       "io.modelcontextprotocol/clientCapabilities": {},
                       "io.modelcontextprotocol/clientInfo": {"name": "portable-video-audit", "version": "1"}}
        discovered = send("server/discover", {"_meta": modern_meta})
        assert "2026-07-28" in discovered["supportedVersions"]
        modern_tools = send("tools/list", {"_meta": modern_meta})
        assert modern_tools["resultType"] == "complete" and len(modern_tools["tools"]) == 6
        presets = call("video_site_presets", {})
        assert len(presets["templates"]) == 8 and presets["weights_finetuned"] is False
        checked = call("video_doctor", {"model_profile": "neodragon"})
        assert checked["ready"] and checked["ready_for_generation"], checked
        assert checked["construction"]["production_ready"] is False
        before_domain_request = job_snapshot(root)
        rejected_site = send("tools/call", {"name": "video_generate", "arguments": {
            "site_template": "rail_lifting_crew", "preset": "quality", "duration_seconds": 2}}, allow_error=True)
        assert rejected_site.get("isError") and "중장비" in rejected_site["content"][0]["text"]
        assert before_domain_request == job_snapshot(root), "Rejected site request created or changed a job"
        site_request = json.loads((SOURCE / "examples" / "construction-reference-request.json").read_text(encoding="utf-8"))
        site_request.update(title="MCP 현장 초안과 출처 확인", diagnostic=True)
        site_submitted = call("video_generate", site_request)
        site_current = call("video_wait", {"job_id": site_submitted["job_id"], "seconds": 45})
        assert site_current["state"] == "completed", site_current
        site_result = site_current["result"]
        assert site_result["reused_scene_count"] == 1, "Site audit must reuse the reviewed draft, not launch new inference"
        assert site_result["continuous"] and site_result["edit_count"] == 0
        assert site_result["weights_finetuned"] is False and site_result["production_realism_verified"] is False
        assert site_result["settings"]["frames"] == 49
        details = Path(site_result["details_path"]).resolve(strict=True)
        assert details.is_relative_to(root) and json.loads(details.read_text(encoding="utf-8"))["production_realism_verified"] is False
        attribution = Path(site_result["attribution_path"]).resolve(strict=True)
        assert attribution.is_relative_to(root) and "CC BY 4.0" in attribution.read_text(encoding="utf-8")
        request = json.loads((SOURCE / "examples" / "beach-request.json").read_text(encoding="utf-8"))
        request["title"] = "MCP 일반 제작과 캐시 검증"
        submitted = call("video_generate", request)
        deadline = time.monotonic()+90
        current = call("video_status", {"job_id": submitted["job_id"]})
        while current["state"] not in control.TERMINAL:
            assert time.monotonic() < deadline, current
            current = call("video_wait", {"job_id": submitted["job_id"], "seconds": 45})
        assert current["state"] == "completed", current
        assert current["result"]["reused_scene_count"] == 1, current
        repeated = call("video_generate", request)
        assert repeated["job_id"] == submitted["job_id"] and repeated["reused_existing_job"]
        # Hold the app's own lock so cancellation is tested before any model is loaded.
        lock = control.EngineLock(inside(root, "engine.lock"))
        assert lock.acquire(), "The engine is busy; repeat the audit after its current job"
        cancelled_request = {**request, "title": "MCP 대기 작업 취소", "seed": 43}
        cancelled = call("video_generate", cancelled_request)
        requested = call("video_cancel", {"job_id": cancelled["job_id"]})
        assert requested.get("cancel_requested") or requested["state"] == "cancelled"
        cancellation = call("video_wait", {"job_id": cancelled["job_id"], "seconds": 10})
        assert cancellation["state"] == "cancelled", cancellation
        lock.release()
        lock = None
        server_working, _ = _memory(process.pid)
        process.stdin.close()
        assert process.wait(timeout=10) == 0
        thread.join(timeout=10)
        leftovers = []
        while not lines.empty():
            value = lines.get_nowait()
            if value is not None:
                leftovers.append(value)
        assert not leftovers, "Unexpected stdout output"
        stderr = process.stderr.read()
        assert not stderr, stderr.decode("utf-8", errors="replace")
        report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "server_version": initialized["serverInfo"]["version"], "tool_count": len(listed["tools"]),
                  "utf8_strict": True, "notification_silent": True, "stderr_bytes": len(stderr),
                  "legacy_handshake_and_modern_stateless_requests": True,
                  "exit_code": process.returncode, "normal_generation_job_id": submitted["job_id"],
                  "normal_generation_used_verified_cache": True, "identical_request_reused_job": True,
                  "construction_production_rejected_without_job_change": True,
                  "construction_diagnostic_cached_job_id": site_submitted["job_id"],
                  "construction_weights_and_realism_not_claimed": True,
                  "construction_attribution_path": str(attribution),
                  "queued_cancellation_job_id": cancelled["job_id"], "queued_cancel_completed": True,
                  "server_working_set_bytes": server_working, "response_bytes": response_bytes,
                  "responses": responses, "cursor_ui_verified": False}
        destination = inside(root, "audits", "mcp-ready-validation.json")
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"passed": True, "report": str(destination), "job_id": submitted["job_id"]}))
    finally:
        if lock is not None:
            lock.release()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
