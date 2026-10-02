"""Verify the installed stdio server without starting failed-model inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import queue
import subprocess
import threading
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_server import SERVER_INFO


def job_snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in sorted((root / "jobs").glob("*")) if folder.is_dir()
            for p in sorted(folder.glob("*.json")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from local_video.storage import default_runtime, runtime_root
    parser.add_argument("--runtime-dir", default=default_runtime())
    args = parser.parse_args()
    root = runtime_root(args.runtime_dir)
    config = json.loads((root / "app" / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["local-video"]
    expected_command = root / "runtime" / "python" / "python.exe"
    expected_arguments = [str(root / "app" / "mcp_server.py")]
    if Path(server["command"]).resolve() != expected_command.resolve() or server["args"] != expected_arguments:
        raise ValueError("The installed MCP configuration does not match this runtime.")
    report_path = (root / "audits" / "mcp-quality-gate-validation.json").resolve()
    if not report_path.is_relative_to(root):
        raise ValueError("Audit output resolves outside the runtime.")
    environment = {**os.environ, **server.get("env", {}),
                   "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
                   "CVL_RUNTIME_DIR": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    before = job_snapshot(root)
    responses, response_bytes = {}, {}
    lines = queue.Queue()
    process = subprocess.Popen([server["command"], *server["args"]], cwd=root / "app",
                               env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def send(ident, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if ident is not None:
            payload["id"] = ident
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        process.stdin.flush()
        if ident is None:
            return None
        raw = lines.get(timeout=30)
        if raw is None:
            raise RuntimeError("MCP exited before replying.")
        reply = json.loads(raw.decode("utf-8", errors="strict"))
        if reply.get("jsonrpc") != "2.0" or reply.get("id") != ident or "error" in reply:
            raise RuntimeError("Unexpected MCP response: " + str(reply))
        responses[str(ident)] = reply
        response_bytes[str(ident)] = len(raw)
        return reply["result"]

    try:
        initialized = send(1, "initialize", {"protocolVersion": "2025-06-18",
                                             "capabilities": {}, "clientInfo": {"name": "quality-audit", "version": "1"}})
        assert initialized["serverInfo"]["version"] == SERVER_INFO["version"]
        send(None, "notifications/initialized", {})
        listed = send(2, "tools/list", {})
        assert len(listed["tools"]) == 6
        checked = send(3, "tools/call", {"name": "video_doctor", "arguments": {"model_profile": "lightning"}})
        doctor = json.loads(checked["content"][0]["text"])
        assert doctor["ready"] is True, doctor.get("errors")
        assert doctor["ready_for_generation"] is False
        assert doctor["quality_status"] == "failed_visual_validation"
        # Send a production request only after the installed server reports the gate.
        rejected = send(4, "tools/call", {"name": "video_generate", "arguments": {
            "title": "실패 모델 제작 차단 확인", "model_profile": "lightning", "scenes": [{"prompt": "Waves roll onto a sandy beach."}]}})
        assert rejected.get("isError") is True
        assert "영상 품질 검증이 실패" in rejected["content"][0]["text"]
        process.stdin.close()
        assert process.wait(timeout=10) == 0
        reader.join(timeout=10)
        remaining = []
        while not lines.empty():
            value = lines.get_nowait()
            if value is not None:
                remaining.append(value)
        assert not remaining, "Unexpected stdout output"
        stderr = process.stderr.read()
        assert not stderr, stderr.decode("utf-8", errors="replace")
        unchanged = before == job_snapshot(root)
        assert unchanged, "MCP production request changed job files"
        report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "command": server["command"], "arguments": server["args"],
                  "server_version": initialized["serverInfo"]["version"],
                  "tool_count": len(listed["tools"]), "response_bytes": response_bytes,
                  "utf8_strict_decode": True, "stderr_bytes": len(stderr), "exit_code": process.returncode,
                  "job_json_files_unchanged": unchanged, "ready": doctor["ready"],
                  "ready_for_generation": doctor["ready_for_generation"],
                  "quality_status": doctor["quality_status"], "production_request_rejected": True,
                  "responses": responses}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"passed": True, "report": str(report_path)}, ensure_ascii=False))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
