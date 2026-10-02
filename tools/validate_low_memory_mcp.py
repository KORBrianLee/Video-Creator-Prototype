"""Check installed Cursor transport, reviewed-clip reuse and site rejection."""
from pathlib import Path
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_video import control
from local_video.storage import runtime_root, default_runtime
from mcp_server import SERVER_INFO


def main():
    root = runtime_root(os.environ.get("CVL_RUNTIME_DIR", default_runtime()))
    control.APP = root / "app"
    server = control.read_json(root / "app" / ".cursor" / "mcp.json")["mcpServers"]["local-video"]
    env = {**os.environ, **server["env"]}
    proc = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=root, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    replies, responses, lock, serial = queue.Queue(), [], None, 0
    def read():
        for line in proc.stdout:
            replies.put(line)
    threading.Thread(target=read, daemon=True).start()
    def send(method, params):
        nonlocal serial
        serial += 1
        proc.stdin.write((json.dumps({"jsonrpc":"2.0", "id":serial, "method":method, "params":params}, ensure_ascii=False)+"\n").encode())
        proc.stdin.flush()
        raw = replies.get(timeout=65)
        answer = json.loads(raw.decode("utf-8", "strict"))
        assert answer["id"] == serial and "error" not in answer, answer
        responses.append({"method":method, "bytes":len(raw)})
        return answer["result"]
    def call(name, arguments):
        reply = send("tools/call", {"name":name, "arguments":arguments})
        assert not reply.get("isError"), reply
        return json.loads(reply["content"][0]["text"])
    try:
        info = send("initialize", {"protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"low-memory-audit","version":"1"}})
        assert info["serverInfo"] == SERVER_INFO
        assert len(send("tools/list", {})["tools"]) == 6
        checked = call("video_doctor", {})
        assert checked["ready"] and checked["ready_for_generation"], checked
        assert checked["device_plan"]["gpu_scope"] == "video_transformer_linear_layers_only"
        assert len(call("video_site_presets", {})["templates"]) == 8
        before = {p.name for p in (root/"jobs").iterdir()}
        rejected = send("tools/call", {"name":"video_generate", "arguments":{"site_template":"rail_lifting_crew"}})
        assert rejected.get("isError") and "중장비" in rejected["content"][0]["text"]
        assert before == {p.name for p in (root/"jobs").iterdir()}
        request = control.read_json(root/"audits"/"low-memory-gpu-beach-request.json")
        request.update(title="MCP GPU 일반 장면 캐시 확인", diagnostic=False)
        created = call("video_generate", request)
        current = call("video_wait", {"job_id":created["job_id"], "seconds":45})
        assert current["state"] == "completed", current
        assert current["result"]["gpu_inference"] and current["result"]["reused_scene_count"] == 1
        details = control.read_json(Path(current["result"]["details_path"]))
        assert details["target_iris_generation_verified"] is False
        metrics = [p["gpu"] for s in details["shots"] for p in s["phases"] if p.get("gpu")]
        assert metrics and all(p["kernel_calls"] > 0 and p["peak_explicit_gpu_buffer_bytes"] <= 64*2**20 for p in metrics)
        repeated = call("video_generate", request)
        assert repeated["job_id"] == created["job_id"] and repeated["reused_existing_job"]
        lock = control.EngineLock(root/"engine.lock")
        assert lock.acquire()
        pending = call("video_generate", {**request, "title":"대기 작업 취소", "seed":44})
        call("video_cancel", {"job_id":pending["job_id"]})
        assert call("video_wait", {"job_id":pending["job_id"], "seconds":10})["state"] == "cancelled"
        lock.release(); lock = None
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0 and not proc.stderr.read()
        record = {"passed":True, "server_version":SERVER_INFO["version"], "tool_count":6, "utf8_strict":True,
                  "reused_job":created["job_id"], "site_production_rejected_without_job":True,
                  "queued_cancel_passed":True, "responses":responses, "target_iris_generation_verified":False}
        control.write_json(root/"audits"/"low-memory-mcp.json", record)
        print(json.dumps({"passed":True, "report":str(root/"audits"/"low-memory-mcp.json")}))
    finally:
        if lock: lock.release()
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=10)


if __name__ == "__main__":
    main()
