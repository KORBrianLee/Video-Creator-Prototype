"""Local MCP stdio adapter using only stdlib; stdout is reserved for JSON-RPC."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
from local_video import control
SERVER_INFO = {"name": "cursor-video-local", "version": "0.7.1"}
LEGACY_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
MODERN_VERSION = "2026-07-28"
SUPPORTED_VERSIONS = [MODERN_VERSION, *sorted(LEGACY_VERSIONS, reverse=True)]

class ProtocolError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.data = code, data

TOOLS = [
    {"name": "video_doctor", "description": "D 저장 위치·여유 RAM·설치 준비와 실제 영상 품질 검증 상태를 확인한다.", "inputSchema": {"type": "object", "properties": {"model_profile": {"type": "string", "enum": ["neodragon", "lightning", "wan"]}}}},
    {"name": "video_generate", "description": "장면 요청으로 로컬 CPU에서 실제 움직임을 생성한다. 기본 Neodragon: 512×320, 49프레임, 약 2초/장면. 기존 실패 모델은 명시적인 진단만 허용한다.", "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "script": {"type": "string"}, "diagnostic": {"type": "boolean", "default": False}, "preset": {"type": "string", "enum": ["preview", "quality"]}, "seed": {"type": "integer"}, "scenes": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "object", "properties": {"id": {"type": "string"}, "prompt": {"type": "string"}, "negative_prompt": {"type": "string"}, "seed": {"type": "integer"}, "image_path": {"type": "string", "description": "선택 사항: D 실행 폴더 안의 참고 이미지 절대 경로"}}, "required": ["prompt"]}}}, "required": ["scenes"]}},
    {"name": "video_status", "description": "작업 진행·오류 또는 완료 영상 경로를 짧게 읽는다.", "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    {"name": "video_wait", "description": "최대 60초 로컬에서 기다려 잦은 상태 조회와 토큰 소비를 줄인다.", "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}, "seconds": {"type": "integer", "minimum": 0, "maximum": 60}}, "required": ["job_id"]}},
    {"name": "video_cancel", "description": "이 생성기의 지정 작업을 취소한다.", "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}
]
TOOLS[1]["inputSchema"]["properties"]["model_profile"] = {"type": "string", "enum": ["neodragon", "lightning", "wan"], "default": "neodragon"}
from local_video.site_profiles import PROFILES, catalog
TOOLS[0]["description"] = "선택한 SSD·여유 RAM·GPU 선택·설치 준비와 영상 품질 검증 상태를 확인한다."
TOOLS[1]["description"] = "저메모리 로컬 영상 모델로 실제 움직임을 생성한다. auto는 외장 GPU 우선, 없으면 감지한 내장 GPU 사용. 기본 Neo는 영상 행렬을 OpenCL GPU에서 하나씩 계산(명시 버퍼 64MiB), 나머지 CPU. 5GiB 작업 RAM 상한. 현장 사실감·그램 성능 미검증."
schema = TOOLS[1]["inputSchema"]
schema.pop("required")
schema["anyOf"] = [{"required": ["scenes"]}, {"required": ["site_template"]}]
schema["properties"].update({
    "site_template": {"type": "string", "enum": list(PROFILES), "description": "미리 준비한 현장 동작. scenes와 동시에 지정하지 않는다."},
    "continuous": {"type": "boolean", "description": "한 장면에서 이전 움직임을 이어받아 생성. 현장 템플릿 기본 true."},
    "duration_seconds": {"type": "integer", "enum": [2, 4, 8], "description": "연속 모드 길이. 현장 초안 기본 2초. 4초 후반 품질 실패 이력 있음; 8초 미검증. RAM 상한 동일."},
    "domain": {"type": "string", "enum": ["general", "construction"], "description": "중장비·작업자 현장은 construction. 현장 사실감 미달로 diagnostic=true 시험만 허용."},
    "cpu_precision": {"type": "string", "enum": ["int8", "bf16_stream"], "description": "preview 기본 int8, quality 기본 bf16_stream: 원본 행렬을 D에서 매핑하고 현재 계산만 FP32로 변환. RAM 상한 동일."},
})
schema["properties"]["scenes"]["items"]["properties"]["first_frame_prompt"] = {
    "type": "string", "description": "장비·작업자·배치의 첫 이미지용 영문 프롬프트. prompt는 영상 동작용."}
schema["properties"]["scenes"]["items"]["properties"]["prompt_modifier"] = {
    "type": "string", "maxLength": 200, "description": "현장 기록에는 빈 문자열을 사용해 기본 영화 효과 문구를 제외한다."}
schema["properties"]["scenes"]["items"]["properties"]["image_path"]["description"] = "Neo 참고 이미지: 선택한 SSD 실행 폴더 안의 절대 경로. Wan 1.3B에서는 지원하지 않는다."
schema["properties"]["cpu_precision"]["description"] = "CPU preview는 int8, GPU 스트리밍은 원본 bf16_stream을 사용한다. 전체 모델을 GPU에 복사하지 않는다."
TOOLS.append({"name": "video_site_presets", "description": "중장비·작업자 현장용 프롬프트 이름과 검증 범위를 짧게 읽는다.",
              "inputSchema": {"type": "object", "properties": {}}})

def compact(result):
    if "job_id" not in result:
        if "model_files" in result:
            result = dict(result)
            files = result.pop("model_files")
            result["model_file_count"] = len(files)
            result["model_files_ready"] = sum(item["ready"] for item in files)
            result["unready_model_files"] = [item for item in files if not item["ready"]]
        return result
    keys = {"job_id", "state", "stage", "error", "cancel_requested", "reused_existing_job", "scene_index", "scene_count", "scene_id", "step", "steps", "scene_elapsed_seconds", "working_set_gib", "peak_working_set_gib", "free_ram_gib", "log_path", "request_path"}
    brief = {k: v for k, v in result.items() if k in keys}
    if result.get("state") == "completed":
        for name in ("scene_index", "scene_count", "scene_id", "step", "steps", "scene_elapsed_seconds", "working_set_gib", "peak_working_set_gib", "free_ram_gib", "request_path", "log_path"):
            brief.pop(name, None)
    if "result" in result:
        value = result["result"]
        brief["result"] = {k: value[k] for k in ["video_path", "contact_sheet_path", "settings", "elapsed_seconds", "model_profile", "visual_review", "quality_note", "negative_prompt_note", "target_iris_generation_verified", "continuous", "edit_count", "site_template", "weights_finetuned", "continuity", "attribution_path", "cpu_precision", "gpu_inference", "backend_assignment"] if k in value}
        if value.get("device_plan"):
            plan = value["device_plan"]
            brief["result"]["device"] = (plan.get("selected_device") or {}).get("name", "CPU")
            brief["result"]["gpu_scope"] = plan.get("gpu_scope", "temporal_diffusion")
        if value.get("site_template") or value.get("domain") == "construction":
            brief["result"].update(domain="construction", production_realism_verified=False,
                quality_note="현장 모드는 실험 초안입니다. 가중치 재학습과 작업자·장비의 자연스러운 형태·동작 품질은 확보되지 않았습니다.")
        if value.get("negative_prompt_note"):
            brief["result"]["negative_prompt_note"] = "제외 조건은 생성한 첫 화면에만 적용됩니다. 참고 화면과 영상 움직임에는 적용되지 않습니다."
        if isinstance(value.get("continuity"), dict):
            continuity = value["continuity"]
            brief["result"]["continuity"] = {k: v for k, v in continuity.items()
                if k in {"state", "possible_jump_frames", "possible_hold_runs", "possible_darkening"}
                and (k == "state" or v)}
        if value.get("video_path"):
            brief["result"]["details_path"] = str(Path(value["video_path"]).with_name("result.json"))
        brief["result"]["reused_scene_count"] = sum(s.get("cache_hit", False) for s in value.get("shots", []))
        phases = [p for shot in value.get("shots", []) for p in shot.get("phases", [])]
        if phases:
            brief["result"]["peak_generation_working_set_gib"] = round(max(p.get("peak_working_set_bytes", 0) for p in phases)/2**30, 3)
    return brief

def dispatch(method, params):
    if method == "server/discover":
        return {"resultType": "complete", "supportedVersions": SUPPORTED_VERSIONS,
                "capabilities": {"tools": {}},
                "_meta": {"io.modelcontextprotocol/serverInfo": SERVER_INFO}}
    if method == "initialize":
        requested = params.get("protocolVersion", "2024-11-05")
        return {"protocolVersion": requested if requested in LEGACY_VERSIONS else "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        try:
            if name == "video_doctor":
                result = control.doctor(profile=args.get("model_profile"))
            elif name == "video_site_presets":
                result = catalog()
            elif name == "video_generate":
                if not isinstance(args.get("diagnostic", False), bool):
                    raise ValueError("diagnostic은 true 또는 false여야 합니다.")
                if not args.get("diagnostic", False):
                    check = control.doctor(profile=args.get("model_profile"))
                    if not check["ready"]:
                        raise RuntimeError("; ".join(check["errors"]))
                    if not check["ready_for_generation"]:
                        raise RuntimeError("영상 품질 검증이 실패했거나 아직 완료되지 않았습니다. 제작 작업을 시작하지 않았습니다. docs/VALIDATION.md를 확인하세요.")
                result = control.submit(args)
            elif name == "video_status":
                result = control.status(args["job_id"])
            elif name == "video_wait":
                result = control.wait(args["job_id"], args.get("seconds", 45))
            elif name == "video_cancel":
                result = control.cancel(args["job_id"])
            else:
                raise ValueError("알 수 없는 도구입니다.")
            return {"content": [{"type": "text", "text": json.dumps(compact(result), ensure_ascii=False, separators=(",", ":"))}]}
        except Exception as exc:
            return {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
    raise LookupError("Unknown method")

def main():
    for line in sys.stdin.buffer:
        ident = None
        try:
            if len(line) > 1024 * 1024:
                raise ValueError("Message too large")
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str) or not isinstance(message.get("params", {}), dict):
                raise ValueError("Invalid JSON-RPC 2.0 request")
            ident = message.get("id")
            if "id" not in message:
                continue
            if isinstance(ident, bool) or not isinstance(ident, (str, int)):
                raise ValueError("Invalid request ID")
            params = message.get("params", {})
            metadata = params.get("_meta", {})
            if not isinstance(metadata, dict):
                raise ProtocolError(-32602, "Invalid request metadata")
            requested = metadata.get("io.modelcontextprotocol/protocolVersion")
            if requested is not None and requested not in SUPPORTED_VERSIONS:
                raise ProtocolError(-32022, "Unsupported protocol version", {"supported": SUPPORTED_VERSIONS, "requested": requested})
            modern = requested == MODERN_VERSION
            if modern and not isinstance(metadata.get("io.modelcontextprotocol/clientCapabilities"), dict):
                raise ProtocolError(-32602, "Missing or invalid client capabilities")
            result = dispatch(message["method"], params)
            if modern:
                result = {**result, "resultType": "complete",
                          "_meta": {**result.get("_meta", {}), "io.modelcontextprotocol/serverInfo": SERVER_INFO}}
            response = {"jsonrpc": "2.0", "id": ident, "result": result}
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        except ProtocolError as exc:
            response = {"jsonrpc": "2.0", "id": ident, "error": {"code": exc.code, "message": str(exc)}}
            if exc.data is not None:
                response["error"]["data"] = exc.data
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601 if isinstance(exc, LookupError) else -32600, "message": str(exc)}}
        sys.stdout.buffer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

if __name__ == "__main__":
    main()
