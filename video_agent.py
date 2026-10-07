"""Cursor-friendly local CLI: one JSON result, no model imports during chat."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from local_video import control

def main():
    parser = argparse.ArgumentParser(description="선택한 C/D SSD의 로컬 CPU·GPU 영상 생성기")
    sub = parser.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--verify", action="store_true")
    submit = sub.add_parser("submit")
    submit.add_argument("request_file", type=Path)
    for name in ["status", "cancel", "wait", "_worker"]:
        item = sub.add_parser(name)
        item.add_argument("job_id")
        if name == "wait":
            item.add_argument("--seconds", type=int, default=45)
    sub.add_parser("list")
    digest = sub.add_parser("digest", help="완료 작업의 짧은 요약과 화면 붕괴·정지 경고(에이전트용, 2KB 이하)")
    digest.add_argument("job_id")
    plan = sub.add_parser("plan", help="고정 계획을 로컬에서 끝까지 실행하고 요약 하나만 출력(백그라운드 실행용)")
    plan.add_argument("plan_file", type=Path)
    frames = sub.add_parser("frames", help="한 장면의 첫 프레임 후보를 시드별로 만들고 비교 이미지 하나로 묶기")
    frames.add_argument("spec_file", type=Path)
    imports = sub.add_parser("import", help="외부에서 만든 첫 프레임(예: GenerateImage, C 저장)을 실행 폴더로 옮기고 원본 삭제")
    imports.add_argument("images", type=Path, nargs="+")
    compose = sub.add_parser("compose-end", help="끝 키프레임의 고정 영역을 시작 프레임에서 복사(카메라 흔들림 방지)")
    compose.add_argument("start", type=Path)
    compose.add_argument("end", type=Path)
    compose.add_argument("target", type=Path)
    compose.add_argument("--keep", action="append", required=True, help="x0,y0,x1,y1 (0~1), 여러 번 지정 가능")
    paste = sub.add_parser("paste-region", help="한 프레임의 영역(사람 등)을 다른 프레임의 지정 위치에 붙이기(키프레임 위치 지정)")
    paste.add_argument("source", type=Path)
    paste.add_argument("base", type=Path)
    paste.add_argument("target", type=Path)
    paste.add_argument("--box", required=True, help="x0,y0,x1,y1 (0~1) 잘라낼 영역")
    paste.add_argument("--at", required=True, help="x,y (0~1) 붙일 왼쪽 위 위치")
    paste.add_argument("--scale", type=float, default=1.0)
    paste.add_argument("--clean", type=Path, help="source에서 사람을 지운 배경판(사람 모양만 오려 붙이기)")
    paste.add_argument("--threshold", type=int, default=28, help="배경판과의 차이 기준(배경까지 붙으면 높인다)")
    paste.add_argument("--ppe", action="store_true", help="보호구 색(주황 조끼·흰 헬멧·짙은 작업복)으로 작업자 모양 찾기")
    paste.add_argument("--solid", type=float, help="0~1: 이 높이 아래 실루엣 안쪽 구멍을 가로줄 단위로 채우기(손을 든 경우 0.4)")
    args = parser.parse_args()
    try:
        from local_video.storage import cache_environment
        os.environ.update(cache_environment(control.load_config()[1]))
    except Exception:
        pass
    try:
        if args.action == "doctor":
            result = control.doctor(args.verify)
        elif args.action == "submit":
            result = control.submit(control.read_json(args.request_file))
        elif args.action == "_worker":
            control.worker(args.job_id)
            return 0
        elif args.action == "list":
            result = control.list_jobs()
        elif args.action == "wait":
            result = control.wait(args.job_id, args.seconds)
        elif args.action == "digest":
            from local_video.digest import job_digest
            result = job_digest(args.job_id)
        elif args.action == "plan":
            from local_video.plan_runner import main as run_plan
            result = run_plan(args.plan_file)
        elif args.action == "frames":
            from local_video.frame_pick import main as pick_frames
            result = pick_frames(args.spec_file)
        elif args.action == "import":
            from local_video.frame_pick import import_frames
            result = import_frames(args.images, control.load_config()[1])
        elif args.action == "compose-end":
            from local_video.frame_pick import compose_keyframe
            boxes = [tuple(float(v) for v in box.split(",")) for box in args.keep]
            result = compose_keyframe(args.start, args.end, boxes, args.target)
        elif args.action == "paste-region":
            from local_video.frame_pick import paste_region
            result = paste_region(args.source, tuple(float(v) for v in args.box.split(",")), args.base,
                                  tuple(float(v) for v in args.at.split(",")), args.target, args.scale,
                                  clean=args.clean, threshold=args.threshold,
                                  solid=args.solid if args.solid is not None else False, ppe=args.ppe)
        else:
            result = getattr(control, args.action)(args.job_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
