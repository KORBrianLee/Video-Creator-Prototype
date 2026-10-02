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
    args = parser.parse_args()
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
        else:
            result = getattr(control, args.action)(args.job_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
