#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if command -v python >/dev/null 2>&1; then PY=python
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v py >/dev/null 2>&1; then PY="py -3"
else
  echo "Python 3.8 이상이 필요합니다."
  echo "  Windows: winget install --id Python.Python.3.12 -e"
  echo "  macOS:   brew install python"
  exit 1
fi
exec $PY setup.py
