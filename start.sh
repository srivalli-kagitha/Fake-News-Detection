#!/usr/bin/env bash
# TruthLens AI - run "./start.sh" on Linux / macOS.
cd "$(dirname "$0")"
for py in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    exec "$py" start.py "$@"
  fi
done
echo "[TruthLens] ERROR: Python 3.10-3.12 is not installed." >&2
exit 1
