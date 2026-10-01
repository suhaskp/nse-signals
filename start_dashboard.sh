#!/usr/bin/env bash
# macOS / Linux launcher: first run sets up a virtual environment, then keeps the dashboard running.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First start: setting up (only once)..."
  python3 -m venv .venv
fi
if ! cmp -s requirements.txt .venv/req_installed.txt; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/req_installed.txt
fi
[ -d data/superstar ] || { mkdir -p data/superstar; cp -r data_template/superstar/. data/superstar/; }
while true; do
  .venv/bin/python serve.py; code=$?
  if [ "$code" -eq 3 ]; then echo "Already running at http://localhost:8501"; exit 0; fi
  echo "Dashboard stopped; restarting in 15 s (Ctrl+C to quit)"; sleep 15
done
