#!/usr/bin/env bash
# Refresh the static JSON + figures the web app reads, from the latest model run.
# Run this after re-running the Python pipeline (train.py + forecast.py).
set -e
cd "$(dirname "$0")/.."

python3 src/export_web.py
echo "Web data refreshed in web/public/data/ (charts render natively, no figures needed)"
