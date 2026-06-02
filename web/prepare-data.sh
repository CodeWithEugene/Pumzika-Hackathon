#!/usr/bin/env bash
# Refresh the static JSON + figures the web app reads, from the latest model run.
# Run this after re-running the Python pipeline (train.py + forecast.py).
set -e
cd "$(dirname "$0")/.."

python3 src/export_web.py
mkdir -p web/public/figures
cp reports/figures/forecast_vs_actual.png reports/figures/calibration.png web/public/figures/
echo "Web data + figures refreshed in web/public/"
