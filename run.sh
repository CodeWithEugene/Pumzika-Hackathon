#!/usr/bin/env bash
# One-command reproduction of the full pipeline.
set -e
cd "$(dirname "$0")"

echo "[1/4] Generating data ..."
python3 src/generate_data.py

echo "[2/4] Training + back-testing ..."
python3 src/train.py

echo "[3/4] Building 90-day forward forecast ..."
python3 src/forecast.py

echo "[4/4] Launching dashboard ..."
echo "   -> opening http://localhost:8501"
python3 -m streamlit run app/dashboard.py
