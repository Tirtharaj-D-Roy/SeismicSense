#!/usr/bin/env bash
# One-click launcher for macOS / Linux: creates a virtual environment on first run, then starts the app.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Setting up for the first time - this takes 1-3 minutes..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
