@echo off
REM One-click launcher for Windows: creates a virtual environment on first run, then starts the app.
cd /d "%~dp0"
if not exist .venv (
    echo Setting up for the first time - this takes 1-3 minutes...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
streamlit run app.py
pause
