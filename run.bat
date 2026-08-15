@echo off
REM Windows batch launcher for RAG Customer Support Chatbot
REM Automatically checks and activates .venv before running

cd /d "%~dp0"

IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment .venv...
    python -m venv .venv
)

IF NOT DEFINED VIRTUAL_ENV (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [INFO] Starting Customer Support Chatbot...
python main.py

pause
