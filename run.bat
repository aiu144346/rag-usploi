@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo  CS Dispute Resolution RAG Chatbot
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

echo [INFO] Starting Streamlit server...
".venv\Scripts\streamlit.exe" run app.py

pause
