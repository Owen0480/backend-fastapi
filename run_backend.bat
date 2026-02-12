@echo off
REM venv Python 경로가 깨졌을 때: 시스템 Python으로 실행
py -3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
if errorlevel 1 (
  echo py -3 실패. python 직접 시도...
  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
)
pause
