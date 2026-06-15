@echo off
setlocal
set "STARTDIR=%CD%"
echo Starting QED-C Benchmarks Server ...
echo Documentation will be available at http://localhost:8088/site/
echo.
cd /d "%~dp0server"
python -m uvicorn app:app --reload --port 8088
cd /d "%STARTDIR%"
