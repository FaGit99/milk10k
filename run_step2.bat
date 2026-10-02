@echo off
REM Step 2: frozen-feature linear probe. Run from anywhere; uses the folder this file is in.
REM Optional: set PY=C:\path\to\python.exe (e.g. a conda/venv python) before running.
REM Any arguments are passed to step2_extract.py, e.g.:  run_step2.bat --backbones eva02_s dinov2_b
cd /d "%~dp0"
if "%PY%"=="" set PY=python

echo ===== 1/3 environment check
%PY% src\step2_env_check.py
if errorlevel 1 (
  echo Environment check failed - see results\step2\env.txt
  goto end
)

echo ===== 2/3 embedding extraction (GPU)
%PY% src\step2_extract.py %*
if errorlevel 1 (
  echo Extraction failed - see data\emb\extract_log.txt
  goto end
)

echo ===== 3/3 linear probe
%PY% src\step2_probe.py
if errorlevel 1 echo Probe failed - see results\step2\probe_log.txt

:end
echo.
echo Finished. Logs: results\step2\env.txt, data\emb\extract_log.txt, results\step2\probe_log.txt
pause
