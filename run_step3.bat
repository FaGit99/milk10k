@echo off
REM Step 3: full fine-tune, dermoscopy only (PanDerm-B, DINOv2-B), logit-adjusted loss, nested threshold calibration.
REM Optional: set PY=C:\path\to\python.exe before running.
REM Arguments are passed through, e.g.:  run_step3.bat --backbones dinov2_b --seeds 0 1 2
REM Smoke test first (1 epoch, 300 lesions, fold 0, writes nothing to runs\):  run_step3.bat --limit 300 --epochs 1 --folds 0
cd /d "%~dp0"
if "%PY%"=="" set PY=python

echo ===== 1/2 environment check
%PY% src\step2_env_check.py
if errorlevel 1 (
  echo Environment check failed - see results\step2\env.txt
  goto end
)

echo ===== 2/2 fine-tuning
%PY% src\step3_finetune.py %*
if errorlevel 1 echo Fine-tuning failed - see results\step3\train_log.txt

:end
echo.
echo Finished. Log: results\step3\train_log.txt ; results: results\step3\summary.csv, metrics.json
pause
