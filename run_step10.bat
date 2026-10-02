@echo off
REM Step 10: epoch sweep PanDerm-B, der + cli, 40 epochs, 5 folds, seed 0; model saved after EVERY epoch.
REM Disk: 2 views x 5 folds x 40 epochs = 400 checkpoints of ~330-350 MB = ~135-140 GB in results\step10\ckpt.
REM Optional: set PY=C:\path\to\python.exe before running. Arguments are passed through, e.g.:
REM   run_step10.bat --eval_only            (re-run only the evaluation from saved runs)
REM   run_step10.bat --ckpt none            (no checkpoints, predictions only)
cd /d "%~dp0"
if "%PY%"=="" set PY=python
%PY% src\step10_epoch_sweep.py %*
if errorlevel 1 echo Epoch sweep failed - see results\step10\train_log.txt
echo.
echo Finished. Log: results\step10\train_log.txt ; per_epoch_*.csv, metrics.json, curves_*.png, best_model_*.json
pause
