@echo off
cd /d C:\Users\ezycloudx-admin\Documents\MedCLIP-SAMv2
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe -u run_all.py --skip-stage1 > full_run.log 2>&1
echo Done > full_run.done
