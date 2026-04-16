@echo off
call venv\Scripts\activate
cd finetune
python finetune-webui.py
pause