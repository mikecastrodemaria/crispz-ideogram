@echo off
setlocal
cd /d "%~dp0"
set "IDEOGRAM_TRANSFORMER=%~dp0checkpoints\ideogram-instant-nsfw\Fal_Ideo_4_NSFW_BF16.safetensors"
if not exist "%IDEOGRAM_TRANSFORMER%" (
    echo Run test_ideogram_instant.bat first to download and validate the checkpoint.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
