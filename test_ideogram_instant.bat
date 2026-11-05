@echo off
setlocal
cd /d "%~dp0"
echo Close crispz and other GPU applications before this test.
".venv\Scripts\python.exe" tools\ideogram_instant.py --download --render %*
if errorlevel 1 (
    echo Test failed. Read the error above. No application settings were changed.
    pause
    exit /b 1
)
echo Test finished. Image: out\ideogram-instant-test.png
pause
