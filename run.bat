@echo off
REM Launches crispz (the Gradio UI) with hardware detection.
REM Uses .venv when it exists; --no-venv (or --system) forces the current Python.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set USE_VENV=1
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-venv" set USE_VENV=0
if /I "%~1"=="--system" set USE_VENV=0
if /I "%~1"=="--debug" set CRISPZ_LOG_LEVEL=debug
shift
goto argloop
:argdone

REM The base Python
where py >nul 2>&1
if errorlevel 1 (
    set PYCMD=python
) else (
    set PYCMD=py
    for %%v in (3.11 3.12 3.13) do (
        py -%%v -c "import sys" >nul 2>&1
        if not errorlevel 1 set PYCMD=py -%%v
    )
)

set RUNPY=!PYCMD!
REM .venv (install.bat) first, then env (the Pinokio launcher): the same rule as
REM boot_check.bat, so the diagnosis and the app run on the same interpreter.
if "!USE_VENV!"=="1" if exist "env\Scripts\python.exe" set RUNPY=env\Scripts\python.exe
if "!USE_VENV!"=="1" if exist ".venv\Scripts\python.exe" set RUNPY=.venv\Scripts\python.exe

REM ESRGAN_DIR: an existing variable wins, otherwise the sdlibs folder when it exists, otherwise the local one
if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN
    ) else (
        set ESRGAN_DIR=%~dp0upscale_models
    )
)

echo === crispz-flux3 - run ===
echo Python     = !RUNPY!
echo ESRGAN_DIR = !ESRGAN_DIR!
if defined CRISPZ_LOG_LEVEL echo Log level  = !CRISPZ_LOG_LEVEL!  (run.bat --debug)
echo.
echo --- Hardware detection ---
!RUNPY! _hw_check.py
echo.

echo --- Starting the Gradio UI ---
echo Open http://127.0.0.1:7860 in your browser
echo.
!RUNPY! app.py
endlocal
