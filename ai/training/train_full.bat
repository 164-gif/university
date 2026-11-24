@echo off
setlocal
cd /d "%~dp0\..\.."
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  echo Create it with py -3.12 -m venv .venv and install CUDA PyTorch plus ai\training\requirements-full.txt.
  exit /b 2
)
"%PYTHON%" -m ai.training.full_train --directory ai/generated/full %*
exit /b %ERRORLEVEL%
