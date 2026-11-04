@echo off
REM TruthLens AI - double-click this file (or run "start.bat") on Windows.
cd /d "%~dp0"
set PY=
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "print()" >nul 2>nul && set PY=py -%%V
  )
)
if not defined PY (
  where python >nul 2>nul && set PY=python
)
if not defined PY (
  echo [TruthLens] ERROR: Python is not installed. Install Python 3.12 from https://www.python.org/downloads/
  echo             During installation tick "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% start.py %*
if errorlevel 1 pause
