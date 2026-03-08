@echo off
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

python scripts/clean.py

endlocal
