@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

python clean.py

endlocal
