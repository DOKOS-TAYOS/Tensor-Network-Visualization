@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"

:find_root
if exist "%ROOT%\.venv\Scripts\python.exe" goto :root_found
for %%I in ("%ROOT%\..") do set "PARENT=%%~fI"
if /I "%PARENT%"=="%ROOT%" goto :missing_venv
set "ROOT=%PARENT%"
goto :find_root

:root_found
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto :missing_venv

cd /d "%ROOT%"

call :run examples\tensorkrowch_demo.py mps 2d
call :run examples\tensorkrowch_demo.py mps 3d
call :run examples\tensorkrowch_demo.py mpo 2d
call :run examples\tensorkrowch_demo.py mpo 3d
call :run examples\tensorkrowch_demo.py peps 2d
call :run examples\tensorkrowch_demo.py peps 3d
call :run examples\tensorkrowch_demo.py weird 2d
call :run examples\tensorkrowch_demo.py weird 3d
call :run examples\tensorkrowch_demo.py disconnected 2d
call :run examples\tensorkrowch_demo.py disconnected 3d
call :run examples\tensorkrowch_demo.py mps 2d --from-list

call :run examples\tensornetwork_demo.py mps 2d
call :run examples\tensornetwork_demo.py mps 3d
call :run examples\tensornetwork_demo.py mpo 2d
call :run examples\tensornetwork_demo.py mpo 3d
call :run examples\tensornetwork_demo.py peps 2d
call :run examples\tensornetwork_demo.py peps 3d
call :run examples\tensornetwork_demo.py weird 2d
call :run examples\tensornetwork_demo.py weird 3d
call :run examples\tensornetwork_demo.py disconnected 2d
call :run examples\tensornetwork_demo.py disconnected 3d

call :run examples\quimb_demo.py mps 2d
call :run examples\quimb_demo.py mps 3d
call :run examples\quimb_demo.py hyper 2d
call :run examples\quimb_demo.py hyper 3d
call :run examples\quimb_demo.py mpo 2d
call :run examples\quimb_demo.py mpo 3d
call :run examples\quimb_demo.py peps 2d
call :run examples\quimb_demo.py peps 3d
call :run examples\quimb_demo.py weird 2d
call :run examples\quimb_demo.py weird 3d
call :run examples\quimb_demo.py disconnected 2d
call :run examples\quimb_demo.py disconnected 3d
call :run examples\quimb_demo.py mps 2d --from-list

call :run examples\tenpy_demo.py mps 2d
call :run examples\tenpy_demo.py mps 3d
call :run examples\tenpy_demo.py mpo 2d
call :run examples\tenpy_demo.py mpo 3d
call :run examples\tenpy_demo.py imps 2d
call :run examples\tenpy_demo.py imps 3d
call :run examples\tenpy_demo.py impo 2d
call :run examples\tenpy_demo.py impo 3d

call :run examples\einsum_demo.py mps 2d
call :run examples\einsum_demo.py mps 3d
call :run examples\einsum_demo.py mps 2d --mode manual
call :run examples\einsum_demo.py mps 3d --mode manual
call :run examples\einsum_demo.py peps 2d
call :run examples\einsum_demo.py peps 3d
call :run examples\einsum_demo.py disconnected 2d
call :run examples\einsum_demo.py disconnected 3d

call :run examples\tn_tsp.py -n 4 --view 2d
call :run examples\tn_tsp.py -n 4 --view 3d
call :run examples\tn_tsp.py -n 5 --view 2d
call :run examples\tn_tsp.py -n 5 --view 3d
call :run examples\tn_tsp.py -n 6 --view 2d
call :run examples\tn_tsp.py -n 6 --view 3d

echo.
echo All example commands completed successfully.
exit /b 0

:run
echo.
echo Running: "%PYTHON%" %*
"%PYTHON%" %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Command failed with exit code %errorlevel%.
exit /b %errorlevel%

:missing_venv
echo.
echo Could not find ".venv\Scripts\python.exe" in this tree or any parent directory.
exit /b 1
