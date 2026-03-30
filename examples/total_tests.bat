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
if not defined PLOT_DELAY_SECONDS set "PLOT_DELAY_SECONDS=2"
set "HELPER_DIR=%ROOT%\.tmp\total-tests"
set "WRAPPER=%ROOT%\.tmp\total-tests\plot_wrapper.py"
set "VIEWER=%ROOT%\.tmp\total-tests\image_viewer.ps1"
set "IMAGE_PATH=%ROOT%\.tmp\total-tests\current_plot.png"

cd /d "%ROOT%"
call :prepare_helpers

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
call :run examples\tenpy_demo.py purification 2d
call :run examples\tenpy_demo.py purification 3d
call :run examples\tenpy_demo.py uniform 2d
call :run examples\tenpy_demo.py uniform 3d
call :run examples\tenpy_demo.py excitation 2d
call :run examples\tenpy_demo.py excitation 3d

call :run examples\einsum_demo.py mps 2d
call :run examples\einsum_demo.py mps 3d
call :run examples\einsum_demo.py mps 2d --mode manual
call :run examples\einsum_demo.py mps 3d --mode manual
call :run examples\einsum_demo.py peps 2d
call :run examples\einsum_demo.py peps 3d
call :run examples\einsum_demo.py disconnected 2d
call :run examples\einsum_demo.py disconnected 3d

call :run examples\einsum_general.py batch 2d
call :run examples\einsum_general.py batch 3d
call :run examples\einsum_general.py ellipsis 2d
call :run examples\einsum_general.py ellipsis 3d
call :run examples\einsum_general.py mps_short 2d
call :run examples\einsum_general.py mps_short 3d
call :run examples\einsum_general.py nway 2d
call :run examples\einsum_general.py nway 3d
call :run examples\einsum_general.py trace 2d
call :run examples\einsum_general.py trace 3d

call :run examples\tn_tsp.py -n 4 --view 2d
call :run examples\tn_tsp.py -n 4 --view 3d
call :run examples\tn_tsp.py -n 5 --view 2d
call :run examples\tn_tsp.py -n 5 --view 3d
call :run examples\tn_tsp.py -n 6 --view 2d
call :run examples\tn_tsp.py -n 6 --view 3d

echo.
echo All example commands completed successfully.
exit /b 0

:prepare_helpers
if not exist "%HELPER_DIR%" mkdir "%HELPER_DIR%"
> "%WRAPPER%" (
echo from __future__ import annotations
echo.
echo import runpy
echo import sys
echo.
echo import matplotlib
echo matplotlib.use^("Agg"^)
echo import matplotlib.pyplot as plt
echo.
echo delay = float^(sys.argv[1]^)
echo image_path = sys.argv[2]
echo script_argv = sys.argv[3:]
echo if not script_argv:
echo     raise SystemExit^("Missing script path."^)
echo plt.show = lambda *args, **kwargs: None
echo sys.argv = script_argv
echo runpy.run_path^(script_argv[0], run_name="__main__"^)
echo figure_numbers = plt.get_fignums^(^)
echo if not figure_numbers:
echo     raise SystemExit^("No figures were generated."^)
echo plt.figure^(figure_numbers[0]^).savefig^(image_path, bbox_inches="tight"^)
echo plt.close^('all'^)
)
if errorlevel 1 goto :error
> "%VIEWER%" (
echo param^(
echo     [string]$Path,
echo     [double]$Seconds
echo ^)
echo Add-Type -AssemblyName System.Windows.Forms
echo Add-Type -AssemblyName System.Drawing
echo $image = [System.Drawing.Image]::FromFile^($Path^)
echo $form = New-Object System.Windows.Forms.Form
echo $form.StartPosition = 'CenterScreen'
echo $form.Width = [Math]::Min^($image.Width + 40, 1400^)
echo $form.Height = [Math]::Min^($image.Height + 60, 1000^)
echo $picture = New-Object System.Windows.Forms.PictureBox
echo $picture.Dock = 'Fill'
echo $picture.SizeMode = 'Zoom'
echo $picture.Image = $image
echo $form.Controls.Add^($picture^)
echo $timer = New-Object System.Windows.Forms.Timer
echo $timer.Interval = [Math]::Max^([int]^($Seconds * 1000^), 1^)
echo $timer.Add_Tick^({ $timer.Stop^(^); $form.Close^(^) }^)
echo $form.Add_Shown^({ $timer.Start^(^) }^)
echo [void]$form.ShowDialog^(^)
echo $image.Dispose^(^)
echo Remove-Item $Path -ErrorAction SilentlyContinue
)
if errorlevel 1 goto :error
exit /b 0

:run
echo.
echo Running: "%PYTHON%" %*
if exist "%IMAGE_PATH%" del /q "%IMAGE_PATH%"
"%PYTHON%" "%WRAPPER%" "%PLOT_DELAY_SECONDS%" "%IMAGE_PATH%" %*
if errorlevel 1 goto :error
if /I "%TOTAL_TESTS_SKIP_VIEWER%"=="1" exit /b 0
if not exist "%IMAGE_PATH%" goto :missing_image
powershell -NoProfile -ExecutionPolicy Bypass -File "%VIEWER%" "%IMAGE_PATH%" "%PLOT_DELAY_SECONDS%"
if errorlevel 1 goto :error
exit /b 0

:missing_image
echo.
echo Wrapper finished without generating "%IMAGE_PATH%".
exit /b 1

:error
echo.
echo Command failed with exit code %errorlevel%.
exit /b %errorlevel%

:missing_venv
echo.
echo Could not find ".venv\Scripts\python.exe" in this tree or any parent directory.
exit /b 1
