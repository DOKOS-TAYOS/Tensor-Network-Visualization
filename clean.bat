@echo off
setlocal

set "ROOT=%~dp0"

echo Cleaning project caches and temporary files...
echo Keeping .venv untouched.

powershell -NoProfile -ExecutionPolicy Bypass ^
  "$root = [System.IO.Path]::GetFullPath('%ROOT%');" ^
  "$venv = [System.IO.Path]::GetFullPath((Join-Path $root '.venv'));" ^
  "$dirPatterns = @('__pycache__', '.pytest_cache', '.ruff_cache', '.tmp', '.pip_tmp', 'pytest-cache-files-*', 'build', 'dist', '*.egg-info');" ^
  "$filePatterns = @('*.pyc', '*.pyo', '*.pyd');" ^
  "$dirs = foreach ($pattern in $dirPatterns) {" ^
  "  Get-ChildItem -Path $root -Force -Directory -Filter $pattern -ErrorAction SilentlyContinue;" ^
  "  Get-ChildItem -Path $root -Recurse -Force -Directory -Filter $pattern -ErrorAction SilentlyContinue;" ^
  "};" ^
  "$dirs | Where-Object { $_.FullName -notlike ($venv + '*') } | Sort-Object FullName -Unique -Descending | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue; Write-Host ('Removed directory: ' + $_.FullName) };" ^
  "$files = foreach ($pattern in $filePatterns) { Get-ChildItem -Path $root -Recurse -Force -File -Filter $pattern -ErrorAction SilentlyContinue };" ^
  "$files | Where-Object { $_.FullName -notlike ($venv + '*') } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue; Write-Host ('Removed file: ' + $_.FullName) }"

echo Done.
endlocal
