@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start_project.ps1" %*
endlocal
