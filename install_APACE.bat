@echo off
setlocal DisableDelayedExpansion

rem Use the bundled script and source files in this checkout or extracted ZIP.
cd /d "%~dp0"
rem Prevent inherited PowerShell 7 modules from masking Windows PowerShell modules.
set "PSModulePath="

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_APACE.ps1" -InstallDirectory "%~dp0." %*
set "APACE_EXIT_CODE=%ERRORLEVEL%"

if not "%APACE_EXIT_CODE%"=="0" (
    echo.
    echo A-PACE setup failed. Review the message above, then run install_APACE.bat again.
    pause
)

endlocal & exit /b %APACE_EXIT_CODE%
