@echo off
setlocal

cd /d "%~dp0"

set "UV_COMMAND=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\A-PACE-tools\uv\uv.exe" (
        set "UV_COMMAND=%LOCALAPPDATA%\A-PACE-tools\uv\uv.exe"
    ) else (
        echo uv was not found. Run the one-command setup from README.md again.
        echo https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)

start "" /B powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:5000/'"
"%UV_COMMAND%" run --locked python app.py
set "APACE_EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %APACE_EXIT_CODE%
