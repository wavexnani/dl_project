@echo off
setlocal

:: Find node executable
where node >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "NODE_EXE=node"
) else (
    set "NODE_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.19.0-win-x64\node.exe"
)

if not exist "%NODE_EXE%" (
    where "%NODE_EXE%" >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Node.js not found in PATH or standard installation path.
        exit /b 1
    )
)

:: Find Archify CLI entrypoint
set "ARCHIFY_CLI=%~dp0.agents\skills\archify\bin\archify.mjs"
if not exist "%ARCHIFY_CLI%" (
    set "ARCHIFY_CLI=%USERPROFILE%\.gemini\config\skills\archify\bin\archify.mjs"
)

"%NODE_EXE%" "%ARCHIFY_CLI%" %*
