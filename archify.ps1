# Archify PowerShell Wrapper
$ErrorActionPreference = "Stop"

$nodeExe = Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $nodeExe) {
    $wingetNode = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.19.0-win-x64\node.exe"
    if (Test-Path $wingetNode) {
        $nodeExe = $wingetNode
    } else {
        Write-Error "Node.js executable was not found. Please ensure Node.js is installed."
        exit 1
    }
}

$cliPath = Join-Path $PSScriptRoot ".agents\skills\archify\bin\archify.mjs"
if (-not (Test-Path $cliPath)) {
    $cliPath = "$env:USERPROFILE\.gemini\config\skills\archify\bin\archify.mjs"
}

if (-not (Test-Path $cliPath)) {
    Write-Error "Archify CLI not found at $cliPath"
    exit 1
}

& $nodeExe $cliPath @args
