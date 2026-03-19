# Windows build script for the pivotal JupyterLab extension.
# Use this instead of 'jlpm build' on Windows, since jlpm cannot find
# Node.js when launched via Python subprocesses (Unix-style PATH issue).
#
# Usage (from repo root or editors/jupyterlab):
#   .\editors\jupyterlab\build.ps1          # production build
#   .\editors\jupyterlab\build.ps1 --dev    # development build (source maps)

param(
    [switch]$dev
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Find node.exe — check common install locations
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
$nodeCandidates = @(
    "C:\Program Files\nodejs\node.exe",
    "$env:APPDATA\nvm\current\node.exe"
)
if ($nodeCmd) { $nodeCandidates += $nodeCmd.Source }
$node = $nodeCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $node) {
    Write-Error "node.exe not found. Install Node.js from https://nodejs.org"
    exit 1
}

# Resolve JupyterLab staging path via Python (always works regardless of PATH format).
# Try conda env python first, fall back to whatever python is on PATH.
$condaPython = "$env:USERPROFILE\.conda\envs\pivotal\python.exe"
$pythonCmd = if (Test-Path $condaPython) { $condaPython } else { "python" }
$corePath = & $pythonCmd -c "import jupyterlab, os; print(os.path.join(os.path.dirname(jupyterlab.__file__), 'staging'))"
if (-not $corePath) {
    Write-Error "Could not find JupyterLab staging path. Is the pivotal conda env active?"
    exit 1
}

Write-Host "Building TypeScript..."
& $node node_modules/typescript/bin/tsc
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building labextension (webpack)..."
$buildArgs = @("node_modules/@jupyterlab/builder/lib/build-labextension.js", "--core-path", $corePath, ".")
if ($dev) { $buildArgs += "--development" }
& $node @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Copying assets..."
New-Item -ItemType Directory -Force -Path "pivotal_jupyterlab/labextension/style/icons" | Out-Null
Copy-Item "style/icons/pivotal_logo.svg" "pivotal_jupyterlab/labextension/style/icons/" -Force

Write-Host "Build complete."
