# Start JupyterLab in the pivotal conda environment
# Usage: .\start_pivotal.ps1 [--execute] [path\to\notebook.ipynb]
#   --execute  pre-run all cells before opening

param(
    [switch]$Execute,
    [string]$Notebook = ""
)

# Activate conda env (conda must be initialised in PowerShell — run `conda init powershell` once)
conda activate pivotal

# Kill any running Edge instances
Write-Host "Closing Edge..."
Stop-Process -Name msedge -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Stop any running Jupyter servers
Write-Host "Stopping existing Jupyter servers..."
$servers = jupyter server list 2>$null | Select-String ':\d+' | ForEach-Object { $_.Matches[0].Value.TrimStart(':') }
foreach ($port in $servers) {
    jupyter server stop $port 2>$null
}
Get-Process -Name "jupyter*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Resolve notebook path
$notebookFile = ""
$originalDir = Get-Location
if ($Notebook) {
    $resolved = Resolve-Path $Notebook -ErrorAction SilentlyContinue
    if ($resolved -and (Test-Path $resolved)) {
        $notebookDir = Split-Path $resolved -Parent
        $notebookFile = Split-Path $resolved -Leaf
        Set-Location $notebookDir
        Write-Host "Notebook: $resolved"
    } else {
        Write-Host "Warning: notebook not found: $Notebook — starting normally"
    }
}

# Pre-execute notebook if requested
if ($Execute -and $notebookFile) {
    Write-Host "Executing notebook cells..."
    jupyter nbconvert --to notebook --execute --inplace $notebookFile
    Write-Host "Done."
}

# Start JupyterLab and capture URL
$logFile = [System.IO.Path]::GetTempFileName()
$proc = Start-Process jupyter -ArgumentList "lab --no-browser" -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -NoNewWindow

Write-Host "Starting JupyterLab (pid $($proc.Id))..."

$url = ""
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    $content = Get-Content $logFile -ErrorAction SilentlyContinue
    $match = $content | Select-String 'http://127\.0\.0\.1:\d+/lab\?token=[a-zA-Z0-9]+' | Select-Object -First 1
    if ($match) {
        $url = $match.Matches[0].Value
        break
    }
}

if ($url) {
    $openUrl = $url
    if ($notebookFile) {
        $openUrl = $url -replace '/lab\?', "/lab/tree/$notebookFile?"
    }
    Write-Host "Opening $openUrl"
    Start-Process $openUrl
} else {
    Write-Host "Could not detect server URL — check $logFile"
}

$proc.WaitForExit()
Set-Location $originalDir
