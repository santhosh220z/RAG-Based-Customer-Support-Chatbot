# PowerShell launcher for RAG Customer Support Chatbot
# Automatically checks for and activates the virtual environment

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# 1. Check if virtual environment exists
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating virtual environment at .venv..." -ForegroundColor Cyan
    python -m venv $VenvDir
}

# 2. Check if virtual environment is already active
if ($env:VIRTUAL_ENV -ne $VenvDir) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Green
    if (Test-Path $VenvActivate) {
        & $VenvActivate
    }
}

# 3. Check dependencies
if (Test-Path (Join-Path $ProjectDir "requirements.txt")) {
    $PipList = & $VenvPython -m pip list
    if ($PipList -notmatch "langchain") {
        Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
        & $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")
    }
}

# 4. Run the chatbot
Write-Host "[INFO] Launching Customer Support Chatbot..." -ForegroundColor Cyan
& $VenvPython (Join-Path $ProjectDir "main.py")
