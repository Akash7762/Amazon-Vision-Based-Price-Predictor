# One-click start on Windows; run it through start-app.bat in the repo root.
# Starts the price model (backend) and the web app, opens the app in the
# browser, and stops both when you press Q or close the window.
#
#   start-app.bat                   start everything
#   start-app.bat -NoBrowser        same, without opening the browser
#   start-app.bat -CreateShortcut   put a "Vision Price Predictor" icon on the desktop
#
# The web app runs as a production build, rebuilt first whenever its code has
# changed. Server output goes to logs\api.log and logs\web.log.
param(
    [switch]$NoBrowser,
    [switch]$CreateShortcut
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest crawls with its progress bar on

$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "frontend"
$python = Join-Path $root "venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
$apiLog = Join-Path $logs "api.log"
$webLog = Join-Path $logs "web.log"
$appUrl = "http://localhost:3000"
$apiCheck = "http://127.0.0.1:8000/health"
$webCheck = "http://127.0.0.1:3000/manifest.webmanifest"
$started = @()

if ($CreateShortcut) {
    $path = Join-Path ([Environment]::GetFolderPath("Desktop")) "Vision Price Predictor.lnk"
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    $link.TargetPath = Join-Path $root "start-app.bat"
    $link.WorkingDirectory = $root
    $link.IconLocation = Join-Path $PSScriptRoot "app.ico"
    $link.Description = "Start Vision Price Predictor"
    $link.Save()
    Write-Host "Added to the desktop: $path"
    exit 0
}

function Test-Up($url) {
    try { (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { $false }
}

# /T takes the children too: each server runs under a cmd that redirects its output.
function Stop-Started {
    foreach ($p in $script:started) {
        if (-not $p.HasExited) { cmd /c "taskkill /PID $($p.Id) /T /F >nul 2>&1" }
    }
}

function Fail($message, $log) {
    Write-Host ""
    Write-Host "  $message" -ForegroundColor Red
    if ($log -and (Test-Path $log)) {
        $tail = Get-Content $log -Tail 15
        if ($tail -match "10048|EADDRINUSE") {
            Write-Host "  Another program is using its port. Close it, then start again." -ForegroundColor Red
        }
        Write-Host "  Last lines of $log`:" -ForegroundColor DarkGray
        $tail | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
    Stop-Started
    exit 1
}

# Runs in this window's console, so closing the window stops the server too.
function Start-Server($dir, $command, $log) {
    $p = Start-Process cmd.exe -ArgumentList "/d /c $command > `"$log`" 2>&1" `
        -WorkingDirectory $dir -NoNewWindow -PassThru
    $script:started += $p
    $p
}

function Wait-Up($url, $process, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Up $url) { return $true }
        if ($process.HasExited) { return $false }
        Start-Sleep -Milliseconds 500
    }
    $false
}

# True when there's no production build, or the code changed after it was made.
function Test-BuildStale {
    $id = Join-Path $frontend ".next\BUILD_ID"
    if (-not (Test-Path $id)) { return $true }
    $dirs = "app", "components", "lib", "public" | ForEach-Object { Join-Path $frontend $_ }
    $files = @(Get-ChildItem $dirs -Recurse -File -ErrorAction SilentlyContinue)
    $files += Get-ChildItem $frontend -File | Where-Object {
        ($_.Name -in "next.config.ts", "proxy.ts", "package.json", "package-lock.json", "tsconfig.json") -or ($_.Name -like ".env*")
    }
    $newest = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    $newest -gt (Get-Item $id).LastWriteTime
}

try { $Host.UI.RawUI.WindowTitle = "Vision Price Predictor" } catch {}
Write-Host ""
Write-Host "  Vision Price Predictor" -ForegroundColor Cyan
Write-Host ""

$apiUp = Test-Up $apiCheck
$webUp = Test-Up $webCheck
if ($apiUp -and $webUp) {
    Write-Host "  It's already running, started from another window."
    if (-not $NoBrowser) { Start-Process $appUrl; Write-Host "  Opened $appUrl" }
    Start-Sleep -Seconds 3
    exit 0
}
New-Item -ItemType Directory -Force $logs | Out-Null

if (-not $apiUp) {
    if (-not (Test-Path $python)) { Fail "No Python environment (venv) yet. Set it up first: see Setup in README.md." }
    if (-not (Test-Path (Join-Path $root "backend\models\price_model.onnx"))) {
        Write-Host "  Downloading the model (only the first time, 112 MB)..."
        & $python (Join-Path $root "backend\download_model.py")
        if ($LASTEXITCODE -ne 0) { Fail "Couldn't download the model; see the message above, and backend\README.md." }
    }
    Write-Host "  Starting the price model..."
    $api = Start-Server $root "venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000 --timeout-keep-alive 75" $apiLog
}

if (-not $webUp) {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail "Node.js isn't installed; get it from https://nodejs.org" }
    Push-Location $frontend
    try {
        if (-not (Test-Path "node_modules")) {
            Write-Host "  Installing the web app's packages (only the first time)..."
            & npm.cmd install
            if ($LASTEXITCODE -ne 0) { Fail "npm install failed; see the messages above." }
        }
        if (Test-BuildStale) {
            Write-Host "  Building the web app (only after its code changes, about a minute)..."
            & node node_modules\next\dist\bin\next build
            if ($LASTEXITCODE -ne 0) { Fail "The web app didn't build; see the messages above." }
        }
    } finally {
        Pop-Location
    }
    Write-Host "  Starting the web app..."
    $webApp = Start-Server $frontend "node node_modules\next\dist\bin\next start --port 3000" $webLog
}

if ($api -and -not (Wait-Up $apiCheck $api 90)) { Fail "The price model didn't start." $apiLog }
if ($webApp -and -not (Wait-Up $webCheck $webApp 60)) { Fail "The web app didn't start." $webLog }

Write-Host ""
Write-Host "  Ready: $appUrl" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process $appUrl }
Write-Host ""
Write-Host "  Keep this window open while you use the app."
Write-Host "  To stop it, press Q here or close this window."

$keys = -not [Console]::IsInputRedirected
try {
    if ($keys) { [Console]::TreatControlCAsInput = $true }  # so Ctrl+C stops cleanly, like Q
    while ($true) {
        if ($api -and $api.HasExited) { Fail "The price model stopped unexpectedly." $apiLog }
        if ($webApp -and $webApp.HasExited) { Fail "The web app stopped unexpectedly." $webLog }
        if ($keys -and [Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            $ctrlC = $key.Key -eq "C" -and ($key.Modifiers -band [ConsoleModifiers]::Control)
            if ($key.Key -eq "Q" -or $ctrlC) { break }
        }
        Start-Sleep -Milliseconds 300
    }
} finally {
    Stop-Started
}
Write-Host "  Stopped."
