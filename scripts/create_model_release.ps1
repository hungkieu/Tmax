param(
    [string]$Output = "model-artifacts.zip",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[ ] $Message"
}

function Write-Done {
    param([string]$Message)
    Write-Host "[x] $Message"
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Red
}

$requiredFiles = @(
    "artifacts/observations.duckdb",
    "artifacts/heat_risk_model.joblib",
    "artifacts/rkpk_heat_risk_model.joblib",
    "artifacts/rjtt_heat_risk_model.joblib",
    "artifacts/wsss_heat_risk_model.joblib"
)

Write-Host "Model artifacts release checklist"
Write-Host "================================="

Write-Step "Check required files"
$missing = @()
foreach ($file in $requiredFiles) {
    if (Test-Path -LiteralPath $file) {
        $item = Get-Item -LiteralPath $file
        Write-Done "$file ($([math]::Round($item.Length / 1MB, 2)) MB)"
    } else {
        $missing += $file
        Write-Fail "$file missing"
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Fail "Cannot create release zip because required files are missing."
    Write-Host "Build/sync/train locally first, then rerun this script."
    exit 1
}

Write-Step "Check output path"
if ((Test-Path -LiteralPath $Output) -and -not $Force) {
    Write-Fail "$Output already exists. Rerun with -Force to overwrite."
    exit 1
}
Write-Done "Output path ready: $Output"

Write-Step "Create zip"
if (Test-Path -LiteralPath $Output) {
    Remove-Item -LiteralPath $Output -Force
}
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rksi-model-release-" + [System.Guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot "artifacts") | Out-Null
    foreach ($file in $requiredFiles) {
        Copy-Item -LiteralPath $file -Destination (Join-Path $stagingRoot $file) -Force
    }
    Compress-Archive -Path (Join-Path $stagingRoot "artifacts") -DestinationPath $Output -Force
} finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
Write-Done "Created $Output"

Write-Step "Verify zip contents"
$entries = (Get-ChildItem -LiteralPath $Output | Select-Object -First 1)
if ($null -eq $entries) {
    Write-Fail "Zip file was not created."
    exit 1
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Output))
try {
    $zipEntries = $zip.Entries | ForEach-Object { $_.FullName -replace "\\", "/" }
    foreach ($file in $requiredFiles) {
        $expected = $file -replace "\\", "/"
        if ($zipEntries -contains $expected) {
            Write-Done "Zip contains $expected"
        } else {
            Write-Fail "Zip missing $expected"
            exit 1
        }
    }
} finally {
    $zip.Dispose()
}

$outputItem = Get-Item -LiteralPath $Output
Write-Host ""
Write-Done "Release zip ready: $($outputItem.FullName)"
Write-Host "Size: $([math]::Round($outputItem.Length / 1MB, 2)) MB"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Keep this zip as a backup or upload it manually if needed."
Write-Host "2. Smoke test Telegram locally:"
Write-Host "   uv run rksi-telegram-report --output artifacts/telegram_report.md --hours 4"
Write-Host "   node scripts/send_telegram_report.mjs artifacts/telegram_report.md"
