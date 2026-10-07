# dev_up.ps1 - PowerShell launcher for Predictive Geofencing Development Environment
# Finds Git Bash and executes dev_up.sh seamlessly in Windows PowerShell / Terminal.

$BashPath = $null
$Candidates = @(
    "D:\Git\bin\bash.exe",
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
)

foreach ($candidate in $Candidates) {
    if (Test-Path $candidate) {
        $BashPath = $candidate
        break
    }
}

if (-not $BashPath) {
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd) {
        $gitDir = Split-Path (Split-Path $gitCmd.Source)
        $potentialBash = Join-Path $gitDir "bin\bash.exe"
        if (Test-Path $potentialBash) {
            $BashPath = $potentialBash
        }
    }
}

if (-not $BashPath) {
    Write-Error "Git Bash (bash.exe) was not found. Please ensure Git for Windows is installed or run dev_up.sh inside Git Bash directly."
    exit 1
}

Set-Location $PSScriptRoot
& $BashPath "$PSScriptRoot/dev_up.sh" @args
