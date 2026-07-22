[CmdletBinding()]
param(
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'

$pythonPath = 'E:\conda\envs\rave\python.exe'
$v2Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$serverScript = Join-Path $PSScriptRoot 'serve_realtime_instrument.py'
$runtimePath = Join-Path $v2Root 'audition\scratch-pad-final\runtime.pt'
$webRoot = Join-Path $v2Root 'realtime-instrument'
$sourceRoot = Join-Path $v2Root 'src'
$artifactRoot = Split-Path $runtimePath -Parent
$pidPath = Join-Path $artifactRoot 'realtime-instrument.pid'
$stdoutLog = Join-Path $artifactRoot 'realtime-instrument.stdout.log'
$stderrLog = Join-Path $artifactRoot 'realtime-instrument.stderr.log'
$statusUrl = 'http://127.0.0.1:8877/api/runtime-status'
$instrumentUrl = 'http://127.0.0.1:8877/'

function Read-TrackedPid {
    $text = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    $serverPid = 0
    if (-not [int]::TryParse($text, [ref]$serverPid) -or $serverPid -le 0) {
        throw "Invalid realtime instrument PID file: $pidPath"
    }
    return $serverPid
}

function Get-TrackedProcess([int]$ServerPid) {
    return Get-CimInstance -ClassName Win32_Process `
        -Filter "ProcessId = $ServerPid" -ErrorAction SilentlyContinue
}

function Assert-RealtimeProcess($Process) {
    $commandLine = [string]$Process.CommandLine
    $executablePath = [string]$Process.ExecutablePath
    $usesServerScript = $commandLine.IndexOf(
        $serverScript,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -ge 0
    $usesPython = $executablePath.Equals(
        $pythonPath,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if (-not $usesServerScript -or -not $usesPython) {
        throw "PID $($Process.ProcessId) does not belong to serve_realtime_instrument.py; refusing to stop it."
    }
}

function Stop-TrackedServer {
    if (-not (Test-Path -LiteralPath $pidPath)) {
        Write-Output 'MidiBrave realtime instrument is not running (no PID file).'
        return
    }

    $serverPid = Read-TrackedPid
    $tracked = Get-TrackedProcess $serverPid
    if ($null -eq $tracked) {
        Remove-Item -LiteralPath $pidPath -Force
        Write-Output "Removed stale realtime instrument PID file for PID $serverPid."
        return
    }

    Assert-RealtimeProcess $tracked
    Stop-Process -Id $serverPid -Force
    $stopDeadline = [DateTime]::UtcNow.AddSeconds(10)
    while ($null -ne (Get-TrackedProcess $serverPid) -and
            [DateTime]::UtcNow -lt $stopDeadline) {
        Start-Sleep -Milliseconds 100
    }
    if ($null -ne (Get-TrackedProcess $serverPid)) {
        throw "Realtime instrument PID $serverPid did not stop."
    }
    Remove-Item -LiteralPath $pidPath -Force
    Write-Output "Stopped MidiBrave realtime instrument PID $serverPid."
}

if ($Stop) {
    Stop-TrackedServer
    return
}

foreach ($requiredPath in @($pythonPath, $serverScript, $runtimePath, $webRoot)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required realtime instrument path is missing: $requiredPath"
    }
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = Read-TrackedPid
    $existing = Get-TrackedProcess $existingPid
    if ($null -ne $existing) {
        Assert-RealtimeProcess $existing
        Write-Output "MidiBrave realtime instrument is already running as PID $existingPid."
        Write-Output "URL: $instrumentUrl"
        return
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$arguments = @(
    '-u',
    $serverScript,
    '--runtime',
    $runtimePath,
    '--root',
    $webRoot,
    '--host',
    '127.0.0.1',
    '--port',
    '8877'
)

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceRoot
    $serverProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $v2Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
} finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
[System.IO.File]::WriteAllText($pidPath, [string]$serverProcess.Id)

$ready = $false
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($serverProcess.HasExited) {
            throw "Realtime instrument exited with code $($serverProcess.ExitCode) during startup."
        }
        try {
            $response = Invoke-WebRequest `
                -UseBasicParsing `
                -Uri $statusUrl `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            # CUDA model loading may take several seconds; keep polling.
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw 'Timed out waiting 60 seconds for the realtime instrument status endpoint.'
    }
} catch {
    if (-not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $serverProcess.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $pidPath) {
        $recordedPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($recordedPid -eq [string]$serverProcess.Id) {
            Remove-Item -LiteralPath $pidPath -Force
        }
    }
    throw "$($_.Exception.Message) Server logs: $stdoutLog ; $stderrLog"
}

Write-Output "MidiBrave realtime instrument PID: $($serverProcess.Id)"
Write-Output "URL: $instrumentUrl"
Start-Process -FilePath $instrumentUrl | Out-Null
