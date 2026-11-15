<#
Start Strata's DeepSeek-V4.1-Flash engine (strata-dsv41) as the one big engine, on the port the Hermes agent expects.

  powershell -NoProfile -File D:\models\engines\strata-ds4.1-flash\ds4\launch-ds4.ps1 [-Port 19100] [-Context 65536]

One big engine at a time: any running Qwen or DeepSeek Strata server or llama-server is stopped first.
Smaller models that run beside it take the next ports (19101, 19102, ...).
The model is the ISTA-DAS IQ1_M set packed in <engine>\models\pack (a link to C:\models\DSV41-IQ1_M on the NVMe drive).
OpenAI-compatible endpoint: http://127.0.0.1:<Port>/v1  (model name: deepseek-v4.1-flash)
#>
param([int]$Port = 19100, [int]$Context = 65536)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$exe = Join-Path $root "build-dsv41\strata-dsv41.exe"
$pack = (Get-Item (Join-Path $root "models")).Target | Select-Object -First 1
if (-not $pack) { $pack = Join-Path $root "models" }
$pack = Join-Path $pack "pack"
$runs = Join-Path $root "ds4\runs"
New-Item -ItemType Directory -Force $runs | Out-Null
$run = Join-Path $runs "strata-ds4-run.json"
if (-not (Test-Path $exe)) { throw "no engine at ${exe}: build it with packaging\build_dsv41_windows.bat" }
if (-not (Test-Path (Join-Path $pack "ds4.txt"))) { throw "no pack at $pack" }

Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq "python.exe" -and $_.CommandLine -like "*serve*server.py*--engine*strata*") -or
    $_.Name -in @("strata.exe", "strata-dsv41.exe", "llama-server.exe")
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
for ($i = 0; $i -lt 30 -and (Get-Process strata, strata-dsv41, llama-server -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 500 }

# setup.py's recipe for an ISTA-DAS set on one CUDA card: --dspark auto (never off), conf 0.65, no vision tower in this pack
$cfg = [ordered]@{
    exe = $exe
    args = @("--pack", $pack, "--max-context", "$Context", "--eos", "1", "--vision", "off", "--dspark", "auto", "--dspark-conf", "0.65")
    cwd = $root
    tokenizer = (Join-Path $pack "tokenizer")
    model_name = "deepseek-v4.1-flash"
    arch = "deepseek41"
    log = (Join-Path $runs "strata-ds4.log")
    lib_dirs = @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin")
    port = $Port
    engine_ready_s = 3600
    # split-KV decode attention: +3.6% real agent turns at 256K, +5-6% at 100K/200K-token prompts on this 3090, last-place
    # rounding only (ds4\lab\RESULTS.md).  The config's env beats the shell's, so a shell DS4_ATTN_SPLIT=0 is passed on here.
    env = @{ DS4_ATTN_SPLIT = $(if ($env:DS4_ATTN_SPLIT) { $env:DS4_ATTN_SPLIT } else { "1" }) }
}
$cfg | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $run

Start-Process -FilePath $py -WorkingDirectory $root -ArgumentList @(
    (Join-Path $root "serve\server.py"), "--engine", "strata", "--config", $run, "--port", "$Port")

$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt 900) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($h.loaded) {
            Write-Host ("DeepSeek ready after {0:N0} s: http://127.0.0.1:{1}/v1 (model deepseek-v4.1-flash)" -f ((Get-Date) - $t0).TotalSeconds, $Port)
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 2
}
throw "DeepSeek did not report loaded within 15 minutes; see $($cfg.log)"
