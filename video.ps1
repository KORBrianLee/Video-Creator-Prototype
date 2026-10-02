param([Parameter(ValueFromRemainingArguments=$true)][string[]]$VideoArguments)
$ErrorActionPreference = 'Stop'
$taskRuntime = 'C:\CursorVideoRuntime'
$taskConfigPath = Join-Path $PSScriptRoot 'video.config.json'
if (Test-Path -LiteralPath $taskConfigPath) {
    $taskRuntime = (Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json).runtime_dir
}
if ($env:CVL_RUNTIME_DIR) { $taskRuntime = $env:CVL_RUNTIME_DIR }
$taskPython = Join-Path $taskRuntime 'runtime\python\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw '실행 환경을 준비하거나 설정한 SSD 위치를 확인하세요.' }
$env:CVL_RUNTIME_DIR = $taskRuntime
$env:TEMP = Join-Path $taskRuntime 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $taskPython -B (Join-Path $PSScriptRoot 'video_agent.py') @VideoArguments
exit $LASTEXITCODE
