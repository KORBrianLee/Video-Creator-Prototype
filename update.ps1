param(
    [string]$RuntimeDir = '',
    [switch]$NoPull
)
$ErrorActionPreference = 'Stop'
if (-not $RuntimeDir) {
    $taskSaved = Join-Path $PSScriptRoot 'runtime-dir.txt'
    $taskParent = Split-Path -Parent $PSScriptRoot
    if (Test-Path -LiteralPath $taskSaved -PathType Leaf) {
        $RuntimeDir = (Get-Content -LiteralPath $taskSaved -Raw).Trim()
    } elseif ((Split-Path -Leaf $PSScriptRoot) -eq 'app' -and (Test-Path -LiteralPath (Join-Path $taskParent 'runtime\python\python.exe') -PathType Leaf)) {
        $RuntimeDir = $taskParent
    } else {
        $RuntimeDir = 'D:\VideoCreator\CursorVideoRuntime'
    }
}
if (-not $NoPull) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git이 설치되어 있지 않습니다. git을 설치하거나 -NoPull로 실행하세요.' }
    & git -C $PSScriptRoot pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'git pull에 실패했습니다. 이 폴더의 로컬 수정 사항을 확인하세요.' }
}
$taskPython = Join-Path $RuntimeDir 'runtime\python\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    Write-Host "$RuntimeDir 에 설치가 없어 처음 설치를 진행합니다."
    & (Join-Path $PSScriptRoot 'setup.ps1') -RuntimeDir $RuntimeDir
    exit $LASTEXITCODE
}
$env:TEMP = Join-Path $RuntimeDir 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $taskPython -B (Join-Path $PSScriptRoot 'installer.py') update --runtime-dir $RuntimeDir
if ($LASTEXITCODE -ne 0) { throw '업데이트가 완료되지 않았습니다. 같은 update.cmd로 재시도할 수 있습니다.' }
Write-Host '업데이트 완료. Cursor에서 local-video MCP를 다시 시작하세요.'
