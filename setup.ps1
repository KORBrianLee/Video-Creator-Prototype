param(
    [string]$RuntimeDir = 'C:\CursorVideoRuntime',
    [ValidateSet('wan', 'neodragon', 'lightning')][string]$Profile = 'neodragon',
    [ValidateSet('auto', 'gpu', 'intel-gpu', 'cpu')][string]$Backend = 'auto'
)
$ErrorActionPreference = 'Stop'
if ($RuntimeDir -notmatch '^[CD]:[\\/]' -or ($RuntimeDir -split '[\\/]') -contains '..') { throw 'C 또는 D SSD의 전용 절대 경로를 지정하세요.' }
$taskRuntime = [IO.Path]::GetFullPath($RuntimeDir)
$taskVolume = [IO.Path]::GetPathRoot($taskRuntime)
if ($taskRuntime.TrimEnd('\') -eq $taskVolume.TrimEnd('\')) { throw '드라이브 루트에 설치할 수 없습니다.' }
if (-not (Test-Path -LiteralPath $taskVolume -PathType Container)) { throw '선택한 SSD가 없습니다.' }
$taskPythonDir = Join-Path $taskRuntime 'runtime\python'
$taskPython = Join-Path $taskPythonDir 'python.exe'
$taskDownloads = Join-Path $taskRuntime 'downloads'
$taskTmp = Join-Path $taskRuntime 'tmp'
foreach ($taskTarget in @($taskRuntime, $taskPythonDir, $taskDownloads, $taskTmp)) {
    $taskAncestor = $taskTarget
    while ($taskAncestor -and $taskAncestor -ne $taskVolume) {
        if (Test-Path -LiteralPath $taskAncestor) {
            $taskItem = Get-Item -LiteralPath $taskAncestor -Force
            if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw '설치 폴더의 링크나 junction은 허용하지 않습니다.' }
        }
        $taskAncestor = Split-Path -Parent $taskAncestor
    }
}
New-Item -ItemType Directory -Force -Path $taskPythonDir, $taskDownloads, $taskTmp | Out-Null
$env:TEMP = $taskTmp
$env:TMP = $taskTmp
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    $taskLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime.lock.json') -Raw | ConvertFrom-Json
    $taskArchive = Join-Path $taskDownloads $taskLock.python.filename
    Invoke-WebRequest -UseBasicParsing -Uri $taskLock.python.url -OutFile $taskArchive
    $taskHash = (Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskHash -ne $taskLock.python.sha256) { throw 'Python 설치 파일 검증 실패.' }
    Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskPythonDir -Force
}
& $taskPython -B (Join-Path $PSScriptRoot 'installer.py') all --runtime-dir $taskRuntime --profile $Profile --backend $Backend --with-vulkan
if ($LASTEXITCODE -ne 0) { throw '설치가 완료되지 않았습니다. 같은 setup.cmd로 재시도할 수 있습니다.' }
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-dir.txt') -Value $taskRuntime -Encoding UTF8
Write-Host "영상 환경 준비 완료. Cursor에서 $taskRuntime\app 폴더를 열고 video_doctor로 확인하세요."
