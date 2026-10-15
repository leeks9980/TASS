$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProjectRoot
$taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $taskBundledPython) {
    & $taskBundledPython (Join-Path $taskProjectRoot 'src\circuit_sim_gui.py')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python (Join-Path $taskProjectRoot 'src\circuit_sim_gui.py')
} else {
    throw 'Python이 없습니다. Python 3.12와 requirements.txt의 패키지를 설치하세요.'
}
