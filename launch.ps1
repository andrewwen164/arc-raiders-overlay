param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Missing .venv. Follow README setup first: py -3.11 -m venv .venv'
}
if ($Smoke) {
    & $taskPython -m arc_audio ui --smoke-seconds 8
} else {
    & $taskPython -m arc_audio ui
}
exit $LASTEXITCODE
