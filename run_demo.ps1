$python = if ($env:SAARTHI_PYTHON) { $env:SAARTHI_PYTHON } else { "python" }
Push-Location $PSScriptRoot
try {
    & $python -m saarthi_app @args
    $status = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $status
