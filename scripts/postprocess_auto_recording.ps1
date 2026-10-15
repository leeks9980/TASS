param(
    [string]$RunDirectory
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$logRoot = Join-Path $root 'simulation_logs'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $run = Get-ChildItem -LiteralPath $logRoot -Directory -Filter 'full_fly_*' |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'protocol_status.json') } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $run) {
        throw 'No automatic recording with protocol_status.json was found.'
    }
    $runPath = $run.FullName
} else {
    $runPath = (Resolve-Path -LiteralPath $RunDirectory).Path
}

$protocolPath = Join-Path $runPath 'protocol_status.json'
if (-not (Test-Path -LiteralPath $protocolPath)) {
    throw "protocol_status.json not found: $runPath"
}

$protocol = Get-Content -LiteralPath $protocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $protocol.finished -or $protocol.stopped_by_storage_limit) {
    Write-Warning "Automatic tracing skipped: recording did not finish cleanly. Run: $runPath"
    exit 2
}

$planned = @($protocol.planned_cases)
if ($planned -notcontains 'neutral') {
    throw 'Automatic tracing requires a neutral reference recording.'
}
$cases = @($planned | Where-Object { $_ -ne 'neutral' })
if ($cases.Count -eq 0) {
    throw 'No stimulus cases were recorded.'
}

$output = Join-Path $runPath 'automatic_post_analysis'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$statusPath = Join-Path $output 'status.json'
$responseDirectory = Join-Path $runPath 'visual_response_analysis'
$responseTable = Join-Path $responseDirectory 'stimulus_selectivity.csv'
$responseReport = Join-Path $responseDirectory 'report.json'
$validationDirectory = Join-Path $runPath 'signal_chain_validation'
$validationReport = Join-Path $validationDirectory 'validation_report.json'
$hasCompletedValidation = Test-Path -LiteralPath $validationReport
$hasCompletedResponseAnalysis = (Test-Path -LiteralPath $responseTable) -and
                                (Test-Path -LiteralPath $responseReport)

$status = [ordered]@{
    schema = 1
    run = $runPath
    reference = 'neutral'
    cases = $cases
    started_utc = [DateTime]::UtcNow.ToString('o')
    signal_chain_validation = $(if ($hasCompletedValidation) { 'reused' } else { 'running' })
    response_analysis = $(if ($hasCompletedResponseAnalysis) { 'reused' } else { 'running' })
    traces = [ordered]@{}
    finished = $false
}
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8

if ($hasCompletedValidation) {
    Write-Host "[1/3] Reusing completed signal-chain validation: $validationDirectory"
} else {
    Write-Host "[1/3] Creating signal-chain CSV, JSON, SVG and HTML report: $runPath"
    & $python (Join-Path $root 'src\validate_visual_signal_chain.py') $runPath
    if ($LASTEXITCODE -ne 0) {
        $status.signal_chain_validation = 'failed'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        exit $LASTEXITCODE
    }
    $status.signal_chain_validation = 'completed'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

if ($hasCompletedResponseAnalysis) {
    Write-Host "[2/3] Reusing completed visual response analysis: $responseDirectory"
} else {
    Write-Host "[2/3] Comparing all visual conditions with neutral: $runPath"
    & $python (Join-Path $root 'src\analyze_visual_responses.py') $runPath --reference neutral --cases $cases --top-per-case 3000 --timeline-top 300
    if ($LASTEXITCODE -ne 0) {
        $status.response_analysis = 'failed'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        exit $LASTEXITCODE
    }
    $status.response_analysis = 'completed'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-Host '[3/3] Tracing responding descending neurons back toward photoreceptors.'
$dnRankings = Join-Path $output 'dn_response_rankings'
$dnReport = Join-Path $dnRankings 'report.json'
if (Test-Path -LiteralPath $dnReport) {
    Write-Host "Reusing completed DN rankings: $dnRankings"
} else {
    Write-Host 'Reading recorded snapshots once to rank DNs for all conditions.'
    & $python (Join-Path $root 'src\select_recorded_dn_targets.py') $runPath --output $dnRankings --reference neutral
    if ($LASTEXITCODE -ne 0) {
        foreach ($case in $cases) { $status.traces[$case] = 'failed_dn_ranking' }
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        exit $LASTEXITCODE
    }
}
foreach ($case in $cases) {
    $destination = Join-Path $output $case
    if (Test-Path -LiteralPath $destination) {
        Remove-Item -LiteralPath $destination -Recurse -Force
    }
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    $ranking = Join-Path $dnRankings ($case + '.csv')
    if (-not (Test-Path -LiteralPath $ranking)) {
        $status.traces[$case] = 'failed_no_dn_ranking'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        continue
    }
    Copy-Item -LiteralPath $ranking -Destination (Join-Path $destination 'ranked_descending_neurons.csv')
    $targets = @(Import-Csv -LiteralPath $ranking | Select-Object -First 10 -ExpandProperty bodyId)
    if ($targets.Count -eq 0) {
        $status.traces[$case] = 'skipped_no_dn_targets'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        continue
    }

    Write-Host "Tracing case: $case; DN targets: $($targets -join ', ')"
    $traceArguments = @($runPath, '--case', $case, '--targets') + $targets
    & $python (Join-Path $root 'src\trace_visual_paths.py') @traceArguments
    if ($LASTEXITCODE -ne 0) {
        $status.traces[$case] = 'failed'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        continue
    }

    $source = Join-Path $runPath 'visual_path_analysis'
    if (-not (Test-Path -LiteralPath $source)) {
        $status.traces[$case] = 'failed_no_trace_output'
        $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
        continue
    }
    Copy-Item -LiteralPath (Join-Path $source '*') -Destination $destination -Recurse -Force
    $status.traces[$case] = 'completed'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

$failed = @($status.traces.GetEnumerator() | Where-Object { $_.Value -ne 'completed' })
$status.finished = ($failed.Count -eq 0)
$status.completed_utc = [DateTime]::UtcNow.ToString('o')
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
if ($status.finished) {
    Write-Host "Automatic response analysis and reverse tracing completed: $output"
    exit 0
}
Write-Warning "Post-analysis completed with one or more failed/skipped cases. See: $statusPath"
exit 3
