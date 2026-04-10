param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$PythonExe = 'python',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "Input directory not found: $InputDir"
}

$inputRoot = (Resolve-Path -LiteralPath $InputDir).Path
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$outputRoot = (Resolve-Path -LiteralPath $OutputDir).Path

$jsonlFiles = Get-ChildItem -LiteralPath $inputRoot -Filter *.jsonl -File | Sort-Object Name
if (-not $jsonlFiles -or $jsonlFiles.Count -eq 0) {
    throw "No .jsonl files found in: $inputRoot"
}

$total = $jsonlFiles.Count
$index = 0

Write-Host "Input dir  : $inputRoot"
Write-Host "Output dir : $outputRoot"
Write-Host "Files      : $total"
Write-Host ""

foreach ($file in $jsonlFiles) {
    $index++
    $stem = $file.BaseName
    $fileOutputDir = Join-Path $outputRoot $stem
    if (-not (Test-Path -LiteralPath $fileOutputDir)) {
        New-Item -ItemType Directory -Path $fileOutputDir -Force | Out-Null
    }

    $argsList = @(
        'src/postprocess/validate_generated_grammar_data.py',
        $file.FullName,
        '--output', (Join-Path $fileOutputDir 'train_ready.jsonl'),
        '--invalid-output', (Join-Path $fileOutputDir 'invalid.jsonl'),
        '--duplicates-output', (Join-Path $fileOutputDir 'duplicates.jsonl'),
        '--report-output', (Join-Path $fileOutputDir 'report.json')
    )

    $displayCommand = "$PythonExe " + ($argsList -join ' ')
    Write-Host "[$index/$total] $displayCommand"

    if (-not $WhatIf) {
        & $PythonExe @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $displayCommand"
        }
    }
    else {
        Write-Host '[WhatIf] Skipped execution.'
    }

    Write-Host ''
}

Write-Host 'Validation batch finished.'
