param(
    [string[]]$Phenomena = @(),
    [int]$Count = 42,
    [int]$RepeatsPerPhenomenon = 2,
    [int]$DelaySeconds = 30,
    [string[]]$Prompts = @('prompt_1', 'prompt_2', 'prompt_3'),
    [string[]]$Providers = @('openai', 'grok', 'gemini'),
    [string]$ConfigPath = 'data/topics/prompt_topic_config.json',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$phenomenaFromConfig = @($config.phenomenon_cards | ForEach-Object { $_.name })
if (-not $phenomenaFromConfig -or $phenomenaFromConfig.Count -eq 0) {
    throw "No phenomena found in $ConfigPath"
}

if (-not $Phenomena -or $Phenomena.Count -eq 0) {
    $Phenomena = $phenomenaFromConfig
}

$providerScripts = @{
    openai = 'src/generation/generate_grammaticality_data_openai.py'
    grok   = 'src/generation/generate_grammaticality_data_grok.py'
    gemini = 'src/generation/generate_grammaticality_data_gemini.py'
}

$providerOutputDirs = @{
    openai = 'data/raw/openai'
    grok   = 'data/raw/grok'
    gemini = 'data/raw/gemini'
}

foreach ($provider in $Providers) {
    if (-not $providerScripts.ContainsKey($provider)) {
        throw "Unsupported provider: $provider"
    }
    if (-not $providerOutputDirs.ContainsKey($provider)) {
        throw "Missing output directory mapping for provider: $provider"
    }
}

$totalCommands = $Providers.Count * $Prompts.Count * $Phenomena.Count * $RepeatsPerPhenomenon
$commandIndex = 0

Write-Host "Phenomena       : $($Phenomena -join ', ')"
Write-Host "Phenomena count : $($Phenomena.Count)"
Write-Host "Prompts         : $($Prompts -join ', ')"
Write-Host "Providers       : $($Providers -join ', ')"
Write-Host "Repeats/phenom. : $RepeatsPerPhenomenon"
Write-Host "Count/run       : $Count"
Write-Host "Total commands  : $totalCommands"
Write-Host "Delay seconds   : $DelaySeconds"
Write-Host ""

foreach ($provider in $Providers) {
    $scriptPath = $providerScripts[$provider]

    foreach ($promptId in $Prompts) {
        foreach ($phenomenon in $Phenomena) {
            for ($repeat = 1; $repeat -le $RepeatsPerPhenomenon; $repeat++) {
                $commandIndex++
                $outputDir = $providerOutputDirs[$provider]
                if (-not (Test-Path -LiteralPath $outputDir)) {
                    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
                }
                $outputPath = Join-Path $outputDir "${provider}_${promptId}_${phenomenon}.jsonl"
                $argsList = @(
                    $scriptPath,
                    '--prompt-id', $promptId,
                    '--phenomenon', $phenomenon,
                    '--count', $Count,
                    '--output', $outputPath,
                    '--append'
                )

                $displayCommand = 'python ' + ($argsList -join ' ')
                Write-Host "[$commandIndex/$totalCommands] $displayCommand"

                if (-not $WhatIf) {
                    & python @argsList
                    if ($LASTEXITCODE -ne 0) {
                        throw "Command failed with exit code ${LASTEXITCODE}: $displayCommand"
                    }

                    Write-Host "Completed. Waiting $DelaySeconds seconds..."
                    Start-Sleep -Seconds $DelaySeconds
                }
                else {
                    Write-Host '[WhatIf] Skipped execution.'
                }

                Write-Host ''
            }
        }
    }
}

Write-Host 'All generation commands finished.'
