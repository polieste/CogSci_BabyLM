#!/usr/bin/env bash

set -euo pipefail

COUNT=42
REPEATS_PER_PHENOMENON=2
DELAY_SECONDS=30
CONFIG_PATH="data/topics/prompt_topic_config.json"
WHAT_IF=0

PROMPTS=("prompt_1" "prompt_2" "prompt_3")
PROVIDERS=("grok")
PHENOMENA=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_generation_matrix.sh [options]

Options:
  --phenomena a,b,c        Comma-separated list of phenomena. If omitted, use all from config.
  --prompts a,b,c          Comma-separated list of prompt ids. Default: prompt_1,prompt_2,prompt_3
  --providers a,b,c        Comma-separated list of providers. Default: grok
  --count N                Number of items per run. Default: 42
  --repeats N              Number of repeats per provider/prompt/phenomenon. Default: 2
  --delay-seconds N        Delay after each completed command. Default: 30
  --config-path PATH       Prompt/topic config path. Default: data/topics/prompt_topic_config.json
  --what-if                Print commands without executing them.
  -h, --help               Show this help message.

Examples:
  bash scripts/run_generation_matrix.sh --what-if
  bash scripts/run_generation_matrix.sh --providers openai,grok,gemini --phenomena anaphor_agreement,quantifiers
EOF
}

split_csv() {
  local input="$1"
  local -n out_array=$2
  IFS=',' read -r -a out_array <<< "$input"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phenomena)
      split_csv "$2" PHENOMENA
      shift 2
      ;;
    --prompts)
      split_csv "$2" PROMPTS
      shift 2
      ;;
    --providers)
      split_csv "$2" PROVIDERS
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --repeats)
      REPEATS_PER_PHENOMENON="$2"
      shift 2
      ;;
    --delay-seconds)
      DELAY_SECONDS="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --what-if)
      WHAT_IF=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

mapfile -t PHENOMENA_FROM_CONFIG < <(
  python - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
for card in config.get("phenomenon_cards", []):
    name = card.get("name", "").strip()
    if name:
        print(name)
PY
)

if [[ ${#PHENOMENA_FROM_CONFIG[@]} -eq 0 ]]; then
  echo "No phenomena found in $CONFIG_PATH" >&2
  exit 1
fi

if [[ ${#PHENOMENA[@]} -eq 0 ]]; then
  PHENOMENA=("${PHENOMENA_FROM_CONFIG[@]}")
fi

provider_script() {
  case "$1" in
    openai) echo "src/generation/generate_grammaticality_data_openai.py" ;;
    grok) echo "src/generation/generate_grammaticality_data_grok.py" ;;
    gemini) echo "src/generation/generate_grammaticality_data_gemini.py" ;;
    *)
      echo "Unsupported provider: $1" >&2
      exit 1
      ;;
  esac
}

provider_output_dir() {
  case "$1" in
    openai) echo "data/raw/openai" ;;
    grok) echo "data/raw/grok" ;;
    gemini) echo "data/raw/gemini" ;;
    *)
      echo "Missing output directory mapping for provider: $1" >&2
      exit 1
      ;;
  esac
}

total_commands=$(( ${#PROVIDERS[@]} * ${#PROMPTS[@]} * ${#PHENOMENA[@]} * REPEATS_PER_PHENOMENON ))
command_index=0

echo "Phenomena       : ${PHENOMENA[*]}"
echo "Phenomena count : ${#PHENOMENA[@]}"
echo "Prompts         : ${PROMPTS[*]}"
echo "Providers       : ${PROVIDERS[*]}"
echo "Repeats/phenom. : $REPEATS_PER_PHENOMENON"
echo "Count/run       : $COUNT"
echo "Total commands  : $total_commands"
echo "Delay seconds   : $DELAY_SECONDS"
echo

for provider in "${PROVIDERS[@]}"; do
  script_path="$(provider_script "$provider")"

  for prompt_id in "${PROMPTS[@]}"; do
    for phenomenon in "${PHENOMENA[@]}"; do
      for (( repeat=1; repeat<=REPEATS_PER_PHENOMENON; repeat++ )); do
        command_index=$((command_index + 1))
        output_dir="$(provider_output_dir "$provider")"
        mkdir -p "$output_dir"
        output_path="${output_dir}/${provider}_${prompt_id}_${phenomenon}.jsonl"

        args=(
          "$script_path"
          "--prompt-id" "$prompt_id"
          "--phenomenon" "$phenomenon"
          "--count" "$COUNT"
          "--output" "$output_path"
          "--append"
        )

        display_command="python ${args[*]}"
        echo "[$command_index/$total_commands] $display_command"

        if [[ "$WHAT_IF" -eq 0 ]]; then
          python "${args[@]}"
          echo "Completed. Waiting $DELAY_SECONDS seconds..."
          sleep "$DELAY_SECONDS"
        else
          echo "[WhatIf] Skipped execution."
        fi

        echo
      done
    done
  done
done

echo "All generation commands finished."
