#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="babylm/babyllama-100m-2024"
VALIDATED_ROOT="data/processed/validated_trials"
PYTHON_EXE="python"
WHAT_IF=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_train_eval_validated_trials.sh [options]

Options:
  --validated-root DIR   Root directory containing validated trial subfolders.
                         Default: data/processed/validated_trials
  --python-exe CMD       Python executable to use. Default: python
  --what-if              Print commands without executing them.
  -h, --help             Show this help message.

Expected layout:
  data/processed/validated_trials/<run_id>/train_ready.jsonl

For each subfolder, the script runs:
  1. train_babyllama_grammar.py with --run-id <folder_name>
  2. evaluate_finetuned_babyllama.py with --model-dir artifacts/models/babyllama_2024_<folder_name>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --validated-root)
      VALIDATED_ROOT="$2"
      shift 2
      ;;
    --python-exe)
      PYTHON_EXE="$2"
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

if [[ ! -d "$VALIDATED_ROOT" ]]; then
  echo "Validated trials directory not found: $VALIDATED_ROOT" >&2
  exit 1
fi

mapfile -t TRIAL_DIRS < <(find "$VALIDATED_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#TRIAL_DIRS[@]} -eq 0 ]]; then
  echo "No validated trial folders found in: $VALIDATED_ROOT" >&2
  exit 1
fi

total=${#TRIAL_DIRS[@]}
index=0

echo "Validated root : $VALIDATED_ROOT"
echo "Model          : $MODEL_NAME"
echo "Folders        : $total"
echo

for trial_dir in "${TRIAL_DIRS[@]}"; do
  index=$((index + 1))
  run_id="$(basename "$trial_dir")"
  train_file="$trial_dir/train_ready.jsonl"
  model_dir="artifacts/models/babyllama_2024_${run_id}"

  if [[ ! -f "$train_file" ]]; then
    echo "[$index/$total] Skipping $run_id because train file is missing: $train_file"
    echo
    continue
  fi

  train_args=(
    "src/training/train_babyllama_grammar.py"
    "--model-name" "$MODEL_NAME"
    "--train-file" "$train_file"
    "--run-id" "$run_id"
  )

  eval_args=(
    "src/training/evaluate_finetuned_babyllama.py"
    "--model-dir" "$model_dir"
    "--compare-base"
    "--run-id" "$run_id"
  )

  echo "[$index/$total] run_id=$run_id"
  echo "Train: $PYTHON_EXE ${train_args[*]}"
  echo "Eval : $PYTHON_EXE ${eval_args[*]}"

  if [[ "$WHAT_IF" -eq 0 ]]; then
    "$PYTHON_EXE" "${train_args[@]}"
    "$PYTHON_EXE" "${eval_args[@]}"
  else
    echo "[WhatIf] Skipped execution."
  fi

  echo
done

echo "Train/eval batch finished."
