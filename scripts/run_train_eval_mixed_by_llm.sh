#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="babylm/babyllama-100m-2024"
INPUT_ROOT="data/processed/mixed_final_generated_grammar_data_by_llm"
PYTHON_EXE="python"
WHAT_IF=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_train_eval_mixed_by_llm.sh [options]

Options:
  --input-root DIR      Directory containing input .jsonl files.
                        Default: data/processed/mixed_final_generated_grammar_data_by_llm
  --python-exe CMD      Python executable to use. Default: python
  --what-if             Print commands without executing them.
  -h, --help            Show this help message.

Expected layout:
  data/processed/mixed_final_generated_grammar_data_by_llm/<run_id>.jsonl

For each input file, the script runs:
  1. train_babyllama_grammar.py with --train-file <file> and --run-id <file_stem>
  2. evaluate_finetuned_babyllama.py with --model-dir artifacts/models/babyllama_100m_2024_<file_stem>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
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

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input directory not found: $INPUT_ROOT" >&2
  exit 1
fi

mapfile -t INPUT_FILES < <(find "$INPUT_ROOT" -maxdepth 1 -type f -name '*.jsonl' | sort)

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "No .jsonl files found in: $INPUT_ROOT" >&2
  exit 1
fi

total=${#INPUT_FILES[@]}
index=0

echo "Input root     : $INPUT_ROOT"
echo "Model          : $MODEL_NAME"
echo "Files          : $total"
echo

for input_file in "${INPUT_FILES[@]}"; do
  index=$((index + 1))
  run_id="$(basename "${input_file%.*}")"
  model_dir="artifacts/models/babyllama_100m_2024_${run_id}"

  train_args=(
    "src/training/train_babyllama_grammar.py"
    "--model-name" "$MODEL_NAME"
    "--train-file" "$input_file"
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
