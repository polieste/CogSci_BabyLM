#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
PYTHON_EXE="python"
WHAT_IF=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_validate_directory.sh --input-dir DIR --output-dir DIR [options]

Options:
  --input-dir DIR     Directory containing input .jsonl files.
  --output-dir DIR    Directory where per-file validation outputs will be written.
  --python-exe CMD    Python executable to use. Default: python
  --what-if           Print commands without executing them.
  -h, --help          Show this help message.

Output layout:
  output-dir/<input-stem>/train_ready.jsonl
  output-dir/<input-stem>/invalid.jsonl
  output-dir/<input-stem>/duplicates.jsonl
  output-dir/<input-stem>/report.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
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

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t JSONL_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jsonl' | sort)

if [[ ${#JSONL_FILES[@]} -eq 0 ]]; then
  echo "No .jsonl files found in: $INPUT_DIR" >&2
  exit 1
fi

total=${#JSONL_FILES[@]}
index=0

echo "Input dir  : $INPUT_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Files      : $total"
echo

for file in "${JSONL_FILES[@]}"; do
  index=$((index + 1))
  stem="$(basename "${file%.*}")"
  file_output_dir="$OUTPUT_DIR/$stem"
  mkdir -p "$file_output_dir"

  args=(
    "src/postprocess/validate_generated_grammar_data.py"
    "$file"
    "--output" "$file_output_dir/train_ready.jsonl"
    "--invalid-output" "$file_output_dir/invalid.jsonl"
    "--duplicates-output" "$file_output_dir/duplicates.jsonl"
    "--report-output" "$file_output_dir/report.json"
  )

  echo "[$index/$total] $PYTHON_EXE ${args[*]}"

  if [[ "$WHAT_IF" -eq 0 ]]; then
    "$PYTHON_EXE" "${args[@]}"
  else
    echo "[WhatIf] Skipped execution."
  fi

  echo
done

echo "Validation batch finished."
