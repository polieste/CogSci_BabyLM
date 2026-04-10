# BabyLM Grammaticality Judgment Pipeline

This repository contains a full workflow for:

1. Generating grammaticality judgment data with multiple parent LLMs
2. Merging and cleaning the generated data
3. Exploring the final training dataset
4. Fine-tuning BabyLM-style models on grammaticality judgment pairs
5. Evaluating the fine-tuned model on BLiMP

The overall task setup follows the BabyLM grammaticality judgment formulation:

- each example contains a grammatical sentence `good`
- and an ungrammatical counterpart `bad`
- training encourages the model to assign higher sentence probability to `good`
- evaluation compares sentence probabilities on BLiMP minimal pairs


## Project Files

### Data Generation

- [generate_grammaticality_data_openai.py](./src/generation/generate_grammaticality_data_openai.py)
  Generate data with OpenAI models such as `gpt-5.4-mini`

- [generate_grammaticality_data_grok.py](./src/generation/generate_grammaticality_data_grok.py)
  Generate data with xAI Grok models

- [generate_grammaticality_data_gemini.py](./src/generation/generate_grammaticality_data_gemini.py)
  Generate data with Gemini models

- [prompt_topic_config.json](./data/topics/prompt_topic_config.json)
  Central config file for prompt templates, topics, and phenomenon cards

- [generation_config.py](./src/generation/generation_config.py)
  Shared helper that reads `prompt_topic_config.json` and builds prompts

- [run_generation_matrix.ps1](./scripts/run_generation_matrix.ps1)
  PowerShell batch runner for repeated generation across prompts, phenomena, and providers

- [run_generation_matrix.sh](./scripts/run_generation_matrix.sh)
  Bash batch runner with the same generation loop logic

### Data Processing

- [prepare_generated_grammar_data.py](./src/postprocess/prepare_generated_grammar_data.py)
  Merge one or many generated files into a unified JSONL dataset

- [validate_generated_grammar_data.py](./src/postprocess/validate_generated_grammar_data.py)
  Validate, clean, deduplicate, and export train-ready JSONL

### EDA
- [train_ready_grammar_data_eda.ipynb](./notebooks/eda/train_ready_grammar_data_eda.ipynb)
  Explore the final cleaned training file

- [Language_acquisition.ipynb](./notebooks/benchmark/Language_acquisition.ipynb)
  Base BLiMP benchmark notebook and official word-count method

### Training and Evaluation

- [train_babyllama_grammar.py](./src/training/train_babyllama_grammar.py)
  Fine-tune a causal LM on `good` vs `bad` grammaticality pairs

- [evaluate_finetuned_babyllama.py](./src/training/evaluate_finetuned_babyllama.py)
  Evaluate a fine-tuned model on `data/blimp/blimp_validation.json`


## Installation

Install dependencies:

```powershell
pip install -r requirements.txt
```

Main packages used:

- `torch`
- `transformers`
- `nltk`
- `pandas`
- `openai`
- `google-genai`
- `pypdf`


## API Keys

Store API keys in `.env` or export them in your shell.

Example `.env`:

```env
OPENAI_API_KEY=...
XAI_API_KEY=...
GEMINI_API_KEY=...
```


## Stage 1: Generate Data

All generation scripts support:

- `--prompt-id`
- `--phenomenon`
- `--count`
- `--topics`
- `--output`
- `--append`

If `--output` is omitted, the filename is created automatically using:

```text
{LLM}_{Prompt-id}_{Phenomenon}_{Topic}_{HHMM}.json
```

Default output directories are:

- `data/raw/openai/`
- `data/raw/grok/`
- `data/raw/gemini/`

Examples:

### OpenAI

```powershell
python src/generation/generate_grammaticality_data_openai.py --prompt-id prompt_1 --phenomenon binding --count 20
```

### Grok

```powershell
python src/generation/generate_grammaticality_data_grok.py --prompt-id prompt_2 --phenomenon anaphor_agreement --count 15 --topics family_home,school_classroom
```

### Gemini

```powershell
python src/generation/generate_grammaticality_data_gemini.py --prompt-id prompt_3 --phenomenon ellipsis --count 10
```

### Append new generations to the same file

```powershell
python src/generation/generate_grammaticality_data_openai.py --prompt-id prompt_1 --phenomenon anaphor_agreement --count 42 --output data/raw/openai/openai_prompt_1_anaphor_agreement.jsonl --append
```

Generated records follow the config schema:

```json
{
  "phenomenon": "",
  "topic": "",
  "good": "",
  "bad": "",
  "edit_type": ""
}
```

### Batch generation scripts

Both batch scripts:

- read the phenomenon list from `data/topics/prompt_topic_config.json`
- loop over `provider`, `prompt`, and `phenomenon`
- repeat each configuration multiple times
- append results into one shared file per `LLM + prompt + phenomenon`

Current defaults in the batch scripts are:

- prompts: `prompt_1`, `prompt_2`, `prompt_3`
- providers: `grok`
- count per run: `42`
- repeats per configuration: `2`

PowerShell example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_generation_matrix.ps1 -WhatIf
```

Bash example:

```bash
bash scripts/run_generation_matrix.sh --what-if
```

Example with multiple providers:

```bash
bash scripts/run_generation_matrix.sh --providers openai,grok,gemini --phenomena anaphor_agreement,quantifiers
```


## Stage 2: Merge Generated Files

Use [prepare_generated_grammar_data.py](./src/postprocess/prepare_generated_grammar_data.py) to combine many generated files into normalized JSONL datasets.

It accepts:

- one file
- many files
- a directory

Supported inputs:

- `.json`
- `.jsonl`

Output schema:

```json
{
  "id": "",
  "phenomenon": "",
  "topic": "",
  "good": "",
  "bad": "",
  "edit_type": "",
  "prompt_family": "",
  "parent_llm": ""
}
```

When you run it once, it now creates 3 groups of outputs:

- one merged file containing all records
- one folder of merged files grouped by `parent_llm`
- one folder of merged files grouped by `parent_llm` and `prompt_family`

Example:

```powershell
python src/postprocess/prepare_generated_grammar_data.py data/raw/openai data/raw/grok data/raw/gemini --output data/processed/final_generated_grammar_data.jsonl
```

Default outputs for that example:

- `data/processed/final_generated_grammar_data.jsonl`
- `data/processed/final_generated_grammar_data_by_llm/`
- `data/processed/final_generated_grammar_data_by_llm_prompt/`


## Stage 3: Validate and Clean the Data

Use [validate_generated_grammar_data.py](./src/postprocess/validate_generated_grammar_data.py) on the merged JSONL file.

It will:

- check required fields
- detect empty fields
- remove records where `good == bad`
- remove duplicate training rows
- export:
  - cleaned training file
  - invalid records
  - duplicates
  - summary report

Example:

```powershell
python src/postprocess/validate_generated_grammar_data.py data/processed/final_generated_grammar_data.jsonl
```

Default outputs:

- `data/processed/train_ready_grammar_data.jsonl`
- `data/processed/invalid_generated_grammar_data.jsonl`
- `data/processed/duplicate_generated_grammar_data.jsonl`
- `data/reports/generated_grammar_data_report.json`


## Stage 4: EDA
### Explore Final Train-Ready Data

Open:

- [train_ready_grammar_data_eda.ipynb](./notebooks/eda/train_ready_grammar_data_eda.ipynb)

This notebook focuses on:

- the final cleaned dataset
- final training distribution
- quality checks before fine-tuning
- official word counts for `good` and `bad`


## Stage 5: Train a Model

Use [train_babyllama_grammar.py](./src/training/train_babyllama_grammar.py).

This script:

- loads a causal language model from Hugging Face
- trains on `good` / `bad` sentence pairs
- uses pairwise ranking loss:
  - the model is encouraged to score `good` higher than `bad`
- creates a train/validation split
- tracks validation accuracy
- uses early stopping
- saves the best checkpoint

Supported models include:

- `babylm/babyllama-100m-2024`
- `BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus`
- `BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus`

### Train BabyLLaMA

```powershell
python src/training/train_babyllama_grammar.py --model-name babylm/babyllama-100m-2024 --train-file data/processed/train_ready_grammar_data.jsonl --run-id exp1
```

### Train GPT-BERT 100M Causal Focus

```powershell
python src/training/train_babyllama_grammar.py --model-name BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus --trust-remote-code --train-file data/processed/train_ready_grammar_data.jsonl --run-id exp1
```

### Train GPT-BERT 10M Causal Focus

```powershell
python src/training/train_babyllama_grammar.py --model-name BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus --trust-remote-code --train-file data/processed/train_ready_grammar_data.jsonl --run-id exp1
```

Useful arguments:

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--valid-ratio`
- `--patience`
- `--grad-accum-steps`
- `--trust-remote-code`

Outputs:

- fine-tuned model directory
- tokenizer files
- training report JSON

If `--output-dir` and `--report-file` are omitted, the script builds names automatically from the model family and `--run-id`, for example:

- `artifacts/models/babyllama_2024_exp1`
- `artifacts/models/babyllama_gpt_bert_100m_exp1`
- `artifacts/models/babyllama_gpt_bert_10m_exp1`


## Stage 6: Evaluate on BLiMP

Use [evaluate_finetuned_babyllama.py](./src/training/evaluate_finetuned_babyllama.py).

This script:

- loads the fine-tuned model
- evaluates on `data/blimp/blimp_validation.json`
- uses the same probability comparison rule as the benchmark:
  - choose the sentence with the higher sentence log probability
- can optionally compare against the original base model

### Evaluate a Fine-Tuned BabyLLaMA Model

```powershell
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/babyllama_2024_exp1 --compare-base --run-id exp1
```

### Evaluate a Fine-Tuned GPT-BERT Model

```powershell
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/babyllama_gpt_bert_100m_exp1 --compare-base --base-model-name BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus --trust-remote-code --run-id exp1
```

Default eval report outputs follow the same naming pattern, for example:

- `data/reports/babyllama_2024_exp1_eval.json`
- `data/reports/babyllama_gpt_bert_100m_exp1_eval.json`
- `data/reports/babyllama_gpt_bert_10m_exp1_eval.json`


## Example End-to-End Pipeline

### 1. Generate data

```powershell
python src/generation/generate_grammaticality_data_gemini.py --prompt-id prompt_3 --phenomenon anaphor_agreement --topics family_home --count 50
```

### 2. Merge all generated files

```powershell
python src/postprocess/prepare_generated_grammar_data.py data/raw/openai data/raw/grok data/raw/gemini --output data/processed/final_generated_grammar_data.jsonl
```

### 3. Validate and clean

```powershell
python src/postprocess/validate_generated_grammar_data.py data/processed/final_generated_grammar_data.jsonl
```

### 4. Inspect the final dataset

Open:

- [train_ready_grammar_data_eda.ipynb](./notebooks/eda/train_ready_grammar_data_eda.ipynb)

### 5. Fine-tune

```powershell
python src/training/train_babyllama_grammar.py --model-name babylm/babyllama-100m-2024 --train-file data/processed/train_ready_grammar_data.jsonl --run-id exp1
```

### 6. Evaluate

```powershell
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/babyllama_2024_exp1 --compare-base --run-id exp1
```


## Notes

- The generator outputs are stored with `.json` filenames but may contain JSONL-style one-record-per-line content.
- The batch generation scripts now append repeated runs into a single `.jsonl` file for each `LLM + prompt + phenomenon`.
- The merge script can read both JSONL and standard JSON containers.
- For GPT-BERT BabyLM baselines, `--trust-remote-code` may be required because the Hugging Face model card indicates custom code.
- The training objective is grammaticality judgment oriented, not plain language-model next-token training.
- The BLiMP evaluation uses sentence log probability comparison without length normalization, matching the benchmark setup used in the notebook.
