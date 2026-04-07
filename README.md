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

If `--output` is omitted, the filename is created automatically using:

```text
{LLM}_{Prompt-id}_{Phenomenon}_{Topic}_{HHMM}.json
```

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


## Stage 2: Merge Generated Files

Use [prepare_generated_grammar_data.py](./src/postprocess/prepare_generated_grammar_data.py) to combine many generated files into one normalized JSONL dataset.

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

Example:

```powershell
python src/postprocess/prepare_generated_grammar_data.py data/raw/generated --output data/processed/final_generated_grammar_data.jsonl
```


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
python src/training/train_babyllama_grammar.py --model-name babylm/babyllama-100m-2024 --train-file data/processed/train_ready_grammar_data.jsonl --output-dir artifacts/models/babyllama_grammar_ft
```

### Train GPT-BERT 100M Causal Focus

```powershell
python src/training/train_babyllama_grammar.py --model-name BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus --trust-remote-code --train-file data/processed/train_ready_grammar_data.jsonl --output-dir artifacts/models/gptbert_100m_grammar_ft
```

### Train GPT-BERT 10M Causal Focus

```powershell
python src/training/train_babyllama_grammar.py --model-name BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus --trust-remote-code --train-file data/processed/train_ready_grammar_data.jsonl --output-dir artifacts/models/gptbert_10m_grammar_ft
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
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/babyllama_grammar_ft --compare-base
```

### Evaluate a Fine-Tuned GPT-BERT Model

```powershell
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/gptbert_100m_grammar_ft --compare-base --base-model-name BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus --trust-remote-code
```

Output:

- `data/reports/finetuned_babylm_benchmark_results.json`


## Example End-to-End Pipeline

### 1. Generate data

```powershell
python src/generation/generate_grammaticality_data_gemini.py --prompt-id prompt_3 --phenomenon anaphor_agreement --topics family_home --count 50
```

### 2. Merge all generated files

```powershell
python src/postprocess/prepare_generated_grammar_data.py data/raw/generated --output data/processed/final_generated_grammar_data.jsonl
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
python src/training/train_babyllama_grammar.py --model-name babylm/babyllama-100m-2024 --train-file data/processed/train_ready_grammar_data.jsonl --output-dir artifacts/models/babyllama_grammar_ft
```

### 6. Evaluate

```powershell
python src/training/evaluate_finetuned_babyllama.py --model-dir artifacts/models/babyllama_grammar_ft --compare-base
```


## Notes

- The generator outputs are stored with `.json` filenames but may contain JSONL-style one-record-per-line content.
- The merge script can read both JSONL and standard JSON containers.
- For GPT-BERT BabyLM baselines, `--trust-remote-code` may be required because the Hugging Face model card indicates custom code.
- The training objective is grammaticality judgment oriented, not plain language-model next-token training.
- The BLiMP evaluation uses sentence log probability comparison without length normalization, matching the benchmark setup used in the notebook.
