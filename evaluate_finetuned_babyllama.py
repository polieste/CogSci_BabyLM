import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL_NAME = "babylm/babyllama-100m-2024"


def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(model_path: str | Path, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def sentence_log_probabilities(model, tokenizer, texts: list[str], device: str) -> list[float]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    token_log_probs = F.log_softmax(logits, dim=-1)
    next_token_log_probs = token_log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    scores = (next_token_log_probs * target_mask).sum(dim=1)
    return scores.detach().cpu().tolist()


def evaluate_blimp(model, tokenizer, dataset: list[dict], batch_size: int, device: str) -> dict:
    term_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    all_correct = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        good_scores = sentence_log_probabilities(
            model, tokenizer, [item["sentence_good"] for item in batch], device
        )
        bad_scores = sentence_log_probabilities(
            model, tokenizer, [item["sentence_bad"] for item in batch], device
        )

        for item, good_score, bad_score in zip(batch, good_scores, bad_scores):
            correct = good_score > bad_score
            all_correct.append(correct)
            term_stats[item["linguistics_term"]]["correct"] += int(correct)
            term_stats[item["linguistics_term"]]["total"] += 1

    per_term_accuracy = {
        term: stats["correct"] / stats["total"]
        for term, stats in sorted(term_stats.items())
    }
    return {
        "dataset_size": len(dataset),
        "overall_accuracy": sum(all_correct) / len(all_correct),
        "per_term_accuracy": per_term_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned BabyLLaMA model on BLiMP validation."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("blimp_validation.json"),
        help="Path to the BLiMP validation JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for BLiMP evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("finetuned_babylm_benchmark_results.json"),
        help="Path to save the evaluation results.",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate the base BabyLLaMA model for direct comparison.",
    )
    parser.add_argument(
        "--base-model-name",
        default=DEFAULT_BASE_MODEL_NAME,
        help="Base model name to compare against.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable Hugging Face trust_remote_code for custom architectures such as GPT-BERT baselines.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_json(args.dataset)

    ft_model, ft_tokenizer = load_model_and_tokenizer(
        args.model_dir,
        device,
        trust_remote_code=args.trust_remote_code,
    )
    finetuned_results = evaluate_blimp(ft_model, ft_tokenizer, dataset, args.batch_size, device)

    results = {
        "model_dir": str(args.model_dir),
        "device": device,
        "trust_remote_code": args.trust_remote_code,
        "finetuned": finetuned_results,
    }

    if args.compare_base:
        base_model, base_tokenizer = load_model_and_tokenizer(
            args.base_model_name,
            device,
            trust_remote_code=args.trust_remote_code,
        )
        base_results = evaluate_blimp(base_model, base_tokenizer, dataset, args.batch_size, device)
        results["base"] = base_results
        results["base_model_name"] = args.base_model_name
        results["accuracy_delta"] = (
            finetuned_results["overall_accuracy"] - base_results["overall_accuracy"]
        )

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Finetuned overall accuracy: {finetuned_results['overall_accuracy']:.4f}")
    if args.compare_base:
        print(f"Base overall accuracy: {results['base']['overall_accuracy']:.4f}")
        print(f"Delta: {results['accuracy_delta']:+.4f}")
    print(f"Saved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
