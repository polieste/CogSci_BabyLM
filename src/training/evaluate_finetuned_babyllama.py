import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


DEFAULT_BASE_MODEL_NAME = "babylm/babyllama-100m-2024"


def build_model_run_name(model_name: str, run_id: str | None = None) -> str:
    model_key = model_name.lower()
    if model_key == "babylm/babyllama-100m-2024":
        base_name = "babyllama_2024"
    elif model_key == "babylm-community/babylm-baseline-100m-gpt-bert-causal-focus":
        base_name = "babyllama_gpt_bert_100m"
    elif model_key == "babylm-community/babylm-baseline-10m-gpt-bert-causal-focus":
        base_name = "babyllama_gpt_bert_10m"
    else:
        slug = model_name.split("/")[-1].lower()
        for src, dst in [("-", "_"), (" ", "_"), (".", "_")]:
            slug = slug.replace(src, dst)
        base_name = slug

    if run_id:
        return f"{base_name}_{run_id}"
    return base_name


def _install_tied_weights_compat_shim() -> None:
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _get_all_tied_weights_keys(self):
        tied = getattr(self, "_tied_weights_keys", None)
        if tied is None:
            return {}
        if isinstance(tied, dict):
            return tied
        return {key: [key] for key in tied}

    def _set_all_tied_weights_keys(self, value):
        self.__dict__["all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys,
        _set_all_tied_weights_keys,
    )


_install_tied_weights_compat_shim()


def _install_json_dtype_compat_shim() -> None:
    original_default = json.JSONEncoder.default
    if getattr(json.JSONEncoder, "_codex_dtype_compat", False):
        return

    def _default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        if obj.__class__.__name__ == "dtype":
            return str(obj)
        return original_default(self, obj)

    json.JSONEncoder.default = _default
    json.JSONEncoder._codex_dtype_compat = True


_install_json_dtype_compat_shim()


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


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise TypeError(f"Unsupported model output type for logits extraction: {type(outputs)!r}")


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
    logits = _extract_logits(outputs)[:, :-1, :]
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
        default=Path("data/blimp/blimp_validation.json"),
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
        default=None,
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
        "--run-id",
        default="id",
        help="Suffix used in saved evaluation report names to distinguish runs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable Hugging Face trust_remote_code for custom architectures such as GPT-BERT baselines.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = build_model_run_name(args.base_model_name if args.compare_base else str(args.model_dir), args.run_id)
    if args.output is None:
        args.output = Path("data/reports") / f"{run_name}_eval.json"
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Finetuned overall accuracy: {finetuned_results['overall_accuracy']:.4f}")
    if args.compare_base:
        print(f"Base overall accuracy: {results['base']['overall_accuracy']:.4f}")
        print(f"Delta: {results['accuracy_delta']:+.4f}")
    print(f"Saved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()


