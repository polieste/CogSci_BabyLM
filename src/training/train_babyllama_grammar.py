import argparse
import copy
import inspect
import json
import random
from pathlib import Path

import nltk
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


DEFAULT_MODEL_NAME = "babylm/babyllama-100m-2024"


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


def ensure_nltk_tokenizer() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def count_word(text: str) -> int:
    ensure_nltk_tokenizer()
    return len(word_tokenize(text))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


class GrammarPairDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        item = self.records[index]
        return {
            "id": item["id"],
            "good": item["good"],
            "bad": item["bad"],
            "phenomenon": item["phenomenon"],
            "topic": item["topic"],
        }


def collate_pairs(batch: list[dict]) -> dict:
    return {
        "ids": [item["id"] for item in batch],
        "good_texts": [item["good"] for item in batch],
        "bad_texts": [item["bad"] for item in batch],
        "phenomena": [item["phenomenon"] for item in batch],
        "topics": [item["topic"] for item in batch],
    }


def split_records(records: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0.0, 1.0).")

    shuffled = records.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    valid_size = int(len(shuffled) * valid_ratio)
    if valid_ratio > 0 and valid_size == 0 and len(shuffled) > 1:
        valid_size = 1
    if valid_size >= len(shuffled) and len(shuffled) > 1:
        valid_size = len(shuffled) - 1

    valid_records = shuffled[:valid_size]
    train_records = shuffled[valid_size:]
    return train_records, valid_records


def load_model_and_tokenizer(model_name: str, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    return model, tokenizer


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise TypeError(f"Unsupported model output type for logits extraction: {type(outputs)!r}")


def sentence_log_probabilities(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
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

    return (next_token_log_probs * target_mask).sum(dim=1)


@torch.no_grad()
def evaluate_pair_accuracy(model, tokenizer, records: list[dict], batch_size: int, device: str) -> float:
    if not records:
        return 0.0

    model.eval()
    dataloader = DataLoader(
        GrammarPairDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    total = 0
    correct = 0

    for batch in dataloader:
        good_scores = sentence_log_probabilities(model, tokenizer, batch["good_texts"], device)
        bad_scores = sentence_log_probabilities(model, tokenizer, batch["bad_texts"], device)
        correct += (good_scores > bad_scores).sum().item()
        total += len(batch["good_texts"])

    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_pair_loss(model, tokenizer, records: list[dict], batch_size: int, device: str) -> float:
    if not records:
        return 0.0

    model.eval()
    dataloader = DataLoader(
        GrammarPairDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        good_scores = sentence_log_probabilities(model, tokenizer, batch["good_texts"], device)
        bad_scores = sentence_log_probabilities(model, tokenizer, batch["bad_texts"], device)
        pairwise_loss = -F.logsigmoid(good_scores - bad_scores).mean()
        total_loss += pairwise_loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def compute_training_stats(train_records: list[dict], valid_records: list[dict], num_epochs_run: int) -> dict:
    all_records = train_records + valid_records
    total_good_words = sum(count_word(item["good"]) for item in all_records)
    total_bad_words = sum(count_word(item["bad"]) for item in all_records)
    total_pair_words = total_good_words + total_bad_words
    return {
        "num_train_records": len(train_records),
        "num_valid_records": len(valid_records),
        "total_good_words": total_good_words,
        "total_bad_words": total_bad_words,
        "total_pair_words": total_pair_words,
        "training_datapoints_used": len(train_records) * num_epochs_run,
    }


def _install_config_save_compat_shim(config) -> None:
    config_cls = config.__class__
    to_json_file = getattr(config_cls, "to_json_file", None)
    if to_json_file is None:
        return

    try:
        signature = inspect.signature(to_json_file)
    except (TypeError, ValueError):
        return

    if "use_diff" in signature.parameters:
        return

    if getattr(config_cls, "_codex_to_json_file_compat", False):
        if "to_json_file" in config.__dict__:
            del config.__dict__["to_json_file"]
        return

    original_to_json_file = to_json_file

    def _wrapped_to_json_file(self, json_file_path, use_diff=True):
        return original_to_json_file(self, json_file_path)

    config_cls.to_json_file = _wrapped_to_json_file
    config_cls._codex_to_json_file_compat = True

    if "to_json_file" in config.__dict__:
        del config.__dict__["to_json_file"]


def _clone_shared_tensors_in_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned_state_dict = {}
    seen_storage = {}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            cloned_state_dict[name] = tensor
            continue

        storage_id = (tensor.device.type, tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.size()), tuple(tensor.stride()))
        if storage_id in seen_storage:
            cloned_state_dict[name] = tensor.clone()
        else:
            cloned_state_dict[name] = tensor
            seen_storage[storage_id] = name

    return cloned_state_dict


def save_checkpoint(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_config_save_compat_shim(model.config)
    state_dict = _clone_shared_tensors_in_state_dict(model.state_dict())
    model.save_pretrained(output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(output_dir)


def train(
    model,
    tokenizer,
    train_records: list[dict],
    valid_records: list[dict],
    output_dir: Path,
    device: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    grad_accum_steps: int,
    patience: int,
    eval_before_after: bool,
) -> dict:
    dataloader = DataLoader(
        GrammarPairDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    history = {
        "epoch_train_losses": [],
        "epoch_valid_losses": [],
        "epoch_train_accuracies": [],
        "epoch_valid_accuracies": [],
        "best_epoch": None,
        "best_valid_accuracy": None,
        "stopped_early": False,
        "epochs_completed": 0,
    }

    if eval_before_after:
        history["train_pair_accuracy_before"] = evaluate_pair_accuracy(
            model, tokenizer, train_records, batch_size=batch_size, device=device
        )
        history["valid_pair_accuracy_before"] = evaluate_pair_accuracy(
            model, tokenizer, valid_records, batch_size=batch_size, device=device
        )

    best_valid_accuracy = float("-inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader, start=1):
            good_scores = sentence_log_probabilities(model, tokenizer, batch["good_texts"], device)
            bad_scores = sentence_log_probabilities(model, tokenizer, batch["bad_texts"], device)

            pairwise_loss = -F.logsigmoid(good_scores - bad_scores).mean()
            loss = pairwise_loss / grad_accum_steps
            loss.backward()
            running_loss += pairwise_loss.item()

            if step % grad_accum_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        epoch_train_loss = running_loss / max(len(dataloader), 1)
        epoch_train_acc = evaluate_pair_accuracy(
            model, tokenizer, train_records, batch_size=batch_size, device=device
        )
        epoch_valid_loss = evaluate_pair_loss(
            model, tokenizer, valid_records, batch_size=batch_size, device=device
        )
        epoch_valid_acc = evaluate_pair_accuracy(
            model, tokenizer, valid_records, batch_size=batch_size, device=device
        )

        history["epoch_train_losses"].append(epoch_train_loss)
        history["epoch_valid_losses"].append(epoch_valid_loss)
        history["epoch_train_accuracies"].append(epoch_train_acc)
        history["epoch_valid_accuracies"].append(epoch_valid_acc)
        history["epochs_completed"] = epoch + 1

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"train_loss: {epoch_train_loss:.4f} - "
            f"valid_loss: {epoch_valid_loss:.4f} - "
            f"train_acc: {epoch_train_acc:.4f} - "
            f"valid_acc: {epoch_valid_acc:.4f}"
        )

        if epoch_valid_acc > best_valid_accuracy:
            best_valid_accuracy = epoch_valid_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
            history["best_valid_accuracy"] = epoch_valid_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                history["stopped_early"] = True
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_state_dict)
    save_checkpoint(model, tokenizer, output_dir)

    if eval_before_after:
        history["train_pair_accuracy_after"] = evaluate_pair_accuracy(
            model, tokenizer, train_records, batch_size=batch_size, device=device
        )
        history["valid_pair_accuracy_after"] = evaluate_pair_accuracy(
            model, tokenizer, valid_records, batch_size=batch_size, device=device
        )

    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model on grammaticality judgment pairs."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/train_ready_grammar_data.jsonl"),
        help="Training JSONL file produced by validate_generated_grammar_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the best fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Path to save the training report JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for pairwise training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop if validation accuracy does not improve for this many epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip before/after pair-accuracy evaluation.",
    )
    parser.add_argument(
        "--run-id",
        default="id",
        help="Suffix used in saved model/report names to distinguish runs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable Hugging Face trust_remote_code for custom architectures such as GPT-BERT baselines.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = build_model_run_name(args.model_name, args.run_id)
    if args.output_dir is None:
        args.output_dir = Path("artifacts/models") / run_name
    if args.report_file is None:
        args.report_file = Path("data/reports") / f"{run_name}_report.json"
    all_records = load_jsonl(args.train_file)
    if len(all_records) < 2:
        raise ValueError("Need at least 2 records to create train/validation splits.")

    train_records, valid_records = split_records(all_records, args.valid_ratio, args.seed)
    if not train_records:
        raise ValueError("Training split is empty. Reduce --valid-ratio.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device,
        trust_remote_code=args.trust_remote_code,
    )
    history = train(
        model=model,
        tokenizer=tokenizer,
        train_records=train_records,
        valid_records=valid_records,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        patience=args.patience,
        eval_before_after=not args.skip_eval,
    )

    training_stats = compute_training_stats(
        train_records=train_records,
        valid_records=valid_records,
        num_epochs_run=history["epochs_completed"],
    )

    report = {
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "output_dir": str(args.output_dir),
        "device": device,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs_requested": args.epochs,
        "grad_accum_steps": args.grad_accum_steps,
        "valid_ratio": args.valid_ratio,
        "patience": args.patience,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        **training_stats,
        **history,
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training finished.")
    print(f"Saved best model to: {args.output_dir}")
    print(f"Saved report to: {args.report_file}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(valid_records)}")
    print(f"Training datapoints used: {training_stats['training_datapoints_used']}")
    print(f"Total pair words: {training_stats['total_pair_words']}")


if __name__ == "__main__":
    main()




