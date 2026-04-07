import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = [
    "id",
    "phenomenon",
    "topic",
    "good",
    "bad",
    "edit_type",
    "prompt_family",
    "parent_llm",
]


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


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_record(record: dict) -> list[str]:
    issues = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            issues.append(f"missing_field:{field}")
        elif not normalize_text(record[field]):
            issues.append(f"empty_field:{field}")

    if issues:
        return issues

    if normalize_text(record["good"]) == normalize_text(record["bad"]):
        issues.append("good_equals_bad")

    if "\n" in normalize_text(record["good"]) or "\n" in normalize_text(record["bad"]):
        issues.append("multiline_sentence")

    if normalize_text(record["phenomenon"]).startswith("{") or normalize_text(record["topic"]).startswith("{"):
        issues.append("malformed_metadata")

    return issues


def deduplicate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    kept = []
    removed = []

    for record in records:
        key = (
            normalize_text(record["phenomenon"]).lower(),
            normalize_text(record["topic"]).lower(),
            normalize_text(record["good"]).lower(),
            normalize_text(record["bad"]).lower(),
            normalize_text(record["edit_type"]).lower(),
        )
        if key in seen:
            removed.append(record)
            continue
        seen.add(key)
        kept.append(record)

    return kept, removed


def build_issue_record(record: dict, issues: list[str]) -> dict:
    return {
        "issues": issues,
        "record": record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, clean, and deduplicate generated grammaticality data."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Merged input JSONL file from prepare_generated_grammar_data.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train_ready_grammar_data.jsonl"),
        help="Path to the cleaned training-ready JSONL file.",
    )
    parser.add_argument(
        "--invalid-output",
        type=Path,
        default=Path("data/processed/invalid_generated_grammar_data.jsonl"),
        help="Path to save invalid records with issue labels.",
    )
    parser.add_argument(
        "--duplicates-output",
        type=Path,
        default=Path("data/processed/duplicate_generated_grammar_data.jsonl"),
        help="Path to save removed duplicate records.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/reports/generated_grammar_data_report.json"),
        help="Path to save a validation summary report.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)

    valid_records = []
    invalid_records = []

    for record in records:
        issues = validate_record(record)
        if issues:
            invalid_records.append(build_issue_record(record, issues))
        else:
            valid_records.append(record)

    deduped_records, duplicate_records = deduplicate_records(valid_records)

    write_jsonl(deduped_records, args.output)
    write_jsonl(invalid_records, args.invalid_output)
    write_jsonl(duplicate_records, args.duplicates_output)

    issue_counter = Counter()
    for item in invalid_records:
        issue_counter.update(item["issues"])

    report = {
        "input_file": str(args.input),
        "total_records": len(records),
        "valid_before_dedup": len(valid_records),
        "invalid_records": len(invalid_records),
        "duplicates_removed": len(duplicate_records),
        "final_train_ready_records": len(deduped_records),
        "invalid_issue_breakdown": dict(issue_counter),
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Total records: {len(records)}")
    print(f"Invalid records: {len(invalid_records)}")
    print(f"Duplicates removed: {len(duplicate_records)}")
    print(f"Train-ready records: {len(deduped_records)}")
    print(f"Saved cleaned data to: {args.output}")
    print(f"Saved invalid records to: {args.invalid_output}")
    print(f"Saved duplicates to: {args.duplicates_output}")
    print(f"Saved report to: {args.report_output}")


if __name__ == "__main__":
    main()




