import argparse
import json
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize


SUPPORTED_SUFFIXES = {".json", ".jsonl"}
TEXT_COLUMNS = ["good", "bad"]


def ensure_nltk_tokenizers() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


def count_word(text: str) -> int:
    return len(word_tokenize(text))


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


def collect_input_files(input_root: Path) -> list[Path]:
    return sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def build_report_for_file(path: Path) -> dict:
    records = load_records(path)
    df = pd.DataFrame(records)

    for column in TEXT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    if not df.empty:
        df["good"] = df["good"].fillna("").astype(str)
        df["bad"] = df["bad"].fillna("").astype(str)
        df["good_word_count"] = df["good"].map(count_word)
        df["bad_word_count"] = df["bad"].map(count_word)
        df["pair_word_count"] = df["good_word_count"] + df["bad_word_count"]
        df["word_count_gap"] = (df["good_word_count"] - df["bad_word_count"]).abs()
    else:
        df["good_word_count"] = pd.Series(dtype="int64")
        df["bad_word_count"] = pd.Series(dtype="int64")
        df["pair_word_count"] = pd.Series(dtype="int64")
        df["word_count_gap"] = pd.Series(dtype="int64")

    report = {
        "file_name": path.name,
        "file_path": str(path),
        "num_records": int(len(df)),
        "total_good_words": int(df["good_word_count"].sum()) if not df.empty else 0,
        "total_bad_words": int(df["bad_word_count"].sum()) if not df.empty else 0,
        "total_pair_words": int(df["pair_word_count"].sum()) if not df.empty else 0,
        "avg_good_words": float(df["good_word_count"].mean()) if not df.empty else 0.0,
        "avg_bad_words": float(df["bad_word_count"].mean()) if not df.empty else 0.0,
    }
    return report


def write_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EDA for all JSON/JSONL files in data/processed and save reports."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed JSON/JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/reports/processed_eda"),
        help="Directory where EDA reports will be saved.",
    )
    args = parser.parse_args()

    ensure_nltk_tokenizers()
    input_files = collect_input_files(args.input_root)
    if not input_files:
        raise ValueError(f"No JSON/JSONL files found in {args.input_root}")

    per_file_reports = []
    summary_rows = []

    for path in input_files:
        report = build_report_for_file(path)
        per_file_reports.append(report)

        relative_name = path.relative_to(args.input_root)
        report_path = args.output_root / relative_name.with_suffix(".report.json")
        write_json(report, report_path)

        summary_rows.append(
            {
                "file_name": report["file_name"],
                "relative_path": str(relative_name),
                "num_records": report["num_records"],
                "total_good_words": report["total_good_words"],
                "total_bad_words": report["total_bad_words"],
                "total_pair_words": report["total_pair_words"],
                "avg_good_words": report["avg_good_words"],
                "avg_bad_words": report["avg_bad_words"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["relative_path"])
    summary_csv_path = args.output_root / "processed_eda_summary.csv"
    summary_json_path = args.output_root / "processed_eda_summary.json"
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)

    write_json(
        {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "num_files": len(per_file_reports),
            "total_records": int(sum(report["num_records"] for report in per_file_reports)),
            "files": per_file_reports,
        },
        summary_json_path,
    )

    print(f"Processed {len(per_file_reports)} files from {args.input_root}")
    print(f"Saved per-file reports under: {args.output_root}")
    print(f"Saved summary CSV to: {summary_csv_path}")
    print(f"Saved summary JSON to: {summary_json_path}")


if __name__ == "__main__":
    main()
