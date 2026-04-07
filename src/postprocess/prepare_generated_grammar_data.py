import argparse
import json
import hashlib
from pathlib import Path


SUPPORTED_SUFFIXES = {".json", ".jsonl"}


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def infer_metadata_from_filename(path: Path) -> dict:
    stem_parts = path.stem.split("_")
    metadata = {
        "parent_llm": "",
        "prompt_family": "",
        "phenomenon": "",
        "topic": "",
    }

    if not stem_parts:
        return metadata

    metadata["parent_llm"] = stem_parts[0]

    if len(stem_parts) >= 3 and stem_parts[1] == "prompt":
        metadata["prompt_family"] = f"{stem_parts[1]}_{stem_parts[2]}"
        remaining = stem_parts[3:]
    else:
        remaining = stem_parts[1:]

    if len(remaining) >= 2:
        phenomenon_parts = remaining[:-2]
        if phenomenon_parts:
            metadata["phenomenon"] = "_".join(phenomenon_parts)
        metadata["topic"] = remaining[-2]

    return metadata


def load_records_from_file(path: Path) -> list[dict]:
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
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


def transform_record(record: dict, path: Path, index: int) -> dict:
    inferred = infer_metadata_from_filename(path)
    phenomenon = normalize_text(record.get("phenomenon")) or inferred["phenomenon"]
    topic = normalize_text(record.get("topic")) or inferred["topic"]
    prompt_family = normalize_text(record.get("prompt_family")) or inferred["prompt_family"]
    parent_llm = normalize_text(record.get("parent_llm")) or inferred["parent_llm"]

    if not phenomenon:
        raise ValueError(f"Missing phenomenon for record {index} in {path}")
    if not topic:
        raise ValueError(f"Missing topic for record {index} in {path}")
    if not prompt_family:
        raise ValueError(f"Missing prompt_family for record {index} in {path}")
    if not parent_llm:
        raise ValueError(f"Missing parent_llm for record {index} in {path}")

    raw_id_source = " || ".join(
        [
            parent_llm,
            prompt_family,
            phenomenon,
            topic,
            normalize_text(record.get("good")),
            normalize_text(record.get("bad")),
            normalize_text(record.get("edit_type")),
        ]
    )
    short_hash = hashlib.md5(raw_id_source.encode("utf-8")).hexdigest()[:10]

    return {
        "id": f"{parent_llm}_{prompt_family}_{short_hash}",
        "phenomenon": phenomenon,
        "topic": topic,
        "good": normalize_text(record.get("good")),
        "bad": normalize_text(record.get("bad")),
        "edit_type": normalize_text(record.get("edit_type")),
        "prompt_family": prompt_family,
        "parent_llm": parent_llm,
    }


def collect_input_files(input_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(
                sorted(
                    p for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
        else:
            raise ValueError(f"Unsupported input path: {path}")
    return files


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge generated grammaticality data files into one normalized JSONL dataset."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing generated .json/.jsonl data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/final_generated_grammar_data.jsonl"),
        help="Path to the merged JSONL output file.",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args.inputs)
    normalized_records = []

    for path in input_files:
        raw_records = load_records_from_file(path)
        for index, record in enumerate(raw_records, start=1):
            normalized_records.append(transform_record(record, path, index))

    write_jsonl(normalized_records, args.output)
    print(f"Merged {len(normalized_records)} records from {len(input_files)} files into {args.output}")


if __name__ == "__main__":
    main()

