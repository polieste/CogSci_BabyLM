import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import httpx
from openai import OpenAI
from generation_config import (
    DEFAULT_CONFIG_PATH,
    build_prompt,
    load_generation_config,
    parse_topics_arg,
)


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_client() -> OpenAI:
    load_env_file()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set in the environment or .env.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=httpx.Timeout(3600.0),
    )


def parse_jsonl_output(text: str) -> list[dict]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = "".join(part for part in parts if "json" not in part.lower()).strip()

    items = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))

    return items


def request_items(client: OpenAI, model: str, prompt: str) -> list[dict]:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        store=False,
    )

    items = parse_jsonl_output(response.output_text)
    return items


def write_jsonl(items: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_safe_filename_part(value: str) -> str:
    safe = value.replace(",", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in safe).strip("_")


def build_default_output_path(provider: str, prompt_id: str, phenomenon: str, topics: list[str]) -> Path:
    topic_label = topics[0] if len(topics) == 1 else "mixed_topics"
    timestamp = datetime.now().strftime("%H%M")
    filename = (
        f"{make_safe_filename_part(provider)}_"
        f"{make_safe_filename_part(prompt_id)}_"
        f"{make_safe_filename_part(phenomenon)}_"
        f"{make_safe_filename_part(topic_label)}_"
        f"{timestamp}.json"
    )
    return Path(filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grammaticality judgment minimal pairs with Grok."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the prompt/topic configuration JSON file.",
    )
    parser.add_argument(
        "--model",
        default="grok-4.20-0309-non-reasoning",
        help="xAI model name to use.",
    )
    parser.add_argument(
        "--phenomenon",
        default="anaphor_agreement",
        help="Target grammatical phenomenon to generate.",
    )
    parser.add_argument(
        "--prompt-id",
        default="prompt_1",
        help="Prompt template id from prompt_topic_config.json.",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Optional comma-separated topic list to override the config defaults.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of items to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the JSONL output file.",
    )
    args = parser.parse_args()

    config = load_generation_config(args.config)
    selected_topics = parse_topics_arg(args.topics)
    prompt, allowed_topics = build_prompt(
        config=config,
        prompt_id=args.prompt_id,
        phenomenon=args.phenomenon,
        count=args.count,
        selected_topics=selected_topics,
    )

    client = build_client()
    output_path = args.output or build_default_output_path(
        provider="grok",
        prompt_id=args.prompt_id,
        phenomenon=args.phenomenon,
        topics=allowed_topics,
    )
    items = request_items(client, args.model, prompt)
    write_jsonl(items, output_path)

    print(f"Prompt id: {args.prompt_id}")
    print(f"Allowed topics: {', '.join(allowed_topics)}")
    print(f"Wrote {len(items)} items to {output_path}")


if __name__ == "__main__":
    main()
