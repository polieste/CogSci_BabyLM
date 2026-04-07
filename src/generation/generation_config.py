import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "data" / "topics" / "prompt_topic_config.json"


def load_generation_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_phenomena(config: dict) -> list[str]:
    return sorted(card["name"] for card in config["phenomenon_cards"])


def get_prompt_ids(config: dict) -> list[str]:
    return sorted(config["prompts"].keys())


def get_all_topics(config: dict) -> list[str]:
    return list(config["topics"]["all_topics_pool"])


def get_phenomenon_card(config: dict, phenomenon: str) -> dict:
    for card in config["phenomenon_cards"]:
        if card["name"] == phenomenon:
            return card
    raise ValueError(f"Unknown phenomenon: {phenomenon}")


def validate_prompt_id(config: dict, prompt_id: str) -> None:
    if prompt_id not in config["prompts"]:
        raise ValueError(
            f"Unknown prompt_id '{prompt_id}'. Available prompt ids: {', '.join(get_prompt_ids(config))}"
        )


def validate_phenomenon(config: dict, phenomenon: str) -> None:
    if phenomenon not in get_phenomena(config):
        raise ValueError(
            f"Unknown phenomenon '{phenomenon}'. Available phenomena: {', '.join(get_phenomena(config))}"
        )


def parse_topics_arg(topics_arg: str | None) -> list[str] | None:
    if not topics_arg:
        return None
    return [topic.strip() for topic in topics_arg.split(",") if topic.strip()]


def resolve_allowed_topics(config: dict, prompt_id: str, phenomenon: str, selected_topics: list[str] | None) -> list[str]:
    all_topics = set(get_all_topics(config))
    if selected_topics:
        invalid = [topic for topic in selected_topics if topic not in all_topics]
        if invalid:
            raise ValueError(
                f"Unknown topic(s): {', '.join(invalid)}. Available topics: {', '.join(get_all_topics(config))}"
            )
        return selected_topics

    prompt_cfg = config["prompts"][prompt_id]
    topic_source = prompt_cfg.get("topic_source")

    if topic_source == "all_topics_pool":
        return get_all_topics(config)

    if topic_source == "recommended_pairings_for_each_phenomenon":
        pairings = config["topics"].get("recommended_pairings", {})
        if phenomenon in pairings:
            return list(pairings[phenomenon])

        card = get_phenomenon_card(config, phenomenon)
        return list(card.get("good_topics", []))

    if topic_source in config["topics"]:
        source_value = config["topics"][topic_source]
        if isinstance(source_value, list):
            return list(source_value)

    return get_all_topics(config)


def render_topic_list(topics: list[str]) -> str:
    return "\n".join(f"- {topic}" for topic in topics)


def build_prompt(config: dict, prompt_id: str, phenomenon: str, count: int, selected_topics: list[str] | None = None) -> tuple[str, list[str]]:
    validate_prompt_id(config, prompt_id)
    validate_phenomenon(config, phenomenon)

    allowed_topics = resolve_allowed_topics(config, prompt_id, phenomenon, selected_topics)
    prompt_template = config["prompts"][prompt_id]["template"]["prompt"]
    phenomenon_card = get_phenomenon_card(config, phenomenon)

    prompt = (
        prompt_template.replace("{{N}}", str(count))
        .replace("{{PHENOMENA}}", phenomenon)
        .replace("{{TOPIC_LIST}}", render_topic_list(allowed_topics))
        .replace("{{PHENOMENON_CARD}}", json.dumps(phenomenon_card, indent=2))
    )
    return prompt, allowed_topics


def get_response_item_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "phenomenon": {"type": "string"},
            "topic": {"type": "string"},
            "good": {"type": "string"},
            "bad": {"type": "string"},
            "edit_type": {"type": "string"},
        },
        "required": ["phenomenon", "topic", "good", "bad", "edit_type"],
        "additionalProperties": False,
    }

