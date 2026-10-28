import json
from pathlib import Path


def save_pages_as_json(
    pages: list[dict[str, int | str]],
    output_path: Path,
) -> None:
    """Save extracted PDF pages to a UTF-8 JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)
