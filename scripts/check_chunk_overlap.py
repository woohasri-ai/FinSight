import json
from pathlib import Path

from tokenizers import Tokenizer


def main() -> None:
    """Validate chunk sizes and source-character overlap within each page."""
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    chunks_path = processed_dir / "tcs_annual_report_2025_26_chunks.json"
    pages_path = processed_dir / "tcs_annual_report_2025_26.json"

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk JSON not found: {chunks_path}")
    if not pages_path.is_file():
        raise FileNotFoundError(f"Extracted pages JSON not found: {pages_path}")

    chunks: list[dict[str, int | str]] = json.loads(chunks_path.read_text(encoding="utf-8"))
    pages: list[dict[str, int | str]] = json.loads(pages_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    source_text = {int(page["page_number"]): str(page["text"]) for page in pages}

    grouped: dict[int, list[dict[str, int | str]]] = {}
    for chunk in chunks:
        grouped.setdefault(int(chunk["page_number"]), []).append(chunk)

    token_overlaps: list[int] = []
    character_overlaps: list[int] = []
    continuity_failures = 0
    size_failures = 0
    metadata_failures = 0
    first_problem: str | None = None

    for page_number, page_chunks in grouped.items():
        page_text = source_text.get(page_number)

        if page_text is None:
            continuity_failures += len(page_chunks)
            first_problem = first_problem or (
                f"Page {page_number} is missing from extracted page JSON."
            )
            continue

        for chunk in page_chunks:
            chunk_text = str(chunk["text"])
            start = chunk.get("start_char")
            end = chunk.get("end_char")

            token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False).ids)

            if token_count > 500:
                size_failures += 1

            if not isinstance(start, int) or not isinstance(end, int):
                metadata_failures += 1
                first_problem = first_problem or (f"Missing character offsets: {chunk['chunk_id']}")
                continue

            if (
                start < 0
                or end < start
                or end > len(page_text)
                or page_text[start:end] != chunk_text
            ):
                continuity_failures += 1
                first_problem = first_problem or (
                    f"Chunk text does not match its source span: {chunk['chunk_id']}"
                )

        for first, second in zip(page_chunks, page_chunks[1:], strict=False):
            first_ids = tokenizer.encode(str(first["text"]), add_special_tokens=False).ids
            second_ids = tokenizer.encode(str(second["text"]), add_special_tokens=False).ids

            max_overlap = min(80, len(first_ids), len(second_ids))
            overlap = 0

            for size in range(max_overlap, 0, -1):
                if first_ids[-size:] == second_ids[:size]:
                    overlap = size
                    break

            token_overlaps.append(overlap)

            first_end = first.get("end_char")
            second_start = second.get("start_char")

            if isinstance(first_end, int) and isinstance(second_start, int):
                character_overlap = max(0, first_end - second_start)
                character_overlaps.append(character_overlap)

                if second_start >= first_end:
                    first_problem = first_problem or (
                        f"No character overlap between {first['chunk_id']} "
                        f"and {second['chunk_id']}."
                    )
            else:
                metadata_failures += 1

    print(f"Chunks checked: {len(chunks)}")
    print(f"Adjacent chunk pairs checked: {len(token_overlaps)}")
    print(f"Pairs with at least 80 matching tokens: {sum(n >= 80 for n in token_overlaps)}")
    print(f"Pairs with zero matching tokens: {sum(n == 0 for n in token_overlaps)}")
    print(
        f"Average matching-token overlap: {sum(token_overlaps) / len(token_overlaps):.1f}"
        if token_overlaps
        else "Average matching-token overlap: N/A"
    )
    print(f"Pairs with positive character overlap: {sum(n > 0 for n in character_overlaps)}")
    print(
        f"Minimum character overlap: {min(character_overlaps)}"
        if character_overlaps
        else "Minimum character overlap: N/A"
    )
    print(f"Chunks exceeding 500 tokens: {size_failures}")
    print(f"Invalid source spans: {continuity_failures}")
    print(f"Missing offset metadata: {metadata_failures}")

    if first_problem:
        print(f"First issue found: {first_problem}")


if __name__ == "__main__":
    main()
