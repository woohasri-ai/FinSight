from tokenizers import Tokenizer


def chunk_page_text(
    text: str,
    page_number: int,
    source_doc: str,
    tokenizer: Tokenizer,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict[str, int | str]]:
    """Split page text into bounded chunks with source character offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text.strip():
        return []

    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_spans = [(start, end) for start, end in encoding.offsets if end > start]

    if not token_spans:
        return []

    chunks: list[dict[str, int | str]] = []
    start_token = 0

    while start_token < len(token_spans):
        end_token = min(start_token + chunk_size, len(token_spans))
        chunk_text = ""
        char_start = token_spans[start_token][0]
        char_end = token_spans[end_token - 1][1]

        # Reduce the end boundary until the saved text fits the token limit.
        while end_token > start_token:
            char_end = token_spans[end_token - 1][1]
            candidate = text[char_start:char_end]

            token_count = len(tokenizer.encode(candidate, add_special_tokens=False).ids)

            if token_count <= chunk_size:
                chunk_text = candidate
                break

            end_token -= 1

        if end_token <= start_token or not chunk_text:
            raise ValueError("Unable to create a chunk within the token limit")

        chunks.append(
            {
                "chunk_id": (f"{source_doc}:page-{page_number}:chunk-{len(chunks) + 1}"),
                "text": chunk_text,
                "source_doc": source_doc,
                "page_number": page_number,
                "start_char": char_start,
                "end_char": char_end,
            }
        )

        if end_token >= len(token_spans):
            break

        next_start = max(start_token + 1, end_token - overlap)

        if next_start <= start_token:
            raise ValueError("Chunking failed to advance")

        start_token = next_start

    return chunks
