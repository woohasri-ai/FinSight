
from collections.abc import Sequence
import time

import torch
from sentence_transformers import CrossEncoder
from tokenizers import Tokenizer


RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_MAX_LENGTH = 512
WINDOW_OVERLAP = 64
TOKEN_SAFETY_MARGIN = 20

CPU_THREADS = 8
INTEROP_THREADS = 2


def load_reranker(
    model_name: str = RERANKER_MODEL_NAME,
) -> CrossEncoder:
    """Load the cross-encoder with controlled CPU thread settings."""
    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(INTEROP_THREADS)
    except RuntimeError:
        # Inter-op threads cannot be changed after parallel work starts.
        pass

    print(
        f"Reranker CPU threads | intra-op: {torch.get_num_threads()} | "
        f"inter-op: {torch.get_num_interop_threads()}",
        flush=True,
    )

    return CrossEncoder(model_name, device="cpu")


def _clone_backend_tokenizer(tokenizer: object) -> Tokenizer | None:
    """Create an independent backend tokenizer without truncation or padding."""
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)

    if backend_tokenizer is None:
        return None

    isolated_tokenizer = Tokenizer.from_str(backend_tokenizer.to_str())
    isolated_tokenizer.no_truncation()
    isolated_tokenizer.no_padding()

    return isolated_tokenizer


def _tokenize(
    tokenizer: object,
    text: str,
    backend_tokenizer: Tokenizer | None = None,
) -> list[int]:
    """Tokenize text independently of persistent inference settings."""
    if backend_tokenizer is not None:
        encoded = backend_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return list(encoded.ids)

    tokenize = getattr(tokenizer, "tokenize", None)
    convert_tokens_to_ids = getattr(
        tokenizer,
        "convert_tokens_to_ids",
        None,
    )

    if callable(tokenize) and callable(convert_tokens_to_ids):
        tokens = tokenize(text)
        token_ids = convert_tokens_to_ids(tokens)

        if isinstance(token_ids, int):
            return [token_ids]

        return list(token_ids)

    return list(tokenizer.encode(text, add_special_tokens=False))


def _pair_input_length(
    tokenizer: object,
    query: str,
    passage: str,
    backend_tokenizer: Tokenizer | None = None,
) -> int:
    """Measure encoded length without mutating model tokenizer state."""
    if backend_tokenizer is not None:
        encoded = backend_tokenizer.encode(
            query,
            passage,
            add_special_tokens=True,
        )
        return len(encoded.ids)

    if not callable(tokenizer):
        raise TypeError("Tokenizer must support pair encoding")

    encoded = tokenizer(
        query,
        passage,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )

    return len(encoded["input_ids"])


def _build_text_windows(
    query: str,
    text: str,
    model: CrossEncoder,
    backend_tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Split long text into overlapping windows within the model limit."""
    tokenizer = getattr(model, "tokenizer", None)

    if tokenizer is None:
        return [text]

    if backend_tokenizer is None:
        backend_tokenizer = _clone_backend_tokenizer(tokenizer)

    max_length = (
        getattr(model, "max_seq_length", None)
        or getattr(tokenizer, "model_max_length", None)
        or DEFAULT_MAX_LENGTH
    )

    query_ids = _tokenize(tokenizer, query, backend_tokenizer)
    token_ids = _tokenize(tokenizer, text, backend_tokenizer)

    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    safety_margin = min(TOKEN_SAFETY_MARGIN, max_length // 5)

    window_size = (
        max_length
        - len(query_ids)
        - special_tokens
        - safety_margin
    )

    if window_size <= 0:
        raise ValueError(
            "Query is too long for the reranker model's maximum input length"
        )

    if len(token_ids) <= window_size:
        if (
            _pair_input_length(
                tokenizer,
                query,
                text,
                backend_tokenizer,
            )
            <= max_length
        ):
            return [text]

    overlap = min(WINDOW_OVERLAP, window_size // 4)
    step = max(1, window_size - overlap)

    windows: list[str] = []

    for start in range(0, len(token_ids), step):
        window_ids = token_ids[start : start + window_size]

        while window_ids:
            window = tokenizer.decode(
                window_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            if (
                _pair_input_length(
                    tokenizer,
                    query,
                    window,
                    backend_tokenizer,
                )
                <= max_length
            ):
                break

            window_ids = window_ids[:-1]

        if not window_ids:
            raise ValueError(
                "Unable to create a passage window within the reranker limit"
            )

        windows.append(window)

        if start + window_size >= len(token_ids):
            break

    return windows or [text]


def _aggregate_window_scores(
    scores: Sequence[float],
    indices: Sequence[int],
) -> float:
    """Use the highest window score as the chunk's relevance score."""
    if not indices:
        raise ValueError("A chunk must have at least one window score")

    return max(float(scores[index]) for index in indices)


def rerank_chunks(
    query: str,
    chunks: Sequence[dict[str, int | float | str]],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict[str, int | float | str]]:
    """Rerank chunks using their highest-scoring text window."""
    if not query.strip():
        raise ValueError("query must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    total_started = time.perf_counter()

    tokenizer = getattr(model, "tokenizer", None)
    backend_tokenizer = (
        _clone_backend_tokenizer(tokenizer)
        if tokenizer is not None
        else None
    )

    pairs: list[tuple[str, str]] = []
    chunk_window_indices: list[list[int]] = []

    window_started = time.perf_counter()

    for chunk in chunks:
        windows = _build_text_windows(
            query=query,
            text=str(chunk["text"]),
            model=model,
            backend_tokenizer=backend_tokenizer,
        )

        indices: list[int] = []

        for window in windows:
            indices.append(len(pairs))
            pairs.append((query, window))

        chunk_window_indices.append(indices)

    window_elapsed = time.perf_counter() - window_started

    prediction_started = time.perf_counter()

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    prediction_elapsed = time.perf_counter() - prediction_started

    if len(scores) != len(pairs):
        raise ValueError("Reranker must return one score per chunk")

    aggregation_started = time.perf_counter()

    chunk_scores = [
        _aggregate_window_scores(scores, indices)
        for indices in chunk_window_indices
    ]

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: chunk_scores[index],
        reverse=True,
    )[:top_k]

    results: list[dict[str, int | float | str]] = []

    for index in ranked_indices:
        result = dict(chunks[index])
        result["score"] = chunk_scores[index]
        results.append(result)

    aggregation_elapsed = time.perf_counter() - aggregation_started

    print(
        "Reranker breakdown | "
        f"window preparation: {window_elapsed:.3f}s | "
        f"model prediction: {prediction_elapsed:.3f}s | "
        f"aggregation: {aggregation_elapsed:.3f}s | "
        f"chunks: {len(chunks)} | "
        f"windows: {len(pairs)} | "
        f"total: {time.perf_counter() - total_started:.3f}s",
        flush=True,
    )

    return results
