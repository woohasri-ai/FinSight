
import numpy as np
import pytest

from finsight.retrieval.rerank import rerank_chunks


class FakeReranker:
    def predict(
        self,
        sentences: list[tuple[str, str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        score_map = {
            "TCS revenue increased": 0.9,
            "AI strategy": 0.3,
            "Unrelated passage": 0.1,
        }
        return np.array(
            [score_map.get(text, 0.5) for _, text in sentences],
            dtype=np.float32,
        )


class FakeTokenizer:
    model_max_length = 10

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for token in tokens:
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return " ".join(
            self._id_to_token[token_id] for token_id in token_ids
        )

    def num_special_tokens_to_add(self, pair: bool = True) -> int:
        return 2

    def __call__(
        self,
        query: str,
        passage: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        verbose: bool = False,
    ) -> dict[str, list[int]]:
        query_ids = self.convert_tokens_to_ids(self.tokenize(query))
        passage_ids = self.convert_tokens_to_ids(self.tokenize(passage))
        special_count = 2 if add_special_tokens else 0
        return {
            "input_ids": query_ids + passage_ids + [0] * special_count
        }


class WindowScoringReranker:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 10
        self.seen_passages: list[str] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.seen_passages = [passage for _, passage in sentences]
        scores = [
            0.9 if "ANSWER" in passage else 0.1
            for _, passage in sentences
        ]
        return np.array(scores, dtype=np.float32)


class FixedWindowScoringReranker(WindowScoringReranker):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.seen_passages = [passage for _, passage in sentences]
        scores = [0.9, 0.4, 0.1][:len(sentences)]
        return np.array(scores, dtype=np.float32)


@pytest.fixture
def sample_chunks() -> list[dict[str, int | float | str]]:
    return [
        {
            "chunk_id": "A",
            "text": "AI strategy",
            "source_doc": "report.pdf",
            "page_number": 20,
            "score": 0.016,
        },
        {
            "chunk_id": "B",
            "text": "TCS revenue increased",
            "source_doc": "report.pdf",
            "page_number": 10,
            "score": 0.015,
        },
    ]


def test_rerank_chunks_orders_by_relevance(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    results = rerank_chunks(
        "What is relevant?",
        sample_chunks,
        FakeReranker(),  # type: ignore[arg-type]
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["chunk_id"] == "B"
    assert results[0]["page_number"] == 10
    assert results[0]["score"] > results[1]["score"]


def test_rerank_chunks_rejects_blank_query(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        rerank_chunks(
            "   ",
            sample_chunks,
            FakeReranker(),  # type: ignore[arg-type]
        )


def test_rerank_chunks_rejects_invalid_top_k(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        rerank_chunks(
            "TCS revenue",
            sample_chunks,
            FakeReranker(),  # type: ignore[arg-type]
            top_k=0,
        )


def test_rerank_chunks_returns_empty_for_no_chunks() -> None:
    results = rerank_chunks(
        "TCS revenue",
        [],
        FakeReranker(),  # type: ignore[arg-type]
    )
    assert results == []


def test_rerank_chunks_limits_results_and_preserves_metadata(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    results = rerank_chunks(
        "What is relevant?",
        sample_chunks,
        FakeReranker(),  # type: ignore[arg-type]
        top_k=1,
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == "B"
    assert results[0]["source_doc"] == "report.pdf"
    assert results[0]["page_number"] == 10
    assert results[0]["text"] == "TCS revenue increased"


def test_rerank_chunks_rejects_mismatched_score_count(
    sample_chunks: list[dict[str, int | float | str]],
) -> None:
    class BrokenReranker:
        def predict(
            self,
            sentences: list[tuple[str, str]],
            show_progress_bar: bool = False,
        ) -> np.ndarray:
            return np.array([0.9], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Reranker must return one score per chunk",
    ):
        rerank_chunks(
            "What is relevant?",
            sample_chunks,
            BrokenReranker(),  # type: ignore[arg-type]
            top_k=2,
        )


def test_rerank_chunks_finds_answer_in_later_window() -> None:
    model = WindowScoringReranker()
    chunks = [
        {
            "chunk_id": "LONG",
            "text": (
                "alpha beta gamma delta epsilon "
                "ANSWER zeta eta theta"
            ),
            "source_doc": "report.pdf",
            "page_number": 120,
        }
    ]

    results = rerank_chunks(
        "question",
        chunks,
        model,  # type: ignore[arg-type]
        top_k=1,
    )

    assert len(model.seen_passages) > 1
    assert any("ANSWER" in passage for passage in model.seen_passages)
    assert results[0]["chunk_id"] == "LONG"
    assert results[0]["score"] > 0.1


def test_rerank_chunks_uses_maximum_window_score() -> None:
    model = FixedWindowScoringReranker()
    chunks = [
        {
            "chunk_id": "LONG",
            "text": (
                "alpha beta gamma delta epsilon "
                "zeta eta theta iota kappa"
            ),
            "source_doc": "report.pdf",
            "page_number": 120,
        }
    ]

    results = rerank_chunks(
        "question",
        chunks,
        model,  # type: ignore[arg-type]
        top_k=1,
    )

    assert len(model.seen_passages) == 3
    assert results[0]["score"] == pytest.approx(0.9)
