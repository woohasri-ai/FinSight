import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from finsight.ingestion.chunk import chunk_page_text


def make_test_tokenizer(words: list[str] | None = None) -> Tokenizer:
    """Create a deterministic tokenizer without network access."""
    vocabulary = {"[UNK]": 0}

    if words:
        vocabulary.update({word: index + 1 for index, word in enumerate(words)})

    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_chunk_page_text_preserves_source_and_page():
    tokenizer = make_test_tokenizer()
    chunks = chunk_page_text(
        text="TCS reported strong revenue growth.",
        page_number=12,
        source_doc="tcs_annual_report_2025_26.pdf",
        tokenizer=tokenizer,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) >= 1
    assert chunks[0]["page_number"] == 12
    assert chunks[0]["source_doc"] == "tcs_annual_report_2025_26.pdf"
    assert chunks[0]["chunk_id"].startswith("tcs_annual_report_2025_26.pdf:page-12:")


def test_chunk_page_text_returns_empty_for_blank_text():
    chunks = chunk_page_text(
        text="   ",
        page_number=1,
        source_doc="sample.pdf",
        tokenizer=make_test_tokenizer(),
    )

    assert chunks == []


def test_chunk_page_text_rejects_invalid_chunk_size():
    tokenizer = make_test_tokenizer()

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_page_text(
            text="Some report text.",
            page_number=1,
            source_doc="sample.pdf",
            tokenizer=tokenizer,
            chunk_size=0,
        )


def test_chunk_page_text_rejects_invalid_overlap():
    tokenizer = make_test_tokenizer()

    with pytest.raises(ValueError, match="overlap"):
        chunk_page_text(
            text="Some report text.",
            page_number=1,
            source_doc="sample.pdf",
            tokenizer=tokenizer,
            chunk_size=4,
            overlap=4,
        )


def test_chunks_respect_size_and_overlap():
    words = [f"word{i}" for i in range(10)]
    tokenizer = make_test_tokenizer(words)
    text = " ".join(words)

    chunks = chunk_page_text(
        text=text,
        page_number=1,
        source_doc="sample.pdf",
        tokenizer=tokenizer,
        chunk_size=4,
        overlap=1,
    )

    token_chunks = [tokenizer.encode(str(chunk["text"])).tokens for chunk in chunks]

    assert len(token_chunks) == 3
    assert all(len(chunk) <= 4 for chunk in token_chunks)

    for previous, current in zip(token_chunks, token_chunks[1:], strict=False):
        assert previous[-1:] == current[:1]


def test_chunk_page_text_preserves_original_text():
    text = "TCS Report: Revenue was ₹100 Crore in FY 2025-26!"
    tokenizer = make_test_tokenizer(
        [
            "TCS",
            "Report",
            "Revenue",
            "was",
            "₹100",
            "Crore",
            "in",
            "FY",
            "2025",
            "26",
        ]
    )

    chunks = chunk_page_text(
        text=text,
        page_number=7,
        source_doc="sample.pdf",
        tokenizer=tokenizer,
        chunk_size=20,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == text
