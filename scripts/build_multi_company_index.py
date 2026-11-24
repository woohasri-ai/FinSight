import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from finsight.ingestion.chunk import chunk_page_text
from finsight.ingestion.parse import extract_pdf_pages
from finsight.ingestion.save import save_pages_as_json
from finsight.retrieval.dense import embed_texts, load_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_MASTER_PATH = (
    PROJECT_ROOT / "data" / "reference" / "company_master.json"
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multi_company"

REPORT_STEMS = {
    "HCLTECH": "hcltech",
    "INFY": "infosys",
    "LTM": "ltm",
    "TCS": "tcs",
    "TECHM": "techmahindra",
    "WIPRO": "wipro",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def load_tier_one_companies() -> list[dict[str, str]]:
    """Load the confirmed IT companies from the official company master."""
    if not COMPANY_MASTER_PATH.is_file():
        raise FileNotFoundError(
            f"Company master not found: {COMPANY_MASTER_PATH}"
        )

    companies = json.loads(
        COMPANY_MASTER_PATH.read_text(encoding="utf-8")
    )

    selected = [
        company
        for company in companies
        if company["industry"] == "Information Technology"
    ]

    if not selected:
        raise ValueError("No Information Technology companies found.")

    symbols = {company["symbol"] for company in selected}

    if symbols != set(REPORT_STEMS):
        raise ValueError(
            "The IT company list does not match the confirmed report "
            f"configuration. Found: {sorted(symbols)}"
        )

    return selected


def build_company_chunks(
    company: dict[str, str],
    tokenizer: Tokenizer,
) -> list[dict]:
    """Parse one company report and create source-aware chunks."""
    symbol = company["symbol"]
    stem = REPORT_STEMS[symbol]
    pdf_name = f"{stem}_annual_report_2025_26.pdf"
    pdf_path = RAW_DIR / pdf_name

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Annual report not found: {pdf_path}")

    print(f"\nProcessing {company['company_name']} ({symbol})...")

    pages = extract_pdf_pages(pdf_path)

    if not pages:
        raise ValueError(f"No extractable text found in {pdf_name}")

    pages_path = OUTPUT_DIR / f"{stem}_pages.json"
    save_pages_as_json(pages, pages_path)

    company_chunks = []

    for page in pages:
        page_chunks = chunk_page_text(
            text=str(page["text"]),
            page_number=int(page["page_number"]),
            source_doc=pdf_name,
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk in page_chunks:
            chunk["company_name"] = company["company_name"]
            chunk["symbol"] = symbol
            chunk["industry"] = company["industry"]

        company_chunks.extend(page_chunks)

    if not company_chunks:
        raise ValueError(f"No chunks generated for {symbol}")

    chunks_path = OUTPUT_DIR / f"{stem}_chunks.json"
    chunks_path.write_text(
        json.dumps(company_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(company_chunks)}")

    return company_chunks


def main() -> None:
    """Build the combined six-company retrieval index."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    companies = load_tier_one_companies()

    print(f"Companies to process: {len(companies)}")
    print("Loading BGE tokenizer...")

    tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    all_chunks = []

    for company in companies:
        company_chunks = build_company_chunks(company, tokenizer)
        all_chunks.extend(company_chunks)

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    combined_chunks_path = OUTPUT_DIR / "tier1_combined_chunks.json"
    combined_embeddings_path = OUTPUT_DIR / "tier1_combined_embeddings.npy"

    combined_chunks_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal combined chunks: {len(all_chunks)}")
    print("Loading BGE embedding model...")

    model = load_embedding_model()
    texts = [str(chunk["text"]) for chunk in all_chunks]

    print("Generating combined embeddings...")
    embeddings = embed_texts(texts, model)

    if embeddings.shape[0] != len(all_chunks):
        raise ValueError(
            "Embedding count does not match combined chunk count."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")

    np.save(combined_embeddings_path, embeddings)

    print("\nMulti-company index completed.")
    print(f"Companies processed: {len(companies)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Combined chunks: {combined_chunks_path}")
    print(f"Combined embeddings: {combined_embeddings_path}")


if __name__ == "__main__":
    main()
