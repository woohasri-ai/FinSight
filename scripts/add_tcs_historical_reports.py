import json
import re
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from finsight.ingestion.chunk import chunk_page_text
from finsight.ingestion.parse import extract_pdf_pages
from finsight.retrieval.dense import embed_texts, load_embedding_model

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "multi_company"

CHUNKS_PATH = OUTPUT / "tier1_combined_chunks.json"
EMBEDDINGS_PATH = OUTPUT / "tier1_combined_embeddings.npy"

HISTORICAL_REPORTS = [
    ("tcs_annual_report_2023_24.pdf", "2023-24"),
    ("tcs_annual_report_2024_25.pdf", "2024-25"),
]

def fiscal_year_from_source(source_doc: str) -> str:
    match = re.search(r"annual_report_(20\d{2})_(\d{2})", source_doc)
    if not match:
        raise ValueError(f"Cannot determine fiscal year: {source_doc}")
    return f"{match.group(1)}-{match.group(2)}"


def main():
    if not CHUNKS_PATH.is_file() or not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError("Existing combined index files are missing.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Existing chunk and embedding counts do not match.")

    for chunk in chunks:
        chunk["fiscal_year"] = fiscal_year_from_source(
            str(chunk["source_doc"])
        )

    existing_ids = {chunk["chunk_id"] for chunk in chunks}
    new_chunks = []

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    for filename, fiscal_year in HISTORICAL_REPORTS:
        pdf_path = RAW / filename

        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing report: {pdf_path}")

        print(f"Processing {filename}...")
        pages = extract_pdf_pages(pdf_path)

        if not pages:
            raise ValueError(f"No extractable text: {filename}")

        stem = filename.removesuffix(".pdf")

        (OUTPUT / f"{stem}_pages.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_chunks = []

        for page in pages:
            page_chunks = chunk_page_text(
                text=str(page["text"]),
                page_number=int(page["page_number"]),
                source_doc=filename,
                tokenizer=tokenizer,
                chunk_size=500,
                overlap=80,
            )

            for chunk in page_chunks:
                chunk.update({
                    "company_name": "Tata Consultancy Services Ltd.",
                    "symbol": "TCS",
                    "industry": "Information Technology",
                    "fiscal_year": fiscal_year,
                })

            report_chunks.extend(page_chunks)

        if not report_chunks:
            raise ValueError(f"No chunks generated: {filename}")

        duplicate_ids = {
            chunk["chunk_id"] for chunk in report_chunks
        } & existing_ids

        if duplicate_ids:
            raise ValueError(f"Duplicate chunk IDs: {filename}")

        existing_ids.update(chunk["chunk_id"] for chunk in report_chunks)

        (OUTPUT / f"{stem}_chunks.json").write_text(
            json.dumps(report_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        new_chunks.extend(report_chunks)
        print(f"Pages: {len(pages)} | New chunks: {len(report_chunks)}")

    print(f"Generating embeddings for {len(new_chunks)} historical chunks...")
    model = load_embedding_model()
    new_embeddings = embed_texts(
        [str(chunk["text"]) for chunk in new_chunks],
        model,
    )

    if new_embeddings.shape[1] != embeddings.shape[1]:
        raise ValueError("Historical embedding dimensions do not match.")

    combined_chunks = chunks + new_chunks
    combined_embeddings = np.vstack([embeddings, new_embeddings])

    if len(combined_chunks) != combined_embeddings.shape[0]:
        raise ValueError("Final chunk and embedding counts do not match.")

    if not np.isfinite(combined_embeddings).all():
        raise ValueError("Combined embeddings contain invalid values.")

    shutil.copy2(CHUNKS_PATH, CHUNKS_PATH.with_suffix(".json.bak"))
    shutil.copy2(EMBEDDINGS_PATH, EMBEDDINGS_PATH.with_suffix(".npy.bak"))

    temp_chunks = CHUNKS_PATH.with_suffix(".json.tmp")
    temp_embeddings = EMBEDDINGS_PATH.with_suffix(".npy.tmp")

    temp_chunks.write_text(
        json.dumps(combined_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    with temp_embeddings.open("wb") as file:
        np.save(file, combined_embeddings)

    temp_chunks.replace(CHUNKS_PATH)
    temp_embeddings.replace(EMBEDDINGS_PATH)

    print("\nHistorical indexing completed.")
    print(f"Previous chunks: {len(chunks)}")
    print(f"Historical chunks added: {len(new_chunks)}")
    print(f"Final chunks: {len(combined_chunks)}")
    print(f"Embedding shape: {combined_embeddings.shape}")
    print("Existing index backups created.")


if __name__ == "__main__":
    main()
