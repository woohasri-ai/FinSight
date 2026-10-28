import json
from pathlib import Path

import numpy as np

from finsight.retrieval.dense import embed_texts, load_embedding_model


def main() -> None:
    """Build and save embeddings for all processed report chunks."""
    project_root = Path(__file__).resolve().parent.parent

    chunks_path = project_root / "data" / "processed" / "tcs_annual_report_2025_26_chunks.json"
    embeddings_path = (
        project_root / "data" / "processed" / "tcs_annual_report_2025_26_embeddings.npy"
    )

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("The chunks file contains no chunks.")

    print(f"Chunks to embed: {len(chunks)}")
    print("Loading BGE embedding model...")

    model = load_embedding_model()

    texts = [str(chunk["text"]) for chunk in chunks]

    print("Generating embeddings...")
    embeddings = embed_texts(texts, model)

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    np.save(embeddings_path, embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to: {embeddings_path}")


if __name__ == "__main__":
    main()
