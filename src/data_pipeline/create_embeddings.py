"""
Embedding generation for item catalog.
Uses SentenceTransformers to embed H&M article descriptions and Amazon doc texts.
Saves embeddings to data/processed/ for reuse.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_NAME = "all-MiniLM-L6-v2"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer embedding model (cached locally after first load)."""
    print(f"[Embeddings] Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def generate_hm_embeddings(
    articles: pd.DataFrame,
    model: SentenceTransformer = None,
    top_n: int = 10_000,
    batch_size: int = 64,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate dense vector embeddings for the top N most popular H&M articles.

    Returns:
        embeddings: np.ndarray of shape (N, 384)
        catalog: The filtered article DataFrame with matching index
    """
    emb_path = os.path.join(PROCESSED_DIR, "hm_catalog_embeddings.npy")
    catalog_path = os.path.join(PROCESSED_DIR, "hm_catalog.parquet")

    # Use cached embeddings if available
    if os.path.exists(emb_path) and os.path.exists(catalog_path):
        print("[Embeddings] Loading cached H&M embeddings...")
        embeddings = np.load(emb_path)
        catalog = pd.read_parquet(catalog_path)
        return embeddings, catalog

    if model is None:
        model = get_embedding_model()

    # Take top N by popularity for tractable local computation
    catalog = (
        articles.sort_values("purchase_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    texts = catalog["semantic_text"].tolist()
    print(f"[Embeddings] Encoding {len(texts):,} articles (batch_size={batch_size})…")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Cache to disk
    np.save(emb_path, embeddings)
    catalog.to_parquet(catalog_path, index=False)
    print(f"[Embeddings] Saved embeddings {embeddings.shape} -> {emb_path}")

    return embeddings, catalog


def generate_query_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
    """Encode a single user query string."""
    return model.encode([query])[0]


if __name__ == "__main__":
    from src.data_pipeline.preprocess import load_hm_data

    articles, _ = load_hm_data()
    model = get_embedding_model()
    embeddings, catalog = generate_hm_embeddings(articles, model)
    print("Embeddings shape:", embeddings.shape)
    print("Catalog sample:\n", catalog[["article_id", "prod_name", "purchase_count"]].head(3))
