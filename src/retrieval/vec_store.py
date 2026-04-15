"""
Vector store / retrieval layer.
Uses cosine similarity against pre-computed catalog embeddings as a
lightweight local alternative to Pinecone/Milvus for student-level compute.

To upgrade to Pinecone, set PINECONE_API_KEY in your environment and
set USE_PINECONE = True.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

USE_PINECONE = False  # Flip to True and set PINECONE_API_KEY env var to use cloud vector DB


class LocalVectorStore:
    """In-memory cosine-similarity vector store backed by NumPy arrays."""

    def __init__(self, embeddings: np.ndarray, catalog: pd.DataFrame):
        self.embeddings = embeddings  # (N, D)
        self.catalog = catalog.reset_index(drop=True)

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> pd.DataFrame:
        """
        Retrieve the top_k most semantically similar items for a query vector.

        Returns a DataFrame with the retrieved candidates + their similarity scores.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.catalog.iloc[top_indices].copy()
        results["semantic_score"] = scores[top_indices]
        return results.reset_index(drop=True)


class PineconeVectorStore:
    """Pinecone cloud vector store (requires PINECONE_API_KEY env var)."""

    def __init__(self, index_name: str = "hm-recommendations"):
        import pinecone
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set PINECONE_API_KEY in your environment or .env file.")
        pc = pinecone.Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)

    def upsert_catalog(self, embeddings: np.ndarray, catalog: pd.DataFrame):
        """Upsert all catalog embeddings into Pinecone."""
        batch_size = 100
        for i in range(0, len(catalog), batch_size):
            batch_emb = embeddings[i : i + batch_size]
            batch_meta = catalog.iloc[i : i + batch_size]
            vectors = [
                (str(row["article_id"]), emb.tolist(), {"prod_name": row["prod_name"]})
                for emb, (_, row) in zip(batch_emb, batch_meta.iterrows())
            ]
            self.index.upsert(vectors=vectors)
        print(f"[Pinecone] Upserted {len(catalog):,} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list:
        response = self.index.query(
            vector=query_vector.tolist(), top_k=top_k, include_metadata=True
        )
        return response["matches"]


def get_vector_store(embeddings: np.ndarray = None, catalog: pd.DataFrame = None):
    """Factory: returns the appropriate vector store based on USE_PINECONE flag."""
    if USE_PINECONE:
        return PineconeVectorStore()
    return LocalVectorStore(embeddings, catalog)
