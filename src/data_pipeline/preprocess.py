"""
Data preprocessing pipeline.
Loads and validates the Amazon & H&M sampled datasets from data/raw/.
"""

import os
import pandas as pd
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def load_amazon_data(sample: int = 1_000_000) -> pd.DataFrame:
    """Load and preprocess the Amazon pre-processed parquet dataset."""
    parquet_path = os.path.join(DATA_DIR, "amazon", "train_data_prepared.parquet")
    test_path = os.path.join(DATA_DIR, "amazon", "test-Amazon-C4.csv")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Amazon parquet not found at {parquet_path}")

    train_df = pd.read_parquet(parquet_path, engine="pyarrow")
    test_df = pd.read_csv(test_path)

    # Sample if too large
    if len(train_df) > sample:
        train_df = train_df.sample(sample, random_state=42).reset_index(drop=True)

    # Normalize text columns to lowercase, stripped
    train_df["query_text"] = train_df["query_text"].fillna("").str.strip()
    train_df["doc_text"] = train_df["doc_text"].fillna("").str.strip()

    print(f"[Amazon] Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


def load_hm_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pre-sampled H&M articles and transaction CSVs."""
    articles_path = os.path.join(DATA_DIR, "hm_articles_sampled.csv")
    txn_path = os.path.join(DATA_DIR, "hm_transactions_sampled.csv")

    if not os.path.exists(articles_path):
        raise FileNotFoundError(
            "H&M articles not found. Run notebooks/01_eda.ipynb first to download."
        )
    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            "H&M transactions not found. Run notebooks/01_eda.ipynb first to download."
        )

    articles = pd.read_csv(articles_path)
    transactions = pd.read_csv(txn_path)

    # Fill missing descriptions
    articles["detail_desc"] = articles["detail_desc"].fillna("")
    articles["prod_name"] = articles["prod_name"].fillna("")

    # Create enriched text for semantic embedding
    articles["semantic_text"] = (
        articles["prod_name"]
        + " | "
        + articles["product_group_name"].fillna("")
        + " | "
        + articles["detail_desc"]
    )

    # Parse timestamps
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month
    transactions["season"] = transactions["month"].apply(_month_to_season)

    # Compute item popularity
    popularity = (
        transactions.groupby("article_id")
        .size()
        .rename("purchase_count")
        .reset_index()
    )
    articles = articles.merge(popularity, on="article_id", how="left")
    articles["purchase_count"] = articles["purchase_count"].fillna(0).astype(int)

    print(f"[H&M] Articles: {articles.shape} | Transactions: {transactions.shape}")
    return articles, transactions


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def build_user_profiles(transactions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregate user feature profiles.
    Addresses cold-start by summarising each user's interaction history.
    Returns a DataFrame indexed by customer_id.
    """
    merged = transactions.merge(
        articles[["article_id", "purchase_count", "product_group_name"]],
        on="article_id",
        how="left"
    )

    user_profiles = merged.groupby("customer_id").agg(
        total_purchases=("article_id", "count"),
        avg_price=("price", "mean"),
        favourite_season=("season", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
        favourite_group=("product_group_name", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
    ).reset_index()

    print(f"[Profiles] Built {len(user_profiles):,} user profiles.")
    return user_profiles


if __name__ == "__main__":
    articles, transactions = load_hm_data()
    user_profiles = build_user_profiles(transactions, articles)
    print(user_profiles.head())
