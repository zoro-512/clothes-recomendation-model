"""
Data preprocessing pipeline.
Loads and validates the Amazon & H&M sampled datasets from data/raw/.
"""

import os
import pandas as pd
import numpy as np
import re


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


def load_myntra_data(sample: int = 100_000) -> pd.DataFrame:
    """Load and preprocess the Myntra fashion dataset."""
    import re  # Import re module for regex operations
    
    myntra_path = os.path.join(DATA_DIR, "Myntra Fasion Clothing.csv", "Myntra Fasion Clothing.csv")
    
    if not os.path.exists(myntra_path):
        raise FileNotFoundError(f"Myntra dataset not found at {myntra_path}")
    
    # Load dataset with proper handling
    df = pd.read_csv(myntra_path, low_memory=False)
    
    # Sample if too large
    if len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    
    # Map Myntra columns to H&M compatible names
    df = df.rename(columns={
        'URL': 'product_url',
        'Product_id': 'article_id',
        'BrandName': 'brand_name',
        'Category': 'product_group_name',
        'Individual_category': 'product_type_name',
        'category_by_Gender': 'index_group_name',
        'Description': 'detail_desc',
        'DiscountPrice (in Rs)': 'price',
        'OriginalPrice (in Rs)': 'original_price',
        'DiscountOffer': 'discount_offer',
        'SizeOption': 'size_options',
        'Ratings': 'ratings',
        'Reviews': 'reviews'
    })
    
    # Create product_name from brand and category
    df['prod_name'] = df['brand_name'] + ' ' + df['product_type_name']
    
    # Create color information (simplified to avoid pandas issues)
    df['colour_group_name'] = 'Multi'
    df['perceived_colour_value_name'] = 'multi'
    
    # Try to extract colors from descriptions if available
    if 'detail_desc' in df.columns:
        color_pattern = r'(black|white|blue|red|green|grey|navy|maroon|beige|brown|olive|khaki|cream|pink|purple|yellow|orange|teal)'
        colors = []
        for desc in df['detail_desc'].fillna(''):
            if isinstance(desc, str):
                match = re.search(color_pattern, desc, flags=re.IGNORECASE)
                if match:
                    colors.append(match.group().title())
                else:
                    colors.append('Multi')
            else:
                colors.append('Multi')
        df['colour_group_name'] = colors
        df['perceived_colour_value_name'] = [c.lower() for c in colors]
    
    # Create additional H&M compatible columns
    df['product_code'] = df['article_id'].astype(str)
    df['index_code'] = df['article_id'].astype(str)
    df['index_name'] = df['product_type_name']
    df['department_name'] = df['product_group_name']
    df['section_name'] = df['product_type_name']
    df['garment_group_name'] = df['product_group_name']
    df['graphical_appearance_name'] = df['brand_name']
    df['perceived_colour_master_name'] = df['colour_group_name']
    
    # Ensure price columns are numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
    
    # Fill missing values
    df['detail_desc'] = df['detail_desc'].fillna(df['prod_name'])
    df['prod_name'] = df['prod_name'].fillna("Unknown Product")
    df['price'] = df['price'].fillna(df['original_price']).fillna(999.0)
    
    # Create semantic text for embeddings
    df['semantic_text'] = (
        df['prod_name'] + " | " + 
        df['product_group_name'].fillna("") + " | " + 
        df['detail_desc']
    )
    
    # Create purchase_count (simulate popularity based on ratings and reviews)
    df['purchase_count'] = ((df['ratings'].fillna(3.0) * 100) + (df['reviews'].fillna(0) * 10)).astype(int)
    
    print(f"[Myntra] Loaded {df.shape} products")
    return df


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

    # Standardise column names (sampled data might have 'C' instead of 'article_id')
    if "C" in articles.columns and "article_id" not in articles.columns:
        articles = articles.rename(columns={"C": "article_id"})

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

    # Compute item popularity and average price
    item_stats = (
        transactions.groupby("article_id")
        .agg(
            purchase_count=("article_id", "size"),
            price=("price", "mean")
        )
        .reset_index()
    )
    articles = articles.merge(item_stats, on="article_id", how="left")
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
