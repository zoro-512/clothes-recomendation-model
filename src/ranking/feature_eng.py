"""
Feature engineering for the LightGBM ranking model.
Joins candidate items with user profile features and context.
"""

import pandas as pd
import numpy as np
from typing import Optional


def build_ranking_features(
    candidates: pd.DataFrame,
    user_profile: Optional[pd.Series] = None,
    context: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Create a feature matrix for the ranking model from retrieved candidates.

    Args:
        candidates: DataFrame returned by the vector store (includes 'semantic_score', 'purchase_count').
        user_profile: A single-row Series with columns [total_purchases, avg_price, favourite_season].
        context: Dict with optional query context (e.g., {'season': 'winter', 'query': '...'}).

    Returns:
        Feature DataFrame ready for LightGBM / XGBoost inference.
    """
    features = candidates.copy()

    # ── Item-level features ────────────────────────────────────────────────────
    max_pop = features["purchase_count"].max() if features["purchase_count"].max() > 0 else 1
    features["normalized_popularity"] = features["purchase_count"] / max_pop
    features["semantic_score"] = features["semantic_score"].fillna(0)
    features["desc_length"] = features["detail_desc"].fillna("").apply(lambda x: len(x.split()))

    # ── User-level features ────────────────────────────────────────────────────
    if user_profile is not None:
        features["user_total_purchases"] = user_profile.get("total_purchases", 0)
        features["user_avg_price"] = user_profile.get("avg_price", 0.0)
        features["user_fav_season"] = user_profile.get("favourite_season", "unknown")
    else:
        features["user_total_purchases"] = 0
        features["user_avg_price"] = 0.0
        features["user_fav_season"] = "unknown"

    # ── Context features ───────────────────────────────────────────────────────
    current_season = (context or {}).get("season", "unknown")
    features["season_match"] = (
        features.get("product_group_name", pd.Series([""] * len(features)))
        .fillna("")
        .apply(lambda x: 1 if current_season.lower() in x.lower() else 0)
    )

    # ── RLHF reward-adjusted score ─────────────────────────────────────────────
    # Pulled in from the reward log if the item has prior feedback
    features["rlhf_reward"] = features.get("rlhf_reward", pd.Series([0.0] * len(features))).fillna(0.0)

    # ── Final feature list for model ───────────────────────────────────────────
    FEATURE_COLS = [
        "semantic_score",
        "normalized_popularity",
        "user_total_purchases",
        "user_avg_price",
        "season_match",
        "desc_length",
        "rlhf_reward",
    ]

    return features, FEATURE_COLS
