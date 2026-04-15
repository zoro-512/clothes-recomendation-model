"""
LightGBM Ranking Model — Training & Inference.

Trains a binary classifier to predict user interaction (click/purchase)
using features from feature_eng.py. Persists model to data/processed/.
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_PATH = os.path.join(PROCESSED_DIR, "ranker_lgb.pkl")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── RLHF Reward Weights ────────────────────────────────────────────────────────
REWARD_WEIGHTS = {"click": 1, "cart": 3, "purchase": 5}


def build_training_data(
    candidates: pd.DataFrame,
    feature_cols: list[str],
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create training labels from historical transaction data.
    To prevent single-class errors when using highly popular candidate sets,
    we dynamically label items as 1 (Positive) if their purchase count is 
    above the median of the current candidate pool.
    """
    median_popularity = candidates['purchase_count'].median()
    y = (candidates["purchase_count"] > median_popularity).astype(int)
    
    # CRITICAL FIX for Data Leakage: 
    # Because our target 'y' is derived from popularity, we CANNOT feed 
    # 'normalized_popularity' into X. Otherwise the model cheats and gets 100% accuracy.
    safe_features = [f for f in feature_cols if f != 'normalized_popularity']
    X = candidates[safe_features]
    
    return X, y


def train_ranker(X: pd.DataFrame, y: pd.Series, save: bool = True) -> lgb.Booster:
    """Train and optionally save the LightGBM ranking model."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    # Evaluation
    val_preds = model.predict(X_val)
    if len(set(y_val.tolist())) > 1:
        auc = roc_auc_score(y_val, val_preds)
        print(f"[Ranker] Validation AUC: {auc:.4f}")
    else:
        print("[Ranker] Validation AUC: N/A (single class in val split - increase candidate pool)")

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"[Ranker] Model saved to {MODEL_PATH}")

    return model


def load_ranker() -> lgb.Booster:
    """Load a previously trained LightGBM ranker from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Ranker model not found at {MODEL_PATH}. Run train_ranker() first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def rank_candidates(
    candidates: pd.DataFrame,
    feature_cols: list[str],
    model: lgb.Booster,
) -> pd.DataFrame:
    """Run inference on candidates and return them sorted by predicted score."""
    # Ensure we strictly use the features the model was trained on (e.g. without leaky features)
    safe_features = [f for f in feature_cols if f in model.feature_name()]
    
    X = candidates[safe_features].fillna(0)
    candidates = candidates.copy()
    candidates["rank_score"] = model.predict(X)
    return candidates.sort_values("rank_score", ascending=False).reset_index(drop=True)


# ── RLHF Feedback Store ────────────────────────────────────────────────────────
FEEDBACK_LOG = os.path.join(PROCESSED_DIR, "rlhf_feedback.csv")


def log_feedback(article_id: str, action: str):
    """
    Record a RLHF feedback event.
    action: one of 'click' (+1), 'cart' (+3), 'purchase' (+5)
    """
    reward = REWARD_WEIGHTS.get(action, 0)
    row = pd.DataFrame([{"article_id": article_id, "action": action, "reward": reward}])
    if os.path.exists(FEEDBACK_LOG):
        row.to_csv(FEEDBACK_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(FEEDBACK_LOG, index=False)
    return reward


def get_rlhf_rewards() -> pd.DataFrame:
    """Load aggregated RLHF rewards by article_id."""
    if not os.path.exists(FEEDBACK_LOG):
        return pd.DataFrame(columns=["article_id", "rlhf_reward"])
    df = pd.read_csv(FEEDBACK_LOG)
    return df.groupby("article_id")["reward"].sum().rename("rlhf_reward").reset_index()
