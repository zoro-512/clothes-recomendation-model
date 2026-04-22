"""
LightGBM Ranking Model Training for Myntra Dataset.
Trains a binary classifier to predict user interaction using Myntra data.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_pipeline.preprocess import load_myntra_data, build_user_profiles
from src.ranking.feature_eng import build_ranking_features
from src.ranking.train_ranker import train_ranker, load_ranker

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_PATH = os.path.join(PROCESSED_DIR, "myntra_ranker_lgb.pkl")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def create_training_data_from_myntra(articles: pd.DataFrame, sample_size: int = 50000):
    """Create training data from Myntra dataset."""
    print(f"[Myntra Training] Creating training data from {len(articles)} articles...")
    
    # Sample for training if dataset is too large
    if len(articles) > sample_size:
        articles = articles.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"[Myntra Training] Sampled to {len(articles)} articles for training")
    
    # Create dummy user profiles for training
    dummy_transactions = pd.DataFrame({
        'customer_id': [f"CUST{i:06d}" for i in range(2000)],
        'article_id': articles['article_id'].sample(2000, replace=True).values,
        'price': articles['price'].sample(2000, replace=True).values,
        't_dat': pd.date_range('2023-01-01', periods=2000, freq='D')
    })
    
    # Add month and season columns for user profile building
    dummy_transactions['month'] = dummy_transactions['t_dat'].dt.month
    season_map = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                  6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}
    dummy_transactions['season'] = dummy_transactions['month'].map(season_map)
    
    user_profiles = build_user_profiles(dummy_transactions, articles)
    
    # Add dummy semantic_score for training (since we're not using vector search)
    articles['semantic_score'] = np.random.uniform(0.5, 1.0, len(articles))
    
    # Create training features
    candidates, feature_cols = build_ranking_features(
        articles, 
        user_profiles.iloc[0] if not user_profiles.empty else None, 
        {"season": "all", "gender": "all"}
    )
    
    # Create labels based on popularity (ratings + reviews)
    # Higher ratings and reviews indicate better products
    candidates['popularity_score'] = (
        candidates.get('ratings', 3.0).fillna(3.0) * 20 + 
        candidates.get('reviews', 0).fillna(0)
    )
    
    # Label as positive if above median popularity
    median_popularity = candidates['popularity_score'].median()
    y = (candidates["popularity_score"] > median_popularity).astype(int)
    
    # Remove leaky features
    safe_features = [f for f in feature_cols if f not in ['normalized_popularity', 'popularity_score']]
    X = candidates[safe_features].fillna(0)
    
    print(f"[Myntra Training] Created training data: {X.shape}, Positive ratio: {y.mean():.2f}")
    print(f"[Myntra Training] Features used: {safe_features}")
    
    return X, y, safe_features

def train_myntra_ranker(save: bool = True):
    """Train LightGBM ranker on Myntra dataset."""
    print("[Myntra Training] Starting Myntra ranker training...")
    
    # Load Myntra data
    articles = load_myntra_data(sample=100000)
    
    # Create training data
    X, y, feature_cols = create_training_data_from_myntra(articles)
    
    # Train model
    model = train_ranker(X, y, save=False)  # Don't save yet, we'll save with custom path
    
    if save:
        # Save with Myntra-specific path
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"[Myntra Training] Model saved to {MODEL_PATH}")
        
        # Save feature columns for later use
        feature_cols_path = os.path.join(PROCESSED_DIR, "myntra_feature_cols.pkl")
        with open(feature_cols_path, "wb") as f:
            pickle.dump(feature_cols, f)
        print(f"[Myntra Training] Feature columns saved to {feature_cols_path}")
    
    return model, feature_cols

def load_myntra_ranker():
    """Load Myntra-trained ranker."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Myntra ranker model not found at {MODEL_PATH}. Run train_myntra_ranker() first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_myntra_feature_cols():
    """Load feature columns used for Myntra training."""
    feature_cols_path = os.path.join(PROCESSED_DIR, "myntra_feature_cols.pkl")
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(
            f"Feature columns not found at {feature_cols_path}. Run train_myntra_ranker() first."
        )
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)

def rank_myntra_candidates(candidates: pd.DataFrame, model: lgb.Booster = None) -> pd.DataFrame:
    """Rank candidates using Myntra-trained model."""
    if model is None:
        model = load_myntra_ranker()
    
    feature_cols = load_myntra_feature_cols()
    safe_features = [f for f in feature_cols if f in model.feature_name()]
    
    X = candidates[safe_features].fillna(0)
    candidates = candidates.copy()
    candidates["rank_score"] = model.predict(X)
    return candidates.sort_values("rank_score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    print("=== Myntra Ranker Training ===")
    
    try:
        model, feature_cols = train_myntra_ranker(save=True)
        print("\n=== Training Complete ===")
        print(f"Model features: {len(feature_cols)}")
        print(f"Model saved to: {MODEL_PATH}")
        
        # Test loading
        loaded_model = load_myntra_ranker()
        print("Model loading test: PASSED")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
