"""
Standalone training script for the Myntra recommendation pipeline.
Runs all stages: Data Loading, Embedding Generation, Feature Engineering, and LightGBM Ranking.
"""

import sys
import os

# Allow importing from src/
sys.path.insert(0, os.path.abspath('.'))

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

# ── Stage 1: Load Myntra Data ────────────────────────────────────────────────
banner("Stage 1: Loading & Preprocessing Myntra Data")
from src.data_pipeline.preprocess import load_myntra_data, build_user_profiles
import pandas as pd

articles = load_myntra_data(sample=100000)
print(f"  Loaded {len(articles):,} Myntra articles")

# ── Stage 2: Generate Embeddings ──────────────────────────────────────────────
banner("Stage 2: Generating/Loading Item Embeddings")
from src.data_pipeline.create_embeddings import get_embedding_model, generate_myntra_embeddings

model = get_embedding_model()
embeddings, catalog = generate_myntra_embeddings(articles, model, top_n=10000)
print(f"  Embedding matrix: {embeddings.shape}")

# ── Stage 3: Train Myntra Ranker ──────────────────────────────────────────────
banner("Stage 3: Training Myntra LightGBM Ranker")
from src.ranking.train_myntra_ranker import train_myntra_ranker

model, feature_cols = train_myntra_ranker(save=True)
print(f"  Ranker trained with {len(feature_cols)} features")
print(f"  Model saved to data/processed/myntra_ranker_lgb.pkl")

banner("Myntra Training Pipeline Complete!")
