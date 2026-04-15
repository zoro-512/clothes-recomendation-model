"""
Standalone training script — runs all 9 stages of the pipeline and prints live progress.
Run from the project root with: python scripts/run_training.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

# ── Stage 1: Load & Preprocess Data ───────────────────────────────────────────
banner("Stage 1: Loading & Preprocessing Data")
from src.data_pipeline.preprocess import load_hm_data, build_user_profiles

articles, transactions = load_hm_data()
user_profiles = build_user_profiles(transactions, articles)
print(f"  Articles: {articles.shape} | Transactions: {transactions.shape}")

# ── Stage 2: Generate Embeddings ──────────────────────────────────────────────
banner("Stage 2: Generating Item Embeddings (Top 10K articles)")
from src.data_pipeline.create_embeddings import get_embedding_model, generate_hm_embeddings

model = get_embedding_model()
embeddings, catalog = generate_hm_embeddings(articles, model, top_n=10_000)
print(f"  Embedding matrix: {embeddings.shape}")

# ── Stage 3: Build Vector Store & Retrieve Candidates ─────────────────────────
banner("Stage 3: Semantic Retrieval")
from src.retrieval.vec_store import get_vector_store
from src.data_pipeline.create_embeddings import generate_query_embedding

vector_store = get_vector_store(embeddings, catalog)
query = "I need a warm waterproof jacket for winter hiking"
query_vec = generate_query_embedding(query, model)
candidates = vector_store.search(query_vec, top_k=100)
print(f"  Query: '{query}'")
print(f"  Top match: {candidates.iloc[0]['prod_name']} (score={candidates.iloc[0]['semantic_score']:.3f})")

# ── Stage 4: Feature Engineering ──────────────────────────────────────────────
banner("Stage 4: Feature Engineering")
from src.ranking.feature_eng import build_ranking_features
from src.ranking.train_ranker import get_rlhf_rewards

rewards = get_rlhf_rewards()
candidates['article_id'] = candidates['article_id'].astype(str)
rewards['article_id'] = rewards['article_id'].astype(str)
candidates = candidates.merge(rewards, on='article_id', how='left')
candidates['rlhf_reward'] = candidates['rlhf_reward'].fillna(0.0)

user_profile = user_profiles.iloc[0]
context = {'season': 'winter'}
featured_candidates, FEATURE_COLS = build_ranking_features(candidates, user_profile, context)
print(f"  Feature columns: {FEATURE_COLS}")

# ── Stage 5: Train LightGBM Ranker ────────────────────────────────────────────
banner("Stage 5: Training LightGBM Ranker")
from src.ranking.train_ranker import build_training_data, train_ranker

X, y = build_training_data(featured_candidates, FEATURE_COLS, transactions)
print(f"  Train samples: {len(X)} | Positives: {y.sum()} ({y.mean():.1%})")
ranker = train_ranker(X, y)

# ── Stage 6: Rank Candidates ──────────────────────────────────────────────────
banner("Stage 6: Re-ranking Candidates")
from src.ranking.train_ranker import rank_candidates

final_ranked = rank_candidates(featured_candidates, FEATURE_COLS, ranker)
print("  Top 5 ranked items:")
for i, row in final_ranked.head(5).iterrows():
    print(f"    {i+1}. {row['prod_name']} | rank_score={row['rank_score']:.4f}")

# ── Stage 7: LLM Reasoning ────────────────────────────────────────────────────
banner("Stage 7: LLM Reasoning / Explanation")
from src.reasoning.llm_agent import build_explanation_chain, explain_recommendation

chain, mode = build_explanation_chain()
top_item = final_ranked.iloc[0]
explanation = explain_recommendation(
    query=query,
    item_name=str(top_item['prod_name']),
    item_desc=str(top_item.get('detail_desc', ''))[:300],
    season='winter',
    chain=chain,
    mode=mode,
)
print(f"  LLM Mode: {mode}")
print(f"  Top Item: {top_item['prod_name']}")
print(f"  Explanation: {explanation}")

# ── Stage 8: RLHF Feedback ────────────────────────────────────────────────────
banner("Stage 8: Logging RLHF Feedback")
from src.ranking.train_ranker import log_feedback

for action in ('click', 'cart', 'purchase'):
    reward = log_feedback(str(top_item['article_id']), action)
    print(f"  Logged '{action}' -> +{reward} pts")

# ── Stage 9: Evaluation ───────────────────────────────────────────────────────
banner("Stage 9: Evaluation Metrics")
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = y.reset_index(drop=True)
y_score = ranker.predict(X.fillna(0))

def precision_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[top_k].mean()

def recall_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[top_k].sum() / max(y_true.sum(), 1)

for k in [5, 10, 20]:
    p = precision_at_k(y_true, y_score, k)
    r = recall_at_k(y_true, y_score, k)
    print(f"  Precision@{k}: {p:.4f}  |  Recall@{k}: {r:.4f}")

if len(set(y_true.tolist())) > 1:
    print(f"  ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
else:
    print("  ROC-AUC: N/A (all candidates have same label for this query)")
    print("  Tip: Use a larger candidate pool (top_k=200) for proper AUC evaluation.")

banner("All stages complete! Ranker saved to data/processed/ranker_lgb.pkl")
