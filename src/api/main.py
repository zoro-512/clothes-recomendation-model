
import os
import sys
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    FeedbackRequest,
    FeedbackResponse,
)
from src.data_pipeline.preprocess import load_hm_data, build_user_profiles
from src.data_pipeline.create_embeddings import get_embedding_model, generate_hm_embeddings, generate_query_embedding
from src.retrieval.vec_store import get_vector_store
from src.ranking.feature_eng import build_ranking_features
from src.ranking.train_ranker import load_ranker, rank_candidates, log_feedback, get_rlhf_rewards
from src.reasoning.llm_agent import build_explanation_chain, explain_recommendation, parse_natural_query

# ── App-level state (loaded once at startup) ───────────────────────────────────
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data once at server startup."""
    print("[Startup] Loading H&M data...")
    articles, transactions = load_hm_data()
    app_state["articles"] = articles
    app_state["transactions"] = transactions
    app_state["user_profiles"] = build_user_profiles(transactions, articles)

    print("[Startup] Generating/loading embeddings...")
    model = get_embedding_model()
    embeddings, catalog = generate_hm_embeddings(articles, model)
    app_state["embedding_model"] = model
    app_state["vector_store"] = get_vector_store(embeddings, catalog)

    print("[Startup] Loading ranker...")
    try:
        app_state["ranker"] = load_ranker()
    except FileNotFoundError:
        print("[Startup] No trained ranker found — will use semantic score only.")
        app_state["ranker"] = None

    print("[Startup] Setting up LLM chain...")
    chain, mode = build_explanation_chain()
    app_state["llm_chain"] = chain
    app_state["llm_mode"] = mode

    print("[Startup] Ready!")
    yield
    app_state.clear()


app = FastAPI(
    title="Hybrid E-commerce Recommender API",
    description="Retrieval → Ranking → LLM Reasoning pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": app_state.get("llm_mode", "not_loaded")}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    """Full hybrid recommendation pipeline."""
    query = req.query
    top_k = req.top_k

    # 1. Parse intent
    context = parse_natural_query(query, app_state["llm_chain"], app_state["llm_mode"])
    season = req.season or context.get("season", "unknown")

    # 2. Retrieval — semantic vector search
    query_vec = generate_query_embedding(query, app_state["embedding_model"])
    candidates = app_state["vector_store"].search(query_vec, top_k=100)

    # 3. Inject RLHF rewards into candidates
    rewards = get_rlhf_rewards()
    candidates["article_id"] = candidates["article_id"].astype(str)
    rewards["article_id"] = rewards["article_id"].astype(str)
    candidates = candidates.merge(rewards, on="article_id", how="left")
    candidates["rlhf_reward"] = candidates["rlhf_reward"].fillna(0.0)

    # 4. Build features & rank
    user_profile = None
    if req.user_id:
        profiles = app_state["user_profiles"]
        matched = profiles[profiles["customer_id"].astype(str) == req.user_id]
        if not matched.empty:
            user_profile = matched.iloc[0]

    featured_candidates, feature_cols = build_ranking_features(
        candidates, user_profile, {"season": season}
    )

    if app_state["ranker"] is not None:
        ranked = rank_candidates(featured_candidates, feature_cols, app_state["ranker"])
    else:
        # Fall back to pure semantic score
        ranked = featured_candidates.sort_values("semantic_score", ascending=False).reset_index(drop=True)

    top_items = ranked.head(top_k)

    # 5. Generate LLM explanations for top items
    recommendations = []
    for _, row in top_items.iterrows():
        explanation = explain_recommendation(
            query=query,
            item_name=str(row.get("prod_name", "")),
            item_desc=str(row.get("detail_desc", ""))[:300],
            season=season,
            chain=app_state["llm_chain"],
            mode=app_state["llm_mode"],
        )
        recommendations.append(
            RecommendedItem(
                article_id=str(row["article_id"]),
                product_name=str(row.get("prod_name", "Unknown")),
                product_group=str(row.get("product_group_name", "")),
                description=str(row.get("detail_desc", ""))[:200],
                rank_score=float(row.get("rank_score", row.get("semantic_score", 0))),
                semantic_score=float(row["semantic_score"]),
                explanation=explanation,
            )
        )

    return RecommendationResponse(
        query=query,
        user_id=req.user_id,
        recommendations=recommendations,
        total_candidates_evaluated=len(ranked),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Record RLHF user interaction event."""
    if req.action not in ("click", "cart", "purchase"):
        raise HTTPException(status_code=400, detail="action must be one of: click, cart, purchase")
    reward = log_feedback(req.article_id, req.action)
    return FeedbackResponse(
        article_id=req.article_id,
        action=req.action,
        reward_points=reward,
        message=f"Feedback recorded. Earned {reward} reward points.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
