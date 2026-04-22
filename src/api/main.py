
import os
import sys
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Disable SSL verification for embedding model loading
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'FALSE'

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    FeedbackRequest,
    FeedbackResponse,
)
from src.data_pipeline.preprocess import load_myntra_data, build_user_profiles
from src.data_pipeline.create_embeddings import get_embedding_model, generate_myntra_embeddings, generate_query_embedding
from src.retrieval.vec_store import get_vector_store
from src.ranking.feature_eng import build_ranking_features
from src.ranking.train_myntra_ranker import load_myntra_ranker, rank_myntra_candidates
from src.ranking.train_ranker import log_feedback, get_rlhf_rewards
from src.reasoning.llm_agent import build_explanation_chain, explain_recommendation, parse_natural_query
from src.utils.web_search import get_product_url

# ── App-level state (loaded once at startup) ───────────────────────────────────
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data once at server startup."""
    print("[Startup] Loading Myntra data...")
    articles = load_myntra_data()
    app_state["articles"] = articles
    # Create dummy transactions for user profiles (since Myntra dataset doesn't have transactions)
    dummy_transactions = pd.DataFrame({
        'customer_id': [f"CUST{i:06d}" for i in range(1000)],
        'article_id': articles['article_id'].sample(1000, replace=True).values,
        'price': articles['price'].sample(1000, replace=True).values,
        't_dat': pd.date_range('2023-01-01', periods=1000, freq='D')
    })
    
    # Add month and season columns for user profile building
    dummy_transactions['month'] = dummy_transactions['t_dat'].dt.month
    season_map = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                  6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}
    dummy_transactions['season'] = dummy_transactions['month'].map(season_map)
    app_state["transactions"] = dummy_transactions
    app_state["user_profiles"] = build_user_profiles(dummy_transactions, articles)

    print("[Startup] Generating/loading embeddings...")
    try:
        model = get_embedding_model()
        embeddings, catalog = generate_myntra_embeddings(articles, model)
        app_state["embedding_model"] = model
        app_state["vector_store"] = get_vector_store(embeddings, catalog)
    except Exception as e:
        print(f"[Startup] Error loading embedding model: {e}")
        print("[Startup] Using cached embeddings without model...")
        # Load cached embeddings directly without model
        import numpy as np
        emb_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "myntra_catalog_embeddings.npy")
        catalog_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "myntra_catalog.parquet")
        
        if os.path.exists(emb_path) and os.path.exists(catalog_path):
            embeddings = np.load(emb_path)
            catalog = pd.read_parquet(catalog_path)
            app_state["embedding_model"] = None  # No model for query encoding
            app_state["vector_store"] = get_vector_store(embeddings, catalog)
            print("[Startup] Loaded cached embeddings successfully")
        else:
            raise Exception("No cached embeddings found and model loading failed")

    print("[Startup] Loading Myntra ranker...")
    try:
        app_state["ranker"] = load_myntra_ranker()
    except FileNotFoundError:
        print("[Startup] No trained Myntra ranker found — will use semantic score only.")
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

# Serve Frontend with cache-busting
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.middleware("http")
async def add_cache_busting_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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
    gender = req.gender or "all"

    # 2. Retrieval — semantic vector search
    if app_state["embedding_model"] is not None:
        query_vec = generate_query_embedding(query, app_state["embedding_model"])
        candidates = app_state["vector_store"].search(query_vec, top_k=200) # Get more candidates to allow for filtering
    else:
        # Fallback: return random sample from catalog if no embedding model
        print("[API] No embedding model available, using random sample")
        catalog = app_state["vector_store"].catalog
        candidates = catalog.sample(min(200, len(catalog)), random_state=42).reset_index(drop=True)

    # 3. Apply Gender Filter (Hard Filter for UI Buttons)
    if gender == "men":
        candidates = candidates[candidates["Gender"] == "Men"].copy()
    elif gender == "women":
        candidates = candidates[candidates["Gender"] == "Women"].copy()
    
    candidates = candidates.head(100) # Limit back to 100 for ranking

    # 4. Inject RLHF rewards into candidates
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
        candidates, user_profile, {"season": season, "gender": gender}
    )
    
    # Ensure product_url is preserved from original candidates
    if 'product_url' in candidates.columns:
        featured_candidates['product_url'] = candidates['product_url']

    if app_state["ranker"] is not None:
        ranked = rank_myntra_candidates(featured_candidates, app_state["ranker"])
    else:
        # Fall back to pure semantic score
        ranked = featured_candidates.sort_values("semantic_score", ascending=False).reset_index(drop=True)
    
    # Ensure product_url is preserved after ranking
    if 'product_url' in featured_candidates.columns and 'product_url' not in ranked.columns:
        ranked['product_url'] = featured_candidates['product_url']

    top_items = ranked.head(top_k)
    
    # Debug: Check if product_url is preserved
    print(f"[DEBUG] top_items has product_url: {'product_url' in top_items.columns}")
    if 'product_url' in top_items.columns:
        print(f"[DEBUG] Sample top_items URLs: {top_items['product_url'].head(2).tolist()}")

    # 5. Generate LLM explanations & Get Product URLs for top items
    recommendations = []
    for _, row in top_items.iterrows():
        explanation = explain_recommendation(
            query=query,
            item_name=str(row.get("prod_name", "")),
            item_desc=str(row.get("detail_desc", ""))[:300],
            season=season,
            gender=gender,
            chain=app_state["llm_chain"],
            mode=app_state["llm_mode"],
        )
        
        # Get product URL directly from dataset - this should always work for Myntra
        product_url = str(row.get("product_url", ""))
        
        # Debug: Log what we're getting
        if product_url and product_url != "":
            print(f"[DEBUG] Using dataset URL: {product_url[:100]}...")
        else:
            print(f"[DEBUG] No dataset URL found, generating fallback for {row.get('prod_name', 'Unknown')}")
            # Only generate fallback if no dataset URL
            product_url = get_product_url(
                product_name=str(row.get("prod_name", "")),
                article_id=str(row.get("article_id", "")),
                product_type=str(row.get("product_type_name", "")),
                color=str(row.get("colour_group_name", "")),
                gender=str(row.get("index_group_name", "")),
                description=str(row.get("detail_desc", ""))
            )
        
        recommendations.append(
            RecommendedItem(
                article_id=str(row["article_id"]),
                product_name=str(row.get("prod_name", "Unknown")),
                product_group=str(row.get("product_group_name", "")),
                product_type=str(row.get("product_type_name", "")),
                gender_category=str(row.get("index_group_name", "")),
                description=str(row.get("detail_desc", ""))[:200],
                rank_score=float(row.get("rank_score", row.get("semantic_score", 0))),
                semantic_score=float(row["semantic_score"]),
                product_url=product_url,
                explanation=explanation,
            )
        )

    return RecommendationResponse(
        query=query,
        user_id=req.user_id,
        recommendations=recommendations,
        total_candidates_evaluated=len(ranked),
    )


@app.get("/test-urls")
def test_urls():
    """Test endpoint to return sample URLs for frontend testing."""
    return [
        {
            "article_id": "12345",
            "product_name": "Test Product 1",
            "product_group": "Test Group",
            "product_type": "Test Type",
            "gender_category": "Test Gender",
            "description": "Test description",
            "rank_score": 0.9,
            "semantic_score": 0.8,
            "product_url": "https://www.myntra.com/jeans/roadster/roadster-men-navy-blue-slim-fit-mid-rise-clean-look-jeans/2296012/buy",
            "explanation": "Test explanation"
        },
        {
            "article_id": "67890",
            "product_name": "Test Product 2",
            "product_group": "Test Group",
            "product_type": "Test Type",
            "gender_category": "Test Gender",
            "description": "Test description",
            "rank_score": 0.8,
            "semantic_score": 0.7,
            "product_url": "https://www.myntra.com/tshirts/selected/selected-men-navy-blue-solid-t-shirt/16842278/buy",
            "explanation": "Test explanation"
        }
    ]

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
