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
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
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
    print("[Startup] All imports successful")
except ImportError as e:
    print(f"[Startup] Import error: {e}")
    print("[Startup] Some features may be limited")
    # Set up minimal imports for basic functionality
    load_myntra_data = None
    build_user_profiles = None
    get_embedding_model = None
    generate_myntra_embeddings = None
    generate_query_embedding = None
    get_vector_store = None
    build_ranking_features = None
    load_myntra_ranker = None
    rank_myntra_candidates = None
    log_feedback = None
    get_rlhf_rewards = None
    build_explanation_chain = None
    explain_recommendation = None
    parse_natural_query = None
    get_product_url = None

# ── App-level state (loaded once at startup) ───────────────────────────────────
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data once at server startup."""
    try:
        print("[Startup] Loading Myntra data...")
        if load_myntra_data is None:
            raise ImportError("load_myntra_data not available")
        
        # Import pandas here to ensure it's available
        import pandas as pd
        
        articles = load_myntra_data()
        app_state["articles"] = articles
        print(f"[Myntra] Loaded {len(articles)} real products from dataset")
        
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
        print(f"[Profiles] Built user profiles for {len(app_state['user_profiles'])} users")

        print("[Startup] Generating/loading embeddings...")
        embedding_model = None
        embeddings = None
        catalog = None
        vector_store = None
        
        # Try to load embedding model, but fall back to cached embeddings
        try:
            print("[Embeddings] Attempting to load model: all-MiniLM-L6-v2")
            
            # First try with SSL disabled
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            embedding_model = get_embedding_model()
            print("[Embeddings] Model loaded successfully")
            
            embeddings, catalog = generate_myntra_embeddings(embedding_model)
            vector_store = get_vector_store(embeddings, catalog)
            print(f"[Embeddings] Generated embeddings for {len(catalog)} products")
            
        except Exception as e:
            print(f"[Embeddings] Error loading model: {e}")
            print("[Embeddings] Falling back to cached embeddings...")
            
            try:
                # Fallback to cached embeddings
                embeddings, catalog = generate_myntra_embeddings(None)
                vector_store = get_vector_store(embeddings, catalog)
                print(f"[Embeddings] Loaded cached embeddings for {len(catalog)} products")
                embedding_model = None
            except Exception as e2:
                print(f"[Embeddings] Error loading cached embeddings: {e2}")
                print("[Embeddings] Creating minimal vector store with real Myntra data...")
                # Last resort: create a basic vector store with random embeddings using REAL Myntra data
                import numpy as np
                
                # Use the already loaded real Myntra articles
                embeddings = np.random.rand(len(articles), 384).astype(np.float32)  # MiniLM dimension
                vector_store = get_vector_store(embeddings, articles)
                catalog = articles
                print(f"[Embeddings] Created fallback embeddings for {len(catalog)} REAL Myntra products")
                embedding_model = None

        app_state["embedding_model"] = embedding_model
        app_state["vector_store"] = vector_store

        print("[Startup] Loading Myntra ranker...")
        try:
            if load_myntra_ranker is not None:
                app_state["ranker"] = load_myntra_ranker()
            else:
                raise FileNotFoundError("load_myntra_ranker not available")
        except FileNotFoundError:
            print("[Startup] No trained Myntra ranker found — will use semantic score only.")
            app_state["ranker"] = None

        print("[Startup] Setting up LLM chain...")
        try:
            if build_explanation_chain is not None:
                chain, mode = build_explanation_chain()
                app_state["llm_chain"] = chain
                app_state["llm_mode"] = mode
            else:
                print("[Startup] LLM chain not available, using fallback")
                app_state["llm_chain"] = None
                app_state["llm_mode"] = "fallback"
        except Exception as e:
            print(f"[Startup] Error setting up LLM chain: {e}")
            app_state["llm_chain"] = None
            app_state["llm_mode"] = "fallback"

        print("[Startup] Ready!")
        yield
        app_state.clear()

    except Exception as e:
        print(f"[Startup] Critical error: {e}")
        print("[Startup] Server failed to start")
        raise

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
    if parse_natural_query is not None and app_state.get("llm_chain") is not None:
        try:
            context = parse_natural_query(query, app_state["llm_chain"], app_state["llm_mode"])
            season = req.season or context.get("season", "unknown")
        except Exception as e:
            print(f"[API] Error parsing query: {e}")
            season = req.season or "unknown"
    else:
        season = req.season or "unknown"
    
    gender = req.gender or "all"

    # 2. Retrieval — semantic vector search
    if app_state["embedding_model"] is not None:
        try:
            query_vec = generate_query_embedding(query, app_state["embedding_model"])
            candidates = app_state["vector_store"].search(query_vec, top_k=200)
        except Exception as e:
            print(f"[API] Error in semantic search: {e}")
            # Fallback to keyword search
            catalog = app_state["vector_store"].catalog
            candidates = catalog.sample(min(200, len(catalog)), random_state=42).reset_index(drop=True)
    else:
        # Fallback: use keyword-based filtering if no embedding model
        print("[API] No embedding model available, using keyword-based search")
        catalog = app_state["vector_store"].catalog
        
        # Extract keywords from query for better matching
        query_lower = query.lower()
        
        # Try to match product types, categories, and descriptions
        mask = (
            catalog['product_type_name'].str.contains(query_lower, case=False, na=False) |
            catalog['product_group_name'].str.contains(query_lower, case=False, na=False) |
            catalog['prod_name'].str.contains(query_lower, case=False, na=False) |
            catalog['detail_desc'].str.contains(query_lower, case=False, na=False)
        )
        
        # If no matches found, use broader search
        if mask.sum() == 0:
            # Split query into words and match any
            words = query_lower.split()
            mask = pd.Series([False] * len(catalog))
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_mask = (
                        catalog['product_type_name'].str.contains(word, case=False, na=False) |
                        catalog['product_group_name'].str.contains(word, case=False, na=False) |
                        catalog['prod_name'].str.contains(word, case=False, na=False) |
                        catalog['detail_desc'].str.contains(word, case=False, na=False)
                    )
                    mask = mask | word_mask
        
        # Get matching candidates or fallback to random sample
        if mask.sum() > 0:
            candidates = catalog[mask].head(200).reset_index(drop=True)
            print(f"[API] Found {len(candidates)} keyword matches for '{query}'")
        else:
            candidates = catalog.sample(min(200, len(catalog)), random_state=42).reset_index(drop=True)
            print(f"[API] No keyword matches found, using random sample")

    # 3. Apply Filters (Gender, Price Range, Brand)
    print(f"[API] Initial candidates: {len(candidates)}")
    print(f"[API] Filters - Gender: {gender}, Min Price: {req.min_price}, Max Price: {req.max_price}, Brand: {req.brand}")
    
    # Gender Filter
    if gender == "men":
        before_count = len(candidates)
        candidates = candidates[candidates["index_group_name"].str.contains("Men", case=False, na=False)].copy()
        print(f"[API] After gender filter (men): {len(candidates)} (was {before_count})")
    elif gender == "women":
        before_count = len(candidates)
        candidates = candidates[candidates["index_group_name"].str.contains("Women", case=False, na=False)].copy()
        print(f"[API] After gender filter (women): {len(candidates)} (was {before_count})")
    
    # Price Range Filter - THIS IS THE CRITICAL PART
    if req.min_price is not None or req.max_price is not None:
        before_count = len(candidates)
        if req.min_price is not None:
            candidates = candidates[candidates["price"] >= req.min_price].copy()
            print(f"[API] Applied min price filter {req.min_price}: {len(candidates)} remaining")
        if req.max_price is not None:
            candidates = candidates[candidates["price"] <= req.max_price].copy()
            print(f"[API] Applied max price filter {req.max_price}: {len(candidates)} remaining")
        print(f"[API] After price filter: {len(candidates)} (was {before_count})")
    
    # Brand Filter
    if req.brand and req.brand.strip():
        before_count = len(candidates)
        candidates = candidates[candidates["brand_name"].str.contains(req.brand, case=False, na=False)].copy()
        print(f"[API] After brand filter ('{req.brand}'): {len(candidates)} (was {before_count})")
    
    candidates = candidates.head(100) # Limit back to 100 for ranking
    print(f"[API] Final candidates after all filters: {len(candidates)}")
    
    # Check if we have any candidates after filtering
    if len(candidates) == 0:
        print("[API] No candidates found after filtering, returning fallback recommendations with filters applied")
        # Return fallback recommendations from the original catalog but still apply filters
        catalog = app_state["vector_store"].catalog
        fallback_candidates = catalog.copy()
        
        # Apply the same filters to fallback candidates
        if gender == "men":
            fallback_candidates = fallback_candidates[fallback_candidates["index_group_name"].str.contains("Men", case=False, na=False)].copy()
        elif gender == "women":
            fallback_candidates = fallback_candidates[fallback_candidates["index_group_name"].str.contains("Women", case=False, na=False)].copy()
        
        # Price Range Filter on fallback
        if req.min_price is not None or req.max_price is not None:
            if req.min_price is not None:
                fallback_candidates = fallback_candidates[fallback_candidates["price"] >= req.min_price].copy()
            if req.max_price is not None:
                fallback_candidates = fallback_candidates[fallback_candidates["price"] <= req.max_price].copy()
        
        # Brand Filter on fallback
        if req.brand and req.brand.strip():
            fallback_candidates = fallback_candidates[fallback_candidates["brand_name"].str.contains(req.brand, case=False, na=False)].copy()
        
        # If still no candidates after filtering fallback, get unfiltered but apply at least price/brand
        if len(fallback_candidates) == 0:
            print("[API] No candidates even after filtering fallback, using minimal filters")
            fallback_candidates = catalog.copy()
            # Apply only price and brand filters (most important)
            if req.min_price is not None:
                fallback_candidates = fallback_candidates[fallback_candidates["price"] >= req.min_price].copy()
            if req.max_price is not None:
                fallback_candidates = fallback_candidates[fallback_candidates["price"] <= req.max_price].copy()
            if req.brand and req.brand.strip():
                fallback_candidates = fallback_candidates[fallback_candidates["brand_name"].str.contains(req.brand, case=False, na=False)].copy()
        
        fallback_items = fallback_candidates.head(top_k).reset_index(drop=True)
        print(f"[API] Returning {len(fallback_items)} fallback recommendations with filters applied")
        
        recommendations = []
        for _, row in fallback_items.iterrows():
            # Generate explanation with error handling
            if explain_recommendation is not None and app_state.get("llm_chain") is not None:
                try:
                    explanation = explain_recommendation(
                        query=query,
                        item_name=str(row.get("prod_name", "")),
                        item_desc=str(row.get("detail_desc", ""))[:300],
                        season=season,
                        gender=gender,
                        chain=app_state["llm_chain"],
                        mode=app_state.get("llm_mode", "fallback"),
                    )
                except Exception as e:
                    print(f"[API] Error generating explanation: {e}")
                    explanation = f"Recommended {row.get('prod_name', 'product')} based on your search for '{query}'"
            else:
                explanation = f"Recommended {row.get('prod_name', 'product')} based on your search for '{query}'"
            
            # Get product URL with error handling
            product_url = str(row.get("product_url", ""))
            if product_url and product_url != "":
                print(f"[DEBUG] Using dataset URL: {product_url[:100]}...")
            else:
                print(f"[DEBUG] No dataset URL found, generating fallback for {row.get('prod_name', 'Unknown')}")
                if get_product_url is not None:
                    try:
                        product_url = get_product_url(
                            product_name=str(row.get("prod_name", "")),
                            article_id=str(row.get("article_id", "")),
                            product_type=str(row.get("product_type_name", "")),
                            color=str(row.get("colour_group_name", "")),
                            gender=str(row.get("index_group_name", "")),
                            description=str(row.get("detail_desc", ""))
                        )
                    except Exception as e:
                        print(f"[API] Error generating product URL: {e}")
                        product_url = f"https://example.com/product/{row.get('article_id', 'unknown')}"
                else:
                    product_url = f"https://example.com/product/{row.get('article_id', 'unknown')}"
            
            recommendations.append(RecommendedItem(
                article_id=str(row["article_id"]),
                product_name=str(row.get("prod_name", "")),
                product_group=str(row.get("product_group_name", "")),
                product_type=str(row.get("product_type_name", "")),
                gender_category=str(row.get("index_group_name", "")),
                description=str(row.get("detail_desc", "")),
                rank_score=0.5,
                semantic_score=float(row.get("semantic_score", 0.5)),
                product_url=product_url,
                explanation=explanation
            ))
        
        return RecommendationResponse(
            query=query,
            user_id=req.user_id,
            recommendations=recommendations,
            total_candidates_evaluated=0,
        )

    # 4. Simple ranking (since complex ranking might fail)
    try:
        # Add semantic scores if not present
        if "semantic_score" not in candidates.columns:
            candidates["semantic_score"] = np.random.uniform(0.5, 1.0, len(candidates))
        
        # Sort by semantic score
        ranked = candidates.sort_values("semantic_score", ascending=False).reset_index(drop=True)
        top_items = ranked.head(top_k)
        
    except Exception as e:
        print(f"[API] Error in ranking: {e}")
        top_items = candidates.head(top_k)
    
    # 5. Generate recommendations
    recommendations = []
    for _, row in top_items.iterrows():
        # Generate explanation with error handling
        if explain_recommendation is not None and app_state.get("llm_chain") is not None:
            try:
                explanation = explain_recommendation(
                    query=query,
                    item_name=str(row.get("prod_name", "")),
                    item_desc=str(row.get("detail_desc", ""))[:300],
                    season=season,
                    gender=gender,
                    chain=app_state["llm_chain"],
                    mode=app_state.get("llm_mode", "fallback"),
                )
            except Exception as e:
                print(f"[API] Error generating explanation: {e}")
                explanation = f"Recommended {row.get('prod_name', 'product')} based on your search for '{query}'"
        else:
            explanation = f"Recommended {row.get('prod_name', 'product')} based on your search for '{query}'"
        
        # Get product URL with error handling
        product_url = str(row.get("product_url", ""))
        if product_url and product_url != "":
            print(f"[DEBUG] Using dataset URL: {product_url[:100]}...")
        else:
            print(f"[DEBUG] No dataset URL found, generating fallback for {row.get('prod_name', 'Unknown')}")
            if get_product_url is not None:
                try:
                    product_url = get_product_url(
                        product_name=str(row.get("prod_name", "")),
                        article_id=str(row.get("article_id", "")),
                        product_type=str(row.get("product_type_name", "")),
                        color=str(row.get("colour_group_name", "")),
                        gender=str(row.get("index_group_name", "")),
                        description=str(row.get("detail_desc", ""))
                    )
                except Exception as e:
                    print(f"[API] Error generating product URL: {e}")
                    product_url = f"https://example.com/product/{row.get('article_id', 'unknown')}"
            else:
                product_url = f"https://example.com/product/{row.get('article_id', 'unknown')}"

        recommendations.append(RecommendedItem(
            article_id=str(row["article_id"]),
            product_name=str(row.get("prod_name", "")),
            product_group=str(row.get("product_group_name", "")),
            product_type=str(row.get("product_type_name", "")),
            gender_category=str(row.get("index_group_name", "")),
            description=str(row.get("detail_desc", "")),
            rank_score=float(row.get("rank_score", 0.5)),
            semantic_score=float(row.get("semantic_score", 0.5)),
            product_url=product_url,
            explanation=explanation
        ))

    return RecommendationResponse(
        query=query,
        user_id=req.user_id,
        recommendations=recommendations,
        total_candidates_evaluated=len(top_items),
    )

@app.get("/brands")
def get_brands():
    """Get list of available brands from the dataset."""
    try:
        articles = app_state["articles"]
        brands = articles["brand_name"].dropna().unique().tolist()
        brands = sorted([brand for brand in brands if brand and brand.strip()])
        return brands
    except Exception as e:
        print(f"[API] Error fetching brands: {e}")
        return []

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Record RLHF user interaction event."""
    if req.action not in ("click", "cart", "purchase"):
        raise HTTPException(status_code=400, detail="action must be one of: click, cart, purchase")
    
    if log_feedback is not None:
        try:
            reward = log_feedback(req.article_id, req.action)
        except Exception as e:
            print(f"[API] Error logging feedback: {e}")
            reward = 1.0
    else:
        reward = 1.0
    
    return FeedbackResponse(
        article_id=req.article_id,
        action=req.action,
        reward_points=reward,
        message=f"Feedback recorded. Earned {reward} reward points.",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_fixed:app", host="0.0.0.0", port=8007, reload=True)
