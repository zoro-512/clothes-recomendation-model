# 🚀 Hybrid AI E-Commerce Recommendation System

An end-to-end, production-ready hybrid recommendation engine designed for e-commerce platforms (specifically tuned for Fashion catalogs like H&M). This system goes beyond simple collaborative filtering by combining **Semantic Vector Retrieval**, **LightGBM Machine Learning Ranking**, and **LLM-based Explanations** to deliver highly personalized and explainable results.

---

## 🌟 Key Features

1. **Two-Stage Retrieval-Ranker Architecture:**
   * **Semantic Retrieval (SentenceTransformers):** Quickly fetches the top candidate items from a vector database using dense Natural Language understanding (`all-MiniLM-L6-v2`) instead of rigid keyword matching.
   * **ML Ranker (LightGBM):** Re-ranks the retrieved candidates using a Gradient Boosting binary classifier. Evaluates semantic relevance, item popularity, historic user profiles, and contextual data (e.g., season matching) simultaneously.

2. **User Profiling & Cold-Start Mitigation:** 
   * Aggregates historic transaction telemetry to build profiles (average budget, favorite product types) to provide personalized recommendations even for users with sparse data.

3. **RLHF Integration (Reinforcement Learning from Human Feedback):** 
   * Continuously adjusts recommendation weights based on live implicit feedback (Clicks `+1`, Add to Cart `+3`, Purchases `+5`).

4. **Explainable AI (LangChain + OpenAI):** 
   * Uses `gpt-3.5-turbo` functioning as a "Personal AI Fashion Stylist" to generate a warm, natural-language sentence explaining *why* the #1 item perfectly matches the user's explicit query and implicit profile.

---

## 📂 Repository Structure

```text
recom/
├── data/
│   ├── raw/             # Raw H&M datasets and Amazon backups
│   └── processed/       # Pickled ML models, vector embeddings, and parquet catalogs
├── notebooks/
│   ├── 01_eda.ipynb     # Exploratory Data Analysis & Data Extraction
│   └── 02_model_training.ipynb # Step-by-step interactive ML pipeline & logic
├── scripts/
│   ├── run_training.py  # 🖥️ Standalone batch script to run the 9-stage pipeline
│   └── cli_search.py    # Command-line interface to test the engine manually
├── src/
│   ├── api/             # FastAPI backend for production deployments
│   ├── data_pipeline/   # Embedding generation & dataframe pre-processing
│   ├── ranking/         # LightGBM architecture & Feature Engineering code (avoiding data leakage!)
│   ├── reasoning/       # LangChain configurations and LLM agent
│   └── retrieval/       # Vector Database operations
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Usage & Setup

### 1. Installation
Clone the repository and install the dependencies (Python 3.10+ recommended).
```bash
pip install -r requirements.txt
```

### 2. Generate Embeddings & Train the Model
The easiest way to initialize the system, process the H&M data, generate the semantic vectors, and train the Machine Learning ranker is to run the standalone batch script:
```bash
python scripts/run_training.py
```
*This will execute all 9 architectural stages sequentially and save the final `ranker_lgb.pkl` model to `/data/processed/`.*

### 3. Interactive Notebook Walkthrough
If you want to view feature importance, evaluate the exact accuracy/AUC metrics, and visualize the Confusion Matrix, open the guided notebook:
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

### 4. Running the API
To serve the models locally via a FastAPI endpoint for a frontend:
```bash
uvicorn src.api.main:app --reload
```

---

## 🧠 Design Philosophy & Machine Learning Integrity
* **Data Leakage Prevention:** Built-in safeguards ensure that the LightGBM ranker cannot "cheat" during training. If target labels are generated from dataset popularity thresholds, popularity features are explicitly dropped from the training `X` matrix so the model learns true multivariate relevance.
* **Fallback Mechanisms:** If the OpenAPI key is missing or rate-limited, the system safely triggers a rule-based deterministic explanation fallback.
