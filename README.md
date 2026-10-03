# Latent — Singapore Reddit NLP Analytics Pipeline

> **A self-refinement of the CS5246 Text Mining course project from the National University of Singapore (NUS).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Vite 8](https://img.shields.io/badge/Vite-8-646cff.svg)](https://vitejs.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)

---

## 🏫 Origins & Acknowledgements

This project originated as the **CS5246 Text Mining** final project at the **National University of Singapore (NUS)**. The original collaborators were:

- **[Hawayo](https://github.com/Hawayo)**
- **[Cindy Chin](https://github.com/cindychin)**
- **[Wkkuu](https://github.com/wkkuu)**

Latent represents a **self-refinement** by the author, transforming the academic project into a modular, production-ready NLP system:
- **Clean modular architecture**: Pipeline stages refactored into dedicated Python packages (`preprocessing/`, `modelling/`, `search/`).
- **Dual execution modes**: Unified interactive notebooks with auto-detection for both **local environments** (custom GPU/CPU) and **Google Colab** (free T4 GPU).
- **Separated Full-Stack Dashboard**: Decoupled Flask REST API (`backend/`) and modern React 19 SPA (`frontend/`).
- **Centralized Artifacts**: Standardized outputs, analytical plot generation, and exploratory Streamlit tools in `artifacts/`.
- **Comprehensive Documentation**: Complete tech stack references, architecture maps, and reproducibility guides.

---

## 📖 What This Project Does

Latent analyzes **posts and comments from r/Singapore** to surface public opinions, trending topics, sentiment patterns, and community engagement dynamics. Raw social media text is processed through a **structured 10-stage NLP pipeline** — from scraping to search and recommendation.

### Key Capabilities

| Capability | Description |
|---|---|
| 🕸️ **Data Scraping** | Historical archive extraction (Arctic Shift API) + live incremental scraping (PRAW) |
| 🧹 **Data Cleaning** | URL/bot/noise filtering, deduplication, schema canonicalization |
| 🏷️ **POS & NER Tagging** | Linguistic annotation using spaCy (`en_core_web_sm`) |
| 🇸🇬 **Singlish Handling** | Custom regex particle normalization and dictionary-based English translation |
| 🔤 **Text Normalisation** | Slang expansion, emoji translation, contraction resolution, lemmatization |
| 📐 **Vector Space Models** | TF-IDF matrices, BM25Okapi retrieval indices, and Sentence-BERT embeddings |
| 😊 **Emotion Analysis** | 7-class emotion classification via RoBERTa (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) |
| 📊 **Topic Clustering** | SVD dimensionality reduction, Silhouette-optimized K-Means clustering, and t-SNE |
| 🔍 **Document Search** | Hybrid multi-metric search (TF-IDF + BM25 + SBERT) with Centroid Tree search |
| 📈 **Analytics Dashboard** | Decoupled full-stack React SPA with 20+ interactive visualizations and REST API |

---

## 🏗️ Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                                 │
│  Arctic Shift API (historical archive) ──→ data_scrape/scrape.py       │
│  Reddit API / PRAW (live incremental) ──→ data_scrape/scrape_incremental.py
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     TEXT PREPROCESSING PIPELINE                        │
│  Stage 1: preprocessing/data_cleaning.py (URLs, bots, dedup)           │
│  Stage 2: preprocessing/pos_ner_tagging.py (spaCy POS/NER)             │
│  Stage 3: preprocessing/singlish_normalisation.py (particle regex)     │
│  Stage 4: preprocessing/singlish_to_english.py (lexicon mapping)       │
│  Stage 5: preprocessing/common_normalisation.py (slang, lemmatization) │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     MODELLING & RETRIEVAL                              │
│  Stage 6: modelling/vector_space_model.py (TF-IDF, BM25, SBERT, Index) │
│  Stage 7: modelling/sentiment_analysis.py (RoBERTa 7-class emotions)   │
│  Stage 8: modelling/clustering.py (SVD, K-Means, t-SNE, keywords)      │
│  Stage 9: search/document_search.py (Hybrid Search + Recommendations)  │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      ANALYTICS & VISUALIZATION                         │
│  Backend: backend/app.py (Flask REST API serving data & models)        │
│  Frontend: frontend/ (React 19 SPA + Chakra UI 3 + Recharts/Plotly)    │
│  Artifacts: artifacts/ (Analytical plots & Streamlit explorer)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure Map

```
Latent/
│
├── README.md                                 # Project overview & documentation (this file)
├── AGENTS.md                                 # Contributor rules & AI conventions
├── TODO.md                                   # Project development roadmap & task tracking
├── LICENSE                                   # MIT License
├── Makefile                                  # Root automation (make install / dev / build)
├── requirements.txt                          # Top-level pipeline Python dependencies
├── .gitignore                                # Excludes large datasets, models, caches
│
├── data_scrape/                              # Reddit data collection
│   ├── scrape.py                             #   Historical archive scraper (Arctic Shift API)
│   └── scrape_incremental.py                 #   Incremental live scraper (PRAW / Reddit API)
│
├── preprocessing/                            # Python data preprocessing pipeline (Stages 1–5)
│   ├── __init__.py                           #   Exports cleaning & normalizer classes
│   ├── data_cleaning.py                      #   Stage 1: deduplication, schema, bot removal
│   ├── pos_ner_tagging.py                    #   Stage 2: spaCy POS & NER extractors
│   ├── singlish_normalisation.py             #   Stage 3: Singlish particle normalization
│   ├── singlish_to_english.py                #   Stage 4: Dictionary translation to standard English
│   └── common_normalisation.py               #   Stage 5: Slang expansion, emojis, lemmatization
│
├── modelling/                                # Machine learning & NLP models (Stages 6–8)
│   ├── __init__.py                           #   Exports model builders & clusterers
│   ├── vector_space_model.py                 #   Stage 6: TF-IDF, BM25, SBERT, inverted index
│   ├── sentiment_analysis.py                 #   Stage 7: RoBERTa emotion classification
│   ├── clustering.py                         #   Stage 8: SVD, K-Means clustering, t-SNE
│   └── emotion_inference.py                  #   Standalone RoBERTa batch CLI inference
│
├── search/                                   # Document retrieval & recommendations (Stage 9)
│   ├── __init__.py                           #   Exports CentroidTreeSearch & SearchEngine
│   └── document_search.py                    #   Stage 9: Hybrid multi-metric search & recs
│
├── notebooks/                                # Dual-environment Jupyter notebooks (Local / Colab)
│   ├── README.md                             #   Notebook guide & instructions
│   ├── Stage_0_Introduction.ipynb            #   Stage 0: Pipeline architecture & overview
│   ├── Stage_1_Data_Collection.ipynb         #   Stage 1: Data cleaning & preprocessing
│   ├── Stage_2_POS_NER_Tagging.ipynb         #   Stage 2: POS & NER tagging
│   ├── Stage_3_Singlish_Normalisation.ipynb  #   Stage 3: Singlish normalization
│   ├── Stage_4_Singlish_to_English.ipynb     #   Stage 4: Singlish → English conversion
│   ├── Stage_5_Text_Normalisation.ipynb      #   Stage 5: Slang, emojis, lemmatization
│   ├── Stage_6_Vector_Space_Model.ipynb      #   Stage 6: TF-IDF, BM25, SBERT, inverted index
│   ├── Stage_7_Sentiment_Analysis.ipynb      #   Stage 7: Emotion classification
│   ├── Stage_8_Clustering.ipynb              #   Stage 8: Topic clustering & t-SNE
│   ├── Stage_9_Document_Search.ipynb         #   Stage 9: Search & recommendation engine
│   └── Appendix_1_Sentiment_Benchmark.ipynb  #   Appendix: Zero-shot & benchmark evaluation
│
├── backend/                                  # Flask REST API server
│   ├── app.py                                #   API endpoints & pre-computed analytics
│   ├── pyproject.toml                        #   Python project metadata (uv)
│   ├── requirements.txt                      #   Backend Python dependencies
│   ├── Makefile                              #   Backend-specific commands
│   ├── documentation.md                      #   Full API specification (20 endpoints)
│   ├── data/                                 #   Runtime CSV datasets (gitignored)
│   └── models/                               #   Fitted ML models (.joblib) (gitignored)
│
├── frontend/                                 # React 19 SPA dashboard
│   ├── index.html                            #   HTML entry point
│   ├── package.json                          #   NPM dependencies (Chakra UI, Recharts, Plotly)
│   ├── vite.config.js                        #   Vite config (API proxy to :5000)
│   ├── public/                               #   Static assets (favicon)
│   └── src/                                  #   Components, pages, context, and styles
│
├── artifacts/                                # Output generation & exploratory tools
│   ├── README.md                             #   Artifacts guide
│   ├── emotion_dashboard.py                  #   Streamlit interactive emotion explorer
│   ├── plot_emotion_summary.py               #   Publication-ready chart generator
│   └── emotion_plots/                        #   Saved plot figures (.gitkeep)
│
├── utilities/                                # Shared lingual dictionaries & helpers
│   ├── pp_class.py                           #   RedditPreprocessor class
│   ├── singlish_dictionary.json              #   Singlish → English dictionary
│   ├── singlish_regex_to_text.txt            #   Regex normalization patterns
│   └── slang_dictionary.csv                  #   Internet slang & abbreviations
│
├── scripts/                                  # Repository maintenance
│   └── convert_notebooks.py                  #   Notebook sanitizer & dual-setup validator
│
└── tech/                                     # Architecture documentation
    └── STACK.md                              #   Detailed tech stack reference (30+ tools)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- **uv** (optional, recommended for fast Python package resolution)

### Quick Start (Full-Stack Dev Server)

```bash
# 1. Clone repository
git clone https://github.com/ChaoChienHung/Latent.git
cd Latent

# 2. Install all dependencies (backend + frontend)
make install

# 3. Start backend and frontend concurrently
make dev
```

Visit **http://localhost:5173** to view the interactive dashboard.

---

## 🏃 Running the Pipeline

You can run the NLP pipeline either via **Python scripts** or via **Jupyter notebooks**.

### Method 1: Modular Python Scripts

Run the pipeline stages sequentially from the repository root:

```bash
# 1. Scrape raw data
python data_scrape/scrape.py --subreddit singapore --year 2025

# 2. Clean raw posts and comments
python -m preprocessing.data_cleaning --input-dir data_scrape/data --output-dir intermediate_data

# 3. POS & NER tagging
python -m preprocessing.pos_ner_tagging --input intermediate_data/cleaned_posts.csv --output intermediate_data/pos_posts.csv --pos --ner

# 4. Singlish normalization
python -m preprocessing.singlish_normalisation --input intermediate_data/cleaned_posts.csv --output intermediate_data/singlish_norm_posts.csv

# 5. Singlish to English translation
python -m preprocessing.singlish_to_english --input intermediate_data/singlish_norm_posts.csv --output intermediate_data/translated_posts.csv

# 6. Common text normalization & lemmatization
python -m preprocessing.common_normalisation --input intermediate_data/translated_posts.csv --output intermediate_data/fully_normalized_posts.csv

# 7. Build Vector Space Models (TF-IDF, BM25, SBERT)
python -m modelling.vector_space_model --posts intermediate_data/fully_normalized_posts.csv --output-dir data/ --with-sbert

# 8. Emotion Classification
python -m modelling.sentiment_analysis --input data/PostVault.csv --output intermediate_data/posts_with_emotions.csv

# 9. Topic Clustering
python -m modelling.clustering --matrix data/vector_database/tfidf_fulltext.npz --k 7 --output-labels intermediate_data/cluster_labels.npy

# 10. Document Search & Recommendation
python -m search.document_search --query "cost of living in Singapore"
```

### Method 2: Dual-Environment Notebooks

All notebooks in [`notebooks/`](notebooks/) run seamlessly on both your **local machine** and **Google Colab**:

- **Local Machine**: Run `jupyter notebook notebooks/` — the environment detector configures local file paths automatically.
- **Google Colab**: Click the **"Open In Colab"** badge at the top of any notebook. The startup cell automatically mounts Google Drive, pulls the latest code, and installs all dependencies on a free GPU runtime.

---

## 📊 Analytics Dashboard

The full-stack dashboard provides interactive visual exploration of Reddit sentiment, topics, and search:

```bash
# Start backend only (Flask on port 5000)
make backend

# Start frontend only (Vite on port 5173)
make frontend

# Build frontend production bundle
make build
```

### Dashboard Pages

| Page | Features |
|---|---|
| **Topic Overview** | KPI cards, topic volume breakdown, interactive `d3-cloud` word clouds, engagement scatter plots |
| **Topic Deep Dive** | Deep dive into specific topic clusters with keyword weights, representative posts, and sentiment breakdown |
| **Timeline** | Temporal trends showing topic volume and conversation shifts across months |
| **Emotion Analysis** | 7-class emotion radar profiles, time-of-day/day-of-week dynamics, flair distributions |
| **Document Search** | Real-time hybrid search comparing TF-IDF, BM25, and SBERT relevance scores |
| **Recommendations** | Content-based recommendation of related posts and discussions |
| **Method Comparison** | In-depth algorithmic comparison between probabilistic and vector space retrieval |

---

## 🎨 Artifacts & Exploratory Analysis

The `artifacts/` folder hosts generated analysis outputs and exploratory tools:

```bash
# Generate analytical plots
python artifacts/plot_emotion_summary.py \
  --input backend/data/stopword_lemmatized_posts_0_labels_w_emot.csv \
  --output-dir artifacts/emotion_plots

# Launch interactive Streamlit explorer
streamlit run artifacts/emotion_dashboard.py
```

---

## 📡 Data Sources & Acquisition

| Source | Script | Description |
|---|---|---|
| [Arctic Shift API](https://arctic-shift.photon-reddit.com/) | `data_scrape/scrape.py` | Historical Reddit archive — bulk monthly extraction without rate limits |
| [Reddit API (PRAW)](https://praw.readthedocs.io/) | `data_scrape/scrape_incremental.py` | Official live Reddit API — cron-friendly incremental post collector |

> **Data Storage**: Processed CSV files and fitted model binaries are gitignored. Place generated data into `backend/data/` and `backend/models/` to power the dashboard.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
