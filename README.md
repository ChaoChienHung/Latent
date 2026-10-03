# Latent — Singapore Reddit NLP Analytics Pipeline

> **A self-refinement of the CS5246 Text Mining course project from the National University of Singapore (NUS).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🏫 Origins & Acknowledgements

This project originated as the **CS5246 Text Mining** final project at the **National University of Singapore (NUS)**. The original collaborators were:

- **[Hawayo](https://github.com/Hawayo)**
- **[Cindy Chin](https://github.com/cindychin)**
- **[Wkkuu](https://github.com/wkkuu)**

Latent is a **self-refinement** that restructures, extends, and improves the original project with:
- Cleaner code organization and modular pipeline design
- Dual execution modes (local Python + Google Colab)
- An integrated full-stack analytics dashboard (Flask + React)
- Improved documentation and reproducibility

---

## 📖 What This Project Does

Latent analyzes **posts and comments from r/Singapore** to surface public opinions, trending topics, sentiment patterns, and community engagement dynamics. Raw social media text is processed through a **structured 10-stage NLP pipeline** — from scraping to search and recommendation.

### Key Capabilities

| Capability | Description |
|---|---|
| 🕸️ **Data Scraping** | Historical + incremental Reddit data collection |
| 🧹 **Text Preprocessing** | URL/bot/noise removal, Singlish normalization, spell correction |
| 🏷️ **POS & NER Tagging** | Linguistic annotation with spaCy |
| 🇸🇬 **Singlish Handling** | Custom dictionary-based normalization and translation |
| 📊 **Topic Clustering** | TF-IDF + K-Means with Silhouette-optimized cluster selection |
| 😊 **Emotion Analysis** | 7-class emotion classification via DistilRoBERTa |
| 🔍 **Document Search** | BM25 + TF-IDF + Sentence-BERT retrieval engines |
| 📈 **Analytics Dashboard** | Full-stack React dashboard with 20+ interactive visualizations |

---

## 🏗️ Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│  Arctic Shift API (historical) ──→ scrape.py                       │
│  Reddit API / PRAW (incremental) ──→ scrape_incremental.py         │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TEXT PREPROCESSING PIPELINE                     │
│  Stage 1: Data Cleaning (URLs, bots, duplicates, encoding)         │
│  Stage 2: POS & NER Tagging (spaCy)                                │
│  Stage 3: Singlish Normalisation (custom dictionary)               │
│  Stage 4: Singlish → English Conversion                            │
│  Stage 5: Common Normalisation (slang, spelling, lemmatization)    │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODELLING & ANALYSIS                            │
│  Stage 6: Vector Space Model + Inverted Index (TF-IDF, BM25, BERT)│
│  Stage 7: Sentiment & Emotion Analysis (DistilRoBERTa)            │
│  Stage 8: Topic Clustering (K-Means + t-SNE visualization)        │
│  Stage 9: Document Search & Recommendation Engine                  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ANALYTICS DASHBOARD                            │
│  Flask API (pre-computed analytics) ──→ React SPA (Chakra UI)      │
│  20+ interactive charts: topics, emotions, engagement, search      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure Map

```
Latent/
│
│── ─── Root ───────────────────────────────────────────────────────────
│
├── README.md                                 # Project overview & guide (this file)
├── AGENTS.md                                 # AI agent rules & coding conventions
├── TODO.md                                   # Task tracker with checkboxes
├── LICENSE                                   # MIT License
├── .gitignore                                # Git exclusions (data, models, caches)
├── requirements.txt                          # Python deps for the full pipeline
├── emotion_inference.py                      # Standalone emotion classification CLI
│
│── ─── Data Collection ────────────────────────────────────────────────
│
├── data_scrape/                              # Reddit data collection
│   ├── scrape.py                             #   Historical bulk scrape (Arctic Shift API)
│   └── scrape_incremental.py                 #   Incremental live scrape (PRAW / Reddit API)
│
│── ─── Pipeline Notebooks ─────────────────────────────────────────────
│
├── notebooks/                                # Organized pipeline notebooks
│   ├── README.md                             #   Guide: local vs Colab usage
│   ├── local/                                #   🖥️  Local Jupyter (your GPU)
│   │   ├── Stage_0_Introduction.ipynb
│   │   ├── Stage_1_Data_Collection.ipynb
│   │   ├── Stage_2_POS_NER_Tagging.ipynb
│   │   ├── Stage_3_Singlish_Normalisation.ipynb
│   │   ├── Stage_4_Singlish_to_English.ipynb
│   │   ├── Stage_5_Text_Normalisation.ipynb
│   │   ├── Stage_6_Vector_Space_Model.ipynb
│   │   ├── Stage_7_Sentiment_Analysis.ipynb
│   │   ├── Stage_8_Clustering.ipynb
│   │   ├── Stage_9_Document_Search.ipynb
│   │   └── Appendix_1_Sentiment_Benchmark.ipynb
│   └── colab/                                #   ☁️  Google Colab (free T4 GPU)
│       ├── Stage_0_Introduction.ipynb        #     ↳ Each has Colab badge + auto-setup
│       ├── Stage_1_Data_Collection.ipynb
│       ├── Stage_2_POS_NER_Tagging.ipynb
│       ├── Stage_3_Singlish_Normalisation.ipynb
│       ├── Stage_4_Singlish_to_English.ipynb
│       ├── Stage_5_Text_Normalisation.ipynb
│       ├── Stage_6_Vector_Space_Model.ipynb
│       ├── Stage_7_Sentiment_Analysis.ipynb
│       ├── Stage_8_Clustering.ipynb
│       ├── Stage_9_Document_Search.ipynb
│       └── Appendix_1_Sentiment_Benchmark.ipynb
│
│── ─── Original Notebooks (Legacy) ───────────────────────────────────
│
├── Stage_0_Introduction.ipynb                # Original CS5246 notebooks (kept for reference)
├── Stage_1_Data_Collection_and_Data_Cleaning.ipynb
├── Stage_2_POS_and_NER_Tagging.ipynb
├── Stage_3_Singlish_Normalisation.ipynb
├── Stage_4_Singlish_to_English_Conversion.ipynb
├── Stage_5_Common_Normalisation.ipynb
├── Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb
├── Stage_8_Clustering_and_Visualization.ipynb
├── Stage_9_Document_Search.ipynb
├── Step_Appendix_1_Sentiment Labelling (Benchmark).ipynb
│
│── ─── Utilities & Dictionaries ───────────────────────────────────────
│
├── utilities/                                # Shared helper modules & dictionaries
│   ├── pp_class.py                           #   RedditPreprocessor class (cleaning pipeline)
│   ├── singlish_dictionary.json              #   Singlish → English word mappings
│   ├── singlish_regex_to_text.txt            #   Singlish regex normalization patterns
│   └── slang_dictionary.csv                  #   Internet slang expansion table
│
│── ─── Analytics Dashboard ────────────────────────────────────────────
│
├── dashboard-ui/                             # Full-stack analytics dashboard
│   ├── app.py                                #   Flask API backend (892 lines, 20 endpoints)
│   ├── Makefile                              #   Dev commands (make install/dev/build)
│   ├── pyproject.toml                        #   Python project metadata (uv)
│   ├── requirements.txt                      #   Backend Python dependencies
│   ├── documentation.md                      #   Detailed backend + frontend docs
│   ├── README.md                             #   Dashboard quick-start guide
│   │
│   ├── data/                                 #   📊 Processed datasets (gitignored)
│   │   ├── PostVault.csv                     #     Main dataset (~6k posts)
│   │   ├── stopword_lemmatized_posts_0.csv   #     Preprocessed posts
│   │   └── ..._labels_w_emot.csv             #     Posts + emotion predictions
│   │
│   ├── models/                               #   🤖 Trained ML models (gitignored)
│   │   ├── bm25_fulltext_model.joblib        #     BM25 index (full text)
│   │   ├── bm25_titles_model.joblib          #     BM25 index (titles)
│   │   └── tfidf_posts_vectorizer.joblib     #     Fitted TF-IDF vectorizer
│   │
│   └── dashboard/                            #   ⚛️  React + Vite frontend
│       ├── index.html                        #     HTML entry point
│       ├── package.json                      #     NPM dependencies
│       ├── vite.config.js                    #     Vite config (API proxy → :5000)
│       ├── public/                           #     Static assets (favicon)
│       └── src/
│           ├── main.jsx                      #     React entry point
│           ├── App.jsx                       #     Route definitions
│           ├── theme.js                      #     Chakra UI theme config
│           ├── index.css                     #     Global styles
│           ├── components/
│           │   ├── Layout.jsx                #       Page wrapper + gradient bg
│           │   ├── Navbar.jsx                #       Sticky navigation bar
│           │   └── WordCloud.jsx             #       d3-cloud word cloud
│           ├── data/
│           │   ├── api.jsx                   #       DataProvider context + useData()
│           │   └── mockData.js               #       Fallback data for offline dev
│           └── pages/
│               ├── TopicOverview.jsx          #       Main dashboard (KPIs, charts)
│               ├── TopicDeepDive.jsx          #       Single-topic analysis
│               ├── Timeline.jsx              #       Topic volume over time
│               ├── EmotionAnalysis.jsx        #       7-class emotion breakdown
│               ├── DocumentSearch.jsx         #       BM25 / TF-IDF search UI
│               ├── DataInsights.jsx           #       Heatmaps, distributions
│               ├── BM25DeepDive.jsx           #       BM25 algorithm explainer
│               ├── MethodComparison.jsx       #       BM25 vs TF-IDF comparison
│               └── Recommendations.jsx        #       Post recommendations
│
│── ─── Sentiment Plots (Legacy) ───────────────────────────────────────
│
├── sentiment_plots/                          # Standalone sentiment visualization scripts
│   ├── emotion_dashboard.py                  #   Streamlit emotion dashboard
│   ├── plot_emotion_summary.py               #   Static plot generation
│   └── emotion_plots/                        #   Generated PNGs (gitignored)
│
│── ─── Scripts & Documentation ────────────────────────────────────────
│
├── scripts/                                  # Dev & maintenance scripts
│   └── convert_notebooks.py                  #   Convert Stage notebooks → local + Colab
│
├── tech/                                     # Tech stack documentation
│   └── STACK.md                              #   Complete technology reference (30+ tools)
│
│── ─── Generated at Runtime (gitignored) ──────────────────────────────
│
└── intermediate_data/                        # Pipeline outputs (created when you run stages)
    ├── PostVault.csv                         #   Generated by Stage 1
    ├── CommentVault.csv                      #   Generated by Stage 1
    ├── *_normalized.csv                      #   Generated by Stages 3–5
    ├── *.npz / *.npy / *.joblib             #   Generated by Stage 6
    └── *_labels.csv                          #   Generated by Stage 7
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm** (for the dashboard)
- **GPU (optional)**: NVIDIA GPU with CUDA for faster model inference

### Option A: Local Python (with your own GPU)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Latent.git
cd Latent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up Reddit API credentials for incremental scraping
cat > .env << EOF
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
EOF

# 5. Run the pipeline (see Pipeline Stages below)
```

### Option B: Google Colab (free GPU)

1. Open any notebook from the `notebooks/colab/` directory
2. Click **"Open in Colab"** badge at the top of each notebook
3. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
4. Run cells sequentially — each notebook auto-installs its dependencies

> **Note**: Colab notebooks include `!pip install` cells and mount Google Drive for data persistence between sessions.

---

## 🔧 Pipeline Stages

### Stage 0 — Data Scraping

**Historical scraping** via [Arctic Shift API](https://arctic-shift.photon-reddit.com/):
```bash
python data_scrape/scrape.py --subreddit singapore --year 2025
```
- `--year`: Year to scrape (default: `2025`)
- `--download-media`: Also download media files

**Incremental scraping** via [PRAW](https://praw.readthedocs.io/) (designed for cron jobs):
```bash
python data_scrape/scrape_incremental.py --subreddit singapore --limit 1000
```
- `--limit`: Max posts to fetch (max 1000)
- `--output-dir`: Output directory for CSV files

### Stage 1 — Data Collection & Cleaning

Cleans raw Reddit data by removing noise (URLs, mentions, bots, duplicates, deleted posts) and normalizing text (punctuation, contractions, emojis).

**Output**: `intermediate_data/PostVault.csv`, `intermediate_data/CommentVault.csv`

### Stage 2 — POS and NER Tagging

Applies Part-of-Speech and Named Entity Recognition tagging using spaCy for downstream linguistic analysis.

### Stage 3 — Singlish Normalisation

Standardizes Singlish expressions using a custom dictionary (`utilities/singlish_dictionary.json`) to reduce lexical variation.

### Stage 4 — Singlish → English Conversion

Converts remaining Singlish terms to standard English. Tracks the number of converted terms per post/comment in a `singlish_count` column.

### Stage 5 — Common Text Normalisation

Applies slang expansion, spelling correction, stop word removal, and lemmatization. Produces the final clean text columns used for all downstream modelling.

### Stage 6 — Vector Space Model & Inverted Index

Builds TF-IDF, BM25, and Sentence-BERT representations for posts and comments. Constructs an inverted index for fast retrieval.

**Outputs**: `.npz` matrices, `.joblib` models, `.npy` BERT embeddings, `.json` inverted indices

### Stage 7 — Sentiment & Emotion Analysis

Classifies posts using `j-hartmann/emotion-english-distilroberta-base` into 7 emotion categories: anger, disgust, fear, joy, neutral, sadness, surprise.

```bash
python emotion_inference.py --input intermediate_data/PostVault.csv --batch-size 16
```

### Stage 8 — Topic Clustering & Visualization

Reduces feature dimensionality with SVD, selects optimal cluster count via Silhouette Score, and runs K-Means clustering. Visualizes clusters with t-SNE and word clouds.

**Output**: `tfidf_cluster`, `bm25_cluster`, `bert_cluster` columns added to the data.

### Stage 9 — Document Search & Recommendation

Implements a search engine and recommendation system using TF-IDF, BM25, and BERT embeddings, with heuristic ranking (title weighting, recency, upvotes, comment count).

---

## 📊 Analytics Dashboard

The project includes a **full-stack analytics dashboard** built with Flask (backend) and React + Chakra UI (frontend).

### Running the Dashboard

```bash
cd dashboard-ui

# Install dependencies
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# Start both servers
make dev
# Or manually:
# Terminal 1: python app.py          (Flask API on port 5000)
# Terminal 2: cd dashboard && npm run dev  (React on port 5173)
```

Open **http://localhost:5173** in your browser.

### Dashboard Features

| Page | Description |
|---|---|
| **Topic Overview** | Global KPIs, topic distribution, word clouds, engagement scatter |
| **Topic Deep Dive** | Per-cluster analysis: top posts, keywords, controversies |
| **Timeline** | Stacked area chart of topic volume over time |
| **Emotion Analysis** | 7-class emotion breakdowns by time, day, flair, engagement |
| **Document Search** | BM25 / TF-IDF search with relevance scoring |
| **Data Insights** | Activity heatmaps, score distribution, Singlish usage |

---

## 📡 Data Sources

| Source | Method | Description |
|---|---|---|
| [Arctic Shift API](https://arctic-shift.photon-reddit.com/) | `data_scrape/scrape.py` | Historical Reddit archive — bulk download by year/month |
| [Reddit API (PRAW)](https://praw.readthedocs.io/) | `data_scrape/scrape_incremental.py` | Live Reddit API — incremental scraping (max 1000 posts) |

### How to Get the Data

1. **Historical data**: Run `python data_scrape/scrape.py --subreddit singapore --year 2025`. This hits the Arctic Shift API (no API key required) and saves posts + comments as CSV files.

2. **Incremental data**: Create a Reddit app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps), save credentials to `.env`, then run `python data_scrape/scrape_incremental.py`.

3. **Pre-processed data**: The pipeline stages generate intermediate CSV files in `intermediate_data/`. These are **not committed to the repository** due to size. Re-run the pipeline to regenerate them.

---

## 🛠️ Tech Stack

See [`tech/STACK.md`](tech/STACK.md) for the complete technology reference. Key technologies:

- **NLP**: spaCy, NLTK, Hugging Face Transformers, Sentence-BERT
- **ML**: scikit-learn (TF-IDF, K-Means, SVD), BM25, PyTorch
- **Frontend**: React 19, Vite 8, Chakra UI 3, Recharts, Plotly.js
- **Backend**: Flask 3, Pandas, NumPy
- **Data**: Arctic Shift API, Reddit API (PRAW)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
