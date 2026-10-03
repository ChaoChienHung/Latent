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

## 📂 Project Structure

```
Latent/
├── README.md                              # This file
├── TODO.md                                # Task tracker
├── .gitignore                             # Git exclusions
├── requirements.txt                       # Python dependencies (full pipeline)
│
├── data_scrape/                           # Reddit data collection scripts
│   ├── scrape.py                          #   Historical scraping (Arctic Shift API)
│   └── scrape_incremental.py              #   Incremental scraping (PRAW / Reddit API)
│
├── notebooks/                             # Pipeline notebooks
│   ├── local/                             #   Jupyter versions (local GPU)
│   │   ├── Stage_1_Data_Collection.ipynb
│   │   ├── Stage_2_POS_NER_Tagging.ipynb
│   │   └── ...
│   └── colab/                             #   Google Colab versions (free GPU)
│       ├── Stage_1_Data_Collection.ipynb
│       ├── Stage_2_POS_NER_Tagging.ipynb
│       └── ...
│
├── utilities/                             # Shared helper modules
│   ├── pp_class.py                        #   RedditPreprocessor class
│   ├── singlish_dictionary.json           #   Singlish → English mappings
│   ├── singlish_regex_to_text.txt         #   Singlish regex patterns
│   └── slang_dictionary.csv               #   Internet slang expansions
│
├── dashboard-ui/                          # Full-stack analytics dashboard
│   ├── app.py                             #   Flask API backend
│   ├── dashboard/                         #   React + Vite frontend
│   ├── data/                              #   Processed CSV data (gitignored)
│   ├── models/                            #   Trained ML models (gitignored)
│   └── requirements.txt                   #   Backend Python dependencies
│
├── tech/                                  # Tech stack documentation
│   └── STACK.md                           #   Complete technology reference
│
└── intermediate_data/                     # Pipeline outputs (gitignored)
    ├── PostVault.csv                      #   Generated by Stage 1
    ├── CommentVault.csv                   #   Generated by Stage 1
    └── ...                                #   Various intermediate CSVs
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
