# Tech Stack — Latent

Comprehensive record of all technologies, libraries, and tools used across the project.

---

## Data Collection

| Technology | Purpose | Notes |
|---|---|---|
| [Arctic Shift API](https://arctic-shift.photon-reddit.com/) | Historical Reddit data scraping | Bulk scrape by year/month for r/Singapore |
| [PRAW](https://praw.readthedocs.io/) | Incremental Reddit scraping | Real-time scraping via Reddit API, supports cron jobs |
| Python `requests` | HTTP client for Arctic Shift | With retry logic and exponential backoff |
| `pandas` | Data loading and export | CSV I/O throughout the pipeline |

## NLP & Text Processing

| Technology | Purpose | Notes |
|---|---|---|
| [spaCy](https://spacy.io/) | POS tagging, NER, lemmatization | `en_core_web_sm` / `en_core_web_trf` models |
| [NLTK](https://www.nltk.org/) | Tokenization, stop words | Used for stop word removal and sentence tokenization |
| [ftfy](https://github.com/rspeer/python-ftfy) | Text encoding repair | Fixes Unicode issues in scraped data |
| [emoji](https://pypi.org/project/emoji/) | Emoji handling | Demojize and normalize emoji in posts |
| [langdetect](https://pypi.org/project/langdetect/) | Language detection | Filter non-English posts |
| Custom Singlish Dictionary | Singlish normalisation | `utilities/singlish_dictionary.json` — hand-curated |
| Custom Slang Dictionary | Slang expansion | `utilities/slang_dictionary.csv` — expanded from internet slang lists |
| Regex patterns | Singlish regex normalization | `utilities/singlish_regex_to_text.txt` |

## Machine Learning & Modelling

| Technology | Purpose | Notes |
|---|---|---|
| [scikit-learn](https://scikit-learn.org/) | TF-IDF, K-Means, SVD | Vectorization, clustering, dimensionality reduction |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 ranking | Probabilistic document retrieval |
| [Sentence-BERT](https://www.sbert.net/) | Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/) | Emotion classification | `j-hartmann/emotion-english-distilroberta-base` (7-class) |
| [PyTorch](https://pytorch.org/) | Deep learning runtime | Backend for Transformers and Sentence-BERT |
| `joblib` | Model serialization | Save/load BM25 and TF-IDF models |
| `numpy` / `scipy` | Numerical operations | Sparse matrices, cosine similarity |

## Visualization & Dashboard

| Technology | Purpose | Notes |
|---|---|---|
| [React](https://react.dev/) 19.x | Frontend framework | SPA for the analytics dashboard |
| [Vite](https://vite.dev/) 8.x | Build tool & dev server | Fast HMR, API proxy to Flask backend |
| [Chakra UI](https://www.chakra-ui.com/) 3.x | Component library | Theming, responsive layout |
| [Recharts](https://recharts.org/) | Charts (bar, pie, line, area) | React-native charting library |
| [Plotly.js](https://plotly.com/javascript/) / [react-plotly.js](https://github.com/plotly/react-plotly.js) | Interactive plots | Heatmaps, scatter plots |
| [d3-cloud](https://github.com/jasondavies/d3-cloud) | Word cloud layout | Keyword visualization per cluster |
| [Framer Motion](https://www.framer.com/motion/) | Animations | Page transitions, micro-interactions |
| [Flask](https://flask.palletsprojects.com/) 3.x | Backend API server | REST API serving pre-computed analytics |
| [Flask-CORS](https://flask-cors.readthedocs.io/) | CORS handling | Cross-origin support for development |
| [Streamlit](https://streamlit.io/) | Quick dashboards | Emotion analysis standalone dashboard |

## Infrastructure & Tooling

| Technology | Purpose | Notes |
|---|---|---|
| Python 3.11+ | Runtime | Primary language for pipeline and backend |
| Node.js 18+ | Frontend runtime | For React dashboard build and dev |
| [uv](https://docs.astral.sh/uv/) | Python package manager | Fast, lockfile-based dependency management |
| npm | JS package manager | Frontend dependencies |
| Git | Version control | Repository management |
| Make | Task runner | `Makefile` for common dev commands |
| Jupyter Notebook | Interactive development | Pipeline exploration and experimentation |
| Google Colab | Cloud GPU | Free GPU access for model inference |

## Data Sources

| Source | Type | Description |
|---|---|---|
| [r/Singapore](https://www.reddit.com/r/singapore/) | Reddit subreddit | Primary data source — posts and comments |
| [Arctic Shift](https://arctic-shift.photon-reddit.com/) | Archive API | Historical Reddit data (posts + comments by year) |
| Reddit API (via PRAW) | Live API | Incremental data scraping for recent posts |

---

## Key Models Used

### Emotion Classification
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Architecture**: DistilRoBERTa fine-tuned for 7-class emotion detection
- **Classes**: anger, disgust, fear, joy, neutral, sadness, surprise
- **Source**: [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

### Sentence Embeddings
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: MiniLM fine-tuned for semantic similarity
- **Output**: 384-dimensional dense vectors
- **Source**: [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Topic Clustering
- **Method**: TF-IDF + K-Means (k=20)
- **Dimensionality Reduction**: Truncated SVD
- **Cluster Selection**: Silhouette Score optimization

### Document Retrieval
- **BM25**: Probabilistic term-frequency ranking (Okapi BM25)
- **TF-IDF**: Cosine similarity in TF-IDF vector space
- **BERT**: Cosine similarity in Sentence-BERT embedding space

---

*Last updated: 2026-10-03*
