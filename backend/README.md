# Latent Backend API

Flask REST API providing analytical, clustering, emotion, and search endpoints for the Latent dashboard.

## 🛠️ Overview

- **Server**: Flask 3.0 + Flask-CORS
- **NLP Models**: Pre-loaded BM25Okapi, TF-IDF vectorizers, and RoBERTa emotion probability aggregations.
- **Port**: Runs on `http://127.0.0.1:5000` (proxied by the frontend Vite server).

## 📁 Directory Structure

```text
backend/
├── app.py                # Flask API server & endpoints
├── data/                 # Runtime CSV datasets (gitignored)
├── models/               # Serialized model artifacts (gitignored)
├── documentation.md      # API specification & endpoint documentation
├── pyproject.toml        # uv package configuration
├── requirements.txt      # pip requirements
└── Makefile              # Backend lifecycle targets
```

## 🚀 Getting Started

### Using uv
```bash
cd backend
uv sync
uv run python app.py
```

### Using pip
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 📡 API Endpoints

See [`documentation.md`](documentation.md) for complete details on the 20 available endpoints:
- `GET /api/overview` — High-level summary metrics
- `GET /api/timeline` — Post volume & engagement over time
- `GET /api/topics` — K-Means topic clusters
- `GET /api/topics/<id>` — Topic deep dive and word clouds
- `GET /api/emotion/summary` — 7-class emotion distributions
- `GET /api/search?q=<query>` — Hybrid search query execution
- `GET /api/recommendations/<post_id>` — Content-based post recommendations
