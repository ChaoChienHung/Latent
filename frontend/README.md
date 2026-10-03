# Latent Frontend Dashboard

A modern Single-Page Application (SPA) dashboard for exploring Reddit r/singapore NLP analytics, emotion classifications, topic clusters, and hybrid search.

## 🛠️ Tech Stack

- **Framework**: React 19 + Vite 8
- **UI Components**: Chakra UI 3
- **Visualization**: Recharts, Plotly, `d3-cloud`
- **Routing**: React Router DOM 7
- **HTTP Proxy**: Vite dev server proxies `/api/*` to Flask backend on port 5000

## 🚀 Getting Started

### Install Dependencies
```bash
npm install
```

### Start Development Server
```bash
npm run dev
```
The dashboard will be available at `http://localhost:5173`. Make sure the backend Flask API is running on port 5000 (`make backend` from project root).

### Build for Production
```bash
npm run build
```

### Run Linter
```bash
npm run lint
```

## 📄 Key Pages & Views

- **Overview & Timeline**: Dataset metrics, post volume over time, temporal engagement patterns.
- **Emotion Analysis**: RoBERTa 7-class emotion distributions, radar profiles, daily emotion dynamics.
- **Topic Clusters & Deep Dive**: K-Means topic clusters, interactive word clouds, keyword profiles.
- **Document Search**: Hybrid search engine querying TF-IDF, BM25, and dense embeddings.
- **Recommendations**: Content-based post and comment recommendation system.
- **Method Comparison**: Comparative analysis across retrieval methods (TF-IDF vs BM25 vs SBERT).
