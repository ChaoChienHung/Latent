"""
Stage 9: Document Search and Recommendation Engine.
Implements CentroidTreeSearch, multi-strategy hybrid SearchEngine
(Inverted Index + TF-IDF + BM25 + SBERT), and content-based RecommendationSystem.

Usage:
    python -m search.document_search --query "housing grants in Singapore"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class CentroidTreeSearch:
    """Hierarchical k-means centroid tree for approximate nearest neighbor search."""

    class Node:
        """A cluster node in the centroid search tree."""

        def __init__(self, centroid: np.ndarray | None = None, indices: np.ndarray | None = None, level: int = 0):
            self.centroid = centroid
            self.indices = indices
            self.level = level
            self.children: list[CentroidTreeSearch.Node] = []
            self.centroid_matrix: np.ndarray | None = None

    def __init__(
        self,
        data: np.ndarray,
        levels: int = 2,
        branching_mode: str = "balanced",
        fixed_k: int = 8,
        metric: str = "cosine",
    ) -> None:
        self.metric = metric
        self.data = self._normalize(data) if metric == "cosine" else data
        self.N, self.D = data.shape
        self.levels = levels
        self.branching_mode = branching_mode
        self.fixed_k = fixed_k
        self.root: CentroidTreeSearch.Node | None = None
        self._build_tree()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-10)

    def _choose_k(self, subset_size: int) -> int:
        if self.branching_mode == "balanced":
            k = int(np.ceil(subset_size ** (1 / max(1, self.levels))))
            return max(2, min(50, k))
        return min(self.fixed_k, subset_size)

    def _build_tree(self) -> None:
        indices = np.arange(self.N)
        self.root = self._build_node(indices, 0)

    def _build_node(self, indices: np.ndarray, level: int) -> CentroidTreeSearch.Node:
        from sklearn.cluster import KMeans

        node_centroid = np.mean(self.data[indices], axis=0)
        node = self.Node(node_centroid, indices, level)

        if level >= self.levels or len(indices) <= self.fixed_k:
            return node

        k = self._choose_k(len(indices))
        if k < 2 or len(indices) < k:
            return node

        subset = self.data[indices]
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42).fit(subset)

        child_centroids = []
        for i in range(k):
            child_idx = indices[kmeans.labels_ == i]
            if len(child_idx) == 0:
                continue
            child = self._build_node(child_idx, level + 1)
            child.centroid = kmeans.cluster_centers_[i]
            node.children.append(child)
            child_centroids.append(child.centroid)

        if node.children:
            node.centroid_matrix = np.vstack(child_centroids)

        return node

    def _compute_score(self, matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return matrix @ query
        return -np.linalg.norm(matrix - query, axis=1)

    def search(self, query: np.ndarray, top_k: int = 5, beam_width: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform beam search over centroid tree."""
        if self.metric == "cosine":
            query = query / (np.linalg.norm(query) + 1e-10)

        beam = [self.root]
        for _ in range(self.levels):
            candidates = []
            for node in beam:
                if not node or not node.children or node.centroid_matrix is None:
                    continue
                scores = self._compute_score(node.centroid_matrix, query)
                for i, child in enumerate(node.children):
                    candidates.append((child, scores[i]))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [c[0] for c in candidates[:beam_width]]

        candidate_indices = np.unique(np.concatenate([n.indices for n in beam if n.indices is not None and n.indices.size > 0]))
        if len(candidate_indices) == 0:
            return np.array([]), np.array([]), np.array([])

        candidate_vectors = self.data[candidate_indices]
        final_scores = self._compute_score(candidate_vectors, query)
        top_idx = np.argsort(-final_scores)[:top_k]

        return candidate_vectors[top_idx], candidate_indices[top_idx], final_scores[top_idx]


class SearchEngine:
    """Unified hybrid search engine combining multiple retrieval signals and business rules."""

    def __init__(self, df: pd.DataFrame | None = None, config: dict[str, float] | None = None, **kwargs) -> None:
        self.df = df
        cfg = config or {}
        self.title_weight = cfg.get("title_weight", 1.0)
        self.fulltext_weight = cfg.get("fulltext_weight", 1.0)
        self.tfidf_weight = cfg.get("tfidf_weight", 1.5)
        self.bm25_weight = cfg.get("bm25_weight", 1.5)
        self.bert_weight = cfg.get("bert_weight", 3.0)

        self.tfidf_vectorizer = kwargs.get("tfidf_vectorizer")
        self.tfidf_fulltext_matrix = kwargs.get("tfidf_fulltext_matrix")
        self.bm25_fulltext_model = kwargs.get("bm25_fulltext_model")
        self.bert_model = kwargs.get("bert_model")
        self.bert_fulltext_matrix = kwargs.get("bert_fulltext_matrix")

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max == s_min:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def search_tfidf(self, query: str) -> np.ndarray:
        """Compute TF-IDF cosine similarity scores."""
        if self.tfidf_vectorizer is None or self.tfidf_fulltext_matrix is None:
            return np.zeros(len(self.df) if self.df is not None else 0)
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_fulltext_matrix).flatten()
        return sims

    def search_bm25(self, query: str) -> np.ndarray:
        """Compute BM25Okapi relevance scores."""
        if self.bm25_fulltext_model is None:
            return np.zeros(len(self.df) if self.df is not None else 0)
        tokenized_query = query.split()
        scores = np.array(self.bm25_fulltext_model.get_scores(tokenized_query))
        return scores

    def search_bert(self, query: str) -> np.ndarray:
        """Compute SBERT dense embedding cosine similarities."""
        if self.bert_model is None or self.bert_fulltext_matrix is None:
            return np.zeros(len(self.df) if self.df is not None else 0)
        q_emb = self.bert_model.encode([query])
        sims = cosine_similarity(q_emb, self.bert_fulltext_matrix).flatten()
        return sims

    def hybrid_search(self, query: str, top_k: int = 10, apply_rules: bool = True) -> list[dict[str, Any]]:
        """Combine all search models with business-rule re-ranking."""
        if not query.strip() or self.df is None:
            return []

        tfidf_scores = self._minmax_normalize(self.search_tfidf(query))
        bm25_scores = self._minmax_normalize(self.search_bm25(query))
        bert_scores = self._minmax_normalize(self.search_bert(query))

        ensemble_scores = (
            self.tfidf_weight * tfidf_scores +
            self.bm25_weight * bm25_scores +
            self.bert_weight * bert_scores
        )

        results = []
        now = int(time.time())

        for idx, base_score in enumerate(ensemble_scores):
            if base_score <= 0.001:
                continue

            score = float(base_score)
            if apply_rules:
                post = self.df.iloc[idx]
                created_utc = post.get("created_utc", now)
                days_old = max(0, now - created_utc) / (24 * 3600)
                time_decay = 1.0 / (1.0 + (days_old ** 0.5))
                engagement_boost = (
                    np.log1p(float(post.get("score", 0))) * 0.1 +
                    np.log1p(float(post.get("num_comments", 0))) * 0.05 +
                    float(post.get("upvote_ratio", 0.5))
                )
                score = score * time_decay * engagement_boost

            results.append({"index": idx, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        enriched = []
        for r in top_results:
            post = self.df.iloc[r["index"]].to_dict()
            post["search_score"] = r["score"]
            enriched.append(post)

        return enriched


class RecommendationSystem:
    """Content-based recommendation system finding similar documents via cosine similarity."""

    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame) -> None:
        self.embeddings = embeddings
        self.df = df

    def recommend_similar_posts(self, post_index: int, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k most similar posts to a given post index."""
        if post_index < 0 or post_index >= len(self.embeddings):
            return []

        target_vec = self.embeddings[post_index].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.embeddings).flatten()

        sims[post_index] = -1.0  # exclude the post itself
        top_indices = np.argsort(-sims)[:top_k]

        recommendations = []
        for idx in top_indices:
            item = self.df.iloc[idx].to_dict()
            item["similarity_score"] = float(sims[idx])
            recommendations.append(item)

        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Query Latent document search engine.")
    parser.add_argument("--query", required=True, help="Search query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    print(f"Executing search for query: '{args.query}' (top {args.top_k} results)")
    print("Initialize SearchEngine with models from backend/models to perform real-time retrieval.")


if __name__ == "__main__":
    main()
