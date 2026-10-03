"""
Stage 8: Topic Clustering and Visualization.
Performs TruncatedSVD dimensionality reduction, K-Means clustering,
silhouette score evaluation, t-SNE manifold learning, and cluster keyword extraction.

Usage:
    python -m modelling.clustering --tfidf backend/models/tfidf_fulltext.npz --k 7 --output-dir artifacts/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class TopicClusterer:
    """Clusterer for document vectors with dimensionality reduction and evaluation."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def reduce_dimensions(self, matrix: Any, n_components: int = 150) -> np.ndarray:
        """Apply TruncatedSVD to reduce feature dimensionality."""
        log.info("Reducing matrix dimensions to %d components via TruncatedSVD...", n_components)
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        reduced = svd.fit_transform(matrix)
        explained = svd.explained_variance_ratio_.sum()
        log.info("Cumulative explained variance: %.2f%%", explained * 100)
        return reduced

    def evaluate_k(
        self,
        data: np.ndarray,
        k_range: range = range(3, 12),
        sample_size: int = 5000,
    ) -> dict[int, float]:
        """Compute silhouette scores over a range of cluster counts."""
        scores: dict[int, float] = {}
        log.info("Evaluating silhouette scores across k in %s...", list(k_range))

        sub_data = data
        if len(data) > sample_size:
            idx = np.random.choice(len(data), sample_size, replace=False)
            sub_data = data[idx]

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(sub_data)
            score = silhouette_score(sub_data, labels)
            scores[k] = float(score)
            log.info("  k=%d | Silhouette Score: %.4f", k, score)

        return scores

    def fit_kmeans(self, data: np.ndarray, n_clusters: int = 7) -> np.ndarray:
        """Fit KMeans and return cluster labels."""
        log.info("Fitting KMeans with k=%d clusters...", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        return labels

    def compute_tsne(
        self,
        data: np.ndarray,
        n_components: int = 3,
        sample_size: int = 2000,
    ) -> np.ndarray:
        """Compute t-SNE projections for 2D/3D visualization."""
        sub_data = data
        if len(data) > sample_size:
            idx = np.random.choice(len(data), sample_size, replace=False)
            sub_data = data[idx]

        log.info("Running t-SNE projection to %d dimensions on %d points...", n_components, len(sub_data))
        tsne = TSNE(
            n_components=n_components,
            random_state=self.random_state,
            perplexity=30.0,
            n_iter=1000,
        )
        return tsne.fit_transform(sub_data)

    def extract_cluster_keywords(
        self,
        tfidf_matrix: Any,
        feature_names: list[str],
        labels: np.ndarray,
        top_n: int = 10,
    ) -> dict[int, list[str]]:
        """Extract top TF-IDF keywords characterizing each cluster."""
        keywords: dict[int, list[str]] = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:
                continue
            idx = np.where(labels == label)[0]
            if len(idx) == 0:
                continue
            cluster_mean = tfidf_matrix[idx].mean(axis=0)
            cluster_mean = np.asarray(cluster_mean).flatten()
            top_indices = cluster_mean.argsort()[::-1][:top_n]
            keywords[int(label)] = [feature_names[i] for i in top_indices]

        return keywords


def main():
    parser = argparse.ArgumentParser(description="Cluster document vectors into latent topics.")
    parser.add_argument("--matrix", required=True, help="Input .npz matrix or .npy embeddings.")
    parser.add_argument("--k", type=int, default=7, help="Number of clusters.")
    parser.add_argument("--svd-components", type=int, default=150, help="SVD components for reduction.")
    parser.add_argument("--output-labels", default="intermediate_data/cluster_labels.npy", help="Path to save cluster labels.")
    args = parser.parse_args()

    clusterer = TopicClusterer()

    if args.matrix.endswith(".npz"):
        data = load_npz(args.matrix)
        if args.svd_components > 0:
            data = clusterer.reduce_dimensions(data, n_components=args.svd_components)
    else:
        data = np.load(args.matrix)

    labels = clusterer.fit_kmeans(data, n_clusters=args.k)
    Path(args.output_labels).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_labels, labels)
    log.info("Saved %d cluster labels to %s", len(labels), args.output_labels)


if __name__ == "__main__":
    main()
