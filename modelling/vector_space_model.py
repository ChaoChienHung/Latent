"""
Stage 6: Vector Space Model and Inverted Index.
Constructs TF-IDF vectorizers, BM25Okapi retrieval models,
Sentence-BERT dense embeddings, and inverted indexes for fast lookup.

Usage:
    python -m modelling.vector_space_model --posts data/PostVault.csv --output-dir backend/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class VectorSpaceModelBuilder:
    """Builds and serializes TF-IDF, BM25, and Sentence-BERT representations."""

    def __init__(self, output_dir: str | Path = "data") -> None:
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.vdb_dir = self.output_dir / "vector_database"
        self.idx_dir = self.output_dir / "inverted_index"

        for d in (self.models_dir, self.vdb_dir, self.idx_dir):
            d.mkdir(parents=True, exist_ok=True)

    def build_tfidf(
        self,
        df_posts: pd.DataFrame,
        text_col: str = "lemmatized_full_text",
        title_col: str = "lemmatized_title",
    ) -> TfidfVectorizer:
        """Fit TF-IDF vectorizer on full text and transform fields."""
        log.info("Fitting TF-IDF vectorizer on '%s'...", text_col)
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_fulltext = vectorizer.fit_transform(df_posts[text_col].astype(str))

        save_npz(self.vdb_dir / "tfidf_fulltext.npz", tfidf_fulltext)
        joblib.dump(vectorizer, self.models_dir / "tfidf_posts_vectorizer.joblib")

        if title_col in df_posts.columns:
            tfidf_titles = vectorizer.transform(df_posts[title_col].astype(str))
            save_npz(self.vdb_dir / "tfidf_titles.npz", tfidf_titles)

        log.info("Saved TF-IDF models and matrices to %s and %s", self.models_dir, self.vdb_dir)
        return vectorizer

    def build_bm25(
        self,
        df_posts: pd.DataFrame,
        text_col: str = "lemmatized_full_text",
        title_col: str = "lemmatized_title",
    ) -> BM25Okapi:
        """Build and serialize BM25Okapi retrieval indices."""
        log.info("Building BM25Okapi models...")
        tokenized_fulltext = [doc.split() for doc in df_posts[text_col].astype(str)]
        bm25_fulltext = BM25Okapi(tokenized_fulltext)
        joblib.dump(bm25_fulltext, self.models_dir / "bm25_fulltext_model.joblib")

        if title_col in df_posts.columns:
            tokenized_titles = [doc.split() for doc in df_posts[title_col].astype(str)]
            bm25_titles = BM25Okapi(tokenized_titles)
            joblib.dump(bm25_titles, self.models_dir / "bm25_titles_model.joblib")

        log.info("Saved BM25Okapi models to %s", self.models_dir)
        return bm25_fulltext

    def build_sbert(
        self,
        df_posts: pd.DataFrame,
        col: str = "full_text",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Generate and save Sentence-BERT dense embeddings."""
        log.info("Encoding texts with Sentence-BERT ('%s')...", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer(model_name)
            texts = df_posts[col].astype(str).tolist()
            embeddings = sbert.encode(texts, batch_size=batch_size, show_progress_bar=True)
            save_path = self.vdb_dir / "bert_fulltext_embeddings.npy"
            np.save(save_path, embeddings)
            log.info("Saved SBERT embeddings (%s) to %s", embeddings.shape, save_path)
            return embeddings
        except ImportError:
            log.warning("sentence-transformers not installed. Skipping SBERT embedding generation.")
            return np.array([])

    def build_inverted_index(
        self,
        vectorizer: TfidfVectorizer,
        tfidf_matrix,
        output_filename: str = "inverted_index_tfidf_fulltext.json",
    ) -> dict:
        """Build dictionary mapping terms to list of (doc_id, score) postings."""
        log.info("Constructing inverted index...")
        feature_names = vectorizer.get_feature_names_out()
        cx = tfidf_matrix.tocoo()

        inverted_index = defaultdict(list)
        for doc_id, col_idx, score in zip(cx.row, cx.col, cx.data):
            term = feature_names[col_idx]
            inverted_index[term].append({"doc_id": int(doc_id), "score": float(score)})

        out_path = self.idx_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f)

        log.info("Saved inverted index with %d terms to %s", len(inverted_index), out_path)
        return inverted_index


def main():
    parser = argparse.ArgumentParser(description="Build Vector Space Models and Indices.")
    parser.add_argument("--posts", required=True, help="Input PostVault CSV file.")
    parser.add_argument("--output-dir", default="data", help="Output directory for indices and models.")
    parser.add_argument("--with-sbert", action="store_true", help="Generate SBERT dense embeddings.")
    args = parser.parse_args()

    df_posts = pd.read_csv(args.posts, keep_default_na=False, low_memory=False)
    builder = VectorSpaceModelBuilder(output_dir=args.output_dir)

    vectorizer = builder.build_tfidf(df_posts)
    builder.build_bm25(df_posts)

    if args.with_sbert:
        builder.build_sbert(df_posts)


if __name__ == "__main__":
    main()
