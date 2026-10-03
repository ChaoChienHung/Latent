"""
Stage 2: Part-of-Speech (POS) and Named Entity Recognition (NER) Tagging.
Extracts linguistic tags and entities from Reddit text using spaCy.

Usage:
    python -m preprocessing.pos_ner_tagging --input intermediate_data/cleaned_posts.csv --output intermediate_data/pos_ner_posts.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class NERTagger:
    """NER tagging using spaCy for named entity recognition."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy is required for NER. Install with: pip install spacy") from exc
        try:
            self.nlp = spacy.load(model)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' not found. Install with: python -m spacy download {model}"
            ) from exc

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""
        doc = self.nlp(str(text))
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def tag_dataframe(self, df: pd.DataFrame, text_col: str, output_col: str = "entities") -> pd.DataFrame:
        """Tag named entities in a dataframe column."""
        out = df.copy()
        log.info("Extracting NER tags for column '%s' (%d rows)...", text_col, len(out))
        out[output_col] = out[text_col].astype(str).apply(self.extract_entities)
        return out


class POSTagger:
    """POS tagging using spaCy for Part-of-Speech identification."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy is required for POS tagging. Install with: pip install spacy") from exc
        try:
            self.nlp = spacy.load(model, disable=["parser", "ner"])
            self.nlp.enable_pipe("tagger")
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' not found. Install with: python -m spacy download {model}"
            ) from exc

    def tag_text(self, text: str) -> list[dict[str, Any]]:
        """Performs POS tagging on a given text."""
        doc = self.nlp(str(text))
        return [
            {"token": tok.text, "lemma": tok.lemma_, "pos": tok.pos_, "tag": tok.tag_}
            for tok in doc
            if not tok.is_space
        ]

    def tag_dataframe(self, df: pd.DataFrame, text_col: str, output_col: str = "pos_tags") -> pd.DataFrame:
        """Tag POS in a dataframe column."""
        out = df.copy()
        log.info("Extracting POS tags for column '%s' (%d rows)...", text_col, len(out))
        out[output_col] = out[text_col].astype(str).apply(self.tag_text)
        return out


def main():
    parser = argparse.ArgumentParser(description="Run POS & NER tagging on Reddit text.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--text-col", default="cleaned_title", help="Text column to tag.")
    parser.add_argument("--ner", action="store_true", help="Perform NER tagging.")
    parser.add_argument("--pos", action="store_true", help="Perform POS tagging.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if args.pos:
        pos_tagger = POSTagger()
        df = pos_tagger.tag_dataframe(df, text_col=args.text_col)
    if args.ner:
        ner_tagger = NERTagger()
        df = ner_tagger.tag_dataframe(df, text_col=args.text_col)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved POS/NER tagged dataset to %s", args.output)


if __name__ == "__main__":
    main()
