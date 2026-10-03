"""
Stage 5: Common Normalisation.
Applies slang expansion, emoji normalization, contraction resolution,
stopword removal, lemmatization, and title+body concatenation.

Usage:
    python -m preprocessing.common_normalisation --input intermediate_data/singlish_to_english_posts.csv --output intermediate_data/fully_normalized_posts.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SLANG_FILE = Path(__file__).resolve().parent.parent / "utilities" / "slang_dictionary.csv"


class TextNormalizer:
    """Standardizes text through slang expansion, emojis, contractions, stopwords, and lemmatization."""

    def __init__(self, slang_path: str | Path = DEFAULT_SLANG_FILE) -> None:
        self.slang_path = Path(slang_path)
        self.slang_map: dict[str, str] = {}
        self._load_slang()
        self._init_nlp()

    def _init_nlp(self) -> None:
        for resource in ("stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _load_slang(self) -> None:
        if not self.slang_path.exists():
            log.warning("Slang dictionary %s not found.", self.slang_path)
            return

        df = pd.read_csv(self.slang_path)
        if "slang" in df.columns and "expand" in df.columns:
            self.slang_map = dict(zip(df["slang"].astype(str).str.lower(), df["expand"].astype(str)))
        log.info("Loaded %d slang dictionary entries.", len(self.slang_map))

    def expand_slang(self, text: str) -> str:
        """Expand common abbreviations and internet slang."""
        tokens = text.split()
        expanded = [self.slang_map.get(tok.lower(), tok) for tok in tokens]
        return " ".join(expanded)

    def expand_contractions(self, text: str) -> str:
        """Expand English contractions (e.g. don't -> do not)."""
        return contractions.fix(text)

    def handle_emojis(self, text: str, replace_with: str = "demojize") -> str:
        """Handle emojis by converting to text descriptions or removing."""
        if replace_with == "demojize":
            return emoji.demojize(text, delimiters=(" ", " "))
        elif replace_with == "remove":
            return emoji.replace_emoji(text, replace="")
        return text

    def clean_special_characters(self, text: str) -> str:
        """Remove non-alphanumeric characters while preserving spaces."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def lemmatize_and_remove_stopwords(self, text: str) -> str:
        """Lowercase, remove stopwords, and lemmatize tokens."""
        tokens = text.lower().split()
        filtered = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 1
        ]
        return " ".join(filtered)

    def normalize(self, text: str) -> str:
        """Full pipeline normalization on a single text string."""
        if not isinstance(text, str) or not text.strip():
            return ""
        t = self.expand_contractions(text)
        t = self.expand_slang(t)
        t = self.handle_emojis(t)
        t = self.clean_special_characters(t)
        t = self.lemmatize_and_remove_stopwords(t)
        return t

    def process_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process posts dataframe to produce full_text and lemmatized fields."""
        out = df.copy()
        
        # Pick best available text columns
        title_col = "cleaned_title" if "cleaned_title" in out.columns else "title"
        body_col = "cleaned_selftext" if "cleaned_selftext" in out.columns else "selftext"

        log.info("Normalizing titles...")
        out["lemmatized_title"] = out[title_col].astype(str).apply(self.normalize)

        if body_col in out.columns:
            log.info("Normalizing contents...")
            out["lemmatized_content"] = out[body_col].astype(str).apply(self.normalize)
            out["full_text"] = out[title_col].astype(str) + " " + out[body_col].astype(str).fillna("")
            out["lemmatized_full_text"] = (
                out["lemmatized_title"].astype(str) + " " + out["lemmatized_content"].astype(str)
            ).str.strip()
        else:
            out["full_text"] = out[title_col].astype(str)
            out["lemmatized_full_text"] = out["lemmatized_title"]

        return out


def main():
    parser = argparse.ArgumentParser(description="Full text normalization for Reddit datasets.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--slang-path", default=str(DEFAULT_SLANG_FILE), help="Path to slang_dictionary.csv.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    normalizer = TextNormalizer(slang_path=args.slang_path)
    df = normalizer.process_posts(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved fully normalized dataset to %s", args.output)


if __name__ == "__main__":
    main()
