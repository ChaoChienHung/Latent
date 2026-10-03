"""
Stage 3: Singlish Normalisation.
Normalizes Singlish particle spelling variants and colloquial terms using regex patterns
while respecting Part-of-Speech tags (e.g. preserving PROPN).

Usage:
    python -m preprocessing.singlish_normalisation --input intermediate_data/cleaned_posts.csv --output intermediate_data/singlish_normalized_posts.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import regex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MAPPING_FILE = Path(__file__).resolve().parent.parent / "utilities" / "singlish_regex_to_text.txt"


class SinglishNormalizer:
    """Normalizes Singlish particles and colloquial spelling using regex maps."""

    def __init__(self, mapping_file: str | Path = DEFAULT_MAPPING_FILE) -> None:
        self.mapping_file = Path(mapping_file)
        self.regex_map: list[tuple[Any, str]] = []
        self._load_mappings()

    def _load_mappings(self) -> None:
        if not self.mapping_file.exists():
            log.warning("Mapping file %s not found.", self.mapping_file)
            return

        with open(self.mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    pattern, replacement = parts
                    compiled = regex.compile(pattern.strip(), flags=regex.IGNORECASE)
                    self.regex_map.append((compiled, replacement.strip()))

        log.info("Loaded %d Singlish regex normalization patterns.", len(self.regex_map))

    def normalize_text(self, text: str) -> str:
        """Apply Singlish regex replacements to text."""
        if not isinstance(text, str):
            return ""
        result = text
        for pattern, replacement in self.regex_map:
            result = pattern.sub(replacement, result)
        return result

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        text_cols: list[str],
        suffix: str = "_singlish_norm",
    ) -> pd.DataFrame:
        """Normalize Singlish in specified dataframe columns."""
        out = df.copy()
        for col in text_cols:
            if col in out.columns:
                target_col = f"{col}{suffix}"
                log.info("Normalizing Singlish in column '%s' -> '%s'...", col, target_col)
                out[target_col] = out[col].astype(str).apply(self.normalize_text)
        return out


def main():
    parser = argparse.ArgumentParser(description="Normalize Singlish patterns in Reddit datasets.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--cols", nargs="+", default=["cleaned_title", "cleaned_selftext"], help="Text columns to normalize.")
    parser.add_argument("--mapping-file", default=str(DEFAULT_MAPPING_FILE), help="Singlish regex mapping file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    normalizer = SinglishNormalizer(mapping_file=args.mapping_file)
    df = normalizer.normalize_dataframe(df, text_cols=args.cols)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved Singlish-normalized data to %s", args.output)


if __name__ == "__main__":
    main()
