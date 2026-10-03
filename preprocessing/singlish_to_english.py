"""
Stage 4: Singlish to English Conversion.
Converts Singlish phrases and terms to standard English using a curated dictionary.

Usage:
    python -m preprocessing.singlish_to_english --input intermediate_data/singlish_normalized_posts.csv --output intermediate_data/singlish_to_english_posts.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DICT_PATH = Path(__file__).resolve().parent.parent / "utilities" / "singlish_dictionary.json"


class SinglishTranslator:
    """Translates Singlish terms to standard English using curated dictionary entries."""

    def __init__(self, dict_path: str | Path = DEFAULT_DICT_PATH) -> None:
        self.dict_path = Path(dict_path)
        self.dictionary: dict[str, str] = {}
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        if not self.dict_path.exists():
            log.warning("Singlish dictionary %s not found.", self.dict_path)
            return

        with open(self.dict_path, "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)

        log.info("Loaded %d Singlish dictionary translations.", len(self.dictionary))

    def translate_text(self, text: str) -> tuple[str, int]:
        """Translate Singlish expressions in text and count replacements."""
        if not isinstance(text, str):
            return "", 0

        converted_text = text
        conversion_count = 0

        for pattern, replacement in self.dictionary.items():
            try:
                new_text, count = re.subn(
                    rf"\b{re.escape(pattern)}\b",
                    replacement,
                    converted_text,
                    flags=re.IGNORECASE,
                )
                if count > 0:
                    converted_text = new_text
                    conversion_count += count
            except re.error:
                continue

        return converted_text, conversion_count

    def translate_dataframe(
        self,
        df: pd.DataFrame,
        text_cols: list[str],
        suffix: str = "_translated",
    ) -> pd.DataFrame:
        """Translate Singlish in dataframe columns."""
        out = df.copy()
        total_conversions = 0

        for col in text_cols:
            if col in out.columns:
                target_col = f"{col}{suffix}"
                log.info("Translating Singlish in column '%s' -> '%s'...", col, target_col)
                results = out[col].astype(str).apply(self.translate_text)
                out[target_col] = [r[0] for r in results]
                col_counts = [r[1] for r in results]
                total_conversions += sum(col_counts)

        log.info("Completed Singlish translation (%d total substitutions made).", total_conversions)
        return out


def main():
    parser = argparse.ArgumentParser(description="Translate Singlish to standard English.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--cols", nargs="+", default=["cleaned_title_singlish_norm", "cleaned_selftext_singlish_norm"], help="Columns to translate.")
    parser.add_argument("--dict-path", default=str(DEFAULT_DICT_PATH), help="Path to singlish_dictionary.json.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    translator = SinglishTranslator(dict_path=args.dict_path)
    df = translator.translate_dataframe(df, text_cols=args.cols)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved translated dataset to %s", args.output)


if __name__ == "__main__":
    main()
