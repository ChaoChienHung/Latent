"""
Stage 1: Data Collection and Data Cleaning.
Loads raw Reddit posts and comments, harmonises schema, deduplicates,
filters bots/moderators/NSFW, removes Markdown artifacts, and applies text normalization.

Usage:
    python -m preprocessing.data_cleaning --input-dir scraped_data --output-dir intermediate_data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

from utilities.pp_class import RedditPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def clean_reddit_posts(
    input_path: str | Path | pd.DataFrame,
    min_words: int = 3,
    lang_detect: bool = False,
) -> pd.DataFrame:
    """Clean Reddit posts DataFrame or CSV file using RedditPreprocessor."""
    pp = RedditPreprocessor(min_words=min_words, lang_detect=lang_detect)
    
    if isinstance(input_path, (str, Path)):
        p = Path(input_path)
        if p.is_dir():
            df = pp.load_csvs(str(p), "*post*.csv")
        else:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    else:
        df = input_path.copy()

    if df.empty:
        log.warning("No post data to clean.")
        return df

    log.info("Starting post cleaning for %d records...", len(df))
    df = pp.harmonise_schema(df, pp.POSTS_CANONICAL_COLS)
    df = pp.dedup(df, id_col="id")
    df = pp.drop_removed(df, text_col="selftext")
    df = pp.drop_bots_and_mods(df)
    df = pp.drop_nsfw(df)
    df = pp.apply_text_cleaning(df, text_col="title")
    df = df.rename(columns={"cleaned_text": "cleaned_title"})
    df = pp.apply_text_cleaning(df, text_col="selftext")
    df = df.rename(columns={"cleaned_text": "cleaned_selftext"})
    
    if lang_detect:
        df = pp.filter_english(df, text_col="cleaned_title")
    
    log.info("Post cleaning finished. Final count: %d", len(df))
    return df


def clean_reddit_comments(
    input_path: str | Path | pd.DataFrame,
    min_words: int = 3,
    lang_detect: bool = False,
) -> pd.DataFrame:
    """Clean Reddit comments DataFrame or CSV file using RedditPreprocessor."""
    pp = RedditPreprocessor(min_words=min_words, lang_detect=lang_detect)
    
    if isinstance(input_path, (str, Path)):
        p = Path(input_path)
        if p.is_dir():
            df = pp.load_csvs(str(p), "*comment*.csv")
        else:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    else:
        df = input_path.copy()

    if df.empty:
        log.warning("No comment data to clean.")
        return df

    log.info("Starting comment cleaning for %d records...", len(df))
    df = pp.harmonise_schema(df, pp.COMMENTS_CANONICAL_COLS)
    df = pp.dedup(df, id_col="id")
    df = pp.drop_removed(df, text_col="body")
    df = pp.drop_bots_and_mods(df)
    df = pp.apply_text_cleaning(df, text_col="body")
    
    if lang_detect:
        df = pp.filter_english(df, text_col="cleaned_text")
        
    log.info("Comment cleaning finished. Final count: %d", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean raw Reddit data.")
    parser.add_argument("--input-dir", default="data_scrape/data", help="Input directory containing scraped CSVs.")
    parser.add_argument("--output-dir", default="intermediate_data", help="Output directory for cleaned data.")
    parser.add_argument("--lang-detect", action="store_true", help="Enable language detection filtering.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_dir)
    if input_path.exists():
        posts_df = clean_reddit_posts(input_path, lang_detect=args.lang_detect)
        if not posts_df.empty:
            out_posts = out_dir / "cleaned_posts.csv"
            posts_df.to_csv(out_posts, index=False)
            log.info("Saved cleaned posts to %s", out_posts)

        comments_df = clean_reddit_comments(input_path, lang_detect=args.lang_detect)
        if not comments_df.empty:
            out_comments = out_dir / "cleaned_comments.csv"
            comments_df.to_csv(out_comments, index=False)
            log.info("Saved cleaned comments to %s", out_comments)
    else:
        log.warning("Input directory %s does not exist.", input_path)


if __name__ == "__main__":
    main()
