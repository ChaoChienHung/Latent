from __future__ import annotations

import glob
import html
import logging
import os
import re
from typing import Any

import emoji
import ftfy
import pandas as pd
from langdetect import LangDetectException, detect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class RedditPreprocessor:
    """Single-class Reddit preprocessing pipeline for posts and comments."""

    REMOVED_MARKERS = {"[removed]", "[deleted]", "[deleted by user]"}
    KNOWN_BOTS = {
        "AutoModerator",
        "SG_wormsbot",
        "RepostSleuthBot",
        "RemindMeBot",
        "reddit_repost_sleuth",
        "TotesMessenger",
        "converter-bot",
    }
    SCORE_BINS = [-float("inf"), 0, 5, 25, 100, float("inf")]
    SCORE_LABELS = ["negative", "low", "medium", "high", "viral"]

    POSTS_CANONICAL_COLS = [
        "id",
        "title",
        "selftext",
        "author",
        "score",
        "upvote_ratio",
        "num_comments",
        "created_utc",
        "edited",
        "distinguished",
        "stickied",
        "over_18",
        "spoiler",
        "locked",
        "archived",
        "is_original_content",
        "is_self",
        "permalink",
        "url",
        "domain",
        "subreddit",
        "subreddit_id",
        "link_flair_text",
        "link_flair_css_class",
        "author_flair_text",
        "author_flair_css_class",
        "gilded",
        "total_awards_received",
        "is_video",
        "media",
        "thumbnail",
        "post_hint",
        "treatment_tags",
        "secure_media",
        "preview",
        "downloaded_media",
    ]

    COMMENTS_CANONICAL_COLS = [
        "id",
        "parent_id",
        "post_id",
        "body",
        "author",
        "score",
        "created_utc",
        "edited",
        "distinguished",
        "stickied",
        "is_submitter",
        "author_flair_text",
        "author_flair_css_class",
        "gilded",
        "total_awards_received",
        "permalink",
        "subreddit",
        "subreddit_id",
        "depth",
        "controversiality",
        "collapsed",
        "collapsed_reason",
        "locked",
    ]

    _RE_URL = re.compile(r"https?://\S+|www\.\S+")
    _RE_MENTIONS = re.compile(r"\b[ur]/\w+")
    _RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    _RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    _RE_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}")
    _RE_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
    _RE_INLINE_CODE = re.compile(r"`[^`]+`")
    _RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
    _RE_BLOCKQUOTE = re.compile(r"^>+\s?", re.MULTILINE)
    _RE_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    _RE_HR = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
    _RE_MULTI_UNDERSCORE = re.compile(r"_+")
    _RE_ELONGATION       = re.compile(r"(.)\1{2,}")
    _RE_REPEATED_PUNCT   = re.compile(r"([!?.])\1+")
    _RE_NON_ALPHA        = re.compile(r"[^a-z0-9\s]")

    def __init__(self, min_words: int = 3, lang_detect: bool = True) -> None:
        self.min_words = min_words
        self.lang_detect = lang_detect

    def load_csvs(self, folder: str, pattern: str) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not files:
            log.warning("No files matched %s in %s", pattern, folder)
            return pd.DataFrame()
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f, low_memory=False, on_bad_lines="skip"))
                log.info("Loaded %s", os.path.basename(f))
            except Exception as exc:
                log.warning("Could not read %s: %s", f, exc)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def harmonise_schema(self, df: pd.DataFrame, canonical_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in canonical_cols:
            if col not in out.columns:
                out[col] = None
        return out[canonical_cols]

    def dedup(self, df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
        before = len(df)
        out = df.drop_duplicates(subset=[id_col])
        log.info("Dedup: %d -> %d rows (removed %d)", before, len(out), before - len(out))
        return out

    def drop_removed(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        before = len(df)
        mask = df[text_col].astype(str).str.strip().isin(self.REMOVED_MARKERS)
        out = df[~mask]
        log.info("Drop removed: %d -> %d rows (removed %d)", before, len(out), before - len(out))
        return out

    def drop_bots_and_mods(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        is_bot = df["author"].isin(self.KNOWN_BOTS)
        is_mod = df["distinguished"].astype(str).str.lower().isin({"moderator", "admin"})
        is_stickied = df.get("stickied", pd.Series(False, index=df.index)).astype(bool)
        out = df[~(is_bot | is_mod | is_stickied)]
        log.info("Drop bots/mods/stickied: %d -> %d rows (removed %d)", before, len(out), before - len(out))
        return out

    def drop_nsfw(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        is_nsfw = df["over_18"].astype(str).str.lower() == "true"
        out = df[~is_nsfw]
        log.info("Drop NSFW: %d -> %d rows (removed %d)", before, len(out), before - len(out))
        return out

    def strip_markdown(self, text: str) -> str:
        text = html.unescape(text)
        text = self._RE_CODE_BLOCK.sub(" ", text)
        text = self._RE_MD_IMAGE.sub(" ", text)
        text = self._RE_MD_LINK.sub(r"\1", text)
        text = self._RE_URL.sub(" ", text)
        text = self._RE_MENTIONS.sub(" ", text)
        text = self._RE_BOLD_ITALIC.sub(r"\1", text)
        text = self._RE_STRIKETHROUGH.sub(r"\1", text)
        text = self._RE_INLINE_CODE.sub(" ", text)
        text = self._RE_BLOCKQUOTE.sub(" ", text)
        text = self._RE_HEADER.sub(" ", text)
        text = self._RE_HR.sub(" ", text)
        return text

    @staticmethod
    def fix_encoding(text: str) -> str:
        return ftfy.fix_text(text)

    @staticmethod
    def normalise_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def normalise_for_classification(self, text: str) -> str:
        """Lowercase, collapse elongation, strip non-alphanumeric. Run after strip_markdown."""
        text = text.lower()
        text = self._RE_ELONGATION.sub(r"\1\1", text)      # coooool -> cool
        text = self._RE_REPEATED_PUNCT.sub(r"\1", text)    # !!! -> !
        text = self._RE_NON_ALPHA.sub(" ", text)           # keep a-z, 0-9, spaces
        return re.sub(r"\s+", " ", text).strip()

    def apply_text_cleaning(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        out["cleaned_text"] = (
            out[text_col].astype(str).apply(self.strip_markdown).apply(self.fix_encoding).apply(self.normalise_text)
        )
        return out

    def apply_text_cleaning_for_classification(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Like apply_text_cleaning but adds lowercase, elongation, and non-alpha stripping."""
        out = df.copy()
        out["cleaned_text"] = (
            out[text_col]
            .astype(str)
            .apply(self.strip_markdown)
            .apply(self.fix_encoding)
            .apply(self.normalise_text)
            .apply(self.normalise_for_classification)
        )
        return out

    def drop_short(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        before = len(df)
        word_count = df[text_col].astype(str).str.split().str.len()
        out = df[word_count >= self.min_words]
        log.info(
            "Drop short (<%d words): %d -> %d rows (removed %d)",
            self.min_words,
            before,
            len(out),
            before - len(out),
        )
        return out

    def _emoji_to_english(self, text: str) -> str:
        """Translate emojis to short English labels"""
        demojized = emoji.demojize(str(text), delimiters=(" ", " "))
        demojized = self._RE_MULTI_UNDERSCORE.sub("_", demojized)
        demojized = demojized.replace("_", " ")
        return demojized

    def process_emoji(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        out["has_emoji"] = out[text_col].astype(str).apply(lambda t: bool(emoji.emoji_list(t)))
        out[text_col] = out[text_col].astype(str).apply(self._emoji_to_english).apply(self.normalise_text)
        return out

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return detect(str(text))
        except LangDetectException:
            return "unknown"

    def add_language(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        log.info("Detecting language for %d rows (this may take a while)...", len(out))
        out["lang"] = out[text_col].astype(str).apply(self.detect_language)
        counts = out["lang"].value_counts().head(5).to_dict()
        log.info("Top languages: %s", counts)
        return out

    def extract_features(self, df: pd.DataFrame, is_posts: bool) -> pd.DataFrame:
        out = df.copy()
        out["created_utc"] = pd.to_datetime(out["created_utc"], utc=True, errors="coerce")
        out["year"] = out["created_utc"].dt.year
        out["month"] = out["created_utc"].dt.month
        out["day_of_week"] = out["created_utc"].dt.day_name()
        out["hour"] = out["created_utc"].dt.hour
        out["score_bucket"] = pd.cut(out["score"].astype(float), bins=self.SCORE_BINS, labels=self.SCORE_LABELS)
        out["text_length"] = out["cleaned_text"].astype(str).str.len()
        out["word_count"] = out["cleaned_text"].astype(str).str.split().str.len()

        if is_posts:
            out["has_body"] = out["selftext"].notna() & (out["selftext"].astype(str).str.strip() != "")
        else:
            depth = out["depth"].fillna(0).astype(float)
            out["depth_bucket"] = pd.cut(
                depth, bins=[-1, 0, 2, 5, float("inf")], labels=["top_level", "shallow", "mid", "deep"]
            )
        return out

    def preprocess_posts_df(self, posts_df: pd.DataFrame, drop_nsfw_posts: bool = False) -> pd.DataFrame:
        out = self.harmonise_schema(posts_df, self.POSTS_CANONICAL_COLS)
        out = self.dedup(out, "id")
        out = self.drop_removed(out, "title")
        out = self.drop_bots_and_mods(out)
        if drop_nsfw_posts:
            out = self.drop_nsfw(out)
        out["raw_text"] = (out["title"].fillna("") + " " + out["selftext"].fillna("")).str.strip()
        out = self.apply_text_cleaning_for_classification(out, "raw_text")
        out = self.drop_short(out, "cleaned_text")
        out = self.process_emoji(out, "cleaned_text")
        if self.lang_detect:
            out = self.add_language(out, "cleaned_text")
        return self.extract_features(out, is_posts=True)

    def preprocess_comments_df(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        out = self.harmonise_schema(comments_df, self.COMMENTS_CANONICAL_COLS)
        out = self.dedup(out, "id")
        out = self.drop_removed(out, "body")
        out = self.drop_bots_and_mods(out)
        out = self.apply_text_cleaning_for_classification(out, "body")
        out = self.drop_short(out, "cleaned_text")
        out = self.process_emoji(out, "cleaned_text")
        if self.lang_detect:
            out = self.add_language(out, "cleaned_text")
        return self.extract_features(out, is_posts=False)

    def preprocess_posts_from_folder(
        self, input_dir: str, output_dir: str | None = None, drop_nsfw_posts: bool = False
    ) -> pd.DataFrame:
        posts = self.load_csvs(input_dir, "*_posts_*.csv")
        if posts.empty:
            return posts
        out = self.preprocess_posts_df(posts, drop_nsfw_posts=drop_nsfw_posts)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out.to_csv(os.path.join(output_dir, "posts_cleaned.csv"), index=False)
        return out

    def preprocess_comments_from_folder(self, input_dir: str, output_dir: str | None = None) -> pd.DataFrame:
        comments = self.load_csvs(input_dir, "*_comments_*.csv")
        if comments.empty:
            return comments
        out = self.preprocess_comments_df(comments)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out.to_csv(os.path.join(output_dir, "comments_cleaned.csv"), index=False)
        return out


class POSTagger:
    """POS tagging class using spaCy (loaded lazily)."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy is required for POS tagging. Install with: pip install spacy") from exc
        try:
            self.nlp = spacy.load(model)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' not found. Install with: python -m spacy download {model}"
            ) from exc

    def tag_text(self, text: str) -> list[dict[str, Any]]:
        doc = self.nlp(str(text))
        return [
            {"token": tok.text, "lemma": tok.lemma_, "pos": tok.pos_, "tag": tok.tag_}
            for tok in doc
            if not tok.is_space
        ]

    def tag_dataframe(self, df: pd.DataFrame, text_col: str, output_col: str = "pos_tags") -> pd.DataFrame:
        out = df.copy()
        out[output_col] = out[text_col].astype(str).apply(self.tag_text)
        return out


class NERTagger:
    """NER tagging class using spaCy (loaded lazily)."""

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
        doc = self.nlp(str(text))
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def tag_dataframe(self, df: pd.DataFrame, text_col: str, output_col: str = "entities") -> pd.DataFrame:
        out = df.copy()
        out[output_col] = out[text_col].astype(str).apply(self.extract_entities)
        return out
