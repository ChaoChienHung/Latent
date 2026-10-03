"""
Generate analytical plots from a post-level emotion-labelled CSV.

Usage:
    uv run python plot_emotion_summary.py \
      --input intermediate_data/stopword_lemmatized_posts_0_labels_w_emot.csv \
      [--output-dir data/emotion_plots]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMOTION_PALETTE = {
    "anger": "#d62728",
    "disgust": "#8c564b",
    "fear": "#9467bd",
    "joy": "#2ca02c",
    "neutral": "#7f7f7f",
    "sadness": "#1f77b4",
    "surprise": "#ff7f0e",
}

EMOTIONS = list(EMOTION_PALETTE.keys())
PROB_COLS = [f"prob_{e}" for e in EMOTIONS]

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"predicted_emotion", "year", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df["month_date"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")

    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(
            df["day_of_week"], categories=DOW_ORDER, ordered=True
        )

    log.info("Loaded %d rows from %s", len(df), input_csv)
    return df


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_overall_distribution(df: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart of overall emotion counts and shares."""
    counts = df["predicted_emotion"].value_counts().sort_values()
    total = counts.sum()

    colors = [EMOTION_PALETTE.get(e, "#333333") for e in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(
            val + total * 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({val / total:.1%})",
            va="center",
            fontsize=9,
        )

    ax.set_title("Overall Emotion Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of posts")
    ax.set_xlim(right=counts.max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_emotion_trends(df: pd.DataFrame, out: Path) -> None:
    """Line chart: share of each emotion over year-month."""
    monthly = (
        df.groupby(["month_date", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    monthly["share"] = monthly.groupby("month_date")["count"].transform(
        lambda x: x / x.sum()
    )

    pivot = monthly.pivot(
        index="month_date", columns="predicted_emotion", values="share"
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    for emotion in pivot.columns:
        color = EMOTION_PALETTE.get(emotion, "#333333")
        ax.plot(
            pivot.index, pivot[emotion], marker="o", linewidth=2,
            markersize=4, color=color, label=emotion
        )

    ax.set_title("Monthly Emotion Share Trends", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of posts")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_emotion_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap: emotion × month (normalised share per month)."""
    monthly = (
        df.groupby(["year_month", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    pivot = monthly.pivot(
        index="predicted_emotion", columns="year_month", values="count"
    ).fillna(0)
    # normalise each month to share
    pivot = pivot.div(pivot.sum(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.9), 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        fmt=".1%",
        annot=True,
        annot_kws={"size": 8},
        cbar_kws={"label": "Share of monthly posts"},
    )
    ax.set_title("Emotion Share Heatmap (by Month)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Emotion")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_emotion_by_day_of_week(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: emotion share by day of week."""
    if "day_of_week" not in df.columns:
        log.warning("day_of_week column missing – skipping DOW plot")
        return

    dow = (
        df.groupby(["day_of_week", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    dow["share"] = dow.groupby("day_of_week")["count"].transform(
        lambda x: x / x.sum()
    )
    pivot = dow.pivot(
        index="day_of_week", columns="predicted_emotion", values="share"
    ).fillna(0)
    # keep ordered weekdays
    pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pivot))
    n_emotions = len(pivot.columns)
    width = 0.8 / n_emotions

    for i, emotion in enumerate(pivot.columns):
        color = EMOTION_PALETTE.get(emotion, "#333333")
        offset = (i - n_emotions / 2 + 0.5) * width
        ax.bar(x + offset, pivot[emotion], width=width, label=emotion, color=color)

    ax.set_title("Emotion Share by Day of Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Share of posts")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_emotion_by_hour(df: pd.DataFrame, out: Path) -> None:
    """Heatmap: emotion × hour of day."""
    if "hour" not in df.columns:
        log.warning("hour column missing – skipping hourly plot")
        return

    hourly = (
        df.groupby(["hour", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    pivot = hourly.pivot(
        index="predicted_emotion", columns="hour", values="count"
    ).fillna(0)
    pivot = pivot.div(pivot.sum(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Share per hour"},
        xticklabels=[f"{h:02d}:00" for h in range(24)],
    )
    ax.set_title("Emotion Distribution by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour (SGT)")
    ax.set_ylabel("Emotion")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_emotion_by_score_bucket(df: pd.DataFrame, out: Path) -> None:
    """Stacked bar chart: emotion composition by score bucket."""
    if "score_bucket" not in df.columns:
        log.warning("score_bucket column missing – skipping score-bucket plot")
        return

    BUCKET_ORDER = ["negative", "low", "medium", "high", "viral"]
    bucket = (
        df.groupby(["score_bucket", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    pivot = bucket.pivot(
        index="score_bucket", columns="predicted_emotion", values="count"
    ).fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    pivot = pivot.reindex([b for b in BUCKET_ORDER if b in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot))
    for emotion in pivot.columns:
        color = EMOTION_PALETTE.get(emotion, "#333333")
        ax.bar(pivot.index, pivot[emotion], bottom=bottom, label=emotion, color=color)
        bottom += pivot[emotion].values

    ax.set_title("Emotion Composition by Score Bucket", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score Bucket")
    ax.set_ylabel("Share of posts")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_confidence_distributions(df: pd.DataFrame, out: Path) -> None:
    """Violin plots of max-probability (model confidence) per predicted emotion."""
    available_prob_cols = [c for c in PROB_COLS if c in df.columns]
    if not available_prob_cols:
        log.warning("No prob_* columns found – skipping confidence plot")
        return

    df = df.copy()
    df["max_prob"] = df[available_prob_cols].max(axis=1)

    emotion_order = (
        df.groupby("predicted_emotion")["max_prob"].median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    colors = [EMOTION_PALETTE.get(e, "#333333") for e in emotion_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(
        [df.loc[df["predicted_emotion"] == e, "max_prob"].values for e in emotion_order],
        showmedians=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1)

    ax.set_xticks(range(1, len(emotion_order) + 1))
    ax.set_xticklabels(emotion_order)
    ax.set_title("Model Confidence by Predicted Emotion", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Max class probability")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_monthly_stacked_area(df: pd.DataFrame, out: Path) -> None:
    """Stacked area chart of emotion shares over time."""
    monthly = (
        df.groupby(["month_date", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    monthly["share"] = monthly.groupby("month_date")["count"].transform(
        lambda x: x / x.sum()
    )
    pivot = (
        monthly.pivot(index="month_date", columns="predicted_emotion", values="share")
        .fillna(0.0)
        .sort_index()
    )

    emotion_cols = [e for e in EMOTIONS if e in pivot.columns]
    colors = [EMOTION_PALETTE[e] for e in emotion_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        pivot.index,
        [pivot[e] for e in emotion_cols],
        labels=emotion_cols,
        colors=colors,
        alpha=0.85,
    )
    ax.set_title("Monthly Emotion Share (Stacked Area)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of posts")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analytical emotion plots from a post-level labelled CSV."
    )
    parser.add_argument(
        "--input",
        default="intermediate_data/stopword_lemmatized_posts_0_labels_w_emot.csv",
        help="Path to the post-level CSV with predicted_emotion and prob_* columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/emotion_plots",
        help="Directory for generated PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    stem = input_csv.stem

    plot_overall_distribution(df, output_dir / f"{stem}_1_overall_distribution.png")
    plot_emotion_trends(df, output_dir / f"{stem}_2_monthly_trends.png")
    plot_monthly_stacked_area(df, output_dir / f"{stem}_3_stacked_area.png")
    plot_emotion_heatmap(df, output_dir / f"{stem}_4_heatmap.png")
    plot_emotion_by_day_of_week(df, output_dir / f"{stem}_5_by_day_of_week.png")
    plot_emotion_by_hour(df, output_dir / f"{stem}_6_by_hour.png")
    plot_emotion_by_score_bucket(df, output_dir / f"{stem}_7_by_score_bucket.png")
    plot_confidence_distributions(df, output_dir / f"{stem}_8_confidence.png")

    log.info("All plots written to %s", output_dir)


if __name__ == "__main__":
    main()
