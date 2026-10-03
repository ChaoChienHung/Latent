"""
Run pretrained emotion classification over CSV files in the data folder.

This script is designed for the cleaned Reddit datasets in this repository and
defaults to j-hartmann/emotion-english-distilroberta-base (7 emotions: anger,
disgust, fear, joy, neutral, sadness, surprise). It produces a slim output CSV
matching the format in emotion_output/sampled_labels.csv:
only id, cleaned_text_normalized, and per-model columns.

Usage:
    uv run python emotion_inference.py
    uv run python emotion_inference.py --input intermediate_data/PostVault.csv
    uv run python emotion_inference.py --model-name j-hartmann/emotion-english-distilroberta-base
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
TEXT_COLUMN_PRIORITY = ("cleaned_text_normalized", "cleaned_text", "raw_text", "body")
MAX_TEXT_LENGTH = 256

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Reddit CSV text with a pretrained emotion model."
    )
    parser.add_argument(
        "--input",
        default="intermediate_data/PostVault.csv",
        help="CSV file or directory of CSV files to classify.",
    )
    parser.add_argument(
        "--output-dir",
        default="emotion_output",
        help="Directory for enriched CSV outputs. Default: emotion_output",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size. Lower this if you hit memory limits.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_TEXT_LENGTH,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Inference device. Default selects CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=("auto", "multi-label", "single-label"),
        help="How to interpret model outputs. Auto uses config first, then label count.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=100,
        help="Emit a progress log every N batches.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in (".csv", ".gz"):
            raise ValueError(f"Input file must be a CSV: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.glob("*.csv*") if path.is_file()
            and path.suffix.lower() in (".csv", ".gz")
        )
        if not files:
            raise ValueError(f"No CSV files found in {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


def choose_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_PRIORITY:
        if column in df.columns and df[column].fillna("").astype(str).str.strip().ne("").any():
            return column
    if {"title", "selftext"}.issubset(df.columns):
        return "__combined_title_selftext__"
    if "title" in df.columns:
        return "title"
    raise ValueError(
        "Could not find a usable text column. Expected one of "
        f"{TEXT_COLUMN_PRIORITY} or title/selftext."
    )


def build_text_series(df: pd.DataFrame, text_column: str) -> pd.Series:
    if text_column == "__combined_title_selftext__":
        title = df["title"].fillna("").astype(str).str.strip()
        selftext = df["selftext"].fillna("").astype(str).str.strip()
        return (title + " " + selftext).str.strip()
    return df[text_column].fillna("").astype(str).str.strip()


def iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def detect_multilabel(
    model: AutoModelForSequenceClassification, prediction_mode: str
) -> bool:
    if prediction_mode == "multi-label":
        return True
    if prediction_mode == "single-label":
        return False
    config = model.config
    if getattr(config, "problem_type", None) == "multi_label_classification":
        return True
    if getattr(config, "problem_type", None) == "single_label_classification":
        return False
    if config.num_labels <= 1:
        return False
    return config.num_labels > 10


def get_labels(model: AutoModelForSequenceClassification) -> list[str]:
    id2label = getattr(model.config, "id2label", None) or {}
    return [id2label.get(i, f"label_{i}") for i in range(model.config.num_labels)]



def score_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length: int,
    prediction_mode: str,
    log_every_batches: int,
) -> tuple[list[str], list[list[float]]]:
    labels = get_labels(model)
    is_multilabel = detect_multilabel(model, prediction_mode)
    top_labels: list[str] = []
    all_probs: list[list[float]] = []

    model.eval()
    with torch.inference_mode():
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_index, batch in enumerate(iter_batches(texts, batch_size), start=1):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits

            if log_every_batches > 0 and (
                batch_index == 1
                or batch_index % log_every_batches == 0
                or batch_index == total_batches
            ):
                log.info("Processed batch %d/%d", batch_index, total_batches)

            if is_multilabel:
                probabilities = torch.sigmoid(logits).detach().cpu()
            else:
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()

            for row in probabilities:
                probs = row.tolist()
                top_labels.append(labels[int(row.argmax())])
                all_probs.append(probs)

    return top_labels, all_probs


def classify_file(
    csv_path: Path,
    output_dir: Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length: int,
    prediction_mode: str,
    log_every_batches: int,
) -> Path:
    df = pd.read_csv(csv_path, low_memory=False)
    text_column = choose_text_column(df)
    text_series = build_text_series(df, text_column)
    valid_mask = text_series.str.strip().ne("")

    log.info("Classifying %s using text column %s", csv_path.name, text_column)

    result = df.copy()

    labels = get_labels(model)
    result["predicted_emotion"] = pd.NA
    for label in labels:
        result[f"prob_{label}"] = pd.NA

    valid_texts = text_series[valid_mask].tolist()
    if valid_texts:
        top_labels, all_probs = score_texts(
            texts=valid_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            prediction_mode=prediction_mode,
            log_every_batches=log_every_batches,
        )
        valid_index = result.index[valid_mask]
        result.loc[valid_index, "predicted_emotion"] = top_labels
        for i, label in enumerate(labels):
            result.loc[valid_index, f"prob_{label}"] = [p[i] for p in all_probs]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    output_path = output_dir / f"{stem}_labels.csv"
    result.to_csv(output_path, index=False)
    log.info("Saved %s", output_path)
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    log.info("Loading model %s on %s", args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)

    for csv_path in discover_input_files(input_path):
        classify_file(
            csv_path=csv_path,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            prediction_mode=args.prediction_mode,
            log_every_batches=args.log_every_batches,
        )


if __name__ == "__main__":
    main()
