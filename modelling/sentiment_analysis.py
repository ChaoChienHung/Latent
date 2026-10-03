"""
Stage 7: Sentiment and Emotion Analysis.
Classifies Reddit posts and comments across 7 emotion classes
(anger, disgust, fear, joy, neutral, sadness, surprise) using RoBERTa.

Usage:
    python -m modelling.sentiment_analysis --input data/PostVault.csv --output intermediate_data/posts_with_emotions.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import importlib

try:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
    AutoTokenizer = transformers.AutoTokenizer
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class EmotionClassifier:
    """Pretrained transformer-based emotion classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        if torch is None or AutoModelForSequenceClassification is None:
            raise ImportError(
                "PyTorch and HuggingFace transformers are required for emotion classification. "
                "Install them via: pip install torch transformers"
            )

        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        log.info("Loading tokenizer and model '%s' on %s...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[dict[str, float]]:
        """Predict emotion probabilities for a list of texts."""
        results: list[dict[str, float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t else "" for t in texts[i : i + batch_size]]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p in probs:
                prob_dict = {self.id2label[idx]: float(val) for idx, val in enumerate(p)}
                results.append(prob_dict)

        return results

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "cleaned_title",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Enrich a dataframe with emotion predictions and individual emotion probability columns."""
        out = df.copy()
        log.info("Classifying %d texts from column '%s'...", len(out), text_col)

        texts = out[text_col].fillna("").tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size)

        for emotion in EMOTIONS:
            out[f"prob_{emotion}"] = [p.get(emotion, 0.0) for p in predictions]

        out["predicted_emotion"] = [
            max(p.items(), key=lambda x: x[1])[0] for p in predictions
        ]
        out["emotion_confidence"] = [
            max(p.values()) for p in predictions
        ]

        log.info("Emotion classification complete.")
        return out


def main():
    parser = argparse.ArgumentParser(description="Classify texts using RoBERTa emotion model.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output enriched CSV file.")
    parser.add_argument("--text-col", default="title", help="Column to run inference on.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face model identifier.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    classifier = EmotionClassifier(model_name=args.model_name)
    df_classified = classifier.classify_dataframe(df, text_col=args.text_col, batch_size=args.batch_size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_classified.to_csv(args.output, index=False)
    log.info("Saved emotion-classified data to %s", args.output)


if __name__ == "__main__":
    main()
