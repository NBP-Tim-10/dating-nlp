"""Preprocesiranje recenzija dating aplikacija + sintetičkih razgovora.

Cilj: jedinstveni dataset za fine-grained sentiment + emocije.

Ulaz:
    data/raw/app_reviews/dating_app_reviews.csv (ako je scrape uspio)
    data/synthetic/conversations.csv
Izlaz:
    data/processed/sentiment_reviews_clean.csv
    data/processed/sentiment_conversations_clean.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocessing.text_cleaning import clean_for_sentiment, full_token_pipeline
from src.utils.paths import PROCESSED_DIR, RAW_DIR, SYNTHETIC_DIR

DetectorFactory.seed = 42
tqdm.pandas()

REVIEWS_CSV = RAW_DIR / "app_reviews" / "dating_app_reviews.csv"
CONV_CSV = SYNTHETIC_DIR / "conversations.csv"


def _safe_lang(text: str) -> str:
    try:
        return detect(text) if isinstance(text, str) and len(text) > 15 else "und"
    except LangDetectException:
        return "und"


def _process_reviews() -> Path | None:
    if not REVIEWS_CSV.exists():
        print(f"[sentiment] {REVIEWS_CSV} ne postoji - preskačem recenzije.")
        return None

    df = pd.read_csv(REVIEWS_CSV)
    df = df.dropna(subset=["text"]).copy()
    print(f"[sentiment] učitano {len(df)} recenzija")

    print("[sentiment] detektujem jezik...")
    df["lang"] = df["text"].progress_apply(_safe_lang)
    df = df[df["lang"] == "en"].copy()
    print(f"[sentiment] EN: {len(df)}")

    print("[sentiment] čistim recenzije (sentiment pipeline)...")
    df["text_clean"] = df["text"].progress_apply(clean_for_sentiment)
    df = df[df["text_clean"].str.len() > 0].copy()

    df["tokens_str"] = df["text_clean"].progress_apply(
        lambda t: " ".join(full_token_pipeline(
            t, cleaner=lambda x: x, drop_stopwords=True, do_lemmatize=True
        ))
    )

    df["sentiment_label"] = df["sentiment"].astype(str)
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    print(f"[sentiment] nakon dedupe: {len(df)}, "
          f"distribucija: {df['sentiment_label'].value_counts().to_dict()}")

    out = PROCESSED_DIR / "sentiment_reviews_clean.csv"
    df[["app", "rating", "sentiment_label", "text", "text_clean", "tokens_str"]].to_csv(
        out, index=False, encoding="utf-8"
    )
    print(f"[sentiment] reviews -> {out}")

    train, tmp = train_test_split(
        df, test_size=0.30, stratify=df["sentiment_label"], random_state=42
    )
    val, test = train_test_split(
        tmp, test_size=0.50, stratify=tmp["sentiment_label"], random_state=42
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        p = PROCESSED_DIR / f"sentiment_reviews_{name}.csv"
        part.to_csv(p, index=False, encoding="utf-8")
        print(f"[sentiment] reviews {name}: {len(part)} -> {p}")

    return out


def _process_conversations() -> Path | None:
    if not CONV_CSV.exists():
        print(f"[sentiment] {CONV_CSV} ne postoji - pokreni generate_synthetic prvo.")
        return None

    df = pd.read_csv(CONV_CSV)
    print(f"[sentiment] razgovori: {len(df)} okreta")

    df["text_clean"] = df["text"].progress_apply(clean_for_sentiment)
    df["tokens_str"] = df["text_clean"].progress_apply(
        lambda t: " ".join(full_token_pipeline(
            t, cleaner=lambda x: x, drop_stopwords=True, do_lemmatize=True
        ))
    )

    out = PROCESSED_DIR / "sentiment_conversations_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[sentiment] conversations -> {out}")
    return out


def run() -> dict[str, Path | None]:
    return {
        "reviews": _process_reviews(),
        "conversations": _process_conversations(),
    }


if __name__ == "__main__":
    run()
