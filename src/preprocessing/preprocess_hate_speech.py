"""Preprocesiranje govor-mržnje korpusa.

Ulaz:  data/raw/hate_speech/hate_combined.csv
Izlaz: data/processed/hate_speech_clean.csv  (text_clean, tokens_str, label, source)
       + train/val/test split fajlovi
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocessing.text_cleaning import (
    clean_for_classification,
    full_token_pipeline,
)
from src.utils.paths import PROCESSED_DIR, RAW_DIR

tqdm.pandas()

INPUT_CSV = RAW_DIR / "hate_speech" / "hate_combined.csv"


def run() -> dict[str, Path]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Ne nalazim {INPUT_CSV}. Pokreni prvo:\n"
            "  python -m src.data_collection.download_hate_speech"
        )

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["text"]).copy()
    print(f"[hate] učitano {len(df)} primjera")

    print("[hate] čistim tekst (classification pipeline)...")
    df["text_clean"] = df["text"].progress_apply(clean_for_classification)
    df = df[df["text_clean"].str.len() > 0].copy()

    print("[hate] tokeniziram + lematiziram...")
    df["tokens_str"] = df["text_clean"].progress_apply(
        lambda t: " ".join(full_token_pipeline(
            t, cleaner=lambda x: x, drop_stopwords=True, do_lemmatize=True
        ))
    )

    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    print(f"[hate] nakon dedupe: {len(df)}")
    print(f"       distribucija label: {df['label'].value_counts().to_dict()}")

    out = PROCESSED_DIR / "hate_speech_clean.csv"
    df[["text", "text_clean", "tokens_str", "label", "source"]].to_csv(
        out, index=False, encoding="utf-8"
    )
    print(f"[hate] zapisano: {out}")

    train, tmp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        tmp, test_size=0.50, stratify=tmp["label"], random_state=42
    )
    paths = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        p = PROCESSED_DIR / f"hate_speech_{name}.csv"
        part.to_csv(p, index=False, encoding="utf-8")
        paths[name] = p
        print(f"[hate] {name}: {len(part)} -> {p}")

    paths["full"] = out
    return paths


if __name__ == "__main__":
    run()
