"""Preprocesiranje OkCupid bio sekcija za sistem preporučivanja i topic modeling.

Ulaz:  data/raw/okcupid/okcupid_profiles.csv
Izlaz:
    - data/processed/bios_for_embeddings.csv  (clean_text, tokens)
    - data/processed/bios_for_topics.csv      (clean_text, tokens)

OkCupid CSV ima 10 essay polja (essay0..essay9). Spajamo ih u jedan
"bio" string. Prazne profile uklanjamo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

from src.preprocessing.text_cleaning import (
    clean_for_embeddings,
    full_token_pipeline,
)
from src.utils.paths import PROCESSED_DIR, RAW_DIR

DetectorFactory.seed = 42
tqdm.pandas()

INPUT_CSV = RAW_DIR / "okcupid" / "okcupid_profiles.csv"
ESSAY_COLS = [f"essay{i}" for i in range(10)]


def _safe_lang(text: str) -> str:
    try:
        return detect(text) if isinstance(text, str) and len(text) > 30 else "und"
    except LangDetectException:
        return "und"


def load() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Ne nalazim {INPUT_CSV}. Pokreni prvo:\n"
            "  python -m src.data_collection.download_okcupid"
        )
    df = pd.read_csv(INPUT_CSV)
    print(f"[bios] učitano {len(df)} sirovih profila")
    return df


def merge_essays(df: pd.DataFrame) -> pd.Series:
    available = [c for c in ESSAY_COLS if c in df.columns]
    if not available:
        raise ValueError("Ne postoje essay kolone u OkCupid CSV-u.")
    return df[available].fillna("").agg(" ".join, axis=1).str.strip()


def run() -> dict[str, Path]:
    df = load()
    df["bio_raw"] = merge_essays(df)
    df = df[df["bio_raw"].str.len() > 50].copy()
    print(f"[bios] nakon filtera dužine: {len(df)}")

    print("[bios] detektujem jezik (može potrajati)...")
    df["lang"] = df["bio_raw"].progress_apply(_safe_lang)
    df = df[df["lang"] == "en"].copy()
    print(f"[bios] zadržano EN: {len(df)}")

    print("[bios] čistim tekst (embedding pipeline)...")
    df["bio_clean"] = df["bio_raw"].progress_apply(clean_for_embeddings)
    df = df[df["bio_clean"].str.split().str.len() >= 8].copy()

    print("[bios] tokenizujem + lematiziram (može potrajati)...")
    df["tokens"] = df["bio_clean"].progress_apply(
        lambda t: full_token_pipeline(
            t,
            cleaner=lambda x: x,
            drop_stopwords=True,
            do_lemmatize=True,
        )
    )
    df["tokens_str"] = df["tokens"].apply(lambda toks: " ".join(toks))

    keep_cols = [
        "bio_raw", "bio_clean", "tokens_str",
        "age", "sex", "orientation", "status", "lang",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out_emb = PROCESSED_DIR / "bios_for_embeddings.csv"
    df[keep_cols].to_csv(out_emb, index=False, encoding="utf-8")
    print(f"[bios] zapisano (embeddings): {out_emb} ({len(df)} redova)")

    out_topics = PROCESSED_DIR / "bios_for_topics.csv"
    df[["tokens_str"] + [c for c in ("age", "sex") if c in df.columns]].to_csv(
        out_topics, index=False, encoding="utf-8"
    )
    print(f"[bios] zapisano (topics): {out_topics}")

    stats = {
        "n_rows": int(len(df)),
        "avg_token_len": float(df["tokens"].str.len().mean()),
        "median_token_len": float(df["tokens"].str.len().median()),
    }
    (PROCESSED_DIR / "bios_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[bios] statistike: {stats}")

    return {"embeddings": out_emb, "topics": out_topics}


if __name__ == "__main__":
    run()
