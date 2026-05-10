"""Preprocesiranje SMS Spam + sintetičkih scam profila za bot/scam detekciju.

Strategija:
    - SMS Spam (UCI) - opšti spam obrasci, kratki tekst
    - Sintetički scam profili - dating-specifični obrasci
    - Spojimo ih u zajednički binarni problem (label = 1 ako spam/scam)

Ulaz:
    data/raw/sms_spam/sms_spam.csv
    data/synthetic/scam_profiles.csv
Izlaz:
    data/processed/scam_detection_clean.csv
    + train/val/test split
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
from src.utils.paths import PROCESSED_DIR, RAW_DIR, SYNTHETIC_DIR

tqdm.pandas()

SMS_CSV = RAW_DIR / "sms_spam" / "sms_spam.csv"
SCAM_CSV = SYNTHETIC_DIR / "scam_profiles.csv"


def _load_sms() -> pd.DataFrame:
    if not SMS_CSV.exists():
        raise FileNotFoundError(
            f"Nedostaje {SMS_CSV}. Pokreni: python -m src.data_collection.download_sms_spam"
        )
    df = pd.read_csv(SMS_CSV)
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label_bin"].astype(int),
        "source": "sms_spam",
    })


def _load_scam() -> pd.DataFrame:
    if not SCAM_CSV.exists():
        raise FileNotFoundError(
            f"Nedostaje {SCAM_CSV}. Pokreni: python -m src.data_collection.generate_synthetic"
        )
    df = pd.read_csv(SCAM_CSV)
    return df[["text", "label"]].assign(source="synthetic_scam")


def run() -> dict[str, Path]:
    sms = _load_sms()
    scam = _load_scam()
    df = pd.concat([sms, scam], ignore_index=True).dropna(subset=["text"])
    print(f"[scam] učitano: SMS={len(sms)}, scam={len(scam)}, ukupno={len(df)}")

    print("[scam] čistim tekst...")
    df["text_clean"] = df["text"].progress_apply(clean_for_classification)
    df = df[df["text_clean"].str.len() > 0].copy()

    print("[scam] tokenizujem + lematiziram...")
    df["tokens_str"] = df["text_clean"].progress_apply(
        lambda t: " ".join(full_token_pipeline(
            t, cleaner=lambda x: x, drop_stopwords=True, do_lemmatize=True
        ))
    )

    df["len_chars"] = df["text"].str.len()
    df["len_tokens"] = df["tokens_str"].str.split().str.len()
    df["has_url"] = df["text"].str.contains(r"http|www\.", case=False, regex=True).astype(int)
    df["has_phone"] = df["text"].str.contains(r"\+?\d[\d\-\s]{6,}", regex=True).astype(int)
    df["has_money"] = df["text"].str.contains(r"\$|usd|btc|eur|crypto|inheritance",
                                              case=False, regex=True).astype(int)

    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    print(f"[scam] nakon dedupe: {len(df)}, "
          f"distribucija label: {df['label'].value_counts().to_dict()}")

    out = PROCESSED_DIR / "scam_detection_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[scam] zapisano: {out}")

    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)
    paths = {"full": out}
    for name, part in (("train", train), ("val", val), ("test", test)):
        p = PROCESSED_DIR / f"scam_detection_{name}.csv"
        part.to_csv(p, index=False, encoding="utf-8")
        paths[name] = p
        print(f"[scam] {name}: {len(part)} -> {p}")
    return paths


if __name__ == "__main__":
    run()
