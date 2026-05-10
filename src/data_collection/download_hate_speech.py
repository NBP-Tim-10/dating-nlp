"""Preuzimanje dataseta za detekciju govora mržnje.

Koristimo dva izvora radi balansa i robusnosti:

1) Davidson et al. (2017) "Hate Speech and Offensive Language" - tweet korpus
   sa 3 klase: hate_speech / offensive / neither.
   GitHub raw URL (24,783 tweets):
   https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv

2) Hugging Face `tweet_eval` (subset `hate`) - dodatne tweet labele
   (binarno: hate vs not_hate). Uskladimo ih sa Davidsonom u zajednički
   binarni label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

from src.utils.paths import RAW_DIR

TARGET_DIR = RAW_DIR / "hate_speech"
DAVIDSON_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)


def _download_davidson() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    out = TARGET_DIR / "davidson_labeled.csv"
    if out.exists():
        print(f"[hate] već postoji: {out}")
        return out

    print(f"[hate] preuzimam Davidson -> {out}")
    r = requests.get(DAVIDSON_URL, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)

    df = pd.read_csv(out)
    print(f"[hate] Davidson: {len(df)} primjera, klase: "
          f"{df['class'].value_counts().to_dict()}")
    return out


def _download_tweet_eval() -> Path | None:
    out = TARGET_DIR / "tweet_eval_hate.csv"
    if out.exists():
        print(f"[hate] već postoji: {out}")
        return out

    try:
        from datasets import load_dataset
    except ImportError:
        print("[hate] `datasets` paket nije instaliran - preskačem tweet_eval")
        return None

    print("[hate] preuzimam tweet_eval/hate sa Hugging Face")
    try:
        ds = load_dataset("tweet_eval", "hate")
    except Exception as exc:
        print(f"[hate] greška pri preuzimanju tweet_eval: {exc}")
        return None

    parts = []
    for split in ("train", "validation", "test"):
        if split in ds:
            df = ds[split].to_pandas()
            df["split"] = split
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full.to_csv(out, index=False, encoding="utf-8")
    print(f"[hate] tweet_eval: {len(full)} primjera")
    return out


def harmonize() -> Path:
    """Spoji oba seta u zajednički binarni format (text, label)."""
    davidson_path = TARGET_DIR / "davidson_labeled.csv"
    tweet_eval_path = TARGET_DIR / "tweet_eval_hate.csv"

    frames = []

    if davidson_path.exists():
        d = pd.read_csv(davidson_path)
        d = d.rename(columns={"tweet": "text", "class": "_orig_class"})
        d["label"] = (d["_orig_class"] == 0).astype(int)
        d["source"] = "davidson"
        frames.append(d[["text", "label", "source"]])

    if tweet_eval_path is not None and tweet_eval_path.exists():
        t = pd.read_csv(tweet_eval_path)
        t = t.rename(columns={"label": "label"})
        t["source"] = "tweet_eval"
        frames.append(t[["text", "label", "source"]])

    if not frames:
        raise RuntimeError("Nijedan hate-speech dataset nije preuzet.")

    out = TARGET_DIR / "hate_combined.csv"
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    full.to_csv(out, index=False, encoding="utf-8")
    print(f"[hate] kombinovano: {len(full)} jedinstvenih primjera -> {out}")
    print(f"       distribucija label: {full['label'].value_counts().to_dict()}")
    return out


def download() -> Path:
    _download_davidson()
    _download_tweet_eval()
    return harmonize()


if __name__ == "__main__":
    download()
