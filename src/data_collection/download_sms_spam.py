"""Preuzimanje SMS Spam Collection dataseta sa UCI repozitorija.

Koristi se kao baseline za detekciju prevara/botova.
Sami SMS poruke su kratki tekstovi koji odgovaraju formatu poruka u
dating aplikaciji. Spam poruke (npr. "Free entry...", "Click this link...")
veoma liče na obrasce romance scam-era.

Dopunjavamo ga sintetičkim romance-scam profilima u
`src/data_collection/generate_synthetic.py`.

Izvor:
    https://archive.ics.uci.edu/dataset/228/sms+spam+collection
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.paths import RAW_DIR

TARGET_DIR = RAW_DIR / "sms_spam"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TARGET_DIR / "sms_spam.csv"
    if out_csv.exists():
        print(f"[sms_spam] već postoji: {out_csv}")
        return out_csv

    print(f"[sms_spam] preuzimam {UCI_ZIP_URL}")
    r = requests.get(UCI_ZIP_URL, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        candidates = [n for n in zf.namelist() if "smsspamcollection" in n.lower()]
        if not candidates:
            raise FileNotFoundError(f"Ne nalazim SMSSpamCollection u zip: {zf.namelist()}")
        with zf.open(candidates[0]) as f:
            raw = f.read().decode("latin-1")

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        label, _, text = line.partition("\t")
        rows.append({"label": label.strip(), "text": text.strip()})

    df = pd.DataFrame(rows)
    df["label_bin"] = (df["label"].str.lower() == "spam").astype(int)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[sms_spam] {len(df)} primjera, spam={df['label_bin'].sum()}")
    return out_csv


if __name__ == "__main__":
    download()
