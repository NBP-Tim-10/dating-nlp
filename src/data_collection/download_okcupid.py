"""Preuzimanje OkCupid Profiles dataseta sa Kaggle-a.

Dataset: https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles
- ~60k anonimizovanih profila (essay sekcije = "bio")
- Koristi se za:
    * Sistem za preporučivanje (semantička sličnost bio sekcija)
    * Modeliranje tema / Icebreakers (LDA / BERTopic nad bios)

Preduslov:
    1) `pip install kaggle`
    2) Postaviti API kredencijale: ~/.kaggle/kaggle.json
       (Kaggle -> Account -> Create New API Token)
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from src.utils.paths import RAW_DIR

DATASET_SLUG = "andrewmvd/okcupid-profiles"
TARGET_DIR = RAW_DIR / "okcupid"


def download() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "Nedostaje `kaggle` paket. Instalirati: pip install kaggle\n"
            "Zatim postaviti API token na ~/.kaggle/kaggle.json"
        )

    api = KaggleApi()
    api.authenticate()
    print(f"[okcupid] Preuzimam {DATASET_SLUG} -> {TARGET_DIR}")
    api.dataset_download_files(DATASET_SLUG, path=str(TARGET_DIR), unzip=False, quiet=False)

    zip_path = next(TARGET_DIR.glob("*.zip"), None)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(TARGET_DIR)
        zip_path.unlink()
        print(f"[okcupid] Raspakovano u {TARGET_DIR}")

    csv_path = next(TARGET_DIR.glob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"CSV nije pronađen u {TARGET_DIR}")

    standard = TARGET_DIR / "okcupid_profiles.csv"
    if csv_path != standard:
        shutil.move(str(csv_path), standard)
    print(f"[okcupid] Spremljeno: {standard}")
    return standard


if __name__ == "__main__":
    download()
