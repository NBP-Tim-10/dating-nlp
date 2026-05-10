"""Scraping Google Play recenzija dating aplikacija.

Za task 'Dinamička analiza sentimenta i emocija' potreban je korpus
mišljenja korisnika nad konkretnim dating aplikacijama. Google Play
recenzije sadrže ocjenu (1-5) koja služi kao slabi label za sentiment.

Aplikacije koje skupljamo (pokušaj reprezentativnog uzorka tržišta):
    - com.tinder              (Tinder)
    - com.bumble.app          (Bumble)
    - com.hinge.app           (Hinge)
    - com.coffeemeetsbagel.app (CMB)

Koristi `google-play-scraper` paket (nezvanični wrapper Play store-a).
Ako paket nije instaliran ili scraping ne uspije, skripta gracefully
preskače i logira upozorenje.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.paths import RAW_DIR

TARGET_DIR = RAW_DIR / "app_reviews"

APPS: dict[str, str] = {
    "tinder": "com.tinder",
    "bumble": "com.bumble.app",
    "hinge": "co.hinge.app",
    "coffee_meets_bagel": "com.coffeemeetsbagel",
}

REVIEWS_PER_APP = 2000
LANG = "en"
COUNTRY = "us"


def scrape() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from google_play_scraper import Sort, reviews
    except ImportError:
        sys.exit(
            "Nedostaje `google-play-scraper` paket. "
            "Instalirati: pip install google-play-scraper"
        )

    all_frames: list[pd.DataFrame] = []
    for app_name, app_id in APPS.items():
        print(f"[reviews] {app_name} ({app_id}) - ciljam {REVIEWS_PER_APP} recenzija")
        try:
            result, _ = reviews(
                app_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=REVIEWS_PER_APP,
            )
        except Exception as exc:
            print(f"[reviews] {app_name}: greška -> {exc}")
            continue

        if not result:
            print(f"[reviews] {app_name}: 0 vraćenih recenzija")
            continue

        df = pd.DataFrame(result)[["reviewId", "userName", "score", "content", "at"]]
        df["app"] = app_name
        df["app_id"] = app_id
        df = df.dropna(subset=["content"])
        df = df.rename(columns={"score": "rating", "content": "text", "at": "review_date"})
        all_frames.append(df)
        print(f"[reviews] {app_name}: skinuto {len(df)} recenzija")
        time.sleep(2)

    if not all_frames:
        raise RuntimeError("Nije uspjelo skinuti nijednu recenziju.")

    full = pd.concat(all_frames, ignore_index=True)
    full = full.drop_duplicates(subset=["reviewId"])

    full["sentiment"] = pd.cut(
        full["rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        include_lowest=True,
    )

    out = TARGET_DIR / "dating_app_reviews.csv"
    full.to_csv(out, index=False, encoding="utf-8")
    print(f"[reviews] ukupno {len(full)} recenzija -> {out}")
    print(f"          sentiment: {full['sentiment'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    scrape()
