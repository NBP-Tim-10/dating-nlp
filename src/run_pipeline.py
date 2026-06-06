"""Master skripta - pokreće cijeli pipeline prikupljanja i preprocesiranja.

Pokretanje:
    python -m src.run_pipeline                    # collect + synthetic + preprocess
    python -m src.run_pipeline --skip-scrape      # bez Google Play scraping-a
    python -m src.run_pipeline --only collect
    python -m src.run_pipeline --only synthetic
    python -m src.run_pipeline --only preprocess
    python -m src.run_pipeline --only recommendation

Faze:
    1) collect        - download i scraping svih izvora
    2) synthetic      - sintetičko generisanje
    3) preprocess     - čišćenje, tokenizacija, lematizacija
    4) recommendation - demo i evaluacija bio recommender sistema
"""
from __future__ import annotations

import argparse
import traceback
from typing import Callable


def _run(name: str, fn: Callable[[], object]) -> bool:
    print("\n" + "=" * 70)
    print(f">>> {name}")
    print("=" * 70)
    try:
        fn()
        return True
    except Exception:
        print(f"!!! {name} NIJE USPIO:")
        traceback.print_exc()
        return False


def collect(skip_scrape: bool = False) -> None:
    from src.data_collection.download_okcupid import download as dl_okc
    from src.data_collection.download_hate_speech import download as dl_hate
    from src.data_collection.download_sms_spam import download as dl_sms

    _run("OkCupid (Kaggle)", dl_okc)
    _run("Hate Speech (Davidson + tweet_eval)", dl_hate)
    _run("SMS Spam (UCI)", dl_sms)

    if not skip_scrape:
        from src.data_collection.scrape_tinder_reviews import scrape
        _run("Google Play recenzije dating aplikacija", scrape)


def synthetic() -> None:
    from src.data_collection.generate_synthetic import generate_all
    _run("Sintetičko generisanje", generate_all)


def preprocess() -> None:
    from src.preprocessing.preprocess_bios import run as p_bios
    from src.preprocessing.preprocess_hate_speech import run as p_hate
    from src.preprocessing.preprocess_bot_detection import run as p_scam
    from src.preprocessing.preprocess_sentiment import run as p_sent

    _run("Preprocess: bios (recommendation + topics)", p_bios)
    _run("Preprocess: hate speech", p_hate)
    _run("Preprocess: scam/bot detection", p_scam)
    _run("Preprocess: sentiment & emocije", p_sent)

def recommendation() -> None:
    from src.recommendation.run_bio_recommendation_demo import run_demo
    from src.recommendation.evaluate_bio_recommender import run_evaluation

    _run("Recommendation: bio recommender demo", lambda: run_demo(
        profile_indices=[0, 10, 25],
        top_k=5,
        max_profiles=5000,
    ))

    _run("Recommendation: bio recommender evaluacija", lambda: run_evaluation(
        max_profiles=5000,
        n_queries=30,
        top_k=5,
        random_state=42,
    ))

def main() -> None:
    parser = argparse.ArgumentParser(description="Master pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Preskoči Google Play scraping recenzija.")
    parser.add_argument(
        "--only",
        choices=["collect", "synthetic", "preprocess", "recommendation"],
        help="Pokreni samo jednu fazu.",
    )
    args = parser.parse_args()

    if args.only == "collect":
        collect(skip_scrape=args.skip_scrape)
    elif args.only == "synthetic":
        synthetic()
    elif args.only == "preprocess":
        preprocess()
    elif args.only == "recommendation":
        recommendation()
    else:
        collect(skip_scrape=args.skip_scrape)
        synthetic()
        preprocess()

    print("\n" + "=" * 70)
    print("PIPELINE GOTOV. Provjeri data/processed/ za izlazne CSV-ove.")
    print("=" * 70)


if __name__ == "__main__":
    main()
