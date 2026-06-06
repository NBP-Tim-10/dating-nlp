from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.recommendation.bio_recommender import BioRecommender
from src.utils.paths import REPORTS_DIR


OUT_DIR = REPORTS_DIR / "recommendation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_demo(
    profile_indices: list[int] | None = None,
    top_k: int = 5,
    max_profiles: int = 1000,
) -> dict[str, Path]:
    """Pokreće demo za Task 1: Sistem za preporučivanje bio profila.

    Generiše rezultate za dvije reprezentacije teksta:
    1. TF-IDF
    2. SBERT embeddings

    Rezultati se snimaju u reports/recommendation/.
    """
    if profile_indices is None:
        profile_indices = [0, 10, 25]

    print("=" * 80)
    print("TASK 1 — Bio recommender demo")
    print("=" * 80)

    recommender = BioRecommender.from_csv(max_profiles=max_profiles)
    print(f"Učitano profila: {len(recommender.data)}")

    all_results: list[pd.DataFrame] = []

    print("\n[1/2] Treniranje TF-IDF reprezentacije...")
    recommender.fit_tfidf()

    for profile_index in profile_indices:
        print(f"TF-IDF preporuke za profil index={profile_index}")
        result = recommender.recommend_by_index(
            profile_index=profile_index,
            method="tfidf",
            top_k=top_k,
        )
        result.insert(0, "query_profile_index", profile_index)
        result.insert(1, "method", "tfidf")
        all_results.append(result)

    print("\n[2/2] Računanje SBERT embeddings...")
    recommender.fit_sbert()

    for profile_index in profile_indices:
        print(f"SBERT preporuke za profil index={profile_index}")
        result = recommender.recommend_by_index(
            profile_index=profile_index,
            method="sbert",
            top_k=top_k,
        )
        result.insert(0, "query_profile_index", profile_index)
        result.insert(1, "method", "sbert")
        all_results.append(result)

    demo_results = pd.concat(all_results, ignore_index=True)

    out_csv = OUT_DIR / "bio_recommendation_demo_results.csv"
    demo_results.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Demo rezultati snimljeni u: {out_csv}")
    print("=" * 80)

    return {"demo_results": out_csv}


if __name__ == "__main__":
    run_demo()