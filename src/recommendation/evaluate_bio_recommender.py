from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.recommendation.bio_recommender import BioRecommender, Method
from src.utils.paths import REPORTS_DIR


OUT_DIR = REPORTS_DIR / "recommendation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _city(location: Any) -> str:
    """Iz 'san francisco, california' uzima samo grad."""
    loc = _safe_str(location)
    if not loc:
        return ""
    return loc.split(",")[0].strip()


def _same_value(query_value: Any, candidate_values: pd.Series) -> float:
    query_value = _safe_str(query_value)
    if not query_value:
        return np.nan

    values = candidate_values.apply(_safe_str)
    values = values[values != ""]
    if len(values) == 0:
        return np.nan

    return float((values == query_value).mean())


def _same_city(query_location: Any, candidate_locations: pd.Series) -> float:
    query_city = _city(query_location)
    if not query_city:
        return np.nan

    cities = candidate_locations.apply(_city)
    cities = cities[cities != ""]
    if len(cities) == 0:
        return np.nan

    return float((cities == query_city).mean())


def _avg_age_diff(query_age: Any, candidate_ages: pd.Series) -> float:
    try:
        q_age = float(query_age)
    except Exception:
        return np.nan

    ages = pd.to_numeric(candidate_ages, errors="coerce").dropna()
    if len(ages) == 0:
        return np.nan

    return float((ages - q_age).abs().mean())


def _top_indices_from_scores(scores: np.ndarray, query_index: int, top_k: int) -> np.ndarray:
    scores = scores.copy()
    scores[query_index] = -np.inf
    return np.argsort(scores)[::-1][:top_k]


def _random_indices(
    n_profiles: int,
    query_index: int,
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.array([i for i in range(n_profiles) if i != query_index])
    k = min(top_k, len(candidates))
    return rng.choice(candidates, size=k, replace=False)


def _metadata_metrics(
    data: pd.DataFrame,
    query_index: int,
    recommended_indices: np.ndarray,
    method: str,
    top_k: int,
) -> dict[str, Any]:
    query = data.iloc[query_index]
    recs = data.iloc[recommended_indices]

    metrics: dict[str, Any] = {
        "method": method,
        "query_index": query_index,
        "query_profile_id": query.get("profile_id", query_index),
        "top_k": top_k,
        "avg_age_diff": _avg_age_diff(query.get("age"), recs.get("age", pd.Series(dtype=object))),
    }

    if "location" in data.columns:
        metrics["same_city_at_k"] = _same_city(query.get("location"), recs["location"])

    for col in ["sex", "orientation", "status", "job", "education"]:
        if col in data.columns:
            metrics[f"same_{col}_at_k"] = _same_value(query.get(col), recs[col])

    return metrics


def _evaluate_recommender_method(
    recommender: BioRecommender,
    method: Method,
    query_indices: list[int],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_index in query_indices:
        scores = recommender._scores_for_existing_profile(query_index, method)
        top_indices = _top_indices_from_scores(scores, query_index, top_k)

        rows.append(
            _metadata_metrics(
                data=recommender.data,
                query_index=query_index,
                recommended_indices=top_indices,
                method=method,
                top_k=top_k,
            )
        )

    return pd.DataFrame(rows)


def _evaluate_random_baseline(
    recommender: BioRecommender,
    query_indices: list[int],
    top_k: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_index in query_indices:
        random_top_indices = _random_indices(
            n_profiles=len(recommender.data),
            query_index=query_index,
            top_k=top_k,
            rng=rng,
        )

        rows.append(
            _metadata_metrics(
                data=recommender.data,
                query_index=query_index,
                recommended_indices=random_top_indices,
                method="random_baseline",
                top_k=top_k,
            )
        )

    return pd.DataFrame(rows)


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in details.columns
        if c not in {"method", "query_index", "query_profile_id", "top_k"}
    ]

    summary = (
        details
        .groupby("method")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    summary.insert(1, "n_queries", details["query_index"].nunique())
    summary.insert(2, "top_k", int(details["top_k"].iloc[0]))

    return summary


def run_evaluation(
    max_profiles: int = 1000,
    n_queries: int = 30,
    top_k: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    print("=" * 80)
    print("TASK 1 — Evaluacija bio recommender sistema")
    print("=" * 80)

    rng = np.random.default_rng(random_state)

    recommender = BioRecommender.from_csv(
        max_profiles=max_profiles,
        random_state=random_state,
    )

    print(f"Učitano profila: {len(recommender.data)}")

    n_queries = min(n_queries, len(recommender.data))
    query_indices = rng.choice(len(recommender.data), size=n_queries, replace=False).tolist()

    print(f"Broj query profila za evaluaciju: {len(query_indices)}")
    print(f"Top-K: {top_k}")

    all_details: list[pd.DataFrame] = []

    print("\n[1/3] Random baseline...")
    all_details.append(
        _evaluate_random_baseline(
            recommender=recommender,
            query_indices=query_indices,
            top_k=top_k,
            rng=rng,
        )
    )

    print("\n[2/3] TF-IDF evaluacija...")
    recommender.fit_tfidf()
    all_details.append(
        _evaluate_recommender_method(
            recommender=recommender,
            method="tfidf",
            query_indices=query_indices,
            top_k=top_k,
        )
    )

    print("\n[3/3] SBERT evaluacija...")
    recommender.fit_sbert()
    all_details.append(
        _evaluate_recommender_method(
            recommender=recommender,
            method="sbert",
            query_indices=query_indices,
            top_k=top_k,
        )
    )

    details = pd.concat(all_details, ignore_index=True)
    summary = _summarize(details)

    details_path = OUT_DIR / "bio_recommendation_evaluation_details.csv"
    summary_path = OUT_DIR / "bio_recommendation_evaluation_summary.csv"

    details.to_csv(details_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("SAŽETAK EVALUACIJE")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nSnimljeno:")
    print(f"- {details_path}")
    print(f"- {summary_path}")

    return {
        "details": details_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OkCupid bio recommender")
    parser.add_argument("--max-profiles", type=int, default=1000)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    run_evaluation(
        max_profiles=args.max_profiles,
        n_queries=args.n_queries,
        top_k=args.top_k,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()