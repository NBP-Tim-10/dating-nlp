from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaning import clean_for_embeddings, full_token_pipeline
from src.utils.paths import PROCESSED_DIR


Method = Literal["tfidf", "sbert"]


INPUT_CSV = PROCESSED_DIR / "bios_for_embeddings.csv"


def _preview(text: str, max_len: int = 220) -> str:
    """Kratak prikaz bio teksta u rezultatima."""
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _prepare_query_for_tfidf(text: str) -> str:
    """Isti stil teksta kao tokens_str kolona iz preprocessing-a."""
    cleaned = clean_for_embeddings(text)
    tokens = full_token_pipeline(
        cleaned,
        cleaner=lambda x: x,
        drop_stopwords=True,
        do_lemmatize=True,
    )
    return " ".join(tokens)


@dataclass
class BioRecommender:
    """Content-based recommender za OkCupid bio profile.

    Podržava dvije forme prikaza teksta:
    1. TF-IDF: klasična, sparse reprezentacija teksta.
    2. SBERT: duboko kontekstualni sentence embeddings.

    Preporuke se računaju pomoću cosine similarity.
    """

    data: pd.DataFrame
    tfidf_vectorizer: TfidfVectorizer | None = None
    tfidf_matrix: object | None = None
    sbert_model: object | None = None
    sbert_embeddings: np.ndarray | None = None

    @classmethod
    def from_csv(
        cls,
        path: Path = INPUT_CSV,
        max_profiles: int | None = None,
        random_state: int = 42,
    ) -> "BioRecommender":
        if not path.exists():
            raise FileNotFoundError(
                f"Ne nalazim {path}. Prvo pokreni:\n"
                "python -m src.preprocessing.preprocess_bios"
            )

        df = pd.read_csv(path)

        required = {"bio_clean", "tokens_str"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Nedostaju kolone u {path}: {missing}")

        df = df.dropna(subset=["bio_clean", "tokens_str"]).copy()
        df = df[df["bio_clean"].astype(str).str.strip().str.len() > 0]
        df = df[df["tokens_str"].astype(str).str.strip().str.len() > 0]

        if max_profiles is not None and len(df) > max_profiles:
            df = df.sample(n=max_profiles, random_state=random_state).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if "profile_id" not in df.columns:
            df.insert(0, "profile_id", df.index)

        return cls(data=df)

    def fit_tfidf(
        self,
        max_features: int = 50_000,
        min_df: int = 2,
        max_df: float = 0.85,
    ) -> None:
        """Pravi TF-IDF matricu nad tokens_str kolonom."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )

        texts = self.data["tokens_str"].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def fit_sbert(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        """Pravi Sentence-BERT embeddings nad bio_clean kolonom.

        Prvo pokretanje može potrajati jer se model preuzima.
        """
        from sentence_transformers import SentenceTransformer

        self.sbert_model = SentenceTransformer(model_name)

        texts = self.data["bio_clean"].fillna("").astype(str).tolist()
        self.sbert_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _scores_for_existing_profile(self, profile_index: int, method: Method) -> np.ndarray:
        if profile_index < 0 or profile_index >= len(self.data):
            raise IndexError(f"profile_index mora biti između 0 i {len(self.data) - 1}")

        if method == "tfidf":
            if self.tfidf_matrix is None:
                raise RuntimeError("TF-IDF nije treniran. Pozovi fit_tfidf().")
            scores = cosine_similarity(
                self.tfidf_matrix[profile_index],
                self.tfidf_matrix,
            ).ravel()
            return scores

        if method == "sbert":
            if self.sbert_embeddings is None:
                raise RuntimeError("SBERT embeddings nisu izračunati. Pozovi fit_sbert().")
            query_vec = self.sbert_embeddings[profile_index]
            return self.sbert_embeddings @ query_vec

        raise ValueError(f"Nepoznata metoda: {method}")

    def _scores_for_new_text(self, text: str, method: Method) -> np.ndarray:
        if method == "tfidf":
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                raise RuntimeError("TF-IDF nije treniran. Pozovi fit_tfidf().")

            query_text = _prepare_query_for_tfidf(text)
            query_vec = self.tfidf_vectorizer.transform([query_text])
            return cosine_similarity(query_vec, self.tfidf_matrix).ravel()

        if method == "sbert":
            if self.sbert_model is None or self.sbert_embeddings is None:
                raise RuntimeError("SBERT embeddings nisu izračunati. Pozovi fit_sbert().")

            cleaned = clean_for_embeddings(text)
            query_vec = self.sbert_model.encode(
                [cleaned],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            return self.sbert_embeddings @ query_vec

        raise ValueError(f"Nepoznata metoda: {method}")

    def recommend_by_index(
        self,
        profile_index: int,
        method: Method = "tfidf",
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Vrati top_k najsličnijih profila za postojeći profil."""
        scores = self._scores_for_existing_profile(profile_index, method)

        # Ne želimo da sistem preporuči isti profil sam sebi.
        scores[profile_index] = -np.inf

        return self._format_results(scores, top_k=top_k)

    def recommend_by_text(
        self,
        text: str,
        method: Method = "tfidf",
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Vrati top_k najsličnijih profila za novi uneseni bio tekst."""
        scores = self._scores_for_new_text(text, method)
        return self._format_results(scores, top_k=top_k)

    def _format_results(self, scores: np.ndarray, top_k: int) -> pd.DataFrame:
        top_indices = np.argsort(scores)[::-1][:top_k]

        cols = [
            "profile_id",
            "age",
            "sex",
            "orientation",
            "status",
            "location",
            "job",
            "education",
            "bio_raw",
        ]
        cols = [c for c in cols if c in self.data.columns]

        result = self.data.iloc[top_indices][cols].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        rounded_scores = [round(float(score), 4) for score in scores[top_indices]]
        result.insert(1, "similarity_score", rounded_scores)
        result["bio_preview"] = result["bio_raw"].apply(_preview)

        if "bio_raw" in result.columns:
            result = result.drop(columns=["bio_raw"])

        return result.reset_index(drop=True)


def _print_query_profile(rec: BioRecommender, profile_index: int) -> None:
    row = rec.data.iloc[profile_index]
    print("\n" + "=" * 80)
    print(f"QUERY PROFIL index={profile_index}, profile_id={row.get('profile_id', profile_index)}")
    print("=" * 80)

    meta = []
    for col in ["age", "sex", "orientation", "status", "location", "job"]:
        if col in row and pd.notna(row[col]):
            meta.append(f"{col}={row[col]}")
    if meta:
        print(" | ".join(meta))

    print("\nBIO PREVIEW:")
    print(_preview(row.get("bio_raw", ""), max_len=700))


def _print_results(title: str, results: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    display_cols = [
        "rank",
        "similarity_score",
        "profile_id",
        "age",
        "sex",
        "orientation",
        "status",
        "location",
        "job",
        "bio_preview",
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    for _, row in results[display_cols].iterrows():
        print(f"\n#{row['rank']} | score={row['similarity_score']} | profile_id={row['profile_id']}")

        meta_parts = []
        for col in ["age", "sex", "orientation", "status", "location", "job"]:
            if col in row and pd.notna(row[col]):
                meta_parts.append(f"{col}={row[col]}")
        if meta_parts:
            print(" | ".join(meta_parts))

        print(row["bio_preview"])


def main() -> None:
    parser = argparse.ArgumentParser(description="OkCupid bio recommender")
    parser.add_argument("--method", choices=["tfidf", "sbert", "both"], default="tfidf")
    parser.add_argument("--index", type=int, default=0, help="Index profila iz processed CSV-a.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=5000,
        help="Limit za brži demo. Za full dataset stavi 0.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Opcionalno: novi bio tekst umjesto postojećeg profile index-a.",
    )

    args = parser.parse_args()

    max_profiles = None if args.max_profiles == 0 else args.max_profiles
    rec = BioRecommender.from_csv(max_profiles=max_profiles)

    print(f"Učitano profila: {len(rec.data)}")

    methods: list[Method]
    if args.method == "both":
        methods = ["tfidf", "sbert"]
    else:
        methods = [args.method]  # type: ignore[list-item]

    if args.text is None:
        _print_query_profile(rec, args.index)

    for method in methods:
        if method == "tfidf":
            print("\nTreniram TF-IDF reprezentaciju...")
            rec.fit_tfidf()
        elif method == "sbert":
            print("\nRačunam SBERT embeddings...")
            rec.fit_sbert()

        if args.text:
            results = rec.recommend_by_text(args.text, method=method, top_k=args.top_k)
            title = f"TOP {args.top_k} PREPORUKA ZA UNESENI TEKST — {method.upper()}"
        else:
            results = rec.recommend_by_index(args.index, method=method, top_k=args.top_k)
            title = f"TOP {args.top_k} PREPORUKA ZA PROFIL {args.index} — {method.upper()}"

        _print_results(title, results)


if __name__ == "__main__":
    main()