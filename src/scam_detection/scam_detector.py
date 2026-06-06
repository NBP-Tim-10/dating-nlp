from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.text_cleaning import clean_for_classification
from src.utils.paths import PROCESSED_DIR

Method = Literal["tfidf", "sbert"]

_TRAIN_CSV = PROCESSED_DIR / "scam_detection_train.csv"


class ScamDetector:
    """Drži oba klasifikatora i može klasificirati novi tekst."""

    def __init__(self, train: pd.DataFrame) -> None:
        self._train = train
        self._vectorizer: TfidfVectorizer | None = None
        self._clf_tfidf: LogisticRegression | None = None
        self._sbert = None
        self._clf_sbert: LogisticRegression | None = None

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, csv_path: Path = _TRAIN_CSV) -> "ScamDetector":
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Nedostaje {csv_path}.\n"
                "Pokreni: python -m src.run_pipeline --only preprocess"
            )
        train = pd.read_csv(csv_path)
        return cls(train)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit_tfidf(self) -> None:
        """Trenira TF-IDF vektorizator i logističku regresiju."""
        X = self._train["text_clean"].fillna("").tolist()
        y = self._train["label"].values

        self._vectorizer = TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X_vec = self._vectorizer.fit_transform(X)

        self._clf_tfidf = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        )
        self._clf_tfidf.fit(X_vec, y)

    def fit_sbert(self, progress_callback=None) -> None:
        """Enkodira trening skup SBERT modelom i trenira LR."""
        from sentence_transformers import SentenceTransformer

        self._sbert = SentenceTransformer("all-MiniLM-L6-v2")

        texts = self._train["text"].fillna("").tolist()
        n = len(texts)

        if progress_callback:
            batch_size = 64
            embeddings = []
            for start in range(0, n, batch_size):
                batch = texts[start : start + batch_size]
                embeddings.append(
                    self._sbert.encode(batch, show_progress_bar=False)
                )
                progress_callback(min(start + batch_size, n) / n)
            E_train = np.vstack(embeddings)
        else:
            E_train = self._sbert.encode(texts, batch_size=64, show_progress_bar=False)

        y = self._train["label"].values
        self._clf_sbert = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        )
        self._clf_sbert.fit(E_train, y)

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_tfidf(self, text: str) -> tuple[int, float]:
        """Vraća (label, vjerovatnoća_scam). Zahtijeva fit_tfidf()."""
        if self._vectorizer is None or self._clf_tfidf is None:
            raise RuntimeError("Pozovi fit_tfidf() prije predict_tfidf().")

        cleaned = clean_for_classification(text)
        X = self._vectorizer.transform([cleaned])
        label = int(self._clf_tfidf.predict(X)[0])
        prob = float(self._clf_tfidf.predict_proba(X)[0][1])
        return label, prob

    def predict_sbert(self, text: str) -> tuple[int, float]:
        """Vraća (label, vjerovatnoća_scam). Zahtijeva fit_sbert()."""
        if self._sbert is None or self._clf_sbert is None:
            raise RuntimeError("Pozovi fit_sbert() prije predict_sbert().")

        embedding = self._sbert.encode([text])
        label = int(self._clf_sbert.predict(embedding)[0])
        prob = float(self._clf_sbert.predict_proba(embedding)[0][1])
        return label, prob

    def tfidf_top_scam_features(self, text: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Vraća top N tokena koji najviše indiciraju scam za dati tekst."""
        if self._vectorizer is None or self._clf_tfidf is None:
            return []

        cleaned = clean_for_classification(text)
        X = self._vectorizer.transform([cleaned])
        feature_names = np.array(self._vectorizer.get_feature_names_out())
        coefs = self._clf_tfidf.coef_[0]

        nonzero = X.nonzero()[1]
        if len(nonzero) == 0:
            return []

        scores = [(feature_names[i], float(X[0, i] * coefs[i])) for i in nonzero]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
