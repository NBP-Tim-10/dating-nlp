"""Backend klasa za Task 5 — sentiment + emocije.

Koristi se u Streamlit UI-ju ``src/ui/sentiment_emotions_app.py``.
Implementacija prati notebook ``05_sentiment_emotions.ipynb``:

- sentiment recenzija: TF-IDF + LinearSVC i Sentence-BERT + Logistic Regression
- emocije u razgovorima: demonstracioni TF-IDF + LinearSVC model nad sintetičkim podacima

Važna napomena: emotion classifier je namjerno predstavljen kao demonstracija, jer je
conversation dataset sintetički i sadrži mali broj jedinstvenih tekstualnih obrazaca.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing.text_cleaning import clean_for_sentiment
from src.utils.paths import PROCESSED_DIR


SentimentMethod = Literal["tfidf", "sbert"]

REVIEWS_TRAIN_CSV = PROCESSED_DIR / "sentiment_reviews_train.csv"
REVIEWS_VAL_CSV = PROCESSED_DIR / "sentiment_reviews_val.csv"
REVIEWS_TEST_CSV = PROCESSED_DIR / "sentiment_reviews_test.csv"
CONVERSATIONS_CSV = PROCESSED_DIR / "sentiment_conversations_clean.csv"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

EMOTION_SCORE = {
    "angry": -3.0,
    "frustrated": -2.5,
    "annoyed": -2.0,
    "bored": -1.5,
    "disengaged": -1.0,
    "sarcastic": -0.5,
    "neutral": 0.0,
    "polite": 0.5,
    "interested": 1.0,
    "playful": 1.5,
    "happy": 2.0,
    "excited": 2.5,
}


@dataclass
class PredictionResult:
    """Jednostavan rezultat predikcije za prikaz u UI-ju."""

    label: str
    confidence: float
    scores: dict[str, float]
    note: str = ""


def _softmax(values: np.ndarray) -> np.ndarray:
    """Pretvara LinearSVC margine u pseudo-confidence vrijednosti.

    Ovo nisu kalibrisane vjerovatnoće, nego normalizovani skorovi korisni za UI prikaz.
    """
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = exp_values.sum()
    if denom == 0:
        return np.ones_like(exp_values) / len(exp_values)
    return exp_values / denom


def _clean_text_series(series: pd.Series) -> list[str]:
    return series.fillna("").astype(str).tolist()


class SentimentEmotionAnalyzer:
    """Drži modele za Task 5 i metode za UI predikciju.

    Klasa je napravljena po istom obrascu kao ``BioRecommender`` / ``ScamDetector``:
    ``from_csv`` učitava procesirane podatke, ``fit_*`` trenira reprezentaciju i model,
    a ``predict_*`` metode se koriste iz Streamlit aplikacije.
    """

    def __init__(
        self,
        reviews_train: pd.DataFrame,
        reviews_val: pd.DataFrame,
        reviews_test: pd.DataFrame,
        conversations: pd.DataFrame,
    ) -> None:
        self.reviews_train = reviews_train
        self.reviews_val = reviews_val
        self.reviews_test = reviews_test
        self.conversations = conversations

        self.sentiment_tfidf: Pipeline | None = None
        self.sentiment_dummy: DummyClassifier | None = None
        self.sbert_model = None
        self.sentiment_sbert_clf: LogisticRegression | None = None

        self.conv_train: pd.DataFrame | None = None
        self.conv_val: pd.DataFrame | None = None
        self.conv_test: pd.DataFrame | None = None
        self.emotion_tfidf: Pipeline | None = None

    @classmethod
    def from_csv(
        cls,
        reviews_train_csv: Path = REVIEWS_TRAIN_CSV,
        reviews_val_csv: Path = REVIEWS_VAL_CSV,
        reviews_test_csv: Path = REVIEWS_TEST_CSV,
        conversations_csv: Path = CONVERSATIONS_CSV,
    ) -> "SentimentEmotionAnalyzer":
        missing = [
            str(path)
            for path in [reviews_train_csv, reviews_val_csv, reviews_test_csv, conversations_csv]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Nedostaju potrebni processed fajlovi:\n"
                + "\n".join(missing)
                + "\n\nPokreni: python -m src.run_pipeline --only preprocess"
            )

        reviews_train = pd.read_csv(reviews_train_csv)
        reviews_val = pd.read_csv(reviews_val_csv)
        reviews_test = pd.read_csv(reviews_test_csv)
        conversations = pd.read_csv(conversations_csv)

        return cls(
            reviews_train=reviews_train,
            reviews_val=reviews_val,
            reviews_test=reviews_test,
            conversations=conversations,
        )

    # ------------------------------------------------------------------
    # Sentiment modeli
    # ------------------------------------------------------------------
    def fit_sentiment_dummy(self) -> None:
        self.sentiment_dummy = DummyClassifier(strategy="most_frequent")
        self.sentiment_dummy.fit(
            self.reviews_train["text_clean"].fillna(""),
            self.reviews_train["sentiment_label"],
        )

    def fit_sentiment_tfidf(self) -> None:
        """TF-IDF + LinearSVC, isti osnovni pristup kao u notebooku 05."""
        self.sentiment_tfidf = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=30_000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        X_train = self.reviews_train["text_clean"].fillna("")
        y_train = self.reviews_train["sentiment_label"]
        self.sentiment_tfidf.fit(X_train, y_train)

    def fit_sentiment_sbert(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        """Sentence-BERT embeddings + Logistic Regression.

        Prvo pokretanje može potrajati jer se model preuzima sa Hugging Face-a.
        """
        from sentence_transformers import SentenceTransformer

        self.sbert_model = SentenceTransformer(model_name)

        X_train = _clean_text_series(self.reviews_train["text"])
        y_train = self.reviews_train["sentiment_label"]

        E_train = self.sbert_model.encode(
            X_train,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.sentiment_sbert_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.sentiment_sbert_clf.fit(E_train, y_train)

    def predict_sentiment_tfidf(self, text: str) -> PredictionResult:
        if self.sentiment_tfidf is None:
            raise RuntimeError("Pozovi fit_sentiment_tfidf() prije predikcije.")

        cleaned = clean_for_sentiment(text)
        label = str(self.sentiment_tfidf.predict([cleaned])[0])

        clf: LinearSVC = self.sentiment_tfidf.named_steps["clf"]
        classes = list(clf.classes_)
        margins = self.sentiment_tfidf.decision_function([cleaned])[0]
        probs = _softmax(np.asarray(margins))
        scores = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        return PredictionResult(
            label=label,
            confidence=float(scores.get(label, max(scores.values()))),
            scores=scores,
            note="TF-IDF + LinearSVC koristi pseudo-confidence na osnovu margina, ne kalibrisane vjerovatnoće.",
        )

    def predict_sentiment_sbert(self, text: str) -> PredictionResult:
        if self.sbert_model is None or self.sentiment_sbert_clf is None:
            raise RuntimeError("Pozovi fit_sentiment_sbert() prije predikcije.")

        embedding = self.sbert_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        probs = self.sentiment_sbert_clf.predict_proba(embedding)[0]
        classes = list(self.sentiment_sbert_clf.classes_)
        scores = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        label = max(scores, key=scores.get)

        return PredictionResult(
            label=label,
            confidence=float(scores[label]),
            scores=scores,
            note="Sentence-BERT + Logistic Regression vraća klasne vjerovatnoće logističke regresije.",
        )

    def tfidf_top_sentiment_features(
        self,
        text: str,
        predicted_label: str,
        top_n: int = 8,
    ) -> list[tuple[str, float]]:
        """Tokeni/n-grami koji najviše podržavaju predviđenu sentiment klasu."""
        if self.sentiment_tfidf is None:
            return []

        vectorizer: TfidfVectorizer = self.sentiment_tfidf.named_steps["tfidf"]
        clf: LinearSVC = self.sentiment_tfidf.named_steps["clf"]

        if predicted_label not in clf.classes_:
            return []

        cleaned = clean_for_sentiment(text)
        X = vectorizer.transform([cleaned])
        nonzero = X.nonzero()[1]
        if len(nonzero) == 0:
            return []

        class_idx = list(clf.classes_).index(predicted_label)
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = clf.coef_[class_idx]

        scores = [(feature_names[i], float(X[0, i] * coefs[i])) for i in nonzero]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_n]

    def sentiment_test_summary(self) -> pd.DataFrame:
        """Kratka tabela testnih metrika za modele koji su već fitovani."""
        rows: list[dict[str, float | str]] = []
        y_test = self.reviews_test["sentiment_label"]

        if self.sentiment_dummy is not None:
            pred = self.sentiment_dummy.predict(self.reviews_test["text_clean"].fillna(""))
            rows.append(self._summary_row("Dummy baseline", y_test, pred))

        if self.sentiment_tfidf is not None:
            pred = self.sentiment_tfidf.predict(self.reviews_test["text_clean"].fillna(""))
            rows.append(self._summary_row("TF-IDF + LinearSVC", y_test, pred))

        if self.sbert_model is not None and self.sentiment_sbert_clf is not None:
            E_test = self.sbert_model.encode(
                _clean_text_series(self.reviews_test["text"]),
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            pred = self.sentiment_sbert_clf.predict(E_test)
            rows.append(self._summary_row("Sentence-BERT + LR", y_test, pred))

        return pd.DataFrame(rows)

    @staticmethod
    def _summary_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
        report = classification_report(
            y_true,
            y_pred,
            labels=SENTIMENT_LABELS,
            output_dict=True,
            zero_division=0,
        )
        return {
            "model": model_name,
            "accuracy": round(float(report["accuracy"]), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "f1_negative": round(float(report["negative"]["f1-score"]), 4),
            "f1_neutral": round(float(report["neutral"]["f1-score"]), 4),
            "f1_positive": round(float(report["positive"]["f1-score"]), 4),
        }

    # ------------------------------------------------------------------
    # Emotion demo
    # ------------------------------------------------------------------
    def prepare_emotion_splits(self) -> None:
        """Group split po conversation_id, kao u notebooku 05."""
        if self.conv_train is not None:
            return

        conversations = self.conversations.copy()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, tmp_idx = next(
            gss.split(
                conversations,
                groups=conversations["conversation_id"],
            )
        )
        conv_train = conversations.iloc[train_idx].reset_index(drop=True)
        conv_tmp = conversations.iloc[tmp_idx].reset_index(drop=True)

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(
            gss2.split(
                conv_tmp,
                groups=conv_tmp["conversation_id"],
            )
        )

        self.conv_train = conv_train
        self.conv_val = conv_tmp.iloc[val_idx].reset_index(drop=True)
        self.conv_test = conv_tmp.iloc[test_idx].reset_index(drop=True)

    def fit_emotion_tfidf(self) -> None:
        """Demonstracioni emotion classifier nad sintetičkim razgovorima."""
        self.prepare_emotion_splits()
        assert self.conv_train is not None

        self.emotion_tfidf = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20_000,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        self.emotion_tfidf.fit(
            self.conv_train["text_clean"].fillna(""),
            self.conv_train["emotion"],
        )

    def predict_emotion_tfidf(self, text: str) -> PredictionResult:
        if self.emotion_tfidf is None:
            raise RuntimeError("Pozovi fit_emotion_tfidf() prije predikcije.")

        cleaned = clean_for_sentiment(text)
        label = str(self.emotion_tfidf.predict([cleaned])[0])

        clf: LinearSVC = self.emotion_tfidf.named_steps["clf"]
        classes = list(clf.classes_)
        margins = self.emotion_tfidf.decision_function([cleaned])[0]
        probs = _softmax(np.asarray(margins))
        scores = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

        return PredictionResult(
            label=label,
            confidence=float(scores.get(label, max(scores.values()))),
            scores=scores,
            note=(
                "Ovo je demonstracioni rezultat nad sintetičkim i šablonizovanim razgovorima. "
                "Ne treba ga tumačiti kao pouzdan model za stvarne korisničke razgovore."
            ),
        )

    def emotion_dataset_summary(self) -> dict[str, object]:
        self.prepare_emotion_splits()
        assert self.conv_train is not None and self.conv_val is not None and self.conv_test is not None

        train_texts = set(self.conv_train["text_clean"].fillna(""))
        val_texts = set(self.conv_val["text_clean"].fillna(""))
        test_texts = set(self.conv_test["text_clean"].fillna(""))

        unique_by_emotion = (
            self.conversations.groupby("emotion")["text_clean"]
            .nunique()
            .sort_values()
            .rename("unique_text_count")
            .reset_index()
        )

        overlap = pd.DataFrame(
            [
                {
                    "split": "validation",
                    "unique_texts": len(val_texts),
                    "overlap_with_train": len(val_texts & train_texts),
                    "overlap_percent": round(100 * len(val_texts & train_texts) / max(len(val_texts), 1), 2),
                },
                {
                    "split": "test",
                    "unique_texts": len(test_texts),
                    "overlap_with_train": len(test_texts & train_texts),
                    "overlap_percent": round(100 * len(test_texts & train_texts) / max(len(test_texts), 1), 2),
                },
            ]
        )

        return {
            "n_rows": int(len(self.conversations)),
            "n_conversations": int(self.conversations["conversation_id"].nunique()),
            "n_unique_texts": int(self.conversations["text_clean"].nunique()),
            "n_emotions": int(self.conversations["emotion"].nunique()),
            "n_flows": int(self.conversations["flow"].nunique()),
            "unique_by_emotion": unique_by_emotion,
            "overlap": overlap,
        }

    def emotion_progression(self) -> pd.DataFrame:
        df = self.conversations.copy()
        df["emotion_score"] = df["emotion"].map(EMOTION_SCORE)
        trajectory = (
            df.groupby(["turn_idx", "flow"], as_index=False)["emotion_score"]
            .mean()
            .pivot(index="turn_idx", columns="flow", values="emotion_score")
            .sort_index()
        )
        return trajectory
