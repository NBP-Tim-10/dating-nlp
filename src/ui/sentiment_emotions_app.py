"""Streamlit UI za Task 5: sentiment + emocije.

Pokretanje iz root foldera projekta:
    streamlit run src/ui/sentiment_emotions_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_emotions.sentiment_emotion_analyzer import (  # noqa: E402
    PredictionResult,
    SentimentEmotionAnalyzer,
)


# ── Primjeri za demo ─────────────────────────────────────────────────────────

EXAMPLES_SENTIMENT = {
    "Negativna recenzija": (
        "This app is full of fake profiles and expensive features. "
        "I paid for premium and still got no real matches. Very disappointing."
    ),
    "Neutralna recenzija": (
        "The app is okay so far. I just started using it and I am still trying to understand "
        "whether the paid features are worth it."
    ),
    "Pozitivna recenzija": (
        "Great dating app, easy to use and I met some genuinely nice people here. "
        "The interface is simple and the matches feel relevant."
    ),
}

EXAMPLES_EMOTION = {
    "Zainteresovanost": "That sounds really interesting, tell me more about it.",
    "Sreća": "Haha I love that, this conversation is actually making my day better.",
    "Sarkazam": "Oh sure, because sending one-word replies is exactly how great conversations start.",
    "Frustracija": "I already answered that twice and it feels like you are not listening.",
    "Dosada": "k",
    "Ljubazno odbijanje": "Thanks for the message, but I do not think we are a good match."
}

SENTIMENT_ORDER = ["negative", "neutral", "positive"]


# ── Konfiguracija stranice ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment + emocije",
    page_icon="💬",
    layout="wide",
)


# ── Cache modela ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_base_analyzer() -> SentimentEmotionAnalyzer:
    analyzer = SentimentEmotionAnalyzer.from_csv()
    analyzer.prepare_emotion_splits()
    return analyzer


@st.cache_resource(show_spinner=False)
def load_sentiment_tfidf_analyzer() -> SentimentEmotionAnalyzer:
    analyzer = SentimentEmotionAnalyzer.from_csv()
    analyzer.fit_sentiment_dummy()
    analyzer.fit_sentiment_tfidf()
    return analyzer


@st.cache_resource(show_spinner=False)
def load_sentiment_sbert_analyzer() -> SentimentEmotionAnalyzer:
    analyzer = SentimentEmotionAnalyzer.from_csv()
    analyzer.fit_sentiment_sbert()
    return analyzer


@st.cache_resource(show_spinner=False)
def load_emotion_analyzer() -> SentimentEmotionAnalyzer:
    analyzer = SentimentEmotionAnalyzer.from_csv()
    analyzer.fit_emotion_tfidf()
    return analyzer


# ── Pomoćne funkcije za prikaz ───────────────────────────────────────────────

def _label_style(label: str) -> tuple[str, str, str]:
    mapping = {
        "negative": ("🔴 NEGATIVE", "#c0392b", "#fff5f5"),
        "neutral": ("🟡 NEUTRAL", "#b7950b", "#fffbeb"),
        "positive": ("🟢 POSITIVE", "#1e8449", "#f0fff4"),
    }
    return mapping.get(label, (f"💬 {label.upper()}", "#2c3e50", "#f7f9fb"))


def result_card(result: PredictionResult, title: str) -> None:
    label_title, colour, background = _label_style(result.label)
    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 8px;
            background: {background};
        ">
            <h3 style="color:{colour}; margin:0 0 8px 0;">{label_title}</h3>
            <p style="margin:0; color:#333;">
                Skor pouzdanosti:
                <strong style="color:{colour};">{result.confidence:.1%}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(result.confidence, 0.0), 1.0))
    if result.note:
        st.caption(result.note)


def scores_dataframe(scores: dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [{"label": label, "score": score} for label, score in scores.items()]
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def show_tfidf_features(analyzer: SentimentEmotionAnalyzer, text: str, predicted_label: str) -> None:
    features = analyzer.tfidf_top_sentiment_features(text, predicted_label, top_n=8)
    positive = [(token, weight) for token, weight in features if weight > 0]
    if not positive:
        st.caption("Nisu pronađeni jaki TF-IDF indikatori za ovu klasu.")
        return

    st.caption(f"**Tokeni/n-grami koji podržavaju klasu `{predicted_label}`:**")
    max_weight = max(weight for _, weight in positive)
    for token, weight in positive[:6]:
        bar_pct = min(int(weight / max_weight * 100), 100)
        st.markdown(
            f"`{token}` &nbsp;"
            f"<span style='display:inline-block;width:{bar_pct}%;height:8px;"
            f"background:#566573;border-radius:4px;vertical-align:middle;'></span> "
            f"<small>{weight:.3f}</small>",
            unsafe_allow_html=True,
        )


def set_example_state(state_key: str, examples: dict[str, str], prefix: str) -> None:
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        with col:
            if st.button(name, key=f"{prefix}_{name}", use_container_width=True):
                st.session_state[state_key] = text


# ── Sentiment tab ────────────────────────────────────────────────────────────

def sentiment_tab() -> None:
    st.header("Sentiment analiza Google Play recenzija")
    st.markdown(
        """
        Ovaj dio demonstrira klasifikaciju recenzije dating aplikacije u tri klase:
        **negative**, **neutral** i **positive**. Labela sentimenta je izvedena iz ratinga
        korisnika, pa neutralna klasa ostaje najteža i najmanje pouzdana.
        """
    )

    method_choice = st.selectbox(
        "Metoda klasifikacije",
        options=["Obje metode", "TF-IDF", "Sentence-BERT"],
        index=0,
        key="sentiment_method",
    )

    st.subheader("Unos recenzije")
    set_example_state("sentiment_input", EXAMPLES_SENTIMENT, "sentiment_example")

    input_text = st.text_area(
        "Tekst recenzije",
        value=st.session_state.get("sentiment_input", ""),
        height=150,
        placeholder="Unesi recenziju dating aplikacije ili odaberi primjer iznad...",
        key="sentiment_input",
    )

    analyze_btn = st.button("Analiziraj sentiment", type="primary", disabled=not input_text.strip())

    if not analyze_btn:
        st.info("Unesi tekst ili odaberi primjer, pa klikni **Analiziraj sentiment**.")
        return

    st.divider()
    st.subheader("Rezultati sentiment klasifikacije")

    run_tfidf = method_choice in ("TF-IDF", "Obje metode")
    run_sbert = method_choice in ("Sentence-BERT", "Obje metode")

    tfidf_result = None
    sbert_result = None
    tfidf_analyzer = None

    if run_tfidf:
        with st.spinner("Učitavam / treniram TF-IDF + LinearSVC model..."):
            try:
                tfidf_analyzer = load_sentiment_tfidf_analyzer()
                tfidf_result = tfidf_analyzer.predict_sentiment_tfidf(input_text)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

    if run_sbert:
        with st.spinner(
            "Učitavam Sentence-BERT model... Prvo pokretanje može potrajati jer se model preuzima."
        ):
            try:
                sbert_analyzer = load_sentiment_sbert_analyzer()
                sbert_result = sbert_analyzer.predict_sentiment_sbert(input_text)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except ModuleNotFoundError:
                st.error(
                    "Nedostaje paket sentence-transformers. Instaliraj ga komandom: "
                    "pip install sentence-transformers"
                )
                return

    if run_tfidf and run_sbert:
        col1, col2 = st.columns(2)
        with col1:
            assert tfidf_result is not None
            result_card(tfidf_result, "TF-IDF + LinearSVC")
            st.dataframe(scores_dataframe(tfidf_result.scores), use_container_width=True, hide_index=True)
            if tfidf_analyzer is not None:
                show_tfidf_features(tfidf_analyzer, input_text, tfidf_result.label)
        with col2:
            assert sbert_result is not None
            result_card(sbert_result, "Sentence-BERT + Logistic Regression")
            st.dataframe(scores_dataframe(sbert_result.scores), use_container_width=True, hide_index=True)
    elif run_tfidf:
        assert tfidf_result is not None
        result_card(tfidf_result, "TF-IDF + LinearSVC")
        st.dataframe(scores_dataframe(tfidf_result.scores), use_container_width=True, hide_index=True)
        if tfidf_analyzer is not None:
            show_tfidf_features(tfidf_analyzer, input_text, tfidf_result.label)
    elif run_sbert:
        assert sbert_result is not None
        result_card(sbert_result, "Sentence-BERT + Logistic Regression")
        st.dataframe(scores_dataframe(sbert_result.scores), use_container_width=True, hide_index=True)

    with st.expander("Napomena o neutralnoj klasi"):
        st.markdown(
            """
            Dataset recenzija je nebalansiran: negativne recenzije dominiraju, pozitivnih ima manje,
            a neutralna klasa je najmanja. Neutral labela je izvedena iz ratinga 3, što ne mora uvijek
            značiti stvarno neutralan tekst. Zato je kod evaluacije važniji **macro-F1** nego sama accuracy.
            """
        )


# ── Emotion tab ──────────────────────────────────────────────────────────────

def emotion_tab() -> None:
    st.header("Analiza emocija u sintetičkim razgovorima")
    st.markdown(
        """
        Ovaj dio demonstrira prepoznavanje emocije u jednoj poruci i prikaz emocionalne
        progresije kroz tok razgovora. Dataset razgovora je **sintetički**, pa rezultate
        emotion classifiera treba tumačiti kao demo, a ne kao dokaz pouzdane generalizacije.
        """
    )

    try:
        base = load_base_analyzer()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    summary = base.emotion_dataset_summary()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Redova", summary["n_rows"])
    metric_cols[1].metric("Razgovora", summary["n_conversations"])
    metric_cols[2].metric("Unique tekstova", summary["n_unique_texts"])
    metric_cols[3].metric("Emocija", summary["n_emotions"])
    metric_cols[4].metric("Flow tipova", summary["n_flows"])

    st.warning(
        "Emotion dataset ima mali broj jedinstvenih tekstualnih obrazaca u odnosu na broj redova. "
        "Zato model može naučiti šablone, a ne stvarnu emocionalnu semantiku."
    )

    with st.expander("Provjera raznovrsnosti emotion dataseta"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Unique tekstovi po emociji**")
            st.dataframe(summary["unique_by_emotion"], use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Overlap tekstova sa train skupom**")
            st.dataframe(summary["overlap"], use_container_width=True, hide_index=True)

    st.subheader("Unos poruke za emotion demo")
    set_example_state("emotion_input", EXAMPLES_EMOTION, "emotion_example")

    input_text = st.text_area(
        "Tekst poruke iz razgovora",
        value=st.session_state.get("emotion_input", ""),
        height=130,
        placeholder="Unesi poruku ili odaberi primjer iznad...",
        key="emotion_input",
    )

    analyze_btn = st.button("Analiziraj emociju", type="primary", disabled=not input_text.strip())

    if analyze_btn:
        with st.spinner("Učitavam / treniram demonstracioni emotion classifier..."):
            try:
                analyzer = load_emotion_analyzer()
                result = analyzer.predict_emotion_tfidf(input_text)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

        st.divider()
        result_card(result, "TF-IDF + LinearSVC emotion demo")
        st.dataframe(scores_dataframe(result.scores), use_container_width=True, hide_index=True)
    else:
        st.info("Unesi poruku ili odaberi primjer, pa klikni **Analiziraj emociju**.")

    st.divider()
    st.subheader("Emocionalna progresija kroz flow tipove")
    st.caption(
        "Emotion score je ručno definisana mapa emocija i služi samo za vizualizaciju toka razgovora."
    )
    progression = base.emotion_progression()
    st.line_chart(progression)

    with st.expander("Zašto emotion classifier nije predstavljen kao jak model?"):
        st.markdown(
            """
            Sintetički conversation dataset je koristan za demonstraciju tokova kao što su
            `engagement_drop`, `frustration_escalation`, `sarcasm` i `positive_match`.
            Međutim, pošto se isti ili vrlo slični tekstovi često ponavljaju, klasifikator može
            postići veoma visoke rezultate samo memorisanjem šablona. Zato je ovaj dio UI-ja
            predstavljen kao **demonstracija nad sintetičkim podacima**, dok je sentiment dio
            metodološki jači jer koristi realne Google Play recenzije.
            """
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("💬 Task 5 — Sentiment + emocije")
    st.markdown(
        """
        UI demonstracija za peti NLP task u projektu: analiza korisničkog sentimenta iz
        Google Play recenzija i demonstracija emocija u sintetičkim dating razgovorima.
        """
    )

    st.sidebar.header("O tasku")
    st.sidebar.markdown(
        """
        **Sentiment:** realne Google Play recenzije dating aplikacija  
        **Emocije:** sintetički razgovori sa definisanim flow tipovima  
        **Modeli:** TF-IDF, Sentence-BERT, Logistic Regression, LinearSVC
        """
    )

    tab_sentiment, tab_emotion = st.tabs([
        "Sentiment recenzija",
        "Emocije u razgovoru",
    ])

    with tab_sentiment:
        sentiment_tab()

    with tab_emotion:
        emotion_tab()


if __name__ == "__main__":
    main()
