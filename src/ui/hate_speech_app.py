"""Streamlit UI za demo detekcije govora mržnje.

Pokretanje:
    streamlit run src/ui/hate_speech_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hate_speech.hate_detector import HateDetector

# ── Primjeri teksta ───────────────────────────────────────────────────────────

EXAMPLES_HATE = {
    "Rasistički tvit": (
        "These people are ruining everything. They don't belong here and never will. "
        "Disgusting parasites destroying our country and our values."
    ),
    "Direktna prijetnja grupi": (
        "All those immigrants are criminals. They come here to take our jobs, "
        "assault our women and destroy our culture. They should all be deported now!"
    ),
    "Ksenofobni komentar": (
        "Our neighborhood used to be safe before they moved in. "
        "These people are subhuman and they should go back to where they came from."
    ),
}

EXAMPLES_NOT_HATE = {
    "Politička kritika": (
        "I strongly disagree with the government's immigration policy. "
        "We need better integration programs and more humane border procedures."
    ),
    "Dating profil": (
        "Software engineer who loves hiking and cooking. "
        "Looking for someone to explore the city with and share good food and good conversations."
    ),
    "Normalna poruka": (
        "Had such a great time at the concert last night! "
        "The band was absolutely amazing — can't wait for the next show."
    ),
}

# ── Konfiguracija stranice ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Detekcija govora mržnje",
    page_icon="🛡️",
    layout="wide",
)

# ── Cache — load i fit modela ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_tfidf_detector() -> HateDetector:
    det = HateDetector.from_csv()
    det.fit_tfidf()
    return det


@st.cache_resource(show_spinner=False)
def load_sbert_detector() -> HateDetector:
    det = HateDetector.from_csv()
    det.fit_sbert()
    return det


# ── Pomoćne funkcije ──────────────────────────────────────────────────────────

def _confidence_label(prob: float, threshold: float) -> str:
    distance = abs(prob - threshold)
    if distance < 0.08:
        return "Granični slučaj"
    if distance < 0.20:
        return "Umjerena pouzdanost"
    return "Visoka pouzdanost"


def result_card(label: int, prob: float, threshold: float) -> None:
    distance = abs(prob - threshold)
    borderline = distance < 0.08

    if label == 1:
        colour = "#7d1f1f" if not borderline else "#b7770d"
        bg = "#fff0f0" if not borderline else "#fffbee"
        icon = "🚫 GOVOR MRŽNJE" if not borderline else "⚠️ VJEROVATNO GOVOR MRŽNJE"
    else:
        colour = "#1a5e35" if not borderline else "#b7770d"
        bg = "#f0fff4" if not borderline else "#fffbee"
        icon = "✅ NIJE GOVOR MRŽNJE" if not borderline else "❓ VJEROVATNO NIJE GOVOR MRŽNJE"

    confidence = _confidence_label(prob, threshold)
    threshold_pct = int(threshold * 100)
    prob_pct = int(prob * 100)
    bar_color = colour

    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 14px 18px 10px;
            margin-bottom: 6px;
            background: {bg};
        ">
            <h3 style="color:{colour}; margin:0 0 4px 0; font-size:1.1rem;">{icon}</h3>
            <p style="margin:0; font-size:0.85rem; color:#555;">
                Pouzdanost: <strong>{confidence}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar with threshold marker
    st.markdown(
        f"""
        <div style="position:relative; margin:10px 0 18px;">
            <div style="background:#ddd; height:16px; border-radius:8px; overflow:visible; position:relative;">
                <div style="width:{prob_pct}%; background:{bar_color}; height:16px; border-radius:8px;"></div>
                <div style="position:absolute; left:{threshold_pct}%; top:-5px; width:3px; height:26px;
                            background:#222; border-radius:2px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#777; margin-top:4px;">
                <span>0%</span>
                <span style="position:relative; left:{threshold_pct - 50}%;">▲ prag ({threshold:.0%})</span>
                <span>100%</span>
            </div>
            <p style="text-align:right; font-size:0.9rem; color:{colour}; margin:4px 0 0;">
                <strong>P(hate) = {prob:.1%}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_tfidf_features(detector: HateDetector, text: str) -> None:
    features = detector.tfidf_top_features(text, top_n=8)
    if not features:
        return
    positive = [(tok, w) for tok, w in features if w > 0]
    if not positive:
        st.caption("Nisu pronađeni jaki indikatori govora mržnje u ovom tekstu.")
        return
    st.caption("**Tokeni koji indiciraju govor mržnje:**")
    max_w = max(w for _, w in positive)
    for token, weight in positive[:5]:
        bar_pct = min(int(weight / max_w * 100), 100)
        st.markdown(
            f"`{token}` &nbsp;"
            f"<span style='display:inline-block;width:{bar_pct}%;height:8px;"
            f"background:#7d1f1f;border-radius:4px;vertical-align:middle;'></span>"
            f" <small style='color:#888;'>{weight:.3f}</small>",
            unsafe_allow_html=True,
        )


# ── Glavni UI ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🛡️ Detekcija govora mržnje")
    st.markdown(
        "Demonstracija **binarne klasifikacije** za prepoznavanje govora mržnje u porukama "
        "na platformama za upoznavanje. Ispitana su dva pristupa reprezentaciji teksta — "
        "**TF-IDF** i **Sentence-BERT** — s logističkom regresijom kao klasifikatorom."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Podešavanja")

    st.sidebar.markdown("**Prag odluke**")
    st.sidebar.markdown(
        "<small>Prag određuje granicu između 'nije hate' i 'hate'. "
        "Viši prag = manje lažnih uzbuna, ali moguće propuštanje hate sadržaja.</small>",
        unsafe_allow_html=True,
    )
    tfidf_threshold = st.sidebar.slider(
        "TF-IDF prag", min_value=0.30, max_value=0.80, value=0.50, step=0.05,
        format="%.2f",
    )
    sbert_threshold = st.sidebar.slider(
        "SBERT prag", min_value=0.30, max_value=0.80, value=0.50, step=0.05,
        format="%.2f",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**O modelu**")
    st.sidebar.markdown(
        "Dataset: Davidson 2017 + tweet\\_eval/hate  \n"
        "Split: 70% / 15% / 15%  \n"
        "Testni F1 (hate, prag=0.50):  \n"
        "TF-IDF **0.564** · SBERT **0.529**  \n"
        "Testni AUC: TF-IDF **0.847** · SBERT **0.830**"
    )

    # ── Unos teksta ──────────────────────────────────────────────────────────
    st.subheader("Unos teksta")

    col_hate, col_ok = st.columns(2)
    example_key = None

    with col_hate:
        st.markdown("**Primjeri govora mržnje**")
        for name in EXAMPLES_HATE:
            if st.button(name, key=f"hate_{name}", use_container_width=True):
                example_key = ("hate", name)

    with col_ok:
        st.markdown("**Primjeri legitimnih poruka**")
        for name in EXAMPLES_NOT_HATE:
            if st.button(name, key=f"ok_{name}", use_container_width=True):
                example_key = ("ok", name)

    if example_key is not None:
        category, name = example_key
        st.session_state["hate_input"] = (
            EXAMPLES_HATE[name] if category == "hate" else EXAMPLES_NOT_HATE[name]
        )

    input_text = st.text_area(
        "Tekst poruke za analizu",
        value=st.session_state.get("hate_input", ""),
        height=130,
        placeholder="Unesi tekst poruke ili odaberi primjer iznad...",
        key="hate_input",
    )

    analyze_btn = st.button("Analiziraj", type="primary", disabled=not input_text.strip())

    if not analyze_btn:
        st.info("Unesi tekst ili odaberi primjer, pa klikni **Analiziraj**.")
        return

    # ── Klasifikacija ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Rezultati")

    tab_tfidf, tab_sbert = st.tabs(["TF-IDF + Logistička regresija", "Sentence-BERT + Logistička regresija"])

    with tab_tfidf:
        with st.spinner("Učitavam TF-IDF model..."):
            try:
                det_tfidf = load_tfidf_detector()
                label_t, prob_t = det_tfidf.predict_tfidf(input_text, threshold=tfidf_threshold)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
        result_card(label_t, prob_t, tfidf_threshold)
        show_tfidf_features(det_tfidf, input_text)

    with tab_sbert:
        with st.spinner(
            "Učitavam Sentence-BERT model... "
            "Prvo pokretanje može potrajati nekoliko minuta."
        ):
            try:
                det_sbert = load_sbert_detector()
                label_s, prob_s = det_sbert.predict_sbert(input_text, threshold=sbert_threshold)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
        result_card(label_s, prob_s, sbert_threshold)

    # ── Objašnjenje ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("O klasifikaciji i pragu odluke"):
        st.markdown(
            """
            **TF-IDF** pretvara tekst u rijedak vektor težina tokena. Tokeni koji se
            često pojavljuju u hate speech porukama dobijaju visoke TF-IDF težine i
            utiču na odluku klasifikatora.

            **Sentence-BERT** enkodira cijeli tekst u gust 384-dimenzionalni semantički
            vektor korištenjem modela `all-MiniLM-L6-v2`. Razumije kontekst i semantičku
            sličnost između tekstova koji ne dijele isti vokabular.

            **Prag odluke** određuje na kojoj vjerovatnoći P(hate) model proglašava tekst
            govorom mržnje. Podrazumijevana vrijednost je 0.50, ali može se prilagoditi:
            - Viši prag → manje lažnih uzbuna, ali potencijalno više propuštenih slučajeva
            - Niži prag → osjetljiviji model, više lažnih uzbuna

            **Napomena o metrikama:** Detekcija govora mržnje je inherentno težak zadatak.
            Hate speech dijeli vokabular s legitimnim kritičkim govorom.
            F1 ≈ 0.56 konzistentan je s literaturom za ovaj tip dataseta i metodologije.
            Fine-tuning domenski specifičnog modela (npr. HateBERT) značajno bi poboljšao rezultate.
            """
        )


if __name__ == "__main__":
    main()
