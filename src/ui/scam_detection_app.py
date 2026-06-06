"""Streamlit UI za demo detekcije prevara u dating aplikacijama.

Pokretanje:
    streamlit run src/ui/scam_detection_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scam_detection.scam_detector import ScamDetector

# ── Primjeri teksta ───────────────────────────────────────────────────────────

EXAMPLES_SCAM = {
    "Romance scam profil": (
        "Hello dear, I am James, a widow U.S. Army officer currently deployed offshore. "
        "I am looking for serious relationship and true love. Please contact me on WhatsApp "
        "+1-555-0147. God bless you."
    ),
    "SMS spam": (
        "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
        "Text FA to 87121 to receive entry question std txt rate apply."
    ),
    "Finansijska prevara": (
        "Congratulations! You have been selected for a $1000 reward. "
        "Call 0800-PRIZE now or visit http://claim-reward.net to receive your prize. Limited time!"
    ),
}

EXAMPLES_LEGIT = {
    "Legit profil — hobiji": (
        "Software engineer by day, hiking enthusiast by night. "
        "Big fan of craft beer, indie music and weekend adventures. "
        "Looking for someone curious about the world."
    ),
    "Legit profil — opušteni stil": (
        "Just moved to the city for work. Love trying new restaurants, going to live shows, "
        "and spending lazy Sundays reading. My dog is the real star of my profile."
    ),
    "Normalna SMS poruka": (
        "Ok lar... Joking wif u oni... Did you eat already? "
        "I'm at the gym, will be home by 8. See you later!"
    ),
}

# ── Konfiguracija stranice ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Detekcija prevara",
    page_icon="🔍",
    layout="wide",
)

# ── Cache — load i fit modela ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_tfidf_detector() -> ScamDetector:
    det = ScamDetector.from_csv()
    det.fit_tfidf()
    return det


@st.cache_resource(show_spinner=False)
def load_sbert_detector() -> ScamDetector:
    det = ScamDetector.from_csv()
    det.fit_sbert()
    return det


# ── Pomoćne funkcije ──────────────────────────────────────────────────────────

def result_card(label: int, prob: float, method_name: str) -> None:
    is_scam = label == 1
    colour = "#c0392b" if is_scam else "#27ae60"
    icon = "⚠️ PREVARA" if is_scam else "✅ LEGITIMNO"

    st.markdown(
        f"""
        <div style="
            border: 2px solid {colour};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 8px;
            background: {'#fff5f5' if is_scam else '#f0fff4'};
        ">
            <h3 style="color:{colour}; margin:0 0 8px 0;">{icon}</h3>
            <p style="margin:0; font-size:0.95rem; color:#333;">
                Vjerovatnoća prevare:
                <strong style="color:{colour};">{prob:.1%}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(prob)


def show_tfidf_features(detector: ScamDetector, text: str) -> None:
    features = detector.tfidf_top_scam_features(text, top_n=8)
    if not features:
        return

    positive = [(tok, w) for tok, w in features if w > 0]
    if not positive:
        st.caption("Nisu pronađeni jaki scam indikatori u tekstu.")
        return

    st.caption("**Tokeni koji najviše indiciraju prevaru:**")
    for token, weight in positive[:5]:
        bar_pct = min(int(weight / max(w for _, w in positive) * 100), 100)
        st.markdown(
            f"`{token}` &nbsp; "
            f"<span style='display:inline-block;width:{bar_pct}%;height:8px;"
            f"background:#e74c3c;border-radius:4px;vertical-align:middle;'></span> "
            f"<small>{weight:.3f}</small>",
            unsafe_allow_html=True,
        )


# ── Glavni UI ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🔍 Detekcija prevara u dating aplikacijama")
    st.markdown(
        """
        Demonstracija **binarne klasifikacije teksta** za prepoznavanje romance scam profila
        i spam poruka.

        Ispitana su dva pristupa reprezentaciji teksta:
        - **TF-IDF** — klasična statistička reprezentacija s n-gramima
        - **Sentence-BERT** — duboka kontekstualna reprezentacija (transformer)

        Isti klasifikator (logistička regresija) korišten je u oba slučaja.
        """
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Podešavanja")

    method_choice = st.sidebar.selectbox(
        "Metoda klasifikacije",
        options=["Obje metode", "TF-IDF", "Sentence-BERT"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**O modelu**")
    st.sidebar.markdown(
        """
        Dataset: SMS Spam Collection (UCI) + sintetički romance scam profili
        Split: 70% train / 15% val / 15% test
        Testni F1 (scam): TF-IDF **0.965** · SBERT **0.934**
        Testni AUC: TF-IDF **0.9956** · SBERT **0.9945**
        """
    )

    # ── Unos teksta ──────────────────────────────────────────────────────────
    st.subheader("Unos teksta")

    col_ex1, col_ex2, col_ex3 = st.columns(3)

    example_key = None
    with col_ex1:
        st.markdown("**Primjeri prevara**")
        for name in EXAMPLES_SCAM:
            if st.button(name, key=f"scam_{name}", use_container_width=True):
                example_key = ("scam", name)

    with col_ex2:
        st.markdown("**Primjeri legitimnih**")
        for name in EXAMPLES_LEGIT:
            if st.button(name, key=f"legit_{name}", use_container_width=True):
                example_key = ("legit", name)

    with col_ex3:
        st.markdown("&nbsp;")

    # Popuni text area na osnovu odabranog primjera
    if example_key is not None:
        category, name = example_key
        if category == "scam":
            default_text = EXAMPLES_SCAM[name]
        else:
            default_text = EXAMPLES_LEGIT[name]
        st.session_state["input_text"] = default_text

    input_text = st.text_area(
        "Tekst poruke ili bio profila",
        value=st.session_state.get("input_text", ""),
        height=150,
        placeholder=(
            "Unesi tekst poruke ili bio profila za analizu...\n"
            "Ili odaberi jedan od primjera iznad."
        ),
        key="input_text",
    )

    analyze_btn = st.button("Analiziraj", type="primary", disabled=not input_text.strip())

    if not analyze_btn:
        st.info("Unesi tekst ili odaberi primjer, pa klikni **Analiziraj**.")
        return

    if not input_text.strip():
        st.warning("Tekst ne može biti prazan.")
        return

    # ── Klasifikacija ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Rezultati")

    run_tfidf = method_choice in ("TF-IDF", "Obje metode")
    run_sbert = method_choice in ("Sentence-BERT", "Obje metode")

    tfidf_result = None
    sbert_result = None

    if run_tfidf:
        with st.spinner("Učitavam TF-IDF model..."):
            try:
                det_tfidf = load_tfidf_detector()
                tfidf_result = det_tfidf.predict_tfidf(input_text)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

    if run_sbert:
        with st.spinner(
            "Učitavam Sentence-BERT model... "
            "Prvo pokretanje može potrajati 3–5 minuta (enkodiranje trening skupa)."
        ):
            try:
                det_sbert = load_sbert_detector()
                sbert_result = det_sbert.predict_sbert(input_text)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

    # ── Prikaz rezultata ──────────────────────────────────────────────────────
    if run_tfidf and run_sbert:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### TF-IDF + Logistička regresija")
            result_card(*tfidf_result, "TF-IDF")
            if det_tfidf is not None:
                show_tfidf_features(det_tfidf, input_text)
        with col2:
            st.markdown("### Sentence-BERT + Logistička regresija")
            result_card(*sbert_result, "SBERT")

    elif run_tfidf:
        st.markdown("### TF-IDF + Logistička regresija")
        result_card(*tfidf_result, "TF-IDF")
        if det_tfidf is not None:
            show_tfidf_features(det_tfidf, input_text)

    elif run_sbert:
        st.markdown("### Sentence-BERT + Logistička regresija")
        result_card(*sbert_result, "SBERT")

    # ── Objašnjenje ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Kako radi klasifikacija?"):
        st.markdown(
            """
            **TF-IDF** pretvara tekst u rijedak vektor težina tokena.
            Tokeni karakteristični za scam poruke (npr. `wire transfer`, `<PHONE>`, `God bless`,
            `whatsapp`) dobijaju visoke težine. Logistička regresija tada gleda koje
            tokene tekst sadrži i donosi odluku.

            **Sentence-BERT** enkodira cijeli tekst u gust 384-dimenzionalni vektor
            semantičkog značenja. Slični tekstovi imaju bliske vektore čak i bez
            zajedničkih riječi. Isti tip logističke regresije je primijenjen na tim vektorima.

            Oba modela trenirana su na kombinaciji SMS Spam Collection dataseta (UCI)
            i sintetički generisanih romance scam profila. Klasna neravnoteža (~82% legit,
            ~18% scam) riješena je parametrom `class_weight='balanced'`.
            """
        )


if __name__ == "__main__":
    main()
