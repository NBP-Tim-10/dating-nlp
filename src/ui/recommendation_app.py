from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Omogućava da Streamlit pravilno pronađe src/ module kada se app pokreće iz root foldera.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.recommendation.bio_recommender import BioRecommender, Method


st.set_page_config(
    page_title="Bio Recommender",
    page_icon="💘",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_base_recommender(max_profiles: int, random_state: int) -> BioRecommender:
    return BioRecommender.from_csv(
        max_profiles=max_profiles,
        random_state=random_state,
    )


@st.cache_resource(show_spinner=False)
def load_tfidf_recommender(max_profiles: int, random_state: int) -> BioRecommender:
    rec = BioRecommender.from_csv(
        max_profiles=max_profiles,
        random_state=random_state,
    )
    rec.fit_tfidf()
    return rec


@st.cache_resource(show_spinner=False)
def load_sbert_recommender(max_profiles: int, random_state: int) -> BioRecommender:
    rec = BioRecommender.from_csv(
        max_profiles=max_profiles,
        random_state=random_state,
    )
    rec.fit_sbert()
    return rec


def preview(text: object, max_len: int = 700) -> str:
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def profile_title(row: pd.Series, index: int) -> str:
    profile_id = row.get("profile_id", index)
    age = row.get("age", "")
    sex = row.get("sex", "")
    orientation = row.get("orientation", "")
    location = row.get("location", "")

    parts = [f"index={index}", f"id={profile_id}"]

    if pd.notna(age):
        parts.append(f"{age} god.")
    if pd.notna(sex):
        parts.append(str(sex))
    if pd.notna(orientation):
        parts.append(str(orientation))
    if pd.notna(location):
        parts.append(str(location))

    return " | ".join(parts)


def show_query_profile(rec: BioRecommender, profile_index: int) -> None:
    row = rec.data.iloc[profile_index]

    st.subheader("Odabrani profil")

    meta_cols = st.columns(5)

    meta_cols[0].metric("Profile ID", row.get("profile_id", profile_index))
    meta_cols[1].metric("Age", row.get("age", "N/A"))
    meta_cols[2].metric("Sex", row.get("sex", "N/A"))
    meta_cols[3].metric("Orientation", row.get("orientation", "N/A"))
    meta_cols[4].metric("Status", row.get("status", "N/A"))

    st.caption(
        f"Lokacija: {row.get('location', 'N/A')} | "
        f"Posao: {row.get('job', 'N/A')} | "
        f"Obrazovanje: {row.get('education', 'N/A')}"
    )

    st.text_area(
        "Bio preview",
        value=preview(row.get("bio_raw", ""), max_len=1200),
        height=180,
        disabled=True,
    )


def clean_results_for_display(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()

    preferred_cols = [
        "rank",
        "similarity_score",
        "profile_id",
        "age",
        "sex",
        "orientation",
        "status",
        "location",
        "job",
        "education",
        "bio_preview",
    ]

    cols = [c for c in preferred_cols if c in display.columns]
    display = display[cols]

    if "similarity_score" in display.columns:
        display["similarity_score"] = display["similarity_score"].astype(float).round(4)

    return display


def run_recommendation(
    method: Method,
    query_mode: str,
    profile_index: int,
    custom_text: str,
    top_k: int,
    max_profiles: int,
    random_state: int,
) -> pd.DataFrame:
    if method == "tfidf":
        with st.spinner("Treniram / učitavam TF-IDF model..."):
            rec = load_tfidf_recommender(max_profiles, random_state)

    elif method == "sbert":
        with st.spinner("Računam / učitavam SBERT embeddings... Prvo pokretanje može potrajati."):
            rec = load_sbert_recommender(max_profiles, random_state)

    else:
        raise ValueError(f"Nepoznata metoda: {method}")

    if query_mode == "Postojeći profil":
        results = rec.recommend_by_index(
            profile_index=profile_index,
            method=method,
            top_k=top_k,
        )
    else:
        results = rec.recommend_by_text(
            text=custom_text,
            method=method,
            top_k=top_k,
        )

    results.insert(0, "method", method.upper())
    return results


def main() -> None:
    st.title("💘 Sistem za preporučivanje profila")
    st.markdown(
        """
        Ovaj UI demonstrira **Task 1: Sistem za preporučivanje profila na osnovu bio embeddings**.

        Sistem poredi tekstualne opise korisnika i vraća najsličnije profile.
        Podržane su dvije reprezentacije teksta:

        - **TF-IDF + cosine similarity** — klasična reprezentacija teksta.
        - **SBERT embeddings + cosine similarity** — duboko kontekstualna reprezentacija teksta.
        """
    )

    st.sidebar.header("Podešavanja")

    max_profiles = st.sidebar.slider(
        "Broj profila za demo",
        min_value=300,
        max_value=5000,
        value=1000,
        step=100,
        help="Manji broj radi brže. Za prezentaciju je 1000 sasvim dovoljno.",
    )

    top_k = st.sidebar.slider(
        "Broj preporuka",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
    )

    method_choice = st.sidebar.selectbox(
        "Metoda",
        options=["TF-IDF", "SBERT", "Obje metode"],
        index=2,
    )

    random_state = 42

    try:
        base_rec = load_base_recommender(max_profiles, random_state)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.caption(f"Učitano profila: {len(base_rec.data)}")

    query_mode = st.radio(
        "Način unosa",
        options=["Postojeći profil", "Novi bio tekst"],
        horizontal=True,
    )

    profile_index = 0
    custom_text = ""

    if query_mode == "Postojeći profil":
        profile_index = st.number_input(
            "Index profila",
            min_value=0,
            max_value=len(base_rec.data) - 1,
            value=0,
            step=1,
        )

        show_query_profile(base_rec, profile_index)

    else:
        custom_text = st.text_area(
            "Unesi novi bio tekst",
            value=(
                "I enjoy live music, good books, simple food, running, "
                "quiet weekends, and spending time with friends."
            ),
            height=180,
        )

        if not custom_text.strip():
            st.warning("Unesi bio tekst prije pokretanja preporuka.")

    run_button = st.button("Prikaži preporuke", type="primary")

    if not run_button:
        st.info("Podesi parametre i klikni na dugme za prikaz preporuka.")
        return

    if query_mode == "Novi bio tekst" and not custom_text.strip():
        st.error("Bio tekst ne može biti prazan.")
        return

    selected_methods: list[Method]
    if method_choice == "TF-IDF":
        selected_methods = ["tfidf"]
    elif method_choice == "SBERT":
        selected_methods = ["sbert"]
    else:
        selected_methods = ["tfidf", "sbert"]

    all_results: list[pd.DataFrame] = []

    for method in selected_methods:
        results = run_recommendation(
            method=method,
            query_mode=query_mode,
            profile_index=profile_index,
            custom_text=custom_text,
            top_k=top_k,
            max_profiles=max_profiles,
            random_state=random_state,
        )
        all_results.append(results)

    st.divider()
    st.header("Rezultati preporuka")

    if len(all_results) == 1:
        display = clean_results_for_display(all_results[0])
        st.dataframe(display, use_container_width=True, hide_index=True)

    else:
        tfidf_results = clean_results_for_display(all_results[0])
        sbert_results = clean_results_for_display(all_results[1])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("TF-IDF preporuke")
            st.dataframe(tfidf_results, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("SBERT preporuke")
            st.dataframe(sbert_results, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Kratko objašnjenje")

    st.markdown(
        """
        **TF-IDF** češće vraća profile koji dijele iste ili vrlo slične riječi sa odabranim profilom.

        **SBERT** poredi značenje cijelog bio teksta, pa može pronaći profile koji su semantički slični
        čak i kada ne koriste potpuno iste riječi.
        """
    )


if __name__ == "__main__":
    main()