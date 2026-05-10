"""Inicijalizacija NLTK resursa potrebnih za preprocesiranje.

Pokrenuti jednom prije prvog preprocesiranja:
    python -m src.utils.nltk_setup
"""
from __future__ import annotations

import nltk

REQUIRED_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def download_all() -> None:
    for pkg in REQUIRED_PACKAGES:
        try:
            nltk.download(pkg, quiet=True)
            print(f"[nltk] OK: {pkg}")
        except Exception as exc:
            print(f"[nltk] GREŠKA pri preuzimanju {pkg}: {exc}")


if __name__ == "__main__":
    download_all()
