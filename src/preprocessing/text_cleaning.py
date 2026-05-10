"""Zajedničke funkcije za čišćenje i preprocesiranje teksta.

Koriste se za sve NLP taskove. Različiti taskovi pozivaju različite
kombinacije ovih koraka (npr. sentiment analiza zadržava emotikone i
negacije, dok detekcija govora mržnje agresivnije normalizuje tekst).
"""
from __future__ import annotations

import html
import re
import string
import unicodedata
from typing import Iterable

import contractions
import emoji
import ftfy
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Regex-i koji se koriste više puta - kompajliramo ih jednom radi performansi
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")


def fix_unicode(text: str) -> str:
    """Popravi pokvarenu Unicode kodaciju (mojibake) i HTML entitete."""
    if not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text


def remove_urls_emails(text: str) -> str:
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    return text


def replace_user_mentions(text: str) -> str:
    text = MENTION_RE.sub(" <USER> ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    return text


def expand_contractions(text: str) -> str:
    """don't -> do not, it's -> it is itd."""
    try:
        return contractions.fix(text)
    except Exception:
        return text


def emojis_to_text(text: str) -> str:
    """:) -> :smiling_face: - zadržava signal za sentiment."""
    return emoji.demojize(text, delimiters=(" :", ": "))


def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="")


def collapse_repeated_chars(text: str) -> str:
    """soooo -> soo (ostavlja 2 karaktera za blagi naglasak)."""
    return REPEATED_CHAR_RE.sub(r"\1\1", text)


def strip_punctuation(text: str, keep: str = "") -> str:
    table = str.maketrans("", "", "".join(c for c in string.punctuation if c not in keep))
    return text.translate(table)


def remove_numbers(text: str) -> str:
    return re.sub(r"\b\d+\b", " <NUM> ", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def to_ascii(text: str) -> str:
    """č -> c, ž -> z - korisno kad mješamo bs/hr/sr i en korpuse."""
    return unidecode(text)


# ---------------------------------------------------------------------------
# Konfigurabilne pipelines za različite taskove
# ---------------------------------------------------------------------------
def clean_for_embeddings(text: str) -> str:
    """Pipeline za sistem preporučivanja i topic modeling.

    - zadržava semantiku, ali uklanja URL/email/telefon/spomene
    - emoji se prevodi u tekst (može nositi informaciju o interesovanjima)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = fix_unicode(text)
    text = remove_urls_emails(text)
    text = replace_user_mentions(text)
    text = expand_contractions(text)
    text = emojis_to_text(text)
    text = collapse_repeated_chars(text)
    text = text.lower()
    text = strip_punctuation(text, keep="<>:_-")
    text = normalize_whitespace(text)
    return text


def clean_for_classification(text: str, *, lower: bool = True) -> str:
    """Pipeline za klasifikaciju (govor mržnje, bot detekcija).

    Agresivnija normalizacija - manji vokabular = stabilniji modeli.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = fix_unicode(text)
    text = remove_urls_emails(text)
    text = replace_user_mentions(text)
    text = expand_contractions(text)
    text = emojis_to_text(text)
    text = collapse_repeated_chars(text)
    if lower:
        text = text.lower()
    text = remove_numbers(text)
    text = strip_punctuation(text, keep="<>:_!?")
    text = normalize_whitespace(text)
    return text


def clean_for_sentiment(text: str) -> str:
    """Pipeline za analizu sentimenta i emocija.

    Zadržava emotikone, znakove uzvika i negacije jer nose signal.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = fix_unicode(text)
    text = remove_urls_emails(text)
    text = replace_user_mentions(text)
    text = expand_contractions(text)
    text = emojis_to_text(text)
    text = collapse_repeated_chars(text)
    text = text.lower()
    text = strip_punctuation(text, keep="<>:_!?'-")
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Tokenizacija + uklanjanje stop-words + lematizacija
# ---------------------------------------------------------------------------
_STOPWORDS_CACHE: set[str] | None = None
_LEMMATIZER = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        from nltk.corpus import stopwords

        _STOPWORDS_CACHE = set(stopwords.words("english"))
        _STOPWORDS_CACHE -= {
            "no", "not", "nor", "never",
            "very", "too", "more", "most",
            "but", "however",
        }
    return _STOPWORDS_CACHE


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def tokenize(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    return word_tokenize(text)


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    sw = _get_stopwords()
    return [t for t in tokens if t.lower() not in sw and len(t) > 1]


def lemmatize(tokens: Iterable[str]) -> list[str]:
    lem = _get_lemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def full_token_pipeline(
    text: str,
    *,
    cleaner=clean_for_embeddings,
    drop_stopwords: bool = True,
    do_lemmatize: bool = True,
) -> list[str]:
    cleaned = cleaner(text)
    if not cleaned:
        return []
    tokens = tokenize(cleaned)
    if drop_stopwords:
        tokens = remove_stopwords(tokens)
    if do_lemmatize:
        tokens = lemmatize(tokens)
    return tokens
