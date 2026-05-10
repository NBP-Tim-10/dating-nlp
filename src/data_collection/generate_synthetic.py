"""Sintetičko generisanje podataka koji nedostaju u javnim datasetovima.

Razlozi zašto trebamo sintetičke podatke:
    1) Romance scam profili (legitimni profili + scam profili sa
       anti-pattern obrascima). Pravi scam korpus iz dating aplikacija
       nije javno dostupan iz sigurnosnih razloga.
    2) Icebreaker primjeri - parovi (bio_user_A, bio_user_B, icebreaker)
       koji ne postoje u OkCupid-u.
    3) Razgovori sa eskalacijom emocija (frustracija/sarkazam) koji se
       koriste za fine-grained sentiment.

Generisanje koristi šablone + Faker za varijabilnost. Za potpunu
diversifikaciju u nastavku projekta moguće je generisati i preko LLM-a
(GPT/Llama) - taj korak je dokumentovan ali ne pokrenut automatski.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import pandas as pd
from faker import Faker

from src.utils.paths import SYNTHETIC_DIR

RNG = random.Random(42)
FAKER = Faker(["en_US"])
Faker.seed(42)


# ---------------------------------------------------------------------------
# 1) Bio profili - legit vs scam (za bot/scam detekciju)
# ---------------------------------------------------------------------------
LEGIT_INTERESTS = [
    "hiking", "cooking pasta", "reading sci-fi", "indie movies", "yoga",
    "live music", "board games", "specialty coffee", "travel", "salsa dancing",
    "rock climbing", "photography", "trail running", "vinyl records",
    "craft beer", "video games", "pottery", "skiing", "surfing", "art galleries",
]

LEGIT_TEMPLATES = [
    "I love {a} and {b}. Looking for someone who also enjoys {c}.",
    "Software engineer by day, {a} enthusiast by night. Big fan of {b}.",
    "Just moved to {city} for work. Always down for {a} or trying a new {b} spot.",
    "Two truths: I can quote every line from my favorite movie and I make a mean {a}.",
    "Weekends = {a}, {b}, and good company. Weekdays = surviving on coffee.",
    "Recently got into {a}. Ask me about my unhealthy obsession with {b}.",
    "Looking for a partner in crime for {a}, {b} and bad dad jokes.",
]

SCAM_TEMPLATES = [
    "Hello dear, I am {name}, a widow oil engineer working on rig in Gulf. "
    "My late wife and only daughter passed and I seek honest woman of GOD. "
    "Please WhatsApp me {phone} as I rarely use this app.",

    "I'm {name}, US Army stationed in Syria. I lost my wife in tragic accident. "
    "I have $4.5M USD inheritance trapped in offshore account that I need help "
    "transferring. Email me directly: {email}",

    "Beautiful soul, I am general doctor with United Nations in Yemen. "
    "Looking for serious relationship leading to marriage. I do not check this "
    "app often, kindly add me on telegram @{handle} for faster response.",

    "Hi sweetheart, I am sincere caring man searching true love forever. "
    "I am very rich businessman in crypto, I can help you invest small amount "
    "to make 10x return. DM me on Hangouts: {email}",

    "Greetings my queen. I work on offshore oil platform 6 months no break. "
    "GOD will lead me to right woman. Please contact me {phone} I cannot "
    "use this app on the rig due to bad signal.",
]

US_CITIES = ["NYC", "Chicago", "Austin", "Seattle", "LA", "Boston", "Denver", "Portland"]


def _legit_bio() -> str:
    template = RNG.choice(LEGIT_TEMPLATES)
    return template.format(
        a=RNG.choice(LEGIT_INTERESTS),
        b=RNG.choice(LEGIT_INTERESTS),
        c=RNG.choice(LEGIT_INTERESTS),
        city=RNG.choice(US_CITIES),
    )


def _scam_bio() -> str:
    template = RNG.choice(SCAM_TEMPLATES)
    return template.format(
        name=FAKER.first_name(),
        phone=FAKER.phone_number(),
        email=FAKER.email(),
        handle=FAKER.user_name(),
    )


def generate_scam_dataset(n_legit: int = 1500, n_scam: int = 1500) -> pd.DataFrame:
    rows = []
    for _ in range(n_legit):
        rows.append({"text": _legit_bio(), "label": 0, "source": "synthetic"})
    for _ in range(n_scam):
        rows.append({"text": _scam_bio(), "label": 1, "source": "synthetic"})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    out = SYNTHETIC_DIR / "scam_profiles.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[synthetic] scam profili: {len(df)} -> {out}")
    return df


# ---------------------------------------------------------------------------
# 2) Icebreaker parovi (za topic modeling / generisanje prijedloga)
# ---------------------------------------------------------------------------
ICEBREAKER_TEMPLATES = [
    "Saw you're into {topic} - what's your all-time favorite?",
    "{topic} fan? Recommend me one thing I have to try.",
    "We both love {topic}. Hot take or popular opinion first?",
    "Quick question: best {topic} you've experienced this year?",
    "Your bio says {topic}. Tell me you're a beginner without telling me.",
]


def generate_icebreaker_pairs(n: int = 1200) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        topic = RNG.choice(LEGIT_INTERESTS)
        bio_a = _legit_bio()
        bio_b = _legit_bio()
        ib = RNG.choice(ICEBREAKER_TEMPLATES).format(topic=topic)
        rows.append({
            "bio_a": bio_a,
            "bio_b": bio_b,
            "shared_topic": topic,
            "icebreaker": ib,
        })
    df = pd.DataFrame(rows)
    out = SYNTHETIC_DIR / "icebreaker_pairs.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[synthetic] icebreaker parovi: {len(df)} -> {out}")
    return df


# ---------------------------------------------------------------------------
# 3) Sintetički razgovori sa progresijom emocija (frustracija, sarkazam)
# ---------------------------------------------------------------------------
@dataclass
class TurnTemplate:
    emotion: str
    text: str


CONVERSATION_FLOWS: dict[str, list[TurnTemplate]] = {
    "engagement_drop": [
        TurnTemplate("interested", "Hey! Loved your bio - the part about {topic} got me."),
        TurnTemplate("interested", "What got you into it?"),
        TurnTemplate("neutral", "Cool, makes sense."),
        TurnTemplate("bored", "k"),
        TurnTemplate("disengaged", "yeah maybe later"),
    ],
    "frustration_escalation": [
        TurnTemplate("neutral", "Hi, how's your week going?"),
        TurnTemplate("polite", "Nice. Doing anything fun this weekend?"),
        TurnTemplate("annoyed", "You keep giving one-word answers..."),
        TurnTemplate("frustrated", "Honestly this feels like pulling teeth."),
        TurnTemplate("angry", "Forget it, this isn't going anywhere."),
    ],
    "sarcasm": [
        TurnTemplate("neutral", "Tell me about yourself"),
        TurnTemplate("sarcastic", "Oh sure, let me write my whole life story for the third time today."),
        TurnTemplate("sarcastic", "I just LOVE generic openers, keep them coming."),
        TurnTemplate("playful", "Kidding. Mostly. What do you actually want to know?"),
    ],
    "positive_match": [
        TurnTemplate("interested", "Your hiking pic is unreal - where was that?"),
        TurnTemplate("excited", "No way, I did that trail last summer!"),
        TurnTemplate("happy", "We should compare playlists for the next one."),
        TurnTemplate("happy", "Ok this is the best convo I've had on here."),
    ],
}


def generate_conversations(n: int = 600) -> pd.DataFrame:
    rows = []
    for conv_id in range(n):
        flow_name = RNG.choice(list(CONVERSATION_FLOWS))
        flow = CONVERSATION_FLOWS[flow_name]
        topic = RNG.choice(LEGIT_INTERESTS)
        for turn_idx, turn in enumerate(flow):
            rows.append({
                "conversation_id": conv_id,
                "turn_idx": turn_idx,
                "flow": flow_name,
                "emotion": turn.emotion,
                "text": turn.text.format(topic=topic),
            })
    df = pd.DataFrame(rows)
    out = SYNTHETIC_DIR / "conversations.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[synthetic] razgovori: {len(df)} okreta u {n} razgovora -> {out}")
    return df


# ---------------------------------------------------------------------------
# 4) Manifest svih sintetičkih izvora
# ---------------------------------------------------------------------------
def write_manifest() -> None:
    manifest = {
        "scam_profiles": "scam_profiles.csv",
        "icebreaker_pairs": "icebreaker_pairs.csv",
        "conversations": "conversations.csv",
        "seed": 42,
        "note": (
            "Sintetički podaci su namjerno šablonizovani da bi se mogli "
            "kontrolisati labelovi. Za diversifikaciju u sljedećoj iteraciji "
            "preporučujemo generisanje preko LLM-a (vidi README)."
        ),
    }
    (SYNTHETIC_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_all() -> None:
    generate_scam_dataset()
    generate_icebreaker_pairs()
    generate_conversations()
    write_manifest()
    print("[synthetic] gotovo.")


if __name__ == "__main__":
    generate_all()
