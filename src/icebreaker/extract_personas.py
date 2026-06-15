"""Ekstrakcija i raznoliko uzorkovanje persona iz PersonaChat dataseta.

Pokretanje:
    python -m src.icebreaker.extract_personas
"""
from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from src.icebreaker.config import (
    DATASET,
    EMBED_MODEL,
    ICEBREAKER_DIR,
    N_CLUSTERS,
    PERSONAS_PATH,
    SEED,
    TARGET_N,
    USE_DIVERSITY,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_CANDIDATE_FIELDS = ["persona_b", "persona", "personality", "personas", "your_persona"]

# Regex za flat-text format: "N your persona: <fact>"
_FACT_RE      = re.compile(r"^\d+\s+your persona:\s+(.+)", re.IGNORECASE)
_NEW_BLOCK_RE = re.compile(r"^1\s+your persona:", re.IGNORECASE)


def _signature(facts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(f.strip().lower() for f in facts if f.strip()))


def _detect_format(ds) -> tuple[str, str | None]:
    """Vraca ('structured', field_name) ili ('text', 'text')."""
    for row in ds:
        for field in _CANDIDATE_FIELDS:
            val = row.get(field)
            if isinstance(val, list) and val and isinstance(val[0], str):
                return "structured", field
        # Fallback: bilo koje polje koje je lista stringova
        for field, val in row.items():
            if isinstance(val, list) and val and isinstance(val[0], str):
                return "structured", field
        # Flat-text format: text polje sa "N your persona: ..."
        text = row.get("text", "")
        if isinstance(text, str) and _FACT_RE.match(text):
            return "text", "text"
        break  # provjeri samo prvi red
    return "unknown", None


def _collect_structured(ds, field: str) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []
    for row in ds:
        raw = row.get(field)
        if not raw or not isinstance(raw, list):
            continue
        facts = [f.strip() for f in raw if isinstance(f, str) and f.strip()]
        if not facts:
            continue
        sig = _signature(facts)
        if sig not in seen:
            seen.add(sig)
            result.append(facts)
    return result


def _collect_text(ds) -> list[list[str]]:
    """Parsira flat-text format gdje je svaki red jedna linija konverzacije/persone."""
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []
    current: list[str] = []

    def _flush():
        if current:
            sig = _signature(current)
            if sig not in seen:
                seen.add(sig)
                result.append(list(current))

    for row in ds:
        text = row.get("text", "").strip()
        # "1 your persona: ..." označava početak novog bloka persona
        if _NEW_BLOCK_RE.match(text):
            _flush()
            current = []
        m = _FACT_RE.match(text)
        if m:
            fact = m.group(1).strip()
            if fact:
                current.append(fact)

    _flush()
    return result


def _diverse_sample(
    personas: list[list[str]],
    target_n: int,
    n_clusters: int,
    embed_model: str,
    seed: int,
) -> list[list[str]]:
    actual_k = min(n_clusters, len(personas))
    if actual_k < n_clusters:
        print(f"[warn] Smanjujem broj klastera {n_clusters}→{actual_k} (nedovoljno persona)")

    texts = [" ".join(p) for p in personas]
    print(f"[info] Embedujem {len(texts)} persona sa '{embed_model}'...")
    model = SentenceTransformer(embed_model)
    emb = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"[info] KMeans clustering (k={actual_k})...")
    km = KMeans(n_clusters=actual_k, random_state=seed, n_init="auto")
    labels = km.fit_predict(emb)

    clusters: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(idx)

    rng = random.Random(seed)
    for lst in clusters.values():
        rng.shuffle(lst)

    iters = {k: iter(v) for k, v in clusters.items()}
    keys = list(clusters.keys())
    rng.shuffle(keys)

    # Round-robin: uzimamo po jedan iz svakog klastera
    selected: list[int] = []
    while len(selected) < target_n and iters:
        exhausted = []
        for k in keys:
            if k not in iters:
                continue
            try:
                selected.append(next(iters[k]))
                if len(selected) >= target_n:
                    break
            except StopIteration:
                exhausted.append(k)
        for k in exhausted:
            del iters[k]

    # Dopuni nasumično ako round-robin nije dostigao target_n
    if len(selected) < target_n:
        pool = [i for i in range(len(personas)) if i not in set(selected)]
        rng.shuffle(pool)
        needed = target_n - len(selected)
        selected.extend(pool[:needed])
        print(f"[warn] Dopunjeno {needed} persona nasumično (klasteri iscrpljeni)")

    return [personas[i] for i in selected]


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    ICEBREAKER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] Učitavam '{DATASET}' (train split)...")
    ds = load_dataset(DATASET, split="train")

    # Detekcija formata dataseta
    fmt, field = _detect_format(ds)
    if fmt == "unknown":
        print("[error] Nije prepoznat format dataseta!", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Format dataseta: '{fmt}'" + (f", polje: '{field}'" if field else ""))

    # Ekstrakcija i deduplikacija
    print("[info] Ekstrahujem jedinstvene persone (deduplikacija po sorted(facts))...")
    if fmt == "structured":
        all_personas = _collect_structured(ds, field)
    else:
        all_personas = _collect_text(ds)

    # Prikaz primjera
    if all_personas:
        print(f"[info] Primjer prve persone: {all_personas[0]}")
    print(f"[info] Pronašao {len(all_personas)} jedinstvenih persona.")

    if not all_personas:
        print("[error] Nema valjanih persona!", file=sys.stderr)
        sys.exit(1)

    # Uzorkovanje
    if len(all_personas) <= TARGET_N:
        print(f"[warn] Dostupno {len(all_personas)} persona ≤ TARGET_N={TARGET_N}. Koristim sve.")
        sampled = all_personas
    elif USE_DIVERSITY:
        print(f"[info] Raznoliko uzorkovanje: {TARGET_N} od {len(all_personas)}...")
        sampled = _diverse_sample(all_personas, TARGET_N, N_CLUSTERS, EMBED_MODEL, SEED)
    else:
        rng = random.Random(SEED)
        sampled = rng.sample(all_personas, TARGET_N)

    # Zapis u JSONL
    print(f"[info] Zapisujem {len(sampled)} persona u {PERSONAS_PATH}...")
    with open(PERSONAS_PATH, "w", encoding="utf-8") as f:
        for i, facts in enumerate(sampled):
            record = {"id": i, "facts": facts, "bio": " ".join(facts)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ─ ──────────────────────────────────────────────────────────────
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    ids = [r["id"] for r in rows]
    avg_facts = sum(len(r["facts"]) for r in rows) / len(rows)
    sigs = {tuple(sorted(r["facts"])) for r in rows}

    print("\n[done] Samotest:")
    print(f"  Broj redova         : {len(rows)}")
    print(f"  Jedinstvenih IDs    : {len(set(ids))}")
    print(f"  Jedinstvenih sig    : {len(sigs)}")
    print(f"  Prosj. br. činjenica: {avg_facts:.2f}")

    rng = random.Random(SEED + 1)
    print("  Nasumični primjeri:")
    for s in rng.sample(rows, min(3, len(rows))):
        print(f"    id={s['id']} | {s['facts']}")

    assert len(sigs) == len(rows), "GREŠKA: Pronađeni duplikati po sorted(facts)!"
    assert ids == list(range(len(rows))), "GREŠKA: ID-evi nisu sekvencijalni 0..N-1!"
    print("[done] Sve provjere prošle.")


if __name__ == "__main__":
    main()
