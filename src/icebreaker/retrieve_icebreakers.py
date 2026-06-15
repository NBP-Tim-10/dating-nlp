"""Poređenje reprezentacija za retrieval bio → icebreaker.

TF-IDF (ne-kontekstualna) vs all-MiniLM-L6-v2 (duboko kontekstualna).
Banka = train icebreakeri; test = held-out eval (bio, gold) parovi.
Relevantnost bank stavke za dati bio definisana je MiniLM cosine
sličnošću s gold icebreakerom (oracle, neovisan od sistema koji rangira).

Pokretanje:
    python -m src.icebreaker.retrieve_icebreakers
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from src.icebreaker.config import (
    EMBED_MODEL,
    EVAL_PATH,
    ICEBREAKER_DIR,
    RETRIEVAL_BANK,
    RETRIEVAL_K,
    RETRIEVAL_TESTSET,
    TFIDF_MAX_FEATS,
    TRAIN_PATH,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Threshold za oracle: bank stavka je "relevantna" ako je MiniLM cosine
# sličnost njenog icebreakera sa gold icebreakerom >= ove vrijednosti.
_ORACLE_THRESHOLD = 0.35


# ── Gradnja banke i test skupa ─────────────────────────────────────────────────

def _parse_bio(content: str) -> str:
    text = content[5:] if content.startswith("Bio: ") else content
    return text.split("\n")[0].strip()


def build_bank_and_testset() -> tuple[list[dict], list[dict]]:
    for path in (TRAIN_PATH, EVAL_PATH):
        if not path.exists():
            print(f"[error] {path} ne postoji. Prvo pokreni generate_dataset.", file=sys.stderr)
            sys.exit(1)

    bank: list[dict] = []
    testset: list[dict] = []

    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            bank.append({
                "id": len(bank),
                "persona_id": r["persona_id"],
                "icebreaker": r["messages"][1]["content"],
            })

    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            testset.append({
                "persona_id": r["persona_id"],
                "bio": _parse_bio(r["messages"][0]["content"]),
                "gold_icebreaker": r["messages"][1]["content"],
            })

    ICEBREAKER_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_BANK, "w", encoding="utf-8") as f:
        for entry in bank:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(RETRIEVAL_TESTSET, "w", encoding="utf-8") as f:
        for entry in testset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[info] Banka         : {len(bank)} icebreaker-a  → {RETRIEVAL_BANK.name}")
    print(f"[info] Test skup     : {len(testset)} (bio, gold) parova → {RETRIEVAL_TESTSET.name}")
    return bank, testset


# ── Retrieval ──────────────────────────────────────────────────────────────────

def tfidf_sim(
    bios: list[str], bank_icebreakers: list[str]
) -> np.ndarray:
    """Cosine sličnost TF-IDF vektora. Fitovanje na bio+bank korpusu
    kako bi cross-modal matching imao zajedničke IDF težine."""
    vect = TfidfVectorizer(max_features=TFIDF_MAX_FEATS, sublinear_tf=True)
    vect.fit(bios + bank_icebreakers)
    bank_mat = vect.transform(bank_icebreakers)
    query_mat = vect.transform(bios)
    return sk_cosine(query_mat, bank_mat)  # (n_queries, n_bank)


# ── Evaluacija ─────────────────────────────────────────────────────────────────

def compute_metrics(
    retrieval_sim: np.ndarray,
    oracle_sim: np.ndarray,
    k_values: list[int],
    threshold: float,
) -> dict[str, float]:
    """
    Precision@k : udio relevantnih stavki u top-k.
    Recall@k    : udio pronađenih relevantnih od ukupno relevantnih u banci.
    MRR         : 1/rang prve relevantne stavke (0 ako nema).
    Relevantnost se određuje oracle_sim >= threshold.
    """
    p_at_k: dict[int, list[float]] = {k: [] for k in k_values}
    r_at_k: dict[int, list[float]] = {k: [] for k in k_values}
    mrr: list[float] = []

    for i in range(retrieval_sim.shape[0]):
        ranked     = np.argsort(retrieval_sim[i])[::-1]   # opadajući rang
        relevant   = oracle_sim[i] >= threshold            # bool maska za banku
        total_rel  = max(1, int(relevant.sum()))
        ranked_rel = relevant[ranked]                      # relevantnost po rangu

        for k in k_values:
            top_k = ranked_rel[:k]
            p_at_k[k].append(float(top_k.sum()) / k)
            r_at_k[k].append(float(top_k.sum()) / total_rel)

        hits = np.where(ranked_rel)[0]
        mrr.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    return {
        **{f"P@{k}": float(np.mean(p_at_k[k])) for k in k_values},
        **{f"R@{k}": float(np.mean(r_at_k[k])) for k in k_values},
        "MRR": float(np.mean(mrr)),
    }


# ── Izvještaj ──────────────────────────────────────────────────────────────────

def print_report(
    tfidf_m: dict[str, float],
    minilm_m: dict[str, float],
    k_values: list[int],
    n_bank: int,
    n_test: int,
    avg_rel: float,
) -> None:
    keys = (
        [f"P@{k}" for k in k_values]
        + [f"R@{k}" for k in k_values]
        + ["MRR"]
    )
    w = 10
    sep = "+" + "-" * 14 + "+" + "-" * w + "+" + "-" * w + "+"

    print("\n" + "=" * (14 + w * 2 + 4))
    print(" RETRIEVAL  bio → icebreaker  |  TF-IDF vs MiniLM")
    print("=" * (14 + w * 2 + 4))
    print(f" Banka        : {n_bank} icebreaker-a (train personas)")
    print(f" Test upiti   : {n_test} bio-a (held-out eval personas)")
    print(f" Oracle relev.: MiniLM cosine ≥ {_ORACLE_THRESHOLD}"
          f"  (prosj. {avg_rel:.1f} relevantin/{n_bank} u banci)")
    print()
    print(sep)
    print(f"| {'Metrika':<12} | {'TF-IDF':^{w-2}} | {'MiniLM':^{w-2}} |")
    print(sep)
    for key in keys:
        t = tfidf_m[key]
        m = minilm_m[key]
        mark = " ◄" if m > t + 1e-6 else (" ►" if t > m + 1e-6 else "  ")
        print(f"| {key:<12} | {t:^{w-2}.4f} | {m:^{w-2}.4f}{mark}|")
    print(sep)

    # Zaključak
    print("\n[zaključak]")
    td, md = tfidf_m["MRR"], minilm_m["MRR"]
    if md > td + 1e-4:
        print(f"  MiniLM pobjeđuje TF-IDF po MRR: {md:.4f} vs {td:.4f} "
              f"(+{md-td:.4f}, +{100*(md-td)/max(td,1e-9):.1f}%).")
        print("  Kontekstualni embeddinzi bolje hvataju semantičku vezu bio ↔ icebreaker,")
        print("  jer isti pojmovi u različitim kontekstima dobijaju različite vektore.")
    elif td > md + 1e-4:
        print(f"  TF-IDF pobjeđuje MiniLM po MRR: {td:.4f} vs {md:.4f} "
              f"(+{td-md:.4f}).")
        print("  Direktno leksičko preklapanje bio ↔ icebreaker je dovoljno za ovaj zadatak,")
        print("  ili je broj relevantnih stavki u banci premali za prednost MiniLM-a.")
    else:
        print(f"  TF-IDF i MiniLM postižu gotovo identičan MRR ({td:.4f} vs {md:.4f}).")

    print(f"\n  Napomena: oracle koristi isti model kao MiniLM retrieval ({EMBED_MODEL}),")
    print("  što može dati blagu prednost MiniLM-u u definiciji relevantnosti.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[info] Gradim banku i test skup...")
    bank, testset = build_bank_and_testset()

    bank_icebreakers = [b["icebreaker"] for b in bank]
    bios             = [t["bio"]             for t in testset]
    golds            = [t["gold_icebreaker"] for t in testset]

    # ── Enkodiranje sa MiniLM (jednom, dijeli se između oracle i retrieval) ──
    print(f"\n[info] Učitavam '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[info] Enkodiranje banke (icebreakeri)...")
    bank_embs = model.encode(
        bank_icebreakers, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=64, show_progress_bar=True,
    )

    print("[info] Enkodiranje gold icebreakera (oracle)...")
    gold_embs = model.encode(
        golds, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=64, show_progress_bar=True,
    )
    oracle_sim = gold_embs @ bank_embs.T  # (n_queries, n_bank)

    avg_rel = float(np.mean((oracle_sim >= _ORACLE_THRESHOLD).sum(axis=1)))
    print(f"[info] Prosj. relevantin stavki/upit: {avg_rel:.1f}/{len(bank)}")

    # ── TF-IDF retrieval ──────────────────────────────────────────────────────
    print("\n[info] TF-IDF retrieval...")
    tf_sim = tfidf_sim(bios, bank_icebreakers)
    tfidf_metrics = compute_metrics(tf_sim, oracle_sim, RETRIEVAL_K, _ORACLE_THRESHOLD)

    # ── MiniLM retrieval ──────────────────────────────────────────────────────
    print("\n[info] MiniLM retrieval (enkodiranje bio upita)...")
    bio_embs = model.encode(
        bios, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=64, show_progress_bar=True,
    )
    ml_sim = bio_embs @ bank_embs.T  # (n_queries, n_bank)
    minilm_metrics = compute_metrics(ml_sim, oracle_sim, RETRIEVAL_K, _ORACLE_THRESHOLD)

    # ── Izvještaj ─────────────────────────────────────────────────────────────
    print_report(
        tfidf_metrics, minilm_metrics, RETRIEVAL_K,
        n_bank=len(bank), n_test=len(testset), avg_rel=avg_rel,
    )

    # Primjeri: top-3 TF-IDF vs MiniLM za prvi test bio
    print("\n[info] Primjer (prvi test bio):")
    bio0 = bios[0]
    gold0 = golds[0]
    print(f"  Bio  : {bio0}")
    print(f"  Gold : {gold0}")
    for label, sim_row in [("TF-IDF", tf_sim[0]), ("MiniLM", ml_sim[0])]:
        top3 = np.argsort(sim_row)[::-1][:3]
        print(f"\n  {label} top-3:")
        for rank, idx in enumerate(top3, 1):
            oracle_s = float(oracle_sim[0, idx])
            print(f"    {rank}. [{oracle_s:.3f}] {bank_icebreakers[idx]}")


if __name__ == "__main__":
    main()
