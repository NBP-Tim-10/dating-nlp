"""Generacija trening/eval skupa: persona → (bio, icebreaker) via Claude API.

Pokretanje (test run, 10 persona):
    python -m src.icebreaker.generate_dataset

Pun set (nakon što MAX_PERSONAS=None u config.py):
    python -m src.icebreaker.generate_dataset
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time

from dotenv import load_dotenv

from src.icebreaker.config import (
    EVAL_HOLDOUT,
    EVAL_PATH,
    GEN_MODE,
    GEN_MODEL,
    ICEBREAKER_DIR,
    MAX_PERSONAS,
    MAX_TOKENS_OUT,
    PERSONAS_PATH,
    SEED,
    TEMPERATURES,
    TRAIN_PATH,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

try:
    import anthropic as _ant
except ImportError:
    print("[error] Nedostaje paket 'anthropic'. Instaliraj: pip install anthropic", file=sys.stderr)
    sys.exit(1)

# Haiku 4.5 cijene (jun 2026): $1/$5 per million tokena (input/output)
_PRICE_IN  = 1.00 / 1_000_000
_PRICE_OUT = 5.00 / 1_000_000

# Fajl za čuvanje batch ID-a između pokretanja (samo za batch mode)
_BATCH_STATE = ICEBREAKER_DIR / ".batch_state.json"

_PROMPT = (
    "Persona facts:\n{facts}\n\n"
    "Rewrite as a short, natural first-person dating bio (max 2 sentences), "
    "grounded ONLY in the facts above (do not invent new facts).\n"
    "Then write ONE playful, specific icebreaker someone could send after matching.\n"
    'Return JSON only, no prose: {{"bio": "...", "icebreaker": "..."}}'
)


def _make_prompt(facts: list[str]) -> str:
    return _PROMPT.format(facts="\n".join(f"- {f}" for f in facts))


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict) and d.get("bio") and d.get("icebreaker"):
            return d
    except json.JSONDecodeError:
        pass
    # Pokušaj izvući JSON objekat ako model doda prozu oko njega
    m = re.search(r'\{[^{}]*?"bio"[^{}]*?"icebreaker"[^{}]*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, dict) and d.get("bio") and d.get("icebreaker"):
                return d
        except json.JSONDecodeError:
            pass
    return None


def _make_record(persona: dict, data: dict, temperature: float) -> dict:
    return {
        "persona_id": persona["id"],
        "gen_model": GEN_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": f"Bio: {data['bio']}\nWrite an icebreaker."},
            {"role": "assistant", "content": data["icebreaker"]},
        ],
    }


# ── Sync mode (async parallel) ─────────────────────────────────────────────────

async def _call_one(
    client: _ant.AsyncAnthropic,
    sem: asyncio.Semaphore,
    persona: dict,
    temperature: float,
    counter: dict,
    total: int,
) -> tuple[dict, float, dict | None, dict | None]:
    prompt = _make_prompt(persona["facts"])
    async with sem:
        try:
            resp = await client.messages.create(
                model=GEN_MODEL,
                max_tokens=MAX_TOKENS_OUT,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            usage: dict | None = {
                "input": resp.usage.input_tokens,
                "output": resp.usage.output_tokens,
            }
            data = _parse_json(text)
            if data is None:
                print(f"  [warn] JSON parse fail: id={persona['id']} T={temperature:.1f}")
        except Exception as exc:
            print(f"  [warn] API error: id={persona['id']} T={temperature:.1f}: {exc}")
            return persona, temperature, None, None

    counter["n"] += 1
    if counter["n"] % 20 == 0 or counter["n"] == total:
        print(f"  [{counter['n']}/{total}] generisano...")
    return persona, temperature, data, usage


async def _run_sync(
    personas: list[dict],
    temperatures: list[float],
    concurrency: int = 8,
) -> tuple[list[dict], list[dict]]:
    client = _ant.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    total = len(personas) * len(temperatures)
    counter = {"n": 0}

    tasks = [
        _call_one(client, sem, p, t, counter, total)
        for p in personas
        for t in temperatures
    ]

    results = await asyncio.gather(*tasks)

    records: list[dict] = []
    usages: list[dict] = []
    for persona, temperature, data, usage in results:
        if usage:
            usages.append(usage)
        if data is not None:
            records.append(_make_record(persona, data, temperature))

    return records, usages


# ── Batch mode ─────────────────────────────────────────────────────────────────

def _run_batch(
    personas: list[dict],
    temperatures: list[float],
) -> tuple[list[dict], list[dict]]:
    client = _ant.Anthropic()

    # Nastavi postojeći batch ako postoji sačuvani ID
    batch_id: str | None = None
    if _BATCH_STATE.exists():
        try:
            state = json.loads(_BATCH_STATE.read_text(encoding="utf-8"))
            batch_id = state.get("batch_id")
            print(f"[info] Nastavak postojećeg batcha: {batch_id}")
        except Exception:
            pass

    if batch_id is None:
        requests = []
        for p in personas:
            for t_idx, t in enumerate(temperatures):
                requests.append({
                    "custom_id": f"p{p['id']}_t{t_idx}",
                    "params": {
                        "model": GEN_MODEL,
                        "max_tokens": MAX_TOKENS_OUT,
                        "temperature": t,
                        "messages": [
                            {"role": "user", "content": _make_prompt(p["facts"])}
                        ],
                    },
                })

        print(f"[info] Kreiram batch sa {len(requests)} zahtjeva...")
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        _BATCH_STATE.write_text(
            json.dumps({"batch_id": batch_id}), encoding="utf-8"
        )
        print(f"[info] Batch ID: {batch_id}  (sačuvan u {_BATCH_STATE})")

    # Polling dok batch ne završi (može potrajati do 24h)
    print("[info] Čekam završetak batcha...")
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        print(
            f"  status={status.processing_status} | "
            f"succeeded={counts.succeeded} errored={counts.errored} "
            f"processing={counts.processing}"
        )
        if status.processing_status == "ended":
            break
        time.sleep(60)

    # Prikupljanje rezultata
    persona_map = {p["id"]: p for p in personas}
    temp_map = {i: t for i, t in enumerate(temperatures)}

    records: list[dict] = []
    usages: list[dict] = []

    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  [warn] {result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        text = msg.content[0].text
        usages.append({
            "input": msg.usage.input_tokens,
            "output": msg.usage.output_tokens,
        })
        m = re.match(r"p(\d+)_t(\d+)", result.custom_id)
        if not m:
            continue
        pid, t_idx = int(m.group(1)), int(m.group(2))
        if pid not in persona_map or t_idx not in temp_map:
            continue
        data = _parse_json(text)
        if data is None:
            print(f"  [warn] JSON parse fail: {result.custom_id}")
            continue
        records.append(_make_record(persona_map[pid], data, temp_map[t_idx]))

    if _BATCH_STATE.exists():
        _BATCH_STATE.unlink()

    return records, usages


# ── Cost reporting ─────────────────────────────────────────────────────────────

def _report_cost(usages: list[dict], full_persona_count: int) -> None:
    if not usages:
        return
    avg_in  = sum(u["input"]  for u in usages) / len(usages)
    avg_out = sum(u["output"] for u in usages) / len(usages)
    full_calls = full_persona_count * len(TEMPERATURES)
    est = full_calls * (avg_in * _PRICE_IN + avg_out * _PRICE_OUT)

    print("\n[info] === TEST-FIRST PROCJENA TROŠKA ===")
    print(f"  Izmjerenih poziva       : {len(usages)}")
    print(f"  Prosječno tokena/pozivu : {avg_in:.0f} in + {avg_out:.0f} out")
    print(f"  Procjena za {full_calls} poziva : ${est:.3f}")
    if est > 3.0:
        print("  [warn] Prelazi $3 budžet! Skrati MAX_TOKENS_OUT ili smanji TARGET_N.")
    else:
        print("  [ok]   Unutar $3 budžeta. Postavi MAX_PERSONAS=None za pun set.")
    print("")
    print("  Vizuelno provjeri generirane primjere gore, a zatim:")
    print("  → config.py: MAX_PERSONAS = None")
    print("  → ponovo pokreni: python -m src.icebreaker.generate_dataset")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ICEBREAKER_DIR.mkdir(parents=True, exist_ok=True)

    if not PERSONAS_PATH.exists():
        print(
            f"[error] {PERSONAS_PATH} ne postoji. "
            "Prvo pokreni: python -m src.icebreaker.extract_personas",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(PERSONAS_PATH, encoding="utf-8") as f:
        all_personas = [json.loads(line) for line in f]
    print(f"[info] Učitano {len(all_personas)} persona iz {PERSONAS_PATH}.")

    # Split na nivou persone — PRIJE generacije
    rng = random.Random(SEED)
    shuffled = all_personas.copy()
    rng.shuffle(shuffled)
    eval_personas  = shuffled[:EVAL_HOLDOUT]
    train_personas = shuffled[EVAL_HOLDOUT:]

    assert not ({p["id"] for p in eval_personas} & {p["id"] for p in train_personas}), \
        "Presjek train/eval persona nije prazan!"
    print(f"[info] Split: {len(train_personas)} train + {len(eval_personas)} eval persona.")

    # TEST-FIRST: limitiraj na MAX_PERSONAS iz train skupa
    is_test_run = MAX_PERSONAS is not None
    if is_test_run:
        active = train_personas[:MAX_PERSONAS]
        print(
            f"[info] TEST-FIRST mode: {len(active)} persona "
            f"× {len(TEMPERATURES)} temperature = {len(active) * len(TEMPERATURES)} poziva."
        )
    else:
        active = train_personas + eval_personas
        n_calls = len(active) * len(TEMPERATURES)
        print(f"[info] Pun set: {len(active)} persona × {len(TEMPERATURES)} = {n_calls} poziva.")

    # Generacija
    print(f"[info] Pokrećem generaciju (mode={GEN_MODE})...")
    if GEN_MODE == "sync":
        records, usages = asyncio.run(_run_sync(active, TEMPERATURES))
    elif GEN_MODE == "batch":
        records, usages = _run_batch(active, TEMPERATURES)
    else:
        print(f"[error] Nepoznat GEN_MODE: '{GEN_MODE}'. Dozvoljeno: 'sync', 'batch'.", file=sys.stderr)
        sys.exit(1)

    # Procjena troška — samo nakon test runa
    if is_test_run:
        _report_cost(usages, len(all_personas))

    # Razdvoj po skupu
    train_ids = {p["id"] for p in (train_personas[:MAX_PERSONAS] if is_test_run else train_personas)}
    eval_ids  = {p["id"] for p in ([] if is_test_run else eval_personas)}

    train_records = [r for r in records if r["persona_id"] in train_ids]
    eval_records  = [r for r in records if r["persona_id"] in eval_ids]

    # Zapis
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistike
    total_calls  = len(active) * len(TEMPERATURES)
    rejected     = total_calls - len(records)
    reject_pct   = 100 * rejected / max(1, total_calls)

    print(f"\n[done] Zapisano: {len(train_records)} train + {len(eval_records)} eval primjera.")
    print(f"[done] Odbačeni (nevažeći JSON): {rejected}/{total_calls} ({reject_pct:.1f}%)")

    # Provjera presjeka na nivou persona
    train_pids = {r["persona_id"] for r in train_records}
    eval_pids  = {r["persona_id"] for r in eval_records}
    assert not (train_pids & eval_pids), "GREŠKA: Ista persona u train i eval skupu!"
    print("[done] Provjera presjeka persona: OK")

    # Kratki pregled (prvih 3 generisana primjera)
    if records:
        print("\n[info] Primjeri (prvih 3):")
        for r in records[:3]:
            bio_msg = r["messages"][0]["content"]
            ice_msg = r["messages"][1]["content"]
            print(f"  [pid={r['persona_id']} T={r['temperature']}]")
            print(f"    {bio_msg}")
            print(f"    → {ice_msg}")


if __name__ == "__main__":
    main()
