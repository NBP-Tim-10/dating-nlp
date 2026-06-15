from __future__ import annotations

from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT

SEED = 42

# ── Persona extraction ─────────────────────────────────────────────────────────
DATASET       = "awsaf49/persona-chat"
TARGET_N      = 500
USE_DIVERSITY = True
N_CLUSTERS    = 100
EMBED_MODEL   = "all-MiniLM-L6-v2"

ICEBREAKER_DIR = PROCESSED_DIR / "icebreaker"
PERSONAS_PATH  = ICEBREAKER_DIR / "personas.jsonl"

# ── Dataset generation ─────────────────────────────────────────────────────────
# Haiku 4.5 sync ≈ $1.45 za 1500 primjera (preporučeno za <$3 budžet)
# Alternativa višeg kvaliteta: GEN_MODEL="claude-sonnet-4-6" + GEN_MODE="batch" ≈ $2.2
GEN_MODEL      = "claude-haiku-4-5-20251001"
JUDGE_MODEL    = "gpt-4o-mini"   # GPT sudija — koristi se samo u evaluaciji (druga familija)
TEMPERATURES   = [0.5, 0.8, 1.0]
EVAL_HOLDOUT   = 75
GEN_MODE       = "sync"          # "sync" (paralelno, odmah) ili "batch" (−50%, asinhrono)
MAX_PERSONAS   = None            # TEST-FIRST: izmjeri trošak, pa postavi None za pun set
MAX_TOKENS_OUT = 160

TRAIN_PATH = ICEBREAKER_DIR / "dataset_train.jsonl"
EVAL_PATH  = ICEBREAKER_DIR / "dataset_eval.jsonl"

# ── Retrieval (poređenje reprezentacija) ───────────────────────────────────────
RETRIEVAL_BANK    = ICEBREAKER_DIR / "icebreaker_bank.jsonl"
RETRIEVAL_TESTSET = ICEBREAKER_DIR / "retrieval_testset.jsonl"
RETRIEVAL_K       = [1, 3, 5]
TFIDF_MAX_FEATS   = 5000

# ── Fine-tuning (Colab T4) ─────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN  = 2048
LORA_R       = 16
LORA_ALPHA   = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EPOCHS       = 3
LOAD_IN_4BIT = True
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "adapter"

# ── Inference & eval ───────────────────────────────────────────────────────────
MERGED_PATH = PROJECT_ROOT / "outputs" / "merged_model"
