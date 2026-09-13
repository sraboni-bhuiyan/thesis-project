"""Central configuration. Every script imports paths and frozen experiment settings from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---- Model / inference (frozen for all runs) ----
# Served via OpenRouter (see .env: OPENROUTER_BASE_URL + OPENAI_API_KEY). Override with LLM_MODEL.
MODEL_NAME = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
TEMPERATURE = 0.0
MAX_TOKENS = 1024

# ---- Retrieval ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
PROMPTS_PATH = SRC_PATH / "prompts"
DATA_PATH = ROOT / "data"
GUIDELINES_PATH = DATA_PATH / "guidelines_raw"
CORPUS_FILE = DATA_PATH / "guidelines_clean" / "corpus.jsonl"
INDEX_DIR = DATA_PATH / "indexes"
FAISS_INDEX = INDEX_DIR / "guidelines.index"
ID_MAP_FILE = INDEX_DIR / "id_map.json"
CASES_PATH = DATA_PATH / "cases" / "main.csv"
FAIRNESS_PATH = DATA_PATH / "cases" / "fairness_variants.csv"
RESULTS_PATH = ROOT / "results"

# Raw PhysioNet dataset location (local only, NOT in Git)
PHYSIONET_BASE = ROOT / "mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2"

# ---- Labels ----
# Triage acuity (ESI 1-5) -> urgency colour
URGENCY_MAP = {1: "red", 2: "orange", 3: "yellow", 4: "green", 5: "blue"}
# Ordered least -> most severe
URGENCY_ORDER = ["blue", "green", "yellow", "orange", "red"]
# Frozen fairness binary: high acuity = triage level 1-2
HIGH_ACUITY = {"red", "orange"}

# ---- Protected groups ----
YOUNG_MAX_AGE = 40   # young: age < 40
OLD_MIN_AGE = 65     # old:   age >= 65
YOUNG_VARIANT_AGE = 30
OLD_VARIANT_AGE = 70
