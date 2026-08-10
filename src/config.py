from pathlib import Path
import os

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
TEMPERATURE = 0.0

# Chunking / retrieval params
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

from pathlib import Path
import os

# Repo root
ROOT = Path(__file__).resolve().parent.parent

# Project data folders (inside repo)
DATA_PATH = ROOT / "data"
GUIDELINES_PATH = DATA_PATH / "guidelines_raw"
CORPUS_PATH = DATA_PATH / "corpus.jsonl"
CASES_PATH = DATA_PATH / "cases" / "main.csv"          # derived from MIMIC-IV-Ext
FAIRNESS_PATH = DATA_PATH / "fairness_variants" / "variants.csv"
RESULTS_PATH = ROOT / "results"

# Raw PhysioNet dataset location (on your machine, NOT in Git)
PHYSIONET_BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis") / \
    "mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2"
