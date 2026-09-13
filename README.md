# Fairness-Aware RAG Triage Assistant

Master's thesis code: compares a prompt-only LLM baseline with a retrieval-augmented (RAG) system, and RAG with demographic masking, for predicting emergency **urgency** (red/orange/yellow/green/blue) and **specialty**, and measures fairness across age × gender.

| Config | Retrieval | Age/gender in prompt | Research question |
|---|---|---|---|
| C1 Baseline | no | yes | RQ1–RQ3 |
| C2 RAG | yes | yes | RQ1–RQ3 |
| C3 RAG + masking | yes | no (also removed from HPI text) | RQ4 |

## Project structure

- `data/guidelines_raw/`: triage guideline PDFs
- `data/guidelines_clean/corpus.jsonl`: chunked guideline corpus
- `data/indexes/`: FAISS index + chunk id map
- `data/cases/`: `main.csv` (331 clinician-approved cases) and `fairness_variants.csv` (70 × 4). Derived from MIMIC and **not for redistribution**.
- `src/config.py`: paths and frozen settings (model, temperature, TOP_K, label maps, age buckets)
- `src/common.py`: API client, retries, JSON parsing, label normalization, resumable runner
- `src/demographics.py`: gender/age rewriting (variants) and masking (C3)
- `src/run_baseline.py`, `src/run_rag.py`: inference
- `src/evaluate.py`, `src/fairness_metrics.py`: metrics
- `src/make_summary.py`, `src/retrieval_eval.py`, `src/error_cases.py`: analysis
- `tests/`: hand-checked metric tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Create `.env` (never commit it):

```
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
# optional: LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

Place the PhysioNet dataset folder `mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2/` in the project root (credentialed access required).

## Reproduce

```bash
# 0. tests
python -m pytest tests

# 1. data (run once)
python src/extract_mimic_cases.py            # -> data/cases/main.csv (331 cases; --max-cases 250 for subset)
python src/build_fairness_variants.py        # -> data/cases/fairness_variants.csv (280 rows)
python src/build_index.py                    # -> data/indexes/

# 2. smoke test (expect 0 INVALID / 0 ERROR)
python src/run_baseline.py --limit 20 --output results/smoke_baseline.csv
python src/run_rag.py --limit 20 --output results/smoke_rag.csv --log-file results/smoke_logs.jsonl
python src/evaluate.py -p results/smoke_rag.csv

# 3. six inference runs (resumable: re-run the same command after rate-limit errors)
python src/run_baseline.py
python src/run_rag.py
python src/run_rag.py --mask-demographics
python src/run_baseline.py --cases-file data/cases/fairness_variants.csv --output results/baseline_variants_predictions.csv
python src/run_rag.py --cases-file data/cases/fairness_variants.csv --output results/rag_variants_predictions.csv --log-file results/retrieval_logs_variants.jsonl
python src/run_rag.py --mask-demographics --cases-file data/cases/fairness_variants.csv --output results/rag_masked_variants_predictions.csv --log-file results/retrieval_logs_masked_variants.jsonl

# 4. metrics, summary table and figures
python src/make_summary.py                   # -> results/results_summary.csv, eval_*.json, fairness_*.json, figures/

# 5. retrieval evaluation + qualitative analysis
python src/retrieval_eval.py sample --n 25   # fill `relevant` in results/retrieval_rating_sheet.csv
python src/retrieval_eval.py score --sheet results/retrieval_rating_sheet_rated.csv
python src/error_cases.py --config rag       # -> results/error_cases.csv

# 6. run-to-run noise check (temperature 0 via API is not fully deterministic)
python src/run_baseline.py --cases-file data/cases/fairness_variants.csv --output results/baseline_variants_predictions_rep2.csv
python src/run_rag.py --cases-file data/cases/fairness_variants.csv --output results/rag_variants_predictions_rep2.csv --log-file results/retrieval_logs_variants_rep2.jsonl
python src/repeat_agreement.py --a results/baseline_variants_predictions.csv --b results/baseline_variants_predictions_rep2.csv --output results/noise_baseline.json
python src/repeat_agreement.py --a results/rag_variants_predictions.csv --b results/rag_variants_predictions_rep2.csv --output results/noise_rag.json
```

Add `--sleep 1` to inference commands if the provider rate-limits.

## Metric definitions

- **Urgency**: exact accuracy, ±1 adjacent accuracy, over-/under-triage rate (vs. triage acuity 1–5 mapped to colours). INVALID labels count as wrong; API ERRORs are reported separately.
- **Specialty**: accuracy vs. the first clinician-approved specialty, after normalization and synonym mapping.
- **High acuity** = red/orange (triage 1–2). Groups: young < 40, old ≥ 65, × male/female.
- **DPR**: max − min over the 4 groups of P(pred high acuity).
- **EOD**: max across-group gap in TPR and in FPR for the high-acuity binary.
- **Counterfactual consistency**: fraction of base cases whose 4 variants get identical urgency / specialty / both.
