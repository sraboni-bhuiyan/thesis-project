# Fairness-Aware RAG Triage Assistant

This repository contains the code and data pipeline for my Master’s thesis on a fairness-aware triage assistant that compares a retrieval-augmented (RAG) system to a prompt-only baseline for predicting emergency care urgency and specialty.

## Project structure

- `data/guidelines_raw/` – Original triage guidance PDFs
- `data/guidelines_clean/` – Cleaned, chunked corpus (`corpus.jsonl`)
- `data/cases/` – Evaluation cases (`main.csv`, `fairness_variants.csv`)
- `src/` – Scripts (`build_corpus.py`, `run_baseline.py`, `run_rag.py`, `evaluate.py`, `config.py`, `prompts/`)
- `results/` – Model predictions and fairness results
- `notebooks/` – Optional exploratory notebooks
- `README.md`, `requirements.txt`

## Setup

```bash
# clone
git clone https://github.com/sraboni-bhuiyan/thesis-project.git
cd thesis-project

# create and activate virtual environment (example for Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Basic workflow

1. **Prepare corpus**: place triage PDFs in `data/guidelines_raw/` and run:

   ```bash
   python src/build_corpus.py
   ```

2. **Run baseline**: generate predictions for the prompt-only system:

   ```bash
   python src/run_baseline.py
   ```

3. **Run RAG**: run the retrieval-augmented variant:

   ```bash
   python src/run_rag.py
   ```

4. **Evaluate**: compute performance and fairness metrics:

   ```bash
   python src/evaluate.py
   ```

## Status

Work in progress as part of my Master’s thesis. API keys and sensitive configuration are not included in this repository.