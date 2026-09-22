"""
Performance metrics for one prediction file.

Urgency: exact and +-1 adjacent accuracy, over-/under-triage rate, INVALID and ERROR counts.
Specialty: accuracy vs. the first clinician-approved label and vs. any listed, after
normalization.

Usage:
  python src/evaluate.py -p results/rag_predictions.csv --output results/eval_rag.json
"""
import argparse
import csv
import json
from pathlib import Path

from common import models_in, normalize_specialty, normalize_urgency, read_cases, resolve, row_key
from config import CASES_PATH, FAIRNESS_PATH, URGENCY_ORDER

URGENCY_INDEX = {u: i for i, u in enumerate(URGENCY_ORDER)}


def compare_urgency(pred: str, ref: str) -> str:
    """'correct' | 'over' (more severe than reference) | 'under' | 'invalid'."""
    if pred not in URGENCY_INDEX or ref not in URGENCY_INDEX:
        return "invalid"
    p, r = URGENCY_INDEX[pred], URGENCY_INDEX[ref]
    return "correct" if p == r else ("over" if p > r else "under")


def evaluate(cases: dict, preds: list) -> dict:
    n = n_err = n_invalid = correct = over = under = adjacent = 0
    spec_correct = spec_any = 0
    for row in preds:
        ref = cases.get(row_key(row))
        if ref is None:
            continue
        n += 1
        if row["pred_urgency"] == "ERROR":
            n_err += 1
            continue
        pred_u = normalize_urgency(row["pred_urgency"])
        ref_u = normalize_urgency(ref["ground_truth_urgency"])
        cat = compare_urgency(pred_u, ref_u)
        if cat == "invalid":
            n_invalid += 1
        else:
            correct += cat == "correct"
            over += cat == "over"
            under += cat == "under"
            adjacent += abs(URGENCY_INDEX[pred_u] - URGENCY_INDEX[ref_u]) <= 1

        pred_s = normalize_specialty(row["pred_specialty"])
        spec_correct += pred_s == normalize_specialty(ref["ground_truth_specialty"])
        all_refs = (ref.get("ground_truth_specialty_all") or ref["ground_truth_specialty"]).split("|")
        spec_any += pred_s in {normalize_specialty(s) for s in all_refs}

    answered = n - n_err  # denominator: rows with a model answer (INVALID counts as wrong)
    rate = lambda k: k / answered if answered else None
    return {
        "n_cases": n,
        "n_error": n_err,
        "n_invalid_urgency": n_invalid,
        "urgency_exact_acc": rate(correct),
        "urgency_adjacent_acc": rate(adjacent),
        "over_triage_rate": rate(over),
        "under_triage_rate": rate(under),
        "specialty_acc": rate(spec_correct),
        "specialty_acc_any_listed": rate(spec_any),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate triage predictions.")
    parser.add_argument("-p", "--pred-file", required=True)
    parser.add_argument("--cases-file", default=None,
                        help="Ground-truth CSV. Default: fairness_variants.csv if preds have variant_id, else main.csv")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    pred_path = resolve(args.pred_file)
    preds = read_cases(pred_path)
    is_variants = any(r.get("variant_id") for r in preds)
    cases_path = resolve(args.cases_file) if args.cases_file else (FAIRNESS_PATH if is_variants else CASES_PATH)
    cases = {row_key(r): r for r in read_cases(cases_path)}

    metrics = evaluate(cases, preds)
    metrics.update({"pred_file": str(pred_path.name), "cases_file": str(Path(cases_path).name),
                    "model": models_in(preds)})
    print(json.dumps(metrics, indent=2))
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
