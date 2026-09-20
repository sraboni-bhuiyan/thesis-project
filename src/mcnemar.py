"""
Paired exact McNemar test between two configs on the same cases.

Method (the one used for the comparisons in results/RESULTS_NOTES.md):
two-sided exact binomial test on the discordant pairs, P(success) = 0.5.
Only cases present in BOTH prediction files and in the ground truth are used;
ERROR rows are dropped pairwise. INVALID counts as wrong, as in evaluate.py.

Usage:
  python src/mcnemar.py -a results/baseline_predictions.csv -b results/rag_predictions.csv
  python src/mcnemar.py -a results/rag_predictions.csv -b results/rag_fixed_predictions.csv --metric urgency
"""
import argparse
import json

from scipy.stats import binomtest

from common import normalize_specialty, normalize_urgency, read_cases, resolve, row_key
from config import CASES_PATH, FAIRNESS_PATH


def correctness(cases: dict, preds: list, metric: str) -> dict:
    """{case key: True/False} for rows with a usable answer; ERROR rows are omitted."""
    out = {}
    for row in preds:
        key = row_key(row)
        ref = cases.get(key)
        if ref is None or row["pred_urgency"] == "ERROR":
            continue
        if metric == "urgency":
            out[key] = normalize_urgency(row["pred_urgency"]) == normalize_urgency(ref["ground_truth_urgency"])
        else:
            out[key] = normalize_specialty(row["pred_specialty"]) == normalize_specialty(ref["ground_truth_specialty"])
    return out


def mcnemar(a: dict, b: dict) -> dict:
    """Exact two-sided McNemar on the paired correctness dicts a and b."""
    shared = sorted(set(a) & set(b))
    only_a = sum(a[k] and not b[k] for k in shared)
    only_b = sum(b[k] and not a[k] for k in shared)
    n_disc = only_a + only_b
    p = binomtest(min(only_a, only_b), n_disc, 0.5).pvalue if n_disc else 1.0
    return {
        "n_paired": len(shared),
        "acc_a": sum(a[k] for k in shared) / len(shared) if shared else None,
        "acc_b": sum(b[k] for k in shared) / len(shared) if shared else None,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "n_discordant": n_disc,
        "p_value": p,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired exact McNemar between two prediction files.")
    parser.add_argument("-a", "--pred-a", required=True)
    parser.add_argument("-b", "--pred-b", required=True)
    parser.add_argument("--metric", choices=["urgency", "specialty"], default="urgency")
    parser.add_argument("--cases-file", default=None,
                        help="Ground-truth CSV. Default: fairness_variants.csv if preds have variant_id, else main.csv")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    preds_a, preds_b = read_cases(resolve(args.pred_a)), read_cases(resolve(args.pred_b))
    is_variants = any(r.get("variant_id") for r in preds_a)
    cases_path = resolve(args.cases_file) if args.cases_file else (FAIRNESS_PATH if is_variants else CASES_PATH)
    cases = {row_key(r): r for r in read_cases(cases_path)}

    res = mcnemar(correctness(cases, preds_a, args.metric), correctness(cases, preds_b, args.metric))
    res.update({"metric": args.metric, "pred_a": resolve(args.pred_a).name, "pred_b": resolve(args.pred_b).name,
                "cases_file": cases_path.name})
    print(json.dumps(res, indent=2))
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
