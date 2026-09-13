"""
Run-to-run noise check: compare two prediction files produced with the same config on the same cases.

Temperature 0 through an API is not fully deterministic, so part of any "inconsistency" across
demographic variants is plain noise. This script measures that noise floor.

Usage:
  python src/repeat_agreement.py --a results/rag_variants_predictions.csv --b results/rag_variants_predictions_rep2.csv
"""
import argparse
import json

from common import normalize_specialty, normalize_urgency, read_cases, resolve, row_key
from fairness_metrics import counterfactual_consistency


def main():
    parser = argparse.ArgumentParser(description="Agreement between two runs of the same config.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    a_rows, b_rows = read_cases(resolve(args.a)), read_cases(resolve(args.b))
    a, b = {row_key(r): r for r in a_rows}, {row_key(r): r for r in b_rows}
    keys = [k for k in a if k in b and "ERROR" not in (a[k]["pred_urgency"], b[k]["pred_urgency"])]
    same_u = sum(normalize_urgency(a[k]["pred_urgency"]) == normalize_urgency(b[k]["pred_urgency"]) for k in keys)
    same_s = sum(normalize_specialty(a[k]["pred_specialty"]) == normalize_specialty(b[k]["pred_specialty"]) for k in keys)
    res = {
        "file_a": resolve(args.a).name,
        "file_b": resolve(args.b).name,
        "n_compared": len(keys),
        "urgency_run_agreement": same_u / len(keys) if keys else None,
        "specialty_run_agreement": same_s / len(keys) if keys else None,
    }
    if any(r.get("variant_id") for r in a_rows):
        res["consistency_run_a"] = counterfactual_consistency(a_rows)["urgency_consistency"]
        res["consistency_run_b"] = counterfactual_consistency(b_rows)["urgency_consistency"]
    print(json.dumps(res, indent=2))
    if args.output:
        resolve(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
