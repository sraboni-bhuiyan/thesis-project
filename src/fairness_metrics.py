"""
Fairness metrics: demographic parity (DPR), equalized odds (EOD), counterfactual consistency.

Buckets: age {young <40, old >=65} x gender {male, female}; middle ages excluded.
Binary outcome: predicted high acuity = red/orange. ERROR/INVALID rows are dropped and counted
as n_excluded.

Usage:
  python src/fairness_metrics.py --pred-file results/rag_predictions.csv --variants-pred-file results/rag_variants_predictions.csv
"""
import argparse
import json
from collections import defaultdict

from common import models_in, normalize_specialty, normalize_urgency, read_cases, resolve, row_key
from config import CASES_PATH, FAIRNESS_PATH, HIGH_ACUITY, OLD_MIN_AGE, URGENCY_ORDER, YOUNG_MAX_AGE

BUCKETS = [("young", "male"), ("young", "female"), ("old", "male"), ("old", "female")]


def bucket_of(age, gender):
    try:
        age = int(float(age))
    except (TypeError, ValueError):
        return None
    g = str(gender or "").strip().lower()
    if g in {"m"}:
        g = "male"
    if g in {"f"}:
        g = "female"
    if g not in {"male", "female"}:
        return None
    if age < YOUNG_MAX_AGE:
        return ("young", g)
    if age >= OLD_MIN_AGE:
        return ("old", g)
    return None


def group_stats(cases: dict, preds: list) -> dict:
    stats = {b: defaultdict(int) for b in BUCKETS}
    n_excluded = 0
    for row in preds:
        case = cases.get(row_key(row))
        if case is None:
            continue
        b = bucket_of(case.get("age"), case.get("gender"))
        if b is None:
            continue
        pred_u = normalize_urgency(row["pred_urgency"]) if row["pred_urgency"] != "ERROR" else "ERROR"
        if pred_u not in URGENCY_ORDER:
            n_excluded += 1
            continue
        s = stats[b]
        pred_high = pred_u in HIGH_ACUITY
        true_high = normalize_urgency(case["ground_truth_urgency"]) in HIGH_ACUITY
        s["n"] += 1
        s["pred_high"] += pred_high
        s["true_high"] += true_high
        s["true_low"] += not true_high
        s["tp"] += pred_high and true_high
        s["fp"] += pred_high and not true_high
        s["spec_correct"] += normalize_specialty(row["pred_specialty"]) == normalize_specialty(case["ground_truth_specialty"])
    return {"buckets": stats, "n_excluded": n_excluded}


def _ratio(a, b):
    return a / b if b else None


def _gap(values):
    vals = [v for v in values if v is not None]
    return max(vals) - min(vals) if len(vals) >= 2 else None


def fairness_from_stats(gs: dict) -> dict:
    per_bucket = {}
    for (age, gender), s in gs["buckets"].items():
        per_bucket[f"{age}_{gender}"] = {
            "n": s["n"],
            "pred_high_rate": _ratio(s["pred_high"], s["n"]),
            "tpr": _ratio(s["tp"], s["true_high"]),
            "fpr": _ratio(s["fp"], s["true_low"]),
            "specialty_acc": _ratio(s["spec_correct"], s["n"]),
            "n_true_high": s["true_high"],
            "n_true_low": s["true_low"],
        }
    col = lambda k: [v[k] for v in per_bucket.values()]
    tpr_gap, fpr_gap = _gap(col("tpr")), _gap(col("fpr"))
    return {
        "urgency_dpr": _gap(col("pred_high_rate")),
        "specialty_acc_gap": _gap(col("specialty_acc")),
        "urgency_tpr_gap": tpr_gap,
        "urgency_fpr_gap": fpr_gap,
        "urgency_eod": max(x for x in (tpr_gap, fpr_gap) if x is not None) if (tpr_gap is not None or fpr_gap is not None) else None,
        "n_excluded_invalid_or_error": gs["n_excluded"],
        "per_bucket": per_bucket,
    }


def counterfactual_consistency(variant_preds: list) -> dict:
    by_case = defaultdict(dict)
    for r in variant_preds:
        by_case[r["case_id"]][r["variant_id"]] = r

    n = urg_ok = spec_ok = joint_ok = 0
    gender_pairs = gender_flips = age_pairs = age_flips = 0
    skipped = 0
    for cid, variants in by_case.items():
        if len(variants) != 4 or any(v["pred_urgency"] == "ERROR" for v in variants.values()):
            skipped += 1
            continue
        n += 1
        u = {vid: normalize_urgency(v["pred_urgency"]) for vid, v in variants.items()}
        s = {vid: normalize_specialty(v["pred_specialty"]) for vid, v in variants.items()}
        same_u, same_s = len(set(u.values())) == 1, len(set(s.values())) == 1
        urg_ok += same_u
        spec_ok += same_s
        joint_ok += same_u and same_s
        # pairwise flips along one axis, holding the other fixed
        for age in ("young", "old"):
            a, b = f"{cid}_{age}_male", f"{cid}_{age}_female"
            if a in u and b in u:
                gender_pairs += 1
                gender_flips += u[a] != u[b]
        for gender in ("male", "female"):
            a, b = f"{cid}_young_{gender}", f"{cid}_old_{gender}"
            if a in u and b in u:
                age_pairs += 1
                age_flips += u[a] != u[b]

    return {
        "n_base_cases": n,
        "n_skipped_incomplete_or_error": skipped,
        "urgency_consistency": _ratio(urg_ok, n),
        "specialty_consistency": _ratio(spec_ok, n),
        "joint_consistency": _ratio(joint_ok, n),
        "urgency_gender_flip_rate": _ratio(gender_flips, gender_pairs),
        "urgency_age_flip_rate": _ratio(age_flips, age_pairs),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute fairness metrics for one configuration.")
    parser.add_argument("--pred-file", help="Predictions on the performance set (main.csv).")
    parser.add_argument("--cases-file", default=str(CASES_PATH))
    parser.add_argument("--variants-pred-file", help="Predictions on fairness_variants.csv.")
    parser.add_argument("--variants-cases-file", default=str(FAIRNESS_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not args.pred_file and not args.variants_pred_file:
        parser.error("give --pred-file and/or --variants-pred-file")

    metrics = {}
    if args.pred_file:
        cases = {row_key(r): r for r in read_cases(resolve(args.cases_file))}
        preds = read_cases(resolve(args.pred_file))
        metrics["model"] = models_in(preds)
        metrics["perf_set"] = fairness_from_stats(group_stats(cases, preds))
        metrics["pred_file"] = resolve(args.pred_file).name
    if args.variants_pred_file:
        vcases = {row_key(r): r for r in read_cases(resolve(args.variants_cases_file))}
        vpreds = read_cases(resolve(args.variants_pred_file))
        metrics.setdefault("model", models_in(vpreds))
        metrics["variants_set"] = fairness_from_stats(group_stats(vcases, vpreds))
        metrics["counterfactual_consistency"] = counterfactual_consistency(vpreds)
        metrics["variants_pred_file"] = resolve(args.variants_pred_file).name

    print(json.dumps(metrics, indent=2))
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
