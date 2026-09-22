"""
Candidate cases for the qualitative error analysis -> results/error_cases.csv.

Two row types: rag_flip (RAG urgency differs from baseline, with retrieved chunk ids and both
rationales) and demographic_flip (urgency differs across a case's 4 demographic variants).
Pick 5-10 illustrative rows by hand.

Usage:
  python src/error_cases.py --config rag
"""
import argparse
import csv
import json
from collections import defaultdict

from common import normalize_urgency, read_cases, row_key
from config import CASES_PATH, RESULTS_PATH

FIELDS = ["type", "case_id", "ground_truth_urgency", "ground_truth_specialty", "age", "gender",
          "baseline_urgency", "rag_urgency", "variant_urgencies", "retrieved_chunk_ids",
          "baseline_rationale", "rag_rationale", "symptoms_excerpt"]


def main():
    parser = argparse.ArgumentParser(description="Export candidate cases for qualitative error analysis.")
    parser.add_argument("--config", default="rag", choices=["baseline", "rag", "rag_masked"],
                        help="Config used for the demographic-flip section.")
    args = parser.parse_args()

    cases = {row_key(r): r for r in read_cases(CASES_PATH)}
    out_rows = []

    base_file, rag_file = RESULTS_PATH / "baseline_predictions.csv", RESULTS_PATH / "rag_predictions.csv"
    if base_file.exists() and rag_file.exists():
        base = {row_key(r): r for r in read_cases(base_file)}
        logs = {}
        log_file = RESULTS_PATH / "retrieval_logs.jsonl"
        if log_file.exists():
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    logs[e["case_id"]] = e
        for key, r in {row_key(r): r for r in read_cases(rag_file)}.items():
            b = base.get(key)
            c = cases.get(key)
            if not b or not c:
                continue
            bu, ru = normalize_urgency(b["pred_urgency"]), normalize_urgency(r["pred_urgency"])
            if bu != ru or ru == "INVALID":
                out_rows.append({
                    "type": "rag_flip", "case_id": key, "ground_truth_urgency": c["ground_truth_urgency"],
                    "ground_truth_specialty": c["ground_truth_specialty"], "age": c["age"], "gender": c["gender"],
                    "baseline_urgency": bu, "rag_urgency": ru,
                    "retrieved_chunk_ids": "|".join(logs.get(key, {}).get("retrieved_chunk_ids", [])),
                    "baseline_rationale": b["pred_rationale"], "rag_rationale": r["pred_rationale"],
                    "symptoms_excerpt": (c["symptoms"] or "")[:400],
                })

    var_file = RESULTS_PATH / f"{args.config}_variants_predictions.csv"
    if var_file.exists():
        groups = defaultdict(list)
        for r in read_cases(var_file):
            groups[r["case_id"]].append(r)
        for cid, rows in groups.items():
            urg = {r["variant_id"].split("_", 1)[1]: normalize_urgency(r["pred_urgency"]) for r in rows}
            if len(set(urg.values())) > 1:
                c = cases.get(cid, {})
                out_rows.append({
                    "type": f"demographic_flip_{args.config}", "case_id": cid,
                    "ground_truth_urgency": c.get("ground_truth_urgency", ""),
                    "ground_truth_specialty": c.get("ground_truth_specialty", ""),
                    "variant_urgencies": "; ".join(f"{k}={v}" for k, v in sorted(urg.items())),
                    "rag_rationale": " || ".join(f"{r['variant_id'].split('_', 1)[1]}: {r['pred_rationale']}" for r in rows),
                    "symptoms_excerpt": (c.get("symptoms") or "")[:400],
                })

    out = RESULTS_PATH / "error_cases.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} candidate rows to {out}")


if __name__ == "__main__":
    main()
