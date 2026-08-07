import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "data" / "cases" / "main.csv"
# Default prediction file – can be overridden with --pred-file
DEFAULT_BASELINE_FILE = PROJECT_ROOT / "results" / "baseline_predictions.csv"

URGENCY_ORDER = ["blue", "green", "yellow", "orange", "red"]
URGENCY_INDEX = {u: i for i, u in enumerate(URGENCY_ORDER)}


def load_cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def load_predictions(pred_file: Path):
    with open(pred_file, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_urgency(pred, ref):
    """Return 'correct', 'over', or 'under'."""
    if pred not in URGENCY_INDEX or ref not in URGENCY_INDEX:
        return "invalid"
    p = URGENCY_INDEX[pred]
    r = URGENCY_INDEX[ref]
    if p == r:
        return "correct"
    elif p > r:
        return "over"      # more severe than reference
    else:
        return "under"     # less severe than reference


def main(pred_file: Path):
    cases = load_cases()
    preds = load_predictions(pred_file)

    total = 0
    urg_correct = urg_over = urg_under = urg_invalid = 0
    spec_correct = 0

    for row in preds:
        cid = row["case_id"]
        ref = cases.get(cid)
        if not ref:
            continue

        total += 1

        # urgency evaluation
        pred_u = row["pred_urgency"].strip().lower()
        ref_u = ref["ground_truth_urgency"].strip().lower()
        cat = compare_urgency(pred_u, ref_u)
        if cat == "correct":
            urg_correct += 1
        elif cat == "over":
            urg_over += 1
        elif cat == "under":
            urg_under += 1
        else:
            urg_invalid += 1

        # specialty evaluation (simple exact match for now)
        pred_s = row["pred_specialty"].strip().lower()
        ref_s = ref["ground_truth_specialty"].strip().lower()
        if pred_s == ref_s:
            spec_correct += 1

    print(f"Evaluated predictions from: {pred_file}")
    print(f"Total cases evaluated: {total}")
    if total > 0:
        print(f"Urgency accuracy: {urg_correct/total:.2%}")
        print(f"Specialty accuracy: {spec_correct/total:.2%}")
        print(f"Over-triage: {urg_over} cases")
        print(f"Under-triage: {urg_under} cases")
        print(f"Invalid urgency labels: {urg_invalid} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate triage predictions (baseline, RAG, or any custom file)."
    )
    parser.add_argument(
        "-p",
        "--pred-file",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
        help=f"Path to prediction CSV to evaluate. Default: {DEFAULT_BASELINE_FILE}",
    )
    args = parser.parse_args()
    main(args.pred_file)