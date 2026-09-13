"""
Generate the counterfactual fairness set from the frozen evaluation set.

For each of N base cases (default 70) four variants are created:
    (young, male), (young, female), (old, male), (old, female)
The age/gender fields are set, and gender/age cues inside the HPI text
(Mr./Ms., he/she, man/woman, 'M'/'F', 'NN y/o') are rewritten to match,
so the only difference between variants is the demographic.
"""
import argparse
import csv
import random

from common import read_cases
from config import CASES_PATH, FAIRNESS_PATH, OLD_VARIANT_AGE, YOUNG_VARIANT_AGE
from demographics import swap_demographics

FIELDS = ["case_id", "variant_id", "age_group", "symptoms", "vitals", "age", "gender",
          "ground_truth_urgency", "ground_truth_specialty", "ground_truth_specialty_all"]


def make_variants(row: dict, young_age: int, old_age: int):
    try:
        orig_age = int(float(row["age"]))
    except (TypeError, ValueError):
        orig_age = None
    variants = []
    for age_group, age in [("young", young_age), ("old", old_age)]:
        for gender in ["male", "female"]:
            variants.append({
                "case_id": row["case_id"],
                "variant_id": f"{row['case_id']}_{age_group}_{gender}",
                "age_group": age_group,
                "symptoms": swap_demographics(row.get("symptoms", ""), gender, age, orig_age),
                "vitals": row.get("vitals", ""),
                "age": str(age),
                "gender": gender.capitalize(),
                "ground_truth_urgency": row["ground_truth_urgency"],
                "ground_truth_specialty": row["ground_truth_specialty"],
                "ground_truth_specialty_all": row.get("ground_truth_specialty_all", ""),
            })
    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate age/gender counterfactual variants from main.csv.")
    parser.add_argument("--num-base", type=int, default=70)
    parser.add_argument("--young-age", type=int, default=YOUNG_VARIANT_AGE)
    parser.add_argument("--old-age", type=int, default=OLD_VARIANT_AGE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [r for r in read_cases(CASES_PATH) if (r.get("symptoms") or "").strip()]
    random.seed(args.seed)
    base = random.sample(rows, min(args.num_base, len(rows)))

    variants = [v for row in base for v in make_variants(row, args.young_age, args.old_age)]
    FAIRNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAIRNESS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(variants)
    print(f"Generated {len(variants)} variants from {len(base)} base cases -> {FAIRNESS_PATH}")


if __name__ == "__main__":
    main()
