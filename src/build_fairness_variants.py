"""
Generate fairness‑variant test set by creating age/gender counter‑factuals.

For each base case sampled from `data/cases/main.csv` we create four variants:
    (young, male), (young, female), (old, male), (old, female)
where:
    young  -> age = YOUNG_AGE (default 30)
    old    -> age = OLD_AGE (default 70)

The clinical text (symptoms, vitals) is left unchanged except for explicit
replacements of the original age and gender tokens so that the case text
matches the new age/gender.

Output CSV columns:
    case_id, variant_id, symptoms, vitals, age, gender,
    ground_truth_urgency, ground_truth_specialty
"""
import argparse
import csv
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CSV = PROJECT_ROOT / "data" / "cases" / "main.csv"
OUT_CSV = PROJECT_ROOT / "data" / "cases" / "fairness_variants.csv"

# Default ages for young/old variants (can be overridden via CLI)
DEFAULT_YOUNG_AGE = 30
DEFAULT_OLD_AGE = 70
SEED = 42  # for reproducible sampling


def token_replace(text: str, old: str, new: str) -> str:
    """
    Replace whole-word occurrences of `old` with `new` in `text`.
    Uses word-boundary regex to avoid partial matches (e.g., replace '5' in '15').
    If `old` is empty, returns text unchanged.
    """
    if not old:
        return text
    # Escape for regex and use word boundaries
    pattern = r'\b' + re.escape(old) + r'\b'
    return re.sub(pattern, new, text, flags=re.IGNORECASE)


def process_row(row: dict, young_age: int, old_age: int):
    """
    Generate four variants from a single base case row.
    Returns a list of dicts ready for CSV writing.
    """
    case_id = row["case_id"]
    symptoms = row.get("symptoms", "")
    vitals = row.get("vitals", "")
    try:
        orig_age = int(row["age"]) if row["age"] else None
    except ValueError:
        orig_age = None
    orig_gender = row.get("gender", "")

    # If age missing, we cannot create meaningful young/old variants – skip
    if orig_age is None:
        return []

    variants = []
    for age_label, age_val in [("young", young_age), ("old", old_age)]:
        for gender_label, gender_val in [("male", "Male"), ("female", "Female")]:
            variant_id = f"{case_id}_{age_label}_{gender_label}"

            # Replace age/gender tokens in symptoms and vitals
            new_symptoms = token_replace(symptoms, str(orig_age), str(age_val)) if orig_age is not None else symptoms
            new_symptoms = token_replace(new_symptoms, orig_gender, gender_val) if orig_gender else new_symptoms

            new_vitals = token_replace(vitals, str(orig_age), str(age_val)) if orig_age is not None else vitals
            new_vitals = token_replace(new_vitals, orig_gender, gender_val) if orig_gender else new_vitals

            variant = {
                "case_id": case_id,
                "variant_id": variant_id,
                "symptoms": new_symptoms,
                "vitals": new_vitals,
                "age": str(age_val),
                "gender": gender_val,
                "ground_truth_urgency": row["ground_truth_urgency"],
                "ground_truth_specialty": row["ground_truth_specialty"],
            }
            variants.append(variant)
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Generate age/gender fairness variants from main.csv."
    )
    parser.add_argument(
        "--num-base",
        type=int,
        default=70,
        help="Number of base cases to sample from main.csv (default: 70).",
    )
    parser.add_argument(
        "--young-age",
        type=int,
        default=DEFAULT_YOUNG_AGE,
        help=f"Age to use for 'young' variants (default: {DEFAULT_YOUNG_AGE}).",
    )
    parser.add_argument(
        "--old-age",
        type=int,
        default=DEFAULT_OLD_AGE,
        help=f"Age to use for 'old' variants (default: {DEFAULT_OLD_AGE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducible sampling (default: {SEED}).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all cases
    with open(MAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if args.num_base > len(all_rows):
        print(f"Warning: requested {args.num_base} base cases but only {len(all_rows)} available. Using all.")
        args.num_base = len(all_rows)

    base_rows = random.sample(all_rows, args.num_base)

    all_variants = []
    for row in base_rows:
        all_variants.extend(process_row(row, args.young_age, args.old_age))

    # Write output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "variant_id",
        "symptoms",
        "vitals",
        "age",
        "gender",
        "ground_truth_urgency",
        "ground_truth_specialty",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_variants)

    print(f"Generated {len(all_variants)} variants from {len(base_rows)} base cases.")
    print(f"Output written to: {OUT_CSV}")


if __name__ == "__main__":
    main()