"""
Build the frozen evaluation set (data/cases/main.csv) from MIMIC-IV-Ext CDS.

Rows    = clinician-approved specialty referrals (331 stays).
Urgency = triage acuity (1-5) joined from triage_level.csv on stay_id, mapped to colours.
"""
import argparse
import ast

import pandas as pd

from config import CASES_PATH, PHYSIONET_BASE, URGENCY_MAP

OUTPUT_COLUMNS = ["case_id", "symptoms", "vitals", "age", "gender",
                  "ground_truth_urgency", "ground_truth_specialty", "ground_truth_specialty_all"]


def parse_demo(info):
    """'Gender: Female, Race: WHITE, Age: 80' -> {'gender': 'Female', 'race': 'WHITE', 'age': '80'}"""
    d = {}
    if isinstance(info, str):
        for part in info.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                d[k.strip().lower()] = v.strip()
    return d


def parse_specialties(s):
    """"['Gastroenterology', 'Cardiology']" -> ['Gastroenterology', 'Cardiology'] (order kept, de-duplicated)."""
    if not isinstance(s, str):
        return []
    try:
        items = ast.literal_eval(s)
        items = items if isinstance(items, list) else [items]
    except (ValueError, SyntaxError):
        items = s.strip("[]").split(",")
    out = []
    for it in items:
        it = str(it).strip().strip("'\"")
        if it and it not in out:
            out.append(it)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract clinician-approved triage cases into data/cases/main.csv.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Optional cap (e.g. 250). Default: all clinician-approved cases.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when sampling --max-cases rows.")
    args = parser.parse_args()

    print("Loading MIMIC-IV-Ext tables from:", PHYSIONET_BASE)
    approved = pd.read_csv(PHYSIONET_BASE / "specialty_referral_clinician_approved.csv")
    triage = pd.read_csv(PHYSIONET_BASE / "triage_level.csv", usecols=["stay_id", "triage"])

    approved = approved.drop_duplicates(subset="stay_id")
    df = approved.merge(triage.drop_duplicates(subset="stay_id"), on="stay_id", how="left")

    demo = df["patient_info"].apply(parse_demo)
    specs = df["specialty clinician approved"].apply(parse_specialties)

    out = pd.DataFrame({
        "case_id": df["stay_id"],
        "symptoms": df["HPI"],
        "vitals": df["initial_vitals"],
        "age": pd.to_numeric(demo.apply(lambda d: d.get("age")), errors="coerce").astype("Int64"),
        "gender": demo.apply(lambda d: d.get("gender")),
        "ground_truth_urgency": pd.to_numeric(df["triage"], errors="coerce").map(URGENCY_MAP),
        "ground_truth_specialty": specs.apply(lambda l: l[0] if l else None),
        "ground_truth_specialty_all": specs.apply(lambda l: "|".join(l)),
    })

    n0 = len(out)
    out = out.dropna(subset=["symptoms", "ground_truth_urgency", "ground_truth_specialty"])
    print(f"  {n0} clinician-approved stays, {n0 - len(out)} dropped for missing HPI/urgency/specialty")

    if args.max_cases is not None and len(out) > args.max_cases:
        out = out.sample(n=args.max_cases, random_state=args.seed)
        print(f"  Sampled {args.max_cases} cases (seed={args.seed})")

    out = out.sort_values("case_id")[OUTPUT_COLUMNS]
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CASES_PATH, index=False)
    print(f"Wrote {len(out)} cases to {CASES_PATH}")
    print("Urgency distribution:", out["ground_truth_urgency"].value_counts().to_dict())


if __name__ == "__main__":
    main()
