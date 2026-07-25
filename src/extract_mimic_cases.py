"""
Extract a triage evaluation dataset from MIMIC-IV-Ext CDS tables
and write it to data/cases/main.csv in the thesis-project repo.

You MUST ensure the MIMIC-IV-Ext dataset is downloaded and pointing to the correct path.
"""

import argparse
from pathlib import Path
import pandas as pd


# ---- 1. Configure local paths ----

# Repo root = parent of src/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output inside the repo (local-only, ignored by Git)
OUTPUT_FILE = REPO_ROOT / "data" / "cases" / "main.csv"

# Raw PhysioNet dataset location (on your machine, NOT in Git)
# This should match what's in config.py
MIMIC_BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis") / \
    "mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2"


# ---- 2. Define the target schema for main.csv ----

TARGET_COLUMNS = [
    "case_id",
    "symptoms",          # Expected by run_baseline.py and run_rag.py
    "chief_complaint",   # For compatibility
    "hpi",               # For compatibility
    "vitals",
    "age",
    "gender",
    "ground_truth_urgency",
    "ground_truth_specialty",
]


def parse_demo(info):
    """Parse patient_info string like 'Gender: Female, Race: WHITE, Age: 80'"""
    if isinstance(info, str):
        parts = [p.strip() for p in info.split(',')]
        d = {}
        for p in parts:
            if ':' in p:
                k, v = p.split(':', 1)
                d[k.strip().lower()] = v.strip()
        return d
    return {}


def parse_spec(s):
    """Parse specialty string like \"['Urology']\" or \"['General Surgery']\" """
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # Take first specialty if multiple
        parts = [p.strip().strip("'\"") for p in s.split(',')]
        return parts[0] if parts else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract triage cases from MIMIC-IV-Ext CDS tables "
                    "into data/cases/main.csv for the thesis project."
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=250,
        help="Maximum number of cases to keep (for thesis-scale evaluation).",
    )
    parser.add_argument(
        "--urgency-map",
        type=str,
        default="1:red,2:orange,3:yellow,4:green,5:blue",
        help="Mapping from triage numeric to urgency color (default: 1:red,2:orange,3:yellow,4:green,5:blue)",
    )
    args = parser.parse_args()

    print("Loading MIMIC-IV-Ext tables from:", MIMIC_BASE)

    # Load key tables
    print("  Loading triage_level.csv...")
    triage_df = pd.read_csv(MIMIC_BASE / "triage_level.csv")

    print("  Loading patient_demographics.csv...")
    demo_df = pd.read_csv(MIMIC_BASE / "patient_demographics.csv")

    print("  Loading vital_signs.csv...")
    vitals_df = pd.read_csv(MIMIC_BASE / "vital_signs.csv")

    print("  Loading specialty_referral_clinician_approved.csv...")
    spec_df = pd.read_csv(MIMIC_BASE / "specialty_referral_clinician_approved.csv")

    # Parse demographics
    print("  Parsing demographics...")
    demo_parsed = demo_df['patient_info'].apply(parse_demo)
    demo_df['gender'] = demo_parsed.apply(lambda x: x.get('gender'))
    demo_df['age'] = demo_parsed.apply(lambda x: x.get('age'))
    demo_clean = demo_df[['stay_id', 'gender', 'age']].copy()

    # Parse specialty
    print("  Parsing specialty referrals...")
    spec_df['spec_primary'] = spec_df['specialty clinician approved'].apply(parse_spec)
    spec_clean = spec_df[['stay_id', 'spec_primary']].copy()

    # Prepare triage data (use HPI as symptoms)
    print("  Preparing triage data...")
    triage_clean = triage_df[['stay_id', 'HPI', 'triage']].copy()

    # Prepare vitals
    vitals_clean = vitals_df[['stay_id', 'initial_vitals']].copy()

    # Merge all tables
    print("  Merging tables...")
    merged = triage_clean.merge(demo_clean, on='stay_id', how='left')
    merged = merged.merge(vitals_clean, on='stay_id', how='left')
    merged = merged.merge(spec_clean, on='stay_id', how='left')

    # Map triage to urgency color
    print("  Mapping triage levels to urgency colors...")
    # Parse urgency map
    urgency_map = {}
    for pair in args.urgency_map.split(','):
        if ':' in pair:
            k, v = pair.split(':', 1)
            urgency_map[int(k.strip())] = v.strip()

    merged['urgency_color'] = merged['triage'].map(urgency_map)

    # Build final dataframe with required columns
    print("  Building final dataset...")
    df_out = pd.DataFrame()
    df_out['case_id'] = merged['stay_id']
    # Symptoms column (what the ML models will see as patient symptoms)
    df_out['symptoms'] = merged['HPI']  # Using HPI as the symptom description
    # Keep these for compatibility/reference
    df_out['chief_complaint'] = merged['HPI']
    df_out['hpi'] = merged['HPI']
    df_out['vitals'] = merged['initial_vitals']
    df_out['age'] = merged['age']
    df_out['gender'] = merged['gender']
    df_out['ground_truth_urgency'] = merged['urgency_color']
    df_out['ground_truth_specialty'] = merged['spec_primary']

    # Optional: drop rows with missing key fields
    initial_count = len(df_out)
    df_out = df_out.dropna(subset=["ground_truth_urgency", "ground_truth_specialty"])
    dropped = initial_count - len(df_out)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing urgency or specialty")

    # Limit to thesis-scale number of cases
    if args.max_cases is not None and len(df_out) > args.max_cases:
        print(f"  Limiting to {args.max_cases} cases (from {len(df_out)} available)")
        df_out = df_out.head(args.max_cases)

    # Reorder columns to match TARGET_COLUMNS
    df_out = df_out[TARGET_COLUMNS]

    # Ensure output folder exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save
    print(f"Writing {len(df_out)} cases to: {OUTPUT_FILE}")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("Done. You can now point run_baseline.py and run_rag.py to main.csv")


if __name__ == "__main__":
    main()