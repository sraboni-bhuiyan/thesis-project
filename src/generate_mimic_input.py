import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis\mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2")
OUT_DIR = Path(r"C:\Users\srabo\Desktop\masters-thesis\thesis-project\data\cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "mimic_raw.csv"

print("Loading files...")
triage_df = pd.read_csv(BASE / "triage_level.csv")
demo_df = pd.read_csv(BASE / "patient_demographics.csv")
vitals_df = pd.read_csv(BASE / "vital_signs.csv")
spec_df = pd.read_csv(BASE / "specialty_referral_clinician_approved.csv")

# Parse demo
def parse_demo(info):
    if isinstance(info, str):
        parts = [p.strip() for p in info.split(',')]
        d = {}
        for p in parts:
            if ':' in p:
                k,v = p.split(':',1)
                d[k.strip().lower()] = v.strip()
        return d
    return {}

demo_parsed = demo_df['patient_info'].apply(parse_demo)
demo_df['gender'] = demo_parsed.apply(lambda x: x.get('gender'))
demo_df['age'] = demo_parsed.apply(lambda x: x.get('age'))
demo_clean = demo_df[['stay_id','gender','age']].copy()

# Parse specialty
def parse_spec(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = [p.strip().strip("'\"") for p in s.split(',')]
        return parts[0] if parts else None
    return None

spec_df['spec_primary'] = spec_df['specialty clinician approved'].apply(parse_spec)
spec_clean = spec_df[['stay_id','spec_primary']].copy()

# Triage: keep HPI (we will use for both chief_complaint and hpi)
triage_clean = triage_df[['stay_id','HPI','triage']].copy()

# Vitals
vitals_clean = vitals_df[['stay_id','initial_vitals']].copy()

# Merge
merged = triage_clean.merge(demo_clean, on='stay_id', how='left')
merged = merged.merge(vitals_clean, on='stay_id', how='left')
merged = merged.merge(spec_clean, on='stay_id', how='left')

# Map triage to color (1-5 to red,orange,yellow,green,blue)
triage_to_color = {1:'red',2:'orange',3:'yellow',4:'green',5:'blue'}
merged['urgency_color'] = merged['triage'].map(triage_to_color)

# Build final dataframe with required columns for mapping
df_out = pd.DataFrame()
df_out['case_id'] = merged['stay_id']
df_out['chief_complaint'] = merged['HPI']  # using HPI as chief complaint
df_out['hpi'] = merged['HPI']              # same as chief complaint for simplicity
df_out['vitals'] = merged['initial_vitals']
df_out['age'] = merged['age']
df_out['gender'] = merged['gender']
df_out['ground_truth_urgency'] = merged['urgency_color']
df_out['ground_truth_specialty'] = merged['spec_primary']

print("Total rows:", len(df_out))
print("Missing values:")
print(df_out.isnull().sum())

# Save
df_out.to_csv(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")