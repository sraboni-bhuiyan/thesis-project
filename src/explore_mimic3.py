import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis\mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2")

print("Loading files...")
triage_df = pd.read_csv(BASE / "triage_level.csv")
demo_df = pd.read_csv(BASE / "patient_demographics.csv")
vitals_df = pd.read_csv(BASE / "vital_signs.csv")
clin_path = BASE / "clinical_data.csv" / "clinical_data.csv"
clinical_df = pd.read_csv(clin_path)
spec_path = BASE / "specialty_referral_clinician_approved.csv"
spec_df = pd.read_csv(spec_path)

print("Triage shape:", triage_df.shape)
print("Demo shape:", demo_df.shape)
print("Vitals shape:", vitals_df.shape)
print("Clinical shape:", clinical_df.shape)
print("Spec shape:", spec_df.shape)

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
demo_df['race'] = demo_parsed.apply(lambda x: x.get('race'))

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

# Merge
merged = triage_df.merge(demo_df[['stay_id','gender','age']], on='stay_id', how='left')
print("After merge demo cols:", merged.columns.tolist())
merged = merged.merge(vitals_df[['stay_id','initial_vitals']], on='stay_id', how='left')
print("After merge vitals cols:", merged.columns.tolist())
merged = merged.merge(spec_df[['stay_id','spec_primary']], on='stay_id', how='left')
print("After merge spec cols:", merged.columns.tolist())
merged = merged.merge(clinical_df[['stay_id','HPI']], on='stay_id', how='left', suffixes=('','_clin'))
print("After merge clinical cols:", merged.columns.tolist())
print("Sample rows:")
print(merged.head())

# Map triage to color
triage_to_color = {1:'red',2:'orange',3:'yellow',4:'green',5:'blue'}
merged['urgency_color'] = merged['triage'].map(triage_to_color)

# Final columns for extraction script:
out = merged[['stay_id','HPI','initial_vitals','age','gender','urgency_color','spec_primary']].copy()
out = out.rename(columns={
    'stay_id':'case_id',
    'HPI':'symptoms',
    'initial_vitals':'vitals',
    'urgency_color':'reference_urgency',
    'spec_primary':'reference_specialty'
})
print("\nFinal sample:")
print(out.head())
print("\nMissing values:")
print(out.isnull().sum())

# Save a small sample for testing
out.head(20).to_csv(r"C:\Users\srabo\Desktop\masters-thesis\thesis-project\data\cases\mimic_sample.csv", index=False)
print("\nSaved sample to data/cases/mimic_sample.csv")