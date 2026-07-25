import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis\mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2")

print("Loading files...")
triage_df = pd.read_csv(BASE / "triage_level.csv")
demo_df = pd.read_csv(BASE / "patient_demographics.csv")
vitals_df = pd.read_csv(BASE / "vital_signs.csv")
# clinical data is inside a folder
clin_path = BASE / "clinical_data.csv" / "clinical_data.csv"
clinical_df = pd.read_csv(clin_path)
spec_path = BASE / "specialty_referral_clinician_approved.csv"
spec_df = pd.read_csv(spec_path)

print("Triage shape:", triage_df.shape)
print("Demo shape:", demo_df.shape)
print("Vitals shape:", vitals_df.shape)
print("Clinical shape:", clinical_df.shape)
print("Spec shape:", spec_df.shape)

print("\nTriage columns:", triage_df.columns.tolist())
print("Demo columns:", demo_df.columns.tolist())
print("Vitals columns:", vitals_df.columns.tolist())
print("Clinical columns:", clinical_df.columns.tolist())
print("Spec columns:", spec_df.columns.tolist())

print("\n--- Triage sample ---")
print(triage_df[['stay_id','triage']].head())

print("\n--- Demo sample ---")
print(demo_df[['stay_id','patient_info']].head())

print("\n--- Vitals sample ---")
print(vitals_df[['stay_id','initial_vitals']].head())

print("\n--- Clinical sample HPI ---")
print(clinical_df[['stay_id','HPI']].head())

print("\n--- Spec sample ---")
print(spec_df[['stay_id','specialty clinician approved']].head())

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
print("\nParsed demo sample:")
print(demo_df[['stay_id','gender','age','race']].head())

# Parse specialty
def parse_spec(s):
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # split by commas, but careful about commas inside? assume simple
        parts = [p.strip().strip("'\"") for p in s.split(',')]
        return parts[0] if parts else None
    return None

spec_df['spec_primary'] = spec_df['specialty clinician approved'].apply(parse_spec)
print("\nParsed spec sample:")
print(spec_df[['stay_id','specialty clinician approved','spec_primary']].head())

# Merge
merged = triage_df.merge(demo_df[['stay_id','gender','age']], on='stay_id', how='left')
merged = merged.merge(vitals_df[['stay_id','initial_vitals']], on='stay_id', how='left')
merged = merged.merge(spec_df[['stay_id','spec_primary']], on='stay_id', how='left')
# No need to merge chiefcomplaint; we will use HPI as symptoms
merged = merged.merge(clinical_df[['stay_id','HPI']], on='stay_id', how='left', suffixes=('','_clin'))

print("\nMerged shape:", merged.shape)
print(merged.head())

# Map triage to color: assume 1=resuscitation (red), 2=emergent (orange), 3=urgent (yellow), 4=less urgent (green), 5=non-urgent (blue/white)
triage_to_color = {1:'red',2:'orange',3:'yellow',4:'green',5:'blue'}
merged['urgency_color'] = merged['triage'].map(triage_to_color)
print("\nUrgency distribution:")
print(merged['triage'].value_counts().sort_index())
print("\nMapped urgency colors sample:")
print(merged[['triage','urgency_color']].dropna().head(10))

# Final columns for extraction script:
out = merged[['stay_id','HPI','initial_vitals','age','gender','urgency_color','spec_primary']].copy()
out = out.rename(columns={
    'stay_id':'case_id',
    'HPI':'symptoms',  # using HPI as symptoms
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