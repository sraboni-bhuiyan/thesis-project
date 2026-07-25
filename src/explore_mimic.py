import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\srabo\Desktop\masters-thesis\mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2")

print("Loading files...")
# Load key files
triage_df = pd.read_csv(BASE / "triage_level.csv")
demo_df = pd.read_csv(BASE / "patient_demographics.csv")
vitals_df = pd.read_csv(BASE / "vital_signs.csv")
# clinical data is a directory; need to find the actual csv
import os
clinical_dir = BASE / "clinical_data.csv"
clinical_files = list(clinical_dir.glob("*.csv"))
if clinical_files:
    clinical_path = clinical_files[0]
else:
    clinical_path = clinical_dir / "clinical_data.csv"
print("Clinical file:", clinical_path)
clinical_df = pd.read_csv(clinical_path)

# specialty referral (clinician approved)
spec_df = pd.read_csv(BASE / "specialty_referral_clinician_approved.csv")

print("Triage shape:", triage_df.shape)
print("Demo shape:", demo_df.shape)
print("Vitals shape:", vitals_df.shape)
print("Clinical shape:", clinical_df.shape)
print("Spec shape:", spec_df.head())

# Show columns
print("\nTriage columns:", triage_df.columns.tolist())
print("Demo columns:", demo_df.columns.tolist())
print("Vitals columns:", vitals_df.columns.tolist())
print("Clinical columns:", clinical_df.columns.tolist())
print("Spec columns:", spec_df.columns.tolist())

# Sample rows
print("\n--- Triage sample ---")
print(triage_df.head())
print("\n--- Demo sample ---")
print(demo_df.head())
print("\n--- Vitals sample ---")
print(vitals_df.head())
print("\n--- Clinical sample HPI ---")
print(clinical_df[['stay_id', 'HPI']].head())
print("\n--- Spec sample ---")
print(spec_df[['stay_id', 'specialty']].head())

# Parse demo column
def parse_demo(info):
    # info like '"Gender: Female, Race: WHITE, Age: 80"'
    # remove quotes if present
    if isinstance(info, str):
        # strip outer quotes
        if info.startswith('"') and info.endswith('"'):
            info = info[1:-1]
    parts = info.split(', ')
    gender = None
    age = None
    race = None
    for p in parts:
        if p.startswith('Gender:'):
            gender = p.split(':',1)[1].strip()
        elif p.startswith('Age:'):
            try:
                age = int(p.split(':',1)[1].strip())
            except:
                age = None
        elif p.startswith('Race:'):
            race = p.split(':',1)[1].strip()
    return gender, age, race

demo_df[['gender','age','race']] = demo_df['patient_info'].apply(lambda x: pd.Series(parse_demo(x)))
print("\nParsed demo sample:")
print(demo_df[['stay_id','gender','age','race']].head())

# Merge triage + demo + vitals + specialty
merged = triage_df.merge(demo_df[['stay_id','gender','age']], on='stay_id', how='left')
merged = merged.merge(vitals_df[['stay_id','initial_vitals']], on='stay_id', how='left')
merged = merged.merge(spec_df[['stay_id','specialty']], on='stay_id', how='left')
merged = merged.merge(clinical_df[['stay_id','HPI','chiefcomplaint']], on='stay_id', how='left')

print("\nMerged shape:", merged.shape)
print(merged.head())

# Decide what to map:
# case_id = stay_id
# symptoms = maybe HPI or chiefcomplaint? We'll choose HPI as it's more detailed
# vitals = initial_vitals
# age = age
# gender = gender
# ground_truth_urgency = triage (need mapping to color)
# ground_truth_specialty = specialty (need to extract first from list-like string)

# Parse specialty: it's like "['Neurology', 'Pulmonology']"
def parse_spec(s):
    if isinstance(s, str):
        # Remove brackets and quotes
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # split by ',' and strip quotes
        parts = [p.strip().strip("'\"") for p in s.split(',')]
        return parts[0] if parts else None
    return None

merged['spec_primary'] = merged['specialty'].apply(parse_spec)

# Map triage to color: assuming 1=red,2=orange,3=yellow,4=green,5=blue
triage_to_color = {1:'red',2:'orange',3:'yellow',4:'green',5:'blue'}
merged['urgency_color'] = merged['triage'].map(triage_to_color)
print("\nUrgency mapping check:")
print(merged[['triage','urgency_color']].dropna().head(10))

# Final columns for extraction script:
out = merged[['stay_id','HPI','initial_vitals','age','gender','urgency_color','spec_primary']].copy()
out = out.rename(columns={
    'stay_id':'case_id',
    'HPI':'symptoms',  # we treat HPI as symptoms
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