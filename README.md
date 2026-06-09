# Fairness-Aware RAG Triage Assistant

Short description

This repository contains code, data, and results for a master's thesis comparing a prompt-only baseline to a Retrieval-Augmented Generation (RAG) triage assistant, with a focus on fairness across age and gender.

Structure
- data/: guidelines_raw, guidelines_clean, indexes, cases
- src/: scripts and prompts
- notebooks/: exploratory and analysis notebooks
- results/: predictions, figures, and tables
- reports/: thesis files

Usage
Populate data/guidelines_raw with source PDFs, then run scripts in src/ and notebooks/.


Labeling guide
1. Urgency labels
Use a 5-level urgency scale:

- Red = immediate life-threatening or very unstable.

- Orange = very urgent, needs fast assessment.

- Yellow = urgent, but not immediately dangerous.

- Green = standard care, can wait.

- Blue = non-urgent, minor issue.

2. How to choose urgency
Base the urgency on:

severity of symptoms,

danger signs,

vital signs,

how fast the condition could get worse,

whether the patient may need immediate intervention.

3. Specialty labels
Use one main specialty for each case. Keep the list small and consistent.

Suggested specialty list:

emergency medicine

cardiology

neurology

surgery

orthopedics

ENT

ophthalmology

gastroenterology

infectious disease

psychiatry

obstetrics

dermatology

general practice

4. How to choose specialty
Choose the specialty that would most likely see the patient first.

Rules:

Do not choose a final diagnosis.

Do not choose a treatment department unless it is clearly the first relevant service.

If more than one specialty fits, pick the most immediately relevant one.

If the case is vague, use the broadest reasonable specialty, often emergency medicine or general practice.

5. Consistency rules
To keep labels consistent:

Use the same urgency meaning for every case.

Do not change labels based on wording style.

Do not infer too much from missing details.

Use vitals only when they clearly support higher urgency.

Keep specialty selection conservative and simple.

6. Example decisions
Chest pain with shortness of breath → likely orange or red, specialty: cardiology or emergency medicine.

Mild sore throat and runny nose → likely green, specialty: general practice.

Sudden one-sided weakness and slurred speech → likely red, specialty: neurology or emergency medicine.

Sprained ankle, able to walk → likely blue or green, specialty: orthopedics.

7. Final rule for the pilot
For your first 20 cases, keep the guide strict and simple. The goal is not perfect clinical accuracy yet, but consistent labeling so you can test the baseline system reliably.


## Guidance documents

Collect and save the first 5 public triage guidance documents in `data/guidelines_raw/`:

- Charité Manchester Triage System page.
- NHS England guidance for emergency departments: initial assessment.
- NHS 111 guidance page.
- One emergency department triage FAQ or overview source.
- One additional Manchester Triage System overview source.

These sources are used to define urgency labels, triage logic, and basic routing rules for the project.