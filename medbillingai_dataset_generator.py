import pandas as pd
import numpy as np
import random
from faker import Faker
fake = Faker()

# -------------------------
# Parameters
# -------------------------
NUM_CLAIMS = 5000

# -------------------------
# ICD-10 & CPT sample codes
# -------------------------
ICD10_CODES = ['I10', 'E11', 'J45', 'M54', 'K21', 'N39', 'F32', 'G43', 'L40', 'E66']
CPT_CODES = ['99213', '99214', '99203', '99204', '93000', '80053', '36415', '71020', '84443', '81002']
PROVIDER_TYPES = ['Hospital', 'Clinic', 'Telemedicine']
PROVIDER_SPECIALTY = ['Cardiology', 'Orthopedic', 'Radiology', 'General', 'Pediatrics', 'Oncology']
INSURANCE_TYPES = ['Private', 'Medicare', 'Medicaid']
STATES = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']

# -------------------------
# Generate claims data
# -------------------------
data = []

for i in range(NUM_CLAIMS):
    patient_age = random.randint(0, 90)
    gender = random.choice(['M', 'F'])
    icd_code = random.choice(ICD10_CODES)
    cpt_code = random.choice(CPT_CODES)
    provider_type = random.choice(PROVIDER_TYPES)
    provider_specialty = random.choice(PROVIDER_SPECIALTY)
    insurance_type = random.choice(INSURANCE_TYPES)
    state = random.choice(STATES)
    claim_amount = round(random.uniform(100, 10000), 2)
    prior_auth = random.choice([0, 1])
    num_procedures = random.randint(1, 5)
    num_diagnoses = random.randint(1, 3)
    chronic_flag = random.choice([0, 1])
    high_risk_flag = random.choice([0, 1])
    telehealth = 1 if provider_type == 'Telemedicine' else 0
    in_network = random.choice([0, 1])
    submission_date = fake.date_between(start_date='-2y', end_date='today')
    processing_days = random.randint(1, 30)
    resubmission_count = random.randint(0, 3)

    # Simple denial probability logic
    denial_prob = 0.05 + 0.1*prior_auth + 0.1*telehealth + 0.05*(high_risk_flag) + 0.05*(claim_amount>5000)
    denied = np.random.binomial(1, min(denial_prob, 0.9))
    denial_reason = None
    if denied:
        denial_reason = random.choice(['Invalid CPT', 'Missing Prior Auth', 'Out-of-Network', 'Incomplete Claim'])

    data.append({
        'Claim_ID': f'C{i+1:05d}',
        'Patient_Age': patient_age,
        'Gender': gender,
        'ICD10_Code': icd_code,
        'CPT_Code': cpt_code,
        'Provider_Type': provider_type,
        'Provider_Specialty': provider_specialty,
        'Insurance_Type': insurance_type,
        'State': state,
        'Claim_Amount': claim_amount,
        'Prior_Authorization': prior_auth,
        'Num_Procedures': num_procedures,
        'Num_Diagnoses': num_diagnoses,
        'Chronic_Flag': chronic_flag,
        'High_Risk_Flag': high_risk_flag,
        'Telehealth': telehealth,
        'In_Network': in_network,
        'Submission_Date': submission_date,
        'Processing_Days': processing_days,
        'Resubmission_Count': resubmission_count,
        'Denied': denied,
        'Denial_Reason': denial_reason
    })

# -------------------------
# Save dataset
# -------------------------
df = pd.DataFrame(data)
df.to_csv('claims_data.csv', index=False)
print('Synthetic FHIR-like claims dataset generated: claims_data.csv')
