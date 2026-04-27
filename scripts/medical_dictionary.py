# =========================
# Week4 medical resources
# =========================

MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"],
    "myocardial infarction": ["mi", "heart attack"],
    "heart attack": ["mi", "myocardial infarction"],
    "cvd": ["cardiovascular disease", "heart disease"],
    "cardiovascular disease": ["cvd", "heart disease"],
    "heart disease": ["cvd", "cardiovascular disease"],
    "htn": ["hypertension", "high blood pressure"],
    "hypertension": ["htn", "high blood pressure"],
    "dm": ["diabetes mellitus", "diabetes"],
    "diabetes": ["dm", "diabetes mellitus"],
    "diabetes mellitus": ["dm", "diabetes"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "type 2 diabetes": ["t2dm", "type 2 diabetes mellitus"],
    "cad": ["coronary artery disease"],
    "coronary artery disease": ["cad"],
    "af": ["atrial fibrillation"],
    "atrial fibrillation": ["af"],
    "chf": ["congestive heart failure", "heart failure"],
    "heart failure": ["chf", "congestive heart failure"],
}

MEDICAL_PATTERNS = {
    "drug": (
        r"\b(aspirin|metformin|atorvastatin|warfarin|insulin|statin|"
        r"clopidogrel|heparin|lisinopril|amlodipine)\b"
    ),
    "disease": (
        r"\b(mi|myocardial infarction|heart attack|cvd|"
        r"cardiovascular disease|heart disease|htn|hypertension|"
        r"high blood pressure|dm|diabetes|diabetes mellitus|t2dm|"
        r"type 2 diabetes|type 2 diabetes mellitus|cad|"
        r"coronary artery disease|af|atrial fibrillation|chf|"
        r"congestive heart failure|heart failure|stroke)\b"
    ),
    "study_type": (
        r"\b(rct|randomized controlled trial|cohort study|"
        r"case control|meta analysis|systematic review)\b"
    ),
    "population": (
        r"\b(elderly|older adults|adults|children|pediatric|"
        r"pregnant women|women|men|patients)\b"
    ),
}
