# step4_build_features.py

import json
import csv
from pathlib import Path

IN_PATH  = "extracted_features_normalized.jsonl"
OUT_PATH = "llm_feature_matrix.csv"

# encoding maps

SEVERITY_MAP = {
    "none": 0, "normal": 0, "calm": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "unknown": -1,
}

ONSET_MAP = {
    "sudden": 1,
    "gradual": 0,
    "unknown": -1,
}

DURATION_MAP = {
    "hours":   0,
    "days":    1,
    "weeks":   2,
    "chronic": 3,
    "unknown": -1,
}

TRAJECTORY_MAP = {
    "improving": 0,
    "stable":    1,
    "worsening": 2,
    "unknown":   -1,
}

AGE_MAP = {
    "neonate":          0,
    "infant_under_3mo": 1,
    "child":            2,
    "adolescent":       3,
    "adult":            4,
    "older_adult":      5,
    "unknown":         -1,
}

SEX_MAP = {
    "female":  0,
    "male":    1,
    "other":   2,
    "unknown": -1,
}

RESOURCE_COUNT_MAP = {
    "0":      0,
    "1":      1,
    "2+":     2,
    "unknown": -1,
}

# helper

def get_val(section, field):
    # returns the value from an evidence object, or None if null
    entry = section.get(field)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("value")
    return entry  # shouldn't happen after normalization, but safe fallback

def encode_binary(val):
    # 0/1 → 0/1, None → -1
    if val is None:
        return -1
    return int(val)

def encode_categorical(val, mapping):
    if val is None:
        return -1
    return mapping.get(str(val).lower(), -1)

def encode_vital(val):
    # None → empty string so csv can handle it; classifier will impute
    if val is None:
        return ""
    return val

# column definitions (determines output column order)

SYMPTOM_FIELDS = [
    "chest_pain", "dyspnea", "altered_mental_status", "active_bleeding",
    "seizure", "syncope", "cyanosis", "vomiting", "fever_reported",
    "abdominal_pain", "headache", "rash", "palpitations", "weakness",
    "trauma_mechanism", "respiratory_distress", "suicidal_ideation"
]

RISK_FIELDS = [
    "pregnancy_related", "pediatric_high_risk", "elderly_risk",
    "immunocompromised", "cardiac_history", "anticoagulant_use",
    "oncologic", "mental_health_concern"
]

RESOURCE_BINARY_FIELDS = [
    "needs_labs", "needs_imaging", "needs_iv_access",
    "needs_specialist", "needs_procedure"
]

VITAL_FIELDS = [
    "heart_rate", "resp_rate", "systolic_bp",
    "diastolic_bp", "spo2", "temperature"
]

# build one row per record

def build_row(record):
    feat  = record["extracted_features"]
    vit   = record["objective_vitals"]

    sym   = feat.get("symptoms", {})
    sev   = feat.get("severity", {})
    risk  = feat.get("risk_factors", {})
    temp  = feat.get("temporal_pattern", {})
    res   = feat.get("resource_clues", {})
    demo  = feat.get("demographics", {})

    row = {
        "case_id": record["case_id"],
        "label"  : record["label"],
        "source" : record["source"],
    }

    # demographics
    row["demo_age_group"] = encode_categorical(get_val(demo, "age_group"), AGE_MAP)
    row["demo_sex"]       = encode_categorical(get_val(demo, "sex"), SEX_MAP)

    # symptoms (binary)
    for f in SYMPTOM_FIELDS:
        row[f"sym_{f}"] = encode_binary(get_val(sym, f))

    # severity (ordinal)
    row["sev_pain_severity"]     = encode_categorical(get_val(sev, "pain_severity"), SEVERITY_MAP)
    row["sev_respiratory_effort"]= encode_categorical(get_val(sev, "respiratory_effort"), SEVERITY_MAP)
    row["sev_overall_distress"]  = encode_categorical(get_val(sev, "overall_distress"), SEVERITY_MAP)

    # risk factors (binary)
    for f in RISK_FIELDS:
        row[f"risk_{f}"] = encode_binary(get_val(risk, f))

    # temporal pattern
    row["temp_onset"]      = encode_categorical(get_val(temp, "onset"), ONSET_MAP)
    row["temp_duration"]   = encode_categorical(get_val(temp, "duration"), DURATION_MAP)
    row["temp_trajectory"] = encode_categorical(get_val(temp, "trajectory"), TRAJECTORY_MAP)

    # resource clues (binary)
    for f in RESOURCE_BINARY_FIELDS:
        row[f"res_{f}"] = encode_binary(get_val(res, f))

    row["res_estimated_resources"] = encode_categorical(
        get_val(res, "estimated_resources"), RESOURCE_COUNT_MAP
    )

    # objective vitals (numeric, empty string if missing)
    for f in VITAL_FIELDS:
        row[f"vit_{f}"] = encode_vital(vit.get(f))

    # pain score from patient_reported
    pain = feat.get("patient_reported", {})
    if isinstance(pain, dict):
        row["vit_pain_score"] = encode_vital(pain.get("pain_score"))
    else:
        row["vit_pain_score"] = ""

    return row


# run 
records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

rows = [build_row(r) for r in records]

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

columns = list(rows[0].keys())
with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"saved {len(rows)} rows × {len(columns)} columns → {OUT_PATH}")

# quick feature summary
print("\nfeature coverage (% non-missing, excluding vitals):")
feature_cols = [c for c in columns if c not in ("case_id", "label", "source")
                and not c.startswith("vit_")]
for col in feature_cols:
    vals = [r[col] for r in rows]
    present = sum(1 for v in vals if v != -1)
    print(f"  {col:<35} {present:3d}/174 = {present/174*100:.0f}%")