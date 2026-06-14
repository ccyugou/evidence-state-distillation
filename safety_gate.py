# step6_safety_gate.py
import json
from pathlib import Path

CLASSIFICATION_PATH = "classification_results_v1.jsonl"
FEATURES_PATH       = "extracted_features_normalized.jsonl"
VERIF_PATH          = "verification_results_v1.jsonl"
OUT_PATH            = "final_results.jsonl"

# age-specific vital sign thresholds from ESI Handbook v5
VITAL_THRESHOLDS = {
    "neonate"         : {"hr": 190, "rr": 60},
    "infant_under_3mo": {"hr": 180, "rr": 55},
    "child"           : {"hr": 140, "rr": 40},
    "adolescent"      : {"hr": 100, "rr": 20},
    "adult"           : {"hr": 100, "rr": 20},
    "older_adult"     : {"hr": 100, "rr": 20},
    "unknown"         : {"hr": 100, "rr": 20},  # adult default if age unknown
}


def get_age_group(feat):
    age = feat.get("demographics", {}).get("age_group")
    if isinstance(age, dict):
        return age.get("value", "unknown")
    return "unknown"


def check_safety_gate(predicted_esi, feat, vitals, verif):
    if predicted_esi <= 2:
        return []  # already classified high-acuity, gate not relevant

    triggers = []
    symptoms  = feat.get("symptoms", {})
    severity  = feat.get("severity", {})
    risks     = feat.get("risk_factors", {})

    # ── Step A signals in extracted features ─────────────────────────
    for sym in ["altered_mental_status", "active_bleeding", "respiratory_distress"]:
        val = symptoms.get(sym)
        if isinstance(val, dict) and val.get("value") == 1:
            triggers.append(f"Step A: {sym}")

    # ── Step B signals ────────────────────────────────────────────────
    pain = severity.get("pain_severity")
    if isinstance(pain, dict) and pain.get("value") == "severe":
        triggers.append("Step B: severe pain")

    resp = severity.get("respiratory_effort")
    if isinstance(resp, dict) and resp.get("value") == "severe":
        triggers.append("Step B: severe respiratory effort")

    distress = severity.get("overall_distress")
    if isinstance(distress, dict) and distress.get("value") == "severe":
        triggers.append("Step B: severe distress")

    si = symptoms.get("suicidal_ideation")
    if isinstance(si, dict) and si.get("value") == 1:
        triggers.append("Step B: suicidal ideation")

    # immunocompromised + fever
    immuno = risks.get("immunocompromised")
    temp   = vitals.get("temperature")
    if isinstance(immuno, dict) and immuno.get("value") == 1:
        if temp and temp >= 38.0:
            triggers.append(f"Step B: immunocompromised with fever T={temp}")

    # ── Step D: age-specific vital sign thresholds ────────────────────
    age_group = get_age_group(feat)
    thresh    = VITAL_THRESHOLDS.get(age_group, VITAL_THRESHOLDS["adult"])

    hr   = vitals.get("heart_rate")
    rr   = vitals.get("resp_rate")
    spo2 = vitals.get("spo2")

    if hr and hr > thresh["hr"]:
        triggers.append(f"Step D: HR {hr} > {thresh['hr']} ({age_group})")
    if rr and rr > thresh["rr"]:
        triggers.append(f"Step D: RR {rr} > {thresh['rr']} ({age_group})")
    if spo2 and spo2 < 92:
        triggers.append(f"Step D: SpO2 {spo2}% < 92%")

    # ── Pediatric fever ───────────────────────────────────────────────
    if age_group in ("neonate", "infant_under_3mo") and temp and temp > 38.0:
        triggers.append(f"Pediatric fever: T={temp} in {age_group}")

    # ── Verification agent high/medium confidence findings ────────────
    for sig in verif.get("missed_esi1_signals", []) + verif.get("missed_esi2_signals", []):
        if sig.get("confidence") in ("high", "medium"):
            triggers.append(
                f"Verification [{sig['confidence']}]: {sig.get('criterion','')}"
            )

    return triggers


# ── load all files ────────────────────────────────────────────────────────────
classifications = {}
with open(CLASSIFICATION_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        classifications[r["case_id"]] = r

features = {}
with open(FEATURES_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        features[r["case_id"]] = r

verifs = {}
with open(VERIF_PATH, encoding="utf-8") as f:
    for line in f:
        v = json.loads(line)
        verifs[v["case_id"]] = v

print(f"classifications: {len(classifications)}")
print(f"features: {len(features)}  verifications: {len(verifs)}")

# ── apply safety gate ─────────────────────────────────────────────────────────
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

total_flagged       = 0
sig_under_flagged   = 0
sig_under_not_flagged = 0

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for case_id, cls in classifications.items():
        feat_record  = features.get(case_id, {})
        feat         = feat_record.get("extracted_features", {})
        vitals       = feat_record.get("objective_vitals", {})
        verif        = verifs.get(case_id, {})
        predicted    = cls.get("predicted_esi")
        label        = cls.get("label")

        triggers = check_safety_gate(predicted, feat, vitals, verif)
        flagged  = len(triggers) > 0

        if flagged:
            total_flagged += 1

        # track significant under-triage cases
        is_sig_under = label in (1, 2) and predicted in (3, 4, 5)
        if is_sig_under:
            if flagged:
                sig_under_flagged += 1
            else:
                sig_under_not_flagged += 1

        record = {
            **cls,
            "human_review_required": flagged,
            "safety_triggers"      : triggers,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── summary ───────────────────────────────────────────────────────────────────
total = len(classifications)
print(f"\n Safety Gate Summary ")
print(f"total cases          : {total}")
print(f"human_review_required: {total_flagged} ({total_flagged/total*100:.1f}%)")
print(f"\nSignificant under-triage cases (label 1/2, pred 3-5):")
print(f"  flagged by gate    : {sig_under_flagged}")
print(f"  NOT flagged        : {sig_under_not_flagged}  ← missed by safety gate")
print(f"\nsaved → {OUT_PATH}")