
import json
from pathlib import Path

IN_PATH  = "extracted_features.jsonl"
OUT_PATH = "extracted_features_normalized.jsonl"

BINARY_SECTIONS = {
    "symptoms": [
        "chest_pain", "dyspnea", "altered_mental_status", "active_bleeding",
        "seizure", "syncope", "cyanosis", "vomiting", "fever_reported",
        "abdominal_pain", "headache", "rash", "palpitations", "weakness",
        "trauma_mechanism", "respiratory_distress", "suicidal_ideation"
    ],
    "risk_factors": [
        "pregnancy_related", "pediatric_high_risk", "elderly_risk",
        "immunocompromised", "cardiac_history", "anticoagulant_use",
        "oncologic", "mental_health_concern"
    ],
    "resource_clues": [
        "needs_labs", "needs_imaging", "needs_iv_access",
        "needs_specialist", "needs_procedure"
    ]
}

CATEGORICAL_SECTIONS = {
    "severity"        : ["pain_severity", "respiratory_effort", "overall_distress"],
    "temporal_pattern": ["onset", "duration", "trajectory"],
}


def to_binary_int(val):
    if val in (1, "1", True, "true", "yes"):
        return 1
    if val in (0, "0", False, "false", "no"):
        return 0
    if isinstance(val, str) and val.strip():
        return 1  #Descriptive string = LLM is describing positive features, mapped to 1
    return val

def normalize_duration(val):
    if not isinstance(val, str):
        return val
    v = val.lower().strip()
    if v in ("hours", "hour", "acute", "minutes", "mins"):
        return "hours"
    if v in ("days", "day") or any(
        f"{n} day" in v or f"{n}-day" in v
        for n in range(1, 15)
    ):
        return "days"
    if v in ("weeks", "week") or any(
        f"{n} week" in v for n in range(1, 8)
    ):
        return "weeks"
    if v in ("chronic", "months", "month", "years", "year", "long-standing", "longstanding"):
        return "chronic"
    if v == "unknown":
        return "unknown"
    # free text like "few days", "several days"
    if "day" in v:
        return "days"
    if "week" in v:
        return "weeks"
    if "month" in v or "year" in v:
        return "chronic"
    if "hour" in v or "minute" in v:
        return "hours"
    return "unknown"  # can't map, flag as unknown


def normalize_estimated_resources(val):
    if val in (2, "2", "two or more", "2 or more", ">=2", ">1", "multiple"):
        return "2+"
    if val in (1, "1", "one"):
        return "1"
    if val in (0, "0", "zero", "none", "no resources"):
        return "0"
    if val in ("unknown", "unclear"):
        return "unknown"
    # LLM used acuity language — map conservatively
    if val in ("high", "many", "several"):
        return "2+"
    if val in ("low", "minimal"):
        return "1"
    # LLM returned resource type name or other unexpected string
    # can't reliably map, mark unreliable
    return "unknown"


def wrap_raw(val, normalized_val):
    return {
        "value"        : normalized_val,
        "evidence_text": "",
        "source_layer" : "unknown",
        "_no_evidence" : True   # flagged: LLM gave value but no evidence
    }

def resolve_no_evidence_binary(val_obj):
    # LLM returned raw int without evidence
    # value=0 without evidence → probably "not mentioned" → null
    # value=1 without evidence → keep as low-confidence positive
    if val_obj.get("_no_evidence") and val_obj.get("value") == 0:
        return None
    return val_obj


def normalize_binary_section(feat, section, fields):
    sec = feat.get(section, {})
    for field in fields:
        val = sec.get(field)
        if val is None:
            continue
        if isinstance(val, dict):
            val["value"] = to_binary_int(val["value"])
        elif isinstance(val, (int, float, bool, str)):
            wrapped = wrap_raw(val, to_binary_int(val))
            sec[field] = resolve_no_evidence_binary(wrapped)
        else:
            sec[field] = None


def normalize_onset(val):
    if not isinstance(val, str):
        return "unknown"
    v = val.lower().strip()
    if v in ("sudden", "acute", "immediate", "abrupt", "sudden onset",
             "this morning", "tonight", "today"):
        return "sudden"
    if v in ("gradual", "progressive", "slowly", "insidious"):
        return "gradual"
    if "days ago" in v or "weeks ago" in v:
        return "gradual"
    return "unknown"  # intermittent, other free text


def normalize_trajectory(val):
    if not isinstance(val, str):
        return "unknown"
    v = val.lower().strip()
    if v in ("worsening", "worse", "deteriorating", "increasing", "escalating"):
        return "worsening"
    if v in ("stable", "unchanged", "no change", "same", "unchanged"):
        return "stable"
    if v in ("improving", "better", "resolved", "resolving", "improved"):
        return "improving"
    return "unknown"  # positional, other


def normalize_pain_severity(val):
    # handle int 0 → "none"
    if val in (0, "0"):
        return "none"
    if not isinstance(val, str):
        return val
    v = val.lower().strip()
    # "4/10" style
    if "/" in v:
        try:
            score = float(v.split("/")[0])
            if score == 0:
                return "none"
            if score <= 3:
                return "mild"
            if score <= 6:
                return "moderate"
            return "severe"
        except ValueError:
            return "unknown"
    # wrong field content ("sudden onset" ended up here)
    if "onset" in v or "sudden" in v:
        return "unknown"
    return val


def normalize_respiratory_effort(val):
    if not isinstance(val, str):
        return val
    v = val.lower().strip()
    if v in ("increased", "increased work of breathing", "laboured",
             "labored", "work of breathing"):
        return "moderate"
    if v in ("intubated", "intubation", "mechanically ventilated"):
        return "severe"
    if v in ("wheezing", "stridor"):
        # symptom, not effort level — can't map cleanly
        return "unknown"
    return val


def normalize_overall_distress(val):
    if val in (0, "0"):
        return "calm"
    if not isinstance(val, str):
        return val
    v = val.lower().strip()
    if v in ("none", "calm", "stable", "quiet", "comfortable", "no distress"):
        return "calm"
    if v in ("concerned", "anxious", "worried", "scared", "mild"):
        return "mild"
    if v in ("distressed", "agitated", "restless", "emotional distress",
             "upset", "appears_sick", "appears sick", "distressed"):
        return "moderate"
    if v in ("severe", "extreme"):
        return "severe"
    return "unknown"


CATEGORICAL_NORMALIZERS = {
    "onset"              : normalize_onset,
    "trajectory"         : normalize_trajectory,
    "duration"           : normalize_duration,
    "pain_severity"      : normalize_pain_severity,
    "respiratory_effort" : normalize_respiratory_effort,
    "overall_distress"   : normalize_overall_distress,
}

def normalize_categorical_section(feat, section, fields):
    sec = feat.get(section, {})
    for field in fields:
        val = sec.get(field)
        if val is None:
            continue

        normalizer = CATEGORICAL_NORMALIZERS.get(field)

        if isinstance(val, dict):
            raw = val["value"]
            val["value"] = normalizer(raw) if normalizer else raw
        elif isinstance(val, str):
            normalized = normalizer(val) if normalizer else val
            sec[field] = wrap_raw(val, normalized)
        else:
            sec[field] = None


def normalize_record(record):
    feat = record.get("extracted_features", {})

    for section, fields in BINARY_SECTIONS.items():
        normalize_binary_section(feat, section, fields)

    for section, fields in CATEGORICAL_SECTIONS.items():
        normalize_categorical_section(feat, section, fields)

    # estimated_resources needs special handling
    er_section = feat.get("resource_clues", {})
    er = er_section.get("estimated_resources")
    if er is None:
        pass
    elif isinstance(er, list):
        er_section["estimated_resources"] = None
    elif isinstance(er, dict):
        er["value"] = normalize_estimated_resources(er["value"])
    else:
        er_section["estimated_resources"] = wrap_raw(er, normalize_estimated_resources(er))

    return record


#  run
records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

normalized = [normalize_record(r) for r in records]

# summary
no_evidence_count = 0
for r in normalized:
    feat = r["extracted_features"]
    for section in list(BINARY_SECTIONS.keys()) + list(CATEGORICAL_SECTIONS.keys()):
        for field, val in feat.get(section, {}).items():
            if isinstance(val, dict) and val.get("_no_evidence"):
                no_evidence_count += 1

er_unknown = sum(
    1 for r in normalized
    if isinstance(r["extracted_features"].get("resource_clues", {})
                   .get("estimated_resources"), dict)
    and r["extracted_features"]["resource_clues"]["estimated_resources"]["value"] == "unknown"
)

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in normalized:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"normalized {len(normalized)} records")
print(f"fields wrapped (no evidence provided): {no_evidence_count}")
print(f"estimated_resources mapped to unknown: {er_unknown}")
print(f"\nnote: estimated_resources has semantic issues.")
print(f"      values like 'high'/'imaging' were force-mapped to '2+'/unknown.")
print(f"      consider treating this field as unreliable in the classifier.")
print(f"\nsaved to {OUT_PATH}")