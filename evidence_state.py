import json
import re
from pathlib import Path

IN_PATH    = "clean_cases.jsonl"
OUT_STATES = "evidence_states.jsonl"
OUT_SUMMARY = "evidence_state_summary.json"

# read data
records = []
with open(IN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"loaded {len(records)} records")


# vital extraction functions
# each function returns the value or None
# plausibility checks are inside each function

def get_hr(text):
    m = re.search(r'\bHR\s*[:\s]\s*(\d{2,3})\b', text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 20 <= v <= 300:
            return v
    return None

def get_rr(text):
    m = re.search(r'\bRR\s*[:\s]\s*(\d{1,2})\b', text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 4 <= v <= 80:
            return v
    return None

def get_bp(text):
    m = re.search(r'\bBP\s*[:\s]\s*(\d{2,3})\s*/\s*(\d{2,3})\b', text, re.IGNORECASE)
    if m:
        sbp = int(m.group(1))
        dbp = int(m.group(2))
        if 40 <= sbp <= 300 and 20 <= dbp <= 200:
            return sbp, dbp
    return None, None

def get_spo2(text):
    m = re.search(r'\bSpO2?\s*[:\s]?\s*(\d{2,3})\s*%?', text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 50 <= v <= 100:
            return v
    return None

def get_temperature(text):
    # catch: T 38.5, T: 38.5, Temp 38.5, Temperature 38.5, T 98.6F, T 98.6°F
    m = re.search(
        r'\bT(?:emp(?:erature)?)?\s*[:\s]?\s*([\d]{2,3}(?:\.\d)?)\s*°?\s*([FC])?\b',
        text, re.IGNORECASE
    )
    if not m:
        return None

    val  = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else None

    # convert fahrenheit to celsius
    # if unit is explicitly F, or value is in typical F range (94-108)
    if unit == "F" or (unit is None and val > 45):
        val = round((val - 32) * 5 / 9, 1)

    if 30.0 <= val <= 45.0:
        return val
    return None

def get_pain(text):
    # require pain/rate/score context nearby to avoid matching BP, GCS, fractions
    m = re.search(
        r'(?:pain|rates?|score)[^/\d]{0,25}(\d{1,2})\s*/\s*10\b',
        text, re.IGNORECASE
    )
    if not m:
        # fallback: standalone X/10 only if not preceded by another number (avoids 120/80)
        m = re.search(r'(?<!\d)(\d{1,2})\s*/\s*10\b', text, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 10:
            return v
    return None


print("extracting vitals from all records")
# build evidence states
states = []
parse_warnings_list = []  # actual parsing problems (out-of-range values)
adult_flag_records  = []  # separate: adult-threshold exploratory flags

for r in records:
    text = r["clean_text"]

    hr        = get_hr(text)
    rr        = get_rr(text)
    sbp, dbp  = get_bp(text)
    spo2      = get_spo2(text)
    temp      = get_temperature(text)
    pain      = get_pain(text)

    vitals = {
        "heart_rate"   : hr,
        "resp_rate"    : rr,
        "systolic_bp"  : sbp,
        "diastolic_bp" : dbp,
        "spo2"         : spo2,
        "temperature"  : temp,
    }

    missing = [k for k, v in vitals.items() if v is None]
    if pain is None:
        missing.append("pain_score")

    # adult-threshold exploratory flags — NOT for safety gate, pediatric caveat
    flags = []
    if hr   and (hr > 120 or hr < 50):
        flags.append("heart_rate")
    if rr   and (rr > 24 or rr < 8):
        flags.append("resp_rate")
    if spo2 and spo2 < 92:
        flags.append("spo2")
    if sbp  and (sbp < 90 or sbp > 180):
        flags.append("systolic_bp")
    if temp and (temp >= 38.0 or temp < 35.0):
        flags.append("temperature")
    if pain and pain >= 7:
        flags.append("pain_score")

    if flags:
        adult_flag_records.append({"case_id": r["case_id"], "flags": flags})

    state = {
        "case_id"     : r["case_id"],
        "source"      : r["source"],
        "input_type"  : "vignette",
        "label_system": "ESI",
        "label"       : r["label"],
        "clean_text"  : text,

        "evidence_state": {
            "patient_reported": {
                "chief_complaint"     : None,
                "symptoms"            : [],
                "duration"            : None,
                "pain_description"    : None,
                "pain_score"          : pain,
                "associated_symptoms" : [],
                "negative_symptoms"   : [],
                "risk_history"        : [],
            },
            "nurse_observed": {
                "general_appearance"  : None,
                "work_of_breathing"   : None,
                "mental_status"       : None,
                "skin_signs"          : [],
                "visible_bleeding"    : None,
                "distress_level"      : None,
                "objective_vitals"    : vitals,
            },
            "question_intent" : [],
            "revisions"       : [],
            "missing_information": missing,
        }
    }

    states.append(state)


# summary stats
vital_keys = ["heart_rate", "resp_rate", "systolic_bp", "diastolic_bp",
              "spo2", "temperature", "pain_score"]

extracted_count = {k: 0 for k in vital_keys}
vital_count_dist = {i: 0 for i in range(8)}
label_stats = {}

for s in states:
    ov   = s["evidence_state"]["nurse_observed"]["objective_vitals"]
    pain = s["evidence_state"]["patient_reported"]["pain_score"]
    lbl  = s["label"]

    count = 0
    for k in ["heart_rate", "resp_rate", "systolic_bp", "diastolic_bp",
              "spo2", "temperature"]:
        if ov[k] is not None:
            extracted_count[k] += 1
            count += 1
    if pain is not None:
        extracted_count["pain_score"] += 1
        count += 1

    vital_count_dist[count] = vital_count_dist.get(count, 0) + 1

    if lbl not in label_stats:
        label_stats[lbl] = {"n": 0, "any_vitals": 0, "total_count": 0}
    label_stats[lbl]["n"] += 1
    label_stats[lbl]["total_count"] += count
    if count > 0:
        label_stats[lbl]["any_vitals"] += 1

total = len(states)
print(f"\nvital extraction coverage:")
for k in vital_keys:
    n = extracted_count[k]
    print(f"  {k:<18} {n:3d}/{total} = {n/total*100:.1f}%")

print(f"\nvital count per record:")
for c in range(8):
    print(f"  {c} vitals : {vital_count_dist.get(c, 0):3d} records")

print(f"\ncoverage by ESI label:")
print(f"  {'ESI':>4}  {'n':>4}  {'any_vitals':>10}  {'rate':>6}  {'avg_count':>9}")
for lbl in sorted(label_stats):
    ls = label_stats[lbl]
    rate = ls["any_vitals"] / ls["n"]
    avg  = ls["total_count"] / ls["n"]
    print(f"  {lbl:>4}  {ls['n']:>4}  {ls['any_vitals']:>10}  {rate:>6.1%}  {avg:>9.2f}")

print(f"\nadult-threshold exploratory flags: {len(adult_flag_records)} records")
print("  (not used for safety gate — pediatric cases not excluded)")
print(f"\nplausibility parse warnings: {len(parse_warnings_list)}")


# write output
Path(OUT_STATES).parent.mkdir(parents=True, exist_ok=True)

with open(OUT_STATES, "w", encoding="utf-8") as f:
    for s in states:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

summary = {
    "total_records": total,
    "vital_extraction_coverage": {
        k: {"extracted": extracted_count[k],
            "missing"  : total - extracted_count[k],
            "rate"     : round(extracted_count[k] / total, 3)}
        for k in vital_keys
    },
    "vital_count_distribution": vital_count_dist,
    "coverage_by_label": {
        str(lbl): {
            "n"              : ls["n"],
            "any_vitals"     : ls["any_vitals"],
            "rate"           : round(ls["any_vitals"] / ls["n"], 3),
            "avg_vital_count": round(ls["total_count"] / ls["n"], 2),
        }
        for lbl, ls in sorted(label_stats.items())
    },
    "adult_threshold_exploratory_flags": {
        "note": "adult thresholds only, not used for safety gate, pediatric cases not excluded",
        "flagged_record_count": len(adult_flag_records),
    },
    "plausibility_warnings": {
        "total": len(parse_warnings_list),
    }
}

with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nsaved to {OUT_STATES}")
print(f"saved to {OUT_SUMMARY}")