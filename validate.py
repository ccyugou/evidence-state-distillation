
# run after extraction to check output quality before moving to classifier

import json
from collections import defaultdict

IN_PATH = "extracted_features_normalized.jsonl"

SYMPTOM_FIELDS = [
    "chest_pain", "dyspnea", "altered_mental_status", "active_bleeding",
    "seizure", "syncope", "cyanosis", "vomiting", "fever_reported",
    "abdominal_pain", "headache", "rash", "palpitations", "weakness",
    "trauma_mechanism", "respiratory_distress", "suicidal_ideation"
]

SEVERITY_ENUMS = {
    "pain_severity"      : {"none", "mild", "moderate", "severe", "unknown"},
    "respiratory_effort" : {"normal", "mild", "moderate", "severe", "unknown"},
    "overall_distress"   : {"calm", "mild", "moderate", "severe", "unknown"},
}

TEMPORAL_ENUMS = {
    "onset"      : {"sudden", "gradual", "unknown"},
    "duration"   : {"hours", "days", "weeks", "chronic", "unknown"},
    "trajectory" : {"worsening", "stable", "improving", "unknown"},
}

RISK_FIELDS = [
    "pregnancy_related", "pediatric_high_risk", "elderly_risk",
    "immunocompromised", "cardiac_history", "anticoagulant_use",
    "oncologic", "mental_health_concern"
]

RESOURCE_BINARY = [
    "needs_labs", "needs_imaging", "needs_iv_access",
    "needs_specialist", "needs_procedure"
]

VALID_SOURCES = {
    "patient_reported", "nurse_observed", "objective_vitals",
    "question_intent", "revision", "mixed"
}


def check_binary_field(section, field_name, all_issues):
    val = section.get(field_name)
    if val is None:
        return
    if not isinstance(val, dict):
        all_issues.append(f"{field_name}: expected object or null, got {type(val).__name__}")
        return
    if val.get("value") not in (0, 1):
        all_issues.append(f"{field_name}: value must be 0 or 1, got {val.get('value')}")
    # skip evidence check for _no_evidence flagged fields
    if val.get("_no_evidence"):
        return
    if not val.get("evidence_text"):
        all_issues.append(f"{field_name}: missing evidence_text")
    if val.get("source_layer") not in VALID_SOURCES:
        all_issues.append(f"{field_name}: invalid source_layer '{val.get('source_layer')}'")


def check_categorical_field(section, field_name, allowed, all_issues):
    val = section.get(field_name)
    if val is None:
        return
    if not isinstance(val, dict):
        all_issues.append(f"{field_name}: expected object or null")
        return
    if val.get("value") not in allowed:
        all_issues.append(f"{field_name}: '{val.get('value')}' not in {allowed}")
    if not val.get("evidence_text"):
        all_issues.append(f"{field_name}: missing evidence_text")
    if val.get("source_layer") not in VALID_SOURCES:
        all_issues.append(f"{field_name}: invalid source_layer")


def validate_record(record):
    issues = []
    feat = record.get("extracted_features", {})

    # symptoms
    symptoms = feat.get("symptoms", {})
    for f in SYMPTOM_FIELDS:
        check_binary_field(symptoms, f, issues)

    # severity
    severity = feat.get("severity", {})
    for f, allowed in SEVERITY_ENUMS.items():
        check_categorical_field(severity, f, allowed, issues)

    # risk factors
    risks = feat.get("risk_factors", {})
    for f in RISK_FIELDS:
        check_binary_field(risks, f, issues)

    # temporal pattern
    temporal = feat.get("temporal_pattern", {})
    for f, allowed in TEMPORAL_ENUMS.items():
        check_categorical_field(temporal, f, allowed, issues)

    # resource clues
    resources = feat.get("resource_clues", {})
    for f in RESOURCE_BINARY:
        check_binary_field(resources, f, issues)

    er = resources.get("estimated_resources")
    if er is not None and isinstance(er, dict):
        if str(er.get("value")) not in {"0", "1", "2+", "unknown"}:
            issues.append(f"estimated_resources: invalid value '{er.get('value')}'")

    return issues


# run validation
records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total_issues  = 0
records_clean = 0
issue_counts  = defaultdict(int)

for r in records:
    issues = validate_record(r)
    total_issues += len(issues)
    if not issues:
        records_clean += 1
    for iss in issues:
        field = iss.split(":")[0]
        issue_counts[field] += 1

print(f"validated {len(records)} records")
print(f"  clean (0 issues)  : {records_clean}")
print(f"  with issues       : {len(records) - records_clean}")
print(f"  total issues      : {total_issues}")

if issue_counts:
    print(f"\nmost common issues:")
    for field, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}  {field}")

# evidence coverage: what fraction of positive features have evidence
positive_with_evidence = 0
positive_total = 0

for r in records:
    feat = r.get("extracted_features", {})
    for section in ["symptoms", "risk_factors", "resource_clues"]:
        for field, val in feat.get(section, {}).items():
            if isinstance(val, dict) and val.get("value") in (1, 0):
                positive_total += 1
                if val.get("evidence_text"):
                    positive_with_evidence += 1

if positive_total > 0:
    print(f"\nevidence coverage: {positive_with_evidence}/{positive_total} "
          f"= {positive_with_evidence/positive_total*100:.1f}%")