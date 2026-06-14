# Print the actual error values ​​of fields 

import json

IN_PATH = "extracted_features_normalized.jsonl"

CATEGORICAL_CHECKS = {
    "temporal_pattern": {
        "onset"     : {"sudden", "gradual", "unknown"},
        "trajectory": {"worsening", "stable", "improving", "unknown"},
    },
    "severity": {
        "pain_severity"      : {"none", "mild", "moderate", "severe", "unknown"},
        "respiratory_effort" : {"normal", "mild", "moderate", "severe", "unknown"},
        "overall_distress"   : {"calm", "mild", "moderate", "severe", "unknown"},
    },
}

records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

for section, fields in CATEGORICAL_CHECKS.items():
    for field, allowed in fields.items():
        bad_values = {}
        for r in records:
            val = r["extracted_features"].get(section, {}).get(field)
            if isinstance(val, dict) and val.get("value") not in allowed:
                v = val["value"]
                bad_values[v] = bad_values.get(v, 0) + 1
        if bad_values:
            print(f"{section}.{field}:")
            for v, count in sorted(bad_values.items(), key=lambda x: -x[1]):
                print(f"  {count:3d}x  {repr(v)}")

print("\n trauma_mechanism bad values ")
for r in records:
    val = r["extracted_features"].get("symptoms", {}).get("trauma_mechanism")
    if isinstance(val, dict) and val.get("value") not in (0, 1):
        print(r["case_id"], repr(val.get("value")))