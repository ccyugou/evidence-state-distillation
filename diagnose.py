
# Print the actual values ​​of several problematic fields to see clearly the formatting issues
import json

IN_PATH = "extracted_features.jsonl"

records = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(" chest_pain value format samples ")
count = 0
for r in records:
    val = r["extracted_features"].get("symptoms", {}).get("chest_pain")
    if val is not None:
        print(f"  type={type(val).__name__}  raw={repr(val)}")
        count += 1
        if count >= 15:
            break

print("\n estimated_resources samples ")
count = 0
for r in records:
    val = r["extracted_features"].get("resource_clues", {}).get("estimated_resources")
    if val is not None:
        print(f"  type={type(val).__name__}  raw={repr(val)}")
        count += 1
        if count >= 10:
            break

print("\n duration samples ")
count = 0
for r in records:
    val = r["extracted_features"].get("temporal_pattern", {}).get("duration")
    if val is not None:
        print(f"  value={repr(val.get('value'))}")
        count += 1
        if count >= 10:
            break