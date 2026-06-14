import json
import re
from pathlib import Path

RAW_PATH  = "Train_scenario.txt"
OUT_PATH  = "clean_cases.jsonl"

LEAKAGE_PATTERNS = [
    r"ESI\s*(level)?\s*[1-5]",
    r"immediate\s+life.saving",
    r"one\s+resource",
    r"two\s+or\s+more\s+resources",
    r"no\s+resources",
]

def check_leakage(text):
    return [p for p in LEAKAGE_PATTERNS if re.search(p, text, re.IGNORECASE)]

# file inspection
with open(RAW_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"total lines : {len(lines)}")

col_counts = {}
for line in lines:
    n = len(line.split("\t"))
    col_counts[n] = col_counts.get(n, 0) + 1
print(f"col distribution : {col_counts}")

tab_lines   = sum(1 for l in lines if "\t" in l)
esi_mentions = sum(1 for l in lines if "esi level" in l.lower())
print(f"lines with tab     : {tab_lines}")
print(f"lines with 'ESI level' : {esi_mentions}")

# parse
records = []
skipped = 0

for i, line in enumerate(lines):
    if line.strip() == "" or i == 0:
        continue

    parts = line.rstrip("\n").split("\t")

    if len(parts) != 4:
        print(f"  row {i}: {len(parts)} cols, skipping")
        skipped += 1
        continue

    if parts[1] == "Clinical Vignettes":
        continue

    vignette = parts[1].strip()
    esi_raw  = parts[3].strip()

    if not vignette:
        skipped += 1
        continue

    try:
        label = int(esi_raw)
        assert label in {1, 2, 3, 4, 5}
    except (ValueError, AssertionError):
        print(f"  row {i}: bad ESI label '{esi_raw}', skipping")
        skipped += 1
        continue

    records.append({
        "case_id"       : f"case_{len(records)+1:03d}",
        "clean_text"    : vignette,
        "label"         : label,
        "source"        : "categorized" if parts[0].strip() else "narrative",
        "leakage_flags" : check_leakage(vignette),
    })

# summary 
esi_dist = {}
for r in records:
    esi_dist[r["label"]] = esi_dist.get(r["label"], 0) + 1

flagged = [r for r in records if r["leakage_flags"]]

print(f"\nparsed  : {len(records)}")
print(f"skipped : {skipped}")
print(f"ESI dist: {dict(sorted(esi_dist.items()))}")
print(f"leakage flags: {len(flagged)} records")
for r in flagged:
    print(f"  {r['case_id']} | {r['leakage_flags']} | {r['clean_text'][:80]}")

# write
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nsaved → {OUT_PATH}")