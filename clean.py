import pandas as pd
import re

with open("Train_scenario.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

records = []

for i, line in enumerate(lines):
    if i == 0 or line.strip() == "":
        continue

    parts = line.rstrip("\n").split("\t")

    if len(parts) == 4 and parts[1] != "Clinical Vignettes":
        records.append({
            "row_id": i,
            "category": parts[0].strip(),
            "vignette": parts[1].strip(),
            "explanation": parts[2].strip(),
            "esi": parts[3].strip(),
        })

df = pd.DataFrame(records)

df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
df = df.dropna(subset=["esi"])
df["esi"] = df["esi"].astype(int)
df = df[df["esi"].between(1, 5)].copy()

df["source"] = df["category"].apply(
    lambda x: "categorized" if x != "" else "narrative"
)

ambig_patterns = [
    r"uptriag",
    r"initially esi",
    r"initially level",
    r"could be upgraded",
    r"could be downgraded",
    r"may be upgraded",
    r"may be downgraded",
    r"would be upgraded",
    r"would be downgraded",
    r"prior to vital sign assessment",
    r"depending on",
    r"borderline",
]

def has_pattern(text, patterns):
    text = text.lower()
    return any(re.search(p, text) for p in patterns)

df["label_ambiguous"] = df["explanation"].apply(
    lambda x: has_pattern(x, ambig_patterns)
)

leak_patterns = [
    r"esi level",
    r"assigned to esi",
    r"triaged as esi",
    r"initially esi",
]

df["vignette_leakage"] = df["vignette"].apply(
    lambda x: has_pattern(x, leak_patterns)
)

print("Cleaned data shape:", df.shape)
print("\nSource distribution:")
print(df["source"].value_counts())

print("\nESI by source:")
print(pd.crosstab(df["source"], df["esi"]))

print("\nAmbiguous labels:")
print(df["label_ambiguous"].value_counts())

print("\nVignette leakage check:")
print(df["vignette_leakage"].value_counts())

df_clean = df[
    ["row_id", "category", "vignette", "esi", "source", "label_ambiguous", "vignette_leakage"]
].copy()

df_clean.to_csv("tribot_clean.csv", index=False)
df_clean[
    (~df_clean["label_ambiguous"]) & (~df_clean["vignette_leakage"])
].to_csv("tribot_clean_strict.csv", index=False)