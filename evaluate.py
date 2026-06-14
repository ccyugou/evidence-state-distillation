# step_evaluate.py
import json
from collections import defaultdict

FILES = {
    "Direct LLM (baseline)" : "baseline_results.jsonl",
    "SAFE-Triage v1"        : "classification_results_v1.jsonl",
    "SAFE-Triage + Safety"  : "final_results.jsonl",
}


def load(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_metrics(records):
    total = len(records)
    correct     = sum(1 for r in records if r["label"] == r.get("predicted_esi"))
    discordance = total - correct
    under       = sum(1 for r in records if r.get("predicted_esi") > r["label"])
    over        = sum(1 for r in records if r.get("predicted_esi") < r["label"])
    sig_under   = sum(1 for r in records
                      if r["label"] in (1,2) and r.get("predicted_esi") in (3,4,5))
    sig_over    = sum(1 for r in records
                      if r["label"] in (4,5) and r.get("predicted_esi") in (1,2))
    adj         = sum(1 for r in records
                      if abs(r["label"] - r.get("predicted_esi")) <= 1)
    hi          = [r for r in records if r["label"] in (1,2)]
    hi_recall   = sum(1 for r in hi if r.get("predicted_esi") in (1,2))

    per_esi = {}
    for esi in [1,2,3,4,5]:
        g = [r for r in records if r["label"] == esi]
        c = sum(1 for r in g if r.get("predicted_esi") == esi)
        per_esi[esi] = (c, len(g))

    flagged = sum(1 for r in records if r.get("human_review_required", False))

    return {
        "total"          : total,
        "accuracy"       : correct / total,
        "discordance"    : discordance / total,
        "under_triage"   : under / total,
        "over_triage"    : over / total,
        "sig_under"      : sig_under / total,
        "sig_over"       : sig_over / total,
        "adj_accuracy"   : adj / total,
        "hi_recall"      : hi_recall / len(hi) if hi else 0,
        "per_esi"        : per_esi,
        "flagged"        : flagged,
    }


# ── print comparison table ────────────────────────────────────────────────────
all_metrics = {}
for name, path in FILES.items():
    try:
        records = load(path)
        all_metrics[name] = compute_metrics(records)
        print(f"loaded {name}: {len(records)} records")
    except FileNotFoundError:
        print(f"MISSING: {path}")

print(f"{'Metric':<30}" + "".join(f"{n:>20}" for n in all_metrics))


rows = [
    ("Accuracy",              "accuracy",    True),
    ("Discordance",           "discordance", False),
    ("Under-triage",          "under_triage",False),
    ("Over-triage",           "over_triage", False),
    ("Sig. under-triage",     "sig_under",   False),
    ("Sig. over-triage",      "sig_over",    False),
    ("Adjacent accuracy",     "adj_accuracy",True),
    ("High-acuity recall",    "hi_recall",   True),
]

for label, key, higher_better in rows:
    vals = [all_metrics[n][key] for n in all_metrics]
    best = max(vals) if higher_better else min(vals)
    row  = f"{label:<30}"
    for name in all_metrics:
        v   = all_metrics[name][key]
        fmt = f"{v*100:.1f}%"
        row += f"{fmt:>20}"
    print(row)


print("\nPer-ESI accuracy:")
print(f"{'ESI':<8}" + "".join(f"{n:>20}" for n in all_metrics))
for esi in [1,2,3,4,5]:
    row = f"ESI {esi}  "
    for name in all_metrics:
        c, n = all_metrics[name]["per_esi"][esi]
        row += f"{c}/{n} ({c/n*100:.0f}%):>20".replace(":>20", "").rjust(20)

    # simpler version
    row = f"ESI {esi}   "
    for name in all_metrics:
        c, n = all_metrics[name]["per_esi"][esi]
        fmt = f"{c}/{n}={c/n*100:.0f}%"
        row += f"{fmt:>20}"
    print(row)

print()
print("Per-source discordance:")
for src in ["narrative", "categorized"]:
    row = f"{src:<30}"
    for name, path in FILES.items():
        if name not in all_metrics:
            continue
        try:
            records = load(path)
            g = [r for r in records if r["source"] == src]
            disc = sum(1 for r in g if r["label"] != r.get("predicted_esi"))
            fmt = f"{disc/len(g)*100:.1f}%"
            row += f"{fmt:>20}"
        except:
            row += f"{'N/A':>20}"
    print(row)

print(f"\nhuman_review_required (SAFE-Triage + Safety Gate only):")
if "SAFE-Triage + Safety" in all_metrics:
    flagged = all_metrics["SAFE-Triage + Safety"]["flagged"]
    total   = all_metrics["SAFE-Triage + Safety"]["total"]
    print(f"  {flagged}/{total} = {flagged/total*100:.1f}%")