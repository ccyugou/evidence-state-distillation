# step5_classify.py
import json
import time
from openai import OpenAI
from pathlib import Path

from classification import CLASSIFICATION_SYSTEM_PROMPT, build_classification_prompt

FEATURES_PATH = "extracted_features_normalized.jsonl"
VERIF_PATH    = "verification_results_v1.jsonl"
OUT_PATH      = "classification_results_v1.jsonl"

from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


def call_api(user_prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                print(f"    API error ({e}), retry {attempt+1}...")
                time.sleep(5)
            else:
                raise


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1])
    return json.loads(text)


def basic_validate(result):
    issues = []
    esi = result.get("predicted_esi")
    if esi not in (1, 2, 3, 4, 5):
        issues.append(f"predicted_esi invalid: {esi}")
    if result.get("confidence") not in ("high", "medium", "low"):
        issues.append("confidence invalid")
    if not result.get("primary_evidence"):
        issues.append("primary_evidence is empty")
    if not result.get("reasoning"):
        issues.append("reasoning is empty")
    return issues


# ── load and index ────────────────────────────────────────────────────────────
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

resources = {}
RESOURCE_PATH = "resource_estimates.jsonl"
if Path(RESOURCE_PATH).exists():
    with open(RESOURCE_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            resources[r["case_id"]] = r
    print(f"resources loaded: {len(resources)}")

case_ids = list(features.keys())
print(f"features: {len(features)}  verifications: {len(verifs)}")

missing_verif = [c for c in case_ids if c not in verifs]
if missing_verif:
    print(f"WARNING: {len(missing_verif)} cases have no verification result")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_cases  = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for i, case_id in enumerate(case_ids):
        print(f"[{i+1:03d}/{len(case_ids)}] {case_id} ...", end=" ", flush=True)

        verif = verifs.get(case_id, {
            "missed_esi1_signals" : [],
            "missed_esi2_signals" : [],
            "upgrade_recommendation": {"recommend_upgrade": False}
        })

        try:
            prompt = build_classification_prompt(features[case_id], verif, resources.get(case_id))
            raw    = call_api(prompt)
            result = parse_response(raw)
            issues = basic_validate(result)

            record = {
                "case_id"       : case_id,
                "label"         : features[case_id]["label"],
                "source"        : features[case_id]["source"],
                "predicted_esi" : result.get("predicted_esi"),
                "confidence"    : result.get("confidence"),
                "decision_path" : result.get("decision_path", {}),
                "primary_evidence": result.get("primary_evidence", []),
                "reasoning"     : result.get("reasoning", ""),
                "validation_issues": issues,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1

            status = f"ESI {result.get('predicted_esi')} ({result.get('confidence')}) label={features[case_id]['label']}"
            if issues:
                status += f" [{len(issues)} issues]"
            print(status)

        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
            failed_cases.append(case_id)

        except Exception as e:
            print(f"FAILED: {e}")
            failed_cases.append(case_id)

        time.sleep(0.5)

print(f"\ndone: {success_count} ok, {len(failed_cases)} failed")
if failed_cases:
    print(f"failed: {failed_cases}")
print(f"saved → {OUT_PATH}")