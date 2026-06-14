# step4_verify.py
import json
import time
from openai import OpenAI
from pathlib import Path

from ESI import VERIFICATION_SYSTEM_PROMPT, build_verification_prompt

FEATURES_PATH = "extracted_features_normalized.jsonl"
STATES_PATH   = "evidence_states.jsonl"
OUT_PATH      = "verification_results_v1.jsonl"

from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


def call_api(user_prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1500,
                temperature=0,
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
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
    for signal in result.get("missed_esi1_signals", []) + result.get("missed_esi2_signals", []):
        if not signal.get("evidence_text"):
            issues.append(f"missing evidence_text for criterion: {signal.get('criterion')}")
        if signal.get("confidence") not in ("high", "medium", "low"):
            issues.append(f"invalid confidence: {signal.get('confidence')}")
    rec = result.get("upgrade_recommendation", {})
    if rec.get("recommend_upgrade") and rec.get("to_level") not in (1, 2):
        issues.append("upgrade recommended but to_level is missing or invalid")
    return issues


# load and index both files by case_id 
features = {}
with open(FEATURES_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        features[r["case_id"]] = r

states = {}
with open(STATES_PATH, encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        states[s["case_id"]] = s

case_ids = list(features.keys())
print(f"loaded {len(case_ids)} records\n")

# make sure both files have the same cases
missing_states = [c for c in case_ids if c not in states]
if missing_states:
    print(f"WARNING: {len(missing_states)} case_ids in features but not in states")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_cases  = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for i, case_id in enumerate(case_ids):
        print(f"[{i+1:03d}/{len(case_ids)}] {case_id} ...", end=" ", flush=True)

        if case_id not in states:
            print("SKIP (no evidence state)")
            continue

        try:
            prompt = build_verification_prompt(features[case_id], states[case_id])
            raw    = call_api(prompt)
            result = parse_response(raw)
            issues = basic_validate(result)

            n_esi1 = len(result.get("missed_esi1_signals", []))
            n_esi2 = len(result.get("missed_esi2_signals", []))
            upgrade = result.get("upgrade_recommendation", {}).get("recommend_upgrade", False)

            record = {
                "case_id"              : case_id,
                "label"                : features[case_id]["label"],
                "source"               : features[case_id]["source"],
                "missed_esi1_signals"  : result.get("missed_esi1_signals", []),
                "missed_esi2_signals"  : result.get("missed_esi2_signals", []),
                "upgrade_recommendation": result.get("upgrade_recommendation", {}),
                "validation_issues"    : issues,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1

            status = f"esi1={n_esi1} esi2={n_esi2} upgrade={upgrade}"
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