# step4b_resources.py
import json
import time
from openai import OpenAI
from pathlib import Path

from classB import RESOURCE_ESTIMATION_SYSTEM_PROMPT, build_resource_prompt

STATES_PATH = "evidence_states.jsonl"
OUT_PATH    = "resource_estimates.jsonl"

VALID_TYPES = {
    "labs", "ecg_xray", "advanced_imaging", "iv_fluids",
    "iv_im_neb_meds", "specialty_consult", "simple_procedure", "complex_procedure"
}

from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


def call_api(user_prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=800,
                temperature=0,
                messages=[
                    {"role": "system", "content": RESOURCE_ESTIMATION_SYSTEM_PROMPT},
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


def validate(result):
    issues = []
    if result.get("resource_count") not in ("0", "1", "2+"):
        issues.append(f"invalid resource_count: {result.get('resource_count')}")
    for res in result.get("likely_resources", []):
        if res.get("resource_type") not in VALID_TYPES:
            issues.append(f"unknown resource_type: {res.get('resource_type')}")
        if not res.get("clinical_indication"):
            issues.append("missing clinical_indication")
    return issues


# ── load ──────────────────────────────────────────────────────────────────────
states = {}
with open(STATES_PATH, encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        states[s["case_id"]] = s

case_ids = list(states.keys())
print(f"loaded {len(case_ids)} states\n")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_cases  = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for i, case_id in enumerate(case_ids):
        print(f"[{i+1:03d}/{len(case_ids)}] {case_id} ...", end=" ", flush=True)

        try:
            prompt = build_resource_prompt(states[case_id])
            raw    = call_api(prompt)
            result = parse_response(raw)
            issues = validate(result)

            record = {
                "case_id"          : case_id,
                "label"            : states[case_id]["label"],
                "source"           : states[case_id]["source"],
                "likely_resources" : result.get("likely_resources", []),
                "resource_count"   : result.get("resource_count", "unknown"),
                "resource_reasoning": result.get("resource_reasoning", ""),
                "validation_issues": issues,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
            status = f"count={result.get('resource_count')} types={len(result.get('likely_resources',[]))}"
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