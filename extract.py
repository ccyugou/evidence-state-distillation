# step3_extract.py

import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

from prompts import SYSTEM_PROMPT, build_user_prompt

IN_PATH  = "evidence_states.jsonl"
OUT_PATH = "extracted_features.jsonl"

load_dotenv()

client = OpenAI()


def call_api(user_prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                temperature=0,   # deterministic output for extraction
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
        text = "\n".join(lines[1:-1])
    return json.loads(text)


# main loop
states = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        states.append(json.loads(line))

print(f"loaded {len(states)} records\n")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_cases  = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for i, state in enumerate(states):
        case_id = state["case_id"]
        print(f"[{i+1:03d}/{len(states)}] {case_id} ...", end=" ", flush=True)

        try:
            prompt   = build_user_prompt(state)
            raw      = call_api(prompt)
            features = parse_response(raw)

            record = {
                "case_id"           : case_id,
                "label"             : state["label"],
                "source"            : state["source"],
                "objective_vitals"  : state["evidence_state"]["nurse_observed"]["objective_vitals"],
                "extracted_features": features,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
            print("ok")

        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
            failed_cases.append(case_id)

        except Exception as e:
            print(f"FAILED: {e}")
            failed_cases.append(case_id)

        time.sleep(0.5)

print(f"\ndone: {success_count} ok, {len(failed_cases)} failed")
if failed_cases:
    print(f"failed cases: {failed_cases}")
print(f"saved to {OUT_PATH}")