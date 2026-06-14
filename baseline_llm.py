# step_baseline_llm.py
import json
import time
from openai import OpenAI
from pathlib import Path

STATES_PATH = "evidence_states.jsonl"
OUT_PATH    = "baseline_results.jsonl"

from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are an emergency triage nurse assigning ESI levels.

ESI levels:
  1 = Immediate lifesaving intervention required
  2 = High-risk situation, altered mental status, or severe pain/distress
  3 = Stable but requires 2 or more resources
  4 = Stable, requires 1 resource
  5 = Stable, requires no resources

Assign the single most appropriate ESI level (1-5).
Output pure JSON only, no explanation.
"""


def call_api(text, retries=2):
    user_prompt = f"""Assign an ESI level to this patient presentation:

{text}

Output format:
{{
  "predicted_esi": <integer 1-5>,
  "reasoning": "<one sentence>"
}}
"""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=200,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                print(f"    retry {attempt+1}...")
                time.sleep(5)
            else:
                raise


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1])
    return json.loads(text)


states = {}
with open(STATES_PATH, encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        states[s["case_id"]] = s

case_ids = list(states.keys())
print(f"loaded {len(case_ids)} cases\n")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_cases  = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for i, case_id in enumerate(case_ids):
        state = states[case_id]
        print(f"[{i+1:03d}/{len(case_ids)}] {case_id} ...", end=" ", flush=True)

        try:
            raw    = call_api(state["clean_text"])
            result = parse_response(raw)

            record = {
                "case_id"      : case_id,
                "label"        : state["label"],
                "source"       : state["source"],
                "predicted_esi": result.get("predicted_esi"),
                "reasoning"    : result.get("reasoning", ""),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
            print(f"ESI {result.get('predicted_esi')} label={state['label']}")

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