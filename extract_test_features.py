# extract_test_features.py
# 对每个test文件跑完整的 step1→step4 pipeline
# 在训练集上已经调好的逻辑直接复用

import json
import csv
import time
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI

from prompts import SYSTEM_PROMPT, build_user_prompt
from normalize import normalize_record
from build_features import build_row, FEATURE_COLS

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

TEST_FILES = [
    "Test-1.txt",
    "Test-2.txt",
    "Test-3.txt",
]

OUT_DIR = Path(".")

# ── step1: parse test TSV ─────────────────────────────────────────────────────

def parse_test_file(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "" or i == 0:
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        # test files may have fewer columns — adjust as needed
        vignette = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        esi_raw  = parts[-1].strip()
        try:
            label = int(esi_raw)
            assert label in {1, 2, 3, 4, 5}
        except (ValueError, AssertionError):
            continue
        if not vignette or vignette == "Clinical Vignettes":
            continue
        records.append({
            "case_id"   : f"test_{len(records)+1:03d}",
            "clean_text": vignette,
            "label"     : label,
            "source"    : "test",
        })
    return records

# ── step2: build empty evidence state ────────────────────────────────────────

VITAL_PATTERNS = {
    "heart_rate"  : r"\bHR\s*[:\s]?\s*(\d{2,3})\b",
    "resp_rate"   : r"\bRR\s*[:\s]?\s*(\d{1,2})\b",
    "spo2"        : r"\bSpO2?\s*[:\s]?\s*(\d{2,3})\s*%?",
    "systolic_bp" : r"\bBP\s*[:\s]?\s*(\d{2,3})\s*/\s*\d+",
    "diastolic_bp": r"\bBP\s*[:\s]?\s*\d+\s*/\s*(\d{2,3})",
    "temperature" : r"\bT(?:emp(?:erature)?)?\s*[:\s]?\s*([\d]{2,3}(?:\.\d)?)\s*°?\s*([FC])?\b",
}

def extract_vitals_simple(text):
    vitals = {}
    for name, pattern in VITAL_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        val = float(m.group(1))
        if name == "temperature":
            unit = m.group(2).upper() if m.group(2) else None
            if unit == "F" or val > 45:
                val = round((val - 32) * 5/9, 1)
        vitals[name] = val
    return vitals

def make_evidence_state(record):
    vitals  = extract_vitals_simple(record["clean_text"])
    missing = [k for k in ["heart_rate","resp_rate","spo2","systolic_bp","diastolic_bp","temperature"]
               if k not in vitals]
    return {
        "case_id"    : record["case_id"],
        "source"     : "test",
        "input_type" : "vignette",
        "label_system": "ESI",
        "label"      : record["label"],
        "clean_text" : record["clean_text"],
        "evidence_state": {
            "patient_reported": {
                "pain_score": None,
            },
            "nurse_observed": {
                "objective_vitals": {
                    "heart_rate"   : vitals.get("heart_rate"),
                    "resp_rate"    : vitals.get("resp_rate"),
                    "spo2"         : vitals.get("spo2"),
                    "systolic_bp"  : vitals.get("systolic_bp"),
                    "diastolic_bp" : vitals.get("diastolic_bp"),
                    "temperature"  : vitals.get("temperature"),
                },
            },
            "missing_information": missing,
        }
    }

# ── step3: LLM extraction ─────────────────────────────────────────────────────

def call_api(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                time.sleep(5)
            else:
                raise

def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    return json.loads(text)

# ── step4: build feature row ──────────────────────────────────────────────────
# reuses build_row from step4_build_features.py directly

# ── main: process each test file ─────────────────────────────────────────────

for test_path in TEST_FILES:
    test_name = Path(test_path).stem
    out_csv   = OUT_DIR / f"llm_features_{test_name}.csv"

    print(f"\nprocessing {test_name} ...")
    records = parse_test_file(test_path)
    print(f"  parsed {len(records)} records")

    rows = []
    for i, record in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {record['case_id']}", end=" ", flush=True)
        try:
            state    = make_evidence_state(record)
            state    = normalize_record(state)   # reuse training normalizer
            prompt   = build_user_prompt(state)
            raw      = call_api(prompt)
            features = parse_response(raw)

            state["extracted_features"] = features
            state["objective_vitals"]   = state["evidence_state"]["nurse_observed"]["objective_vitals"]

            row = build_row(state)
            rows.append(row)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(0.5)

    if not rows:
        print(f"  no rows for {test_name}, skipping")
        continue

    columns = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  saved {len(rows)} rows → {out_csv}")