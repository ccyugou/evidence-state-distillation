#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_scan_dataset_extended.py

Raw SAFE-Triage / TRIBOT dataset audit.

Purpose
-------
This script is intentionally placed before 01_timelines. It audits the raw JSON
transcripts BEFORE any downstream agent sees them, with special focus on:

1) unique identity binding:
   instance_id = case_id__run_uuid
   dataset_instance_id = dataset__case_id__run_uuid

2) possible label leakage / noisy leakage:
   - vignette.acuity
   - history[].triage
   - specialisation

3) ground-truth sanity:
   - ground_truth.acuity unique values and distribution
   - ground_truth.pain distribution and anomalies
   - vignette vs ground_truth consistency

4) clinical raw-field statistics useful for later agent design:
   - vitals distribution and abnormal/danger-zone flags
   - chief complaint / arrival transport / specialisation by acuity
   - patient_persona and nurse_persona distributions by acuity
   - history length / actor turn counts
   - scale-question audit to avoid treating fatigue/distress scores as pain

No LLM calls. No external dependencies. Stdlib only.

Default paths follow the existing project convention:
  RAW_DATA_DIR = transcripts/mimic
  OUTPUT_DIR   = outputs/00_dataset_scan

It also writes a compatibility copy of the main summary to:
  outputs/scan_summary.json
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clinical_data_contract import (
    PAIN_MAX,
    PAIN_MIN,
    VITAL_CANONICAL_UNITS,
    VITAL_NAME_ALIASES,
    VITAL_PLAUSIBILITY_RANGES,
    classify_pain_contract,
    classify_vital_contract,
    is_scale_question_text,
)


DEFAULT_RAW_DATA_DIR = "transcripts/mimic"
DEFAULT_OUTPUT_DIR = "outputs/00_dataset_scan"
LEGACY_SUMMARY_FILE = "outputs/scan_summary.json"
AUDIT_SCHEMA_VERSION = "00_raw_dataset_audit_v2.1_audit_hardening"

REQUIRED_TOP_LEVEL_KEYS = [
    "dataset",
    "case_id",
    "run_uuid",
    "vignette",
    "history",
    "ground_truth",
    "patient_persona",
    "nurse_persona",
]

ALLOWED_ACUITY_VALUES = {1, 2, 3, 4, 5}

# Lightweight adult-oriented audit thresholds. These are NOT classification rules.
# They are used only to understand raw dataset structure before agent debugging.
VITAL_AUDIT_THRESHOLDS = {
    "temperature": {
        "low_abnormal": 96.0,
        "high_abnormal": 100.4,
        "low_danger": 95.0,
        "high_danger": 103.0,
    },
    "heartrate": {
        "low_abnormal": 50.0,
        "high_abnormal": 100.0,
        "low_danger": 40.0,
        "high_danger": 130.0,
    },
    "resprate": {
        "low_abnormal": 10.0,
        "high_abnormal": 20.0,
        "low_danger": 8.0,
        "high_danger": 30.0,
    },
    "o2sat": {
        "low_abnormal": 95.0,
        "high_abnormal": None,
        "low_danger": 92.0,
        "high_danger": None,
    },
    "sbp": {
        "low_abnormal": 90.0,
        "high_abnormal": 180.0,
        "low_danger": 90.0,
        "high_danger": 200.0,
    },
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

NURSE_QUESTION_RE = re.compile(
    r"\?|^\s*(?:any|are|is|was|were|do|does|did|have|has|had|can|could|would|will|"
    r"how|what|when|where|why|which|tell me|describe|on a scale|scale)\b",
    re.IGNORECASE,
)

PAIN_CONTEXT_HINTS = ["pain", "ache", "hurt", "hurts", "sore", "discomfort"]
NON_PAIN_SCALE_HINTS = [
    "fatigue", "distress", "tired", "drained", "weakness", "breathlessness",
    "shortness of breath", "breathing difficulty", "anxiety", "nausea",
    "dizziness", "dizzy", "discomfort level", "symptom severity",
]

RAW_LABEL_LIKE_PATTERNS = [
    ("explicit_esi_level", re.compile(r"\besi\s*[- ]?[1-5]\b", re.IGNORECASE)),
    ("explicit_triage_level", re.compile(r"\b(?:triage|acuity)\s*(?:level|category)\s*[:#-]?\s*[1-5]\b", re.IGNORECASE)),
    ("explicit_level_category", re.compile(r"\b(?:level|category)\s*[1-5]\s*(?:triage|acuity)\b", re.IGNORECASE)),
    ("explicit_high_low_acuity", re.compile(r"\b(?:high|low)\s+acuity\b", re.IGNORECASE)),
]
RAW_WORKFLOW_MARKER_RE = re.compile(r"\b(?:triage nurse|triage area|triage desk|triage room)\b", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_key(value: Any) -> str:
    if value is None:
        return "<MISSING>"
    return str(value)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_int_like(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        f = float(text)
    except Exception:
        return None
    if math.isnan(f):
        return None
    if f.is_integer():
        return int(f)
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f):
        return None
    return f


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: (str(kv[0]), kv[1]))}


def counter_to_rows(counter: Counter, key_name: str = "value") -> List[Dict[str, Any]]:
    rows = []
    for k, v in counter.most_common():
        rows.append({key_name: k, "count": int(v)})
    return rows


def nested_counter_to_rows(nested: Dict[Any, Counter], outer_name: str, inner_name: str) -> List[Dict[str, Any]]:
    rows = []
    for outer in sorted(nested.keys(), key=lambda x: str(x)):
        for inner, count in nested[outer].most_common():
            rows.append({outer_name: outer, inner_name: inner, "count": int(count)})
    return rows


def describe_numeric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": round(mean(sorted_values), 4),
        "median": round(median(sorted_values), 4),
    }


def list_json_files(base_dir: str) -> List[str]:
    files = []
    for root, _dirs, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def load_json(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "top_level_not_object"
        return data, None
    except Exception as e:
        return None, str(e)


def make_instance_id(case: Dict[str, Any]) -> str:
    case_id = normalize_key(case.get("case_id"))
    run_uuid = normalize_key(case.get("run_uuid"))
    return f"{case_id}__{run_uuid}"


def make_dataset_instance_id(case: Dict[str, Any]) -> str:
    dataset = normalize_key(case.get("dataset"))
    return f"{dataset}__{make_instance_id(case)}"


def get_gt_acuity(case: Dict[str, Any]) -> Optional[int]:
    return to_int_like(case.get("ground_truth", {}).get("acuity"))


def get_vignette_acuity(case: Dict[str, Any]) -> Optional[int]:
    return to_int_like(case.get("vignette", {}).get("acuity"))


def get_gt_pain(case: Dict[str, Any]) -> Optional[float]:
    return to_float(case.get("ground_truth", {}).get("pain"))


def get_vignette_pain(case: Dict[str, Any]) -> Optional[float]:
    return to_float(case.get("vignette", {}).get("pain"))


def classify_pain_value(value: Any) -> Dict[str, Any]:
    result = classify_pain_contract(value)
    result["numeric_value"] = to_float(value)
    result["quarantine_reason"] = (
        None if result.get("usable_for_clinical_reasoning") else result.get("quarantine_reason")
    )
    return result


def canonical_vital_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return VITAL_NAME_ALIASES.get(normalized)


def classify_vital_measurement(name: str, value: Any) -> Dict[str, Any]:
    result = classify_vital_contract(name, value)
    result["numeric_value"] = to_float(value)
    return result


def classify_raw_text_label_mentions(text: Any) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []
    mentions = []
    workflow_only = bool(RAW_WORKFLOW_MARKER_RE.search(text))
    for label_type, pattern in RAW_LABEL_LIKE_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append({
                "label_type": label_type,
                "matched_text": match.group(0),
                "start_char": match.start(),
                "end_char": match.end(),
                "workflow_context_also_present": workflow_only,
                "requires_manual_leakage_review": True,
            })
    return mentions


def audit_system_vitals(case: Dict[str, Any], vignette: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    history = case.get("history") if isinstance(case.get("history"), list) else []
    by_field: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for history_index, item in enumerate(history):
        if not isinstance(item, dict) or item.get("actor") != "system" or item.get("event") != "vital":
            continue
        field = canonical_vital_name(item.get("name"))
        if not field:
            continue
        validity = classify_vital_measurement(field, item.get("value"))
        row = {
            "instance_id": make_instance_id(case),
            "case_id": normalize_key(case.get("case_id")),
            "run_uuid": normalize_key(case.get("run_uuid")),
            "history_index": history_index,
            "turn": item.get("turn"),
            "field": field,
            "raw_name": item.get("name"),
            "raw_value": item.get("value"),
            **validity,
        }
        by_field[field].append(row)
        rows.append(row)

    field_summary = {}
    for field, field_rows in sorted(by_field.items()):
        values = [row["numeric_value"] for row in field_rows if row["numeric_value"] is not None]
        distinct_values = sorted(set(values))
        vignette_value = to_float(vignette.get(field))
        latest = field_rows[-1] if field_rows else None
        field_summary[field] = {
            "event_count": len(field_rows),
            "distinct_numeric_values": distinct_values,
            "same_value_duplicate_count": max(0, len(values) - len(distinct_values)),
            "multiple_distinct_values": len(distinct_values) > 1,
            "latest_event": latest,
            "vignette_value": vignette_value,
            "latest_matches_vignette": (
                latest is not None and latest.get("numeric_value") == vignette_value
                if latest is not None and vignette_value is not None else None
            ),
        }
    return {
        "system_vital_event_count": len(rows),
        "fields": field_summary,
        "latest_value_policy": "history_order_last_event_is_latest_for_audit_only",
    }, rows


def extract_history_triage_values(history: Any) -> List[int]:
    values = []
    if not isinstance(history, list):
        return values
    for ev in history:
        if not isinstance(ev, dict):
            continue
        if "triage" not in ev:
            continue
        v = to_int_like(ev.get("triage"))
        if v is not None:
            values.append(v)
    return values


def mode_from_values(values: List[int]) -> Optional[int]:
    if not values:
        return None
    c = Counter(values)
    # deterministic tie-break: larger count first, then smaller acuity value
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def get_history_text_stats(history: Any) -> Dict[str, int]:
    stats = {
        "history_events": 0,
        "patient_turns": 0,
        "nurse_turns": 0,
        "system_events": 0,
        "system_vital_events": 0,
        "history_triage_entries": 0,
    }
    if not isinstance(history, list):
        return stats
    stats["history_events"] = len(history)
    for ev in history:
        if not isinstance(ev, dict):
            continue
        actor = ev.get("actor")
        if actor == "patient":
            stats["patient_turns"] += 1
        elif actor == "nurse":
            stats["nurse_turns"] += 1
        elif actor == "system":
            stats["system_events"] += 1
            if ev.get("event") == "vital":
                stats["system_vital_events"] += 1
        if "triage" in ev:
            stats["history_triage_entries"] += 1
    return stats


def classify_vital(name: str, value: Any) -> Tuple[bool, bool, Optional[float]]:
    """Return (is_abnormal, is_danger_zone, numeric_value)."""
    x = to_float(value)
    if x is None:
        return False, False, None
    cfg = VITAL_AUDIT_THRESHOLDS.get(name)
    if not cfg:
        return False, False, x

    abnormal = False
    danger = False

    low_abn = cfg.get("low_abnormal")
    high_abn = cfg.get("high_abnormal")
    low_danger = cfg.get("low_danger")
    high_danger = cfg.get("high_danger")

    if low_abn is not None and x < low_abn:
        abnormal = True
    if high_abn is not None and x > high_abn:
        abnormal = True
    if low_danger is not None and x < low_danger:
        danger = True
    if high_danger is not None and x > high_danger:
        danger = True

    return abnormal, danger, x


def iter_vitals(case: Dict[str, Any]) -> Iterable[Tuple[str, Any, str]]:
    """Yield (name, value, source) for ground_truth and vignette vitals."""
    gt_vitals = case.get("ground_truth", {}).get("vitals", {})
    if isinstance(gt_vitals, dict):
        for name, value in gt_vitals.items():
            yield name, value, "ground_truth"
    vignette = case.get("vignette", {})
    if isinstance(vignette, dict):
        for name in ["temperature", "heartrate", "resprate", "o2sat", "sbp"]:
            if name in vignette:
                yield name, vignette.get(name), "vignette"


def compare_vignette_gt_vitals(case: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    vignette = case.get("vignette", {}) if isinstance(case.get("vignette"), dict) else {}
    gt_vitals = case.get("ground_truth", {}).get("vitals", {})
    if not isinstance(gt_vitals, dict):
        gt_vitals = {}
    for name in ["temperature", "heartrate", "resprate", "o2sat", "sbp"]:
        v1 = to_float(vignette.get(name))
        v2 = to_float(gt_vitals.get(name))
        result[name] = {
            "vignette": v1,
            "ground_truth": v2,
            "matches": (v1 == v2) if (v1 is not None and v2 is not None) else None,
        }
    return result


def clean_chiefcomplaint(text: Any) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_chiefcomplaint(text: str) -> List[str]:
    stop = {
        "with", "and", "or", "of", "the", "a", "an", "to", "for", "in", "on",
        "from", "due", "after", "by", "at", "x", "s/p", "w", "c/o",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return [t for t in tokens if t not in stop and len(t) >= 3]


def previous_nurse_question(history: List[Dict[str, Any]], idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        ev = history[j]
        if isinstance(ev, dict) and ev.get("actor") == "nurse":
            text = normalize_text(ev.get("original") or ev.get("utterance"))
            if NURSE_QUESTION_RE.search(text):
                return text
    return ""


def extract_numbers_0_to_10(text: str) -> List[int]:
    text_lower = text.lower()
    found: List[int] = []

    for m in re.finditer(r"\b(10|[0-9])\b", text_lower):
        try:
            value = int(m.group(1))
        except Exception:
            continue
        if 0 <= value <= 10:
            found.append(value)

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            found.append(value)

    # Preserve unique values in sorted order for compact audit output.
    return sorted(set(found))


def classify_scale_question(question: str) -> str:
    q = question.lower()
    has_scale_hint = is_scale_question_text(q)
    if not has_scale_hint:
        return "not_scale_question"
    has_pain = any(h in q for h in PAIN_CONTEXT_HINTS)
    has_non_pain = any(h in q for h in NON_PAIN_SCALE_HINTS)
    if has_pain and has_non_pain:
        return "multi_target_scale"
    if has_pain:
        return "pain_scale"
    if has_non_pain:
        return "non_pain_scale"
    return "ambiguous_scale"


def audit_scale_mentions(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = case.get("history")
    if not isinstance(history, list):
        return []

    rows = []
    instance_id = make_instance_id(case)
    case_id = normalize_key(case.get("case_id"))
    run_uuid = normalize_key(case.get("run_uuid"))
    acuity = get_gt_acuity(case)
    gt_pain = get_gt_pain(case)

    for i, ev in enumerate(history):
        if not isinstance(ev, dict) or ev.get("actor") != "patient":
            continue
        patient_text = normalize_text(ev.get("original") or ev.get("utterance"))
        nums = extract_numbers_0_to_10(patient_text)
        if not nums:
            continue
        nurse_q = previous_nurse_question(history, i)
        qtype = classify_scale_question(nurse_q)
        if qtype == "not_scale_question":
            continue
        rows.append({
            "instance_id": instance_id,
            "case_id": case_id,
            "run_uuid": run_uuid,
            "ground_truth_acuity": acuity,
            "ground_truth_pain": gt_pain,
            "turn": ev.get("turn"),
            "question_type": qtype,
            "numeric_values_0_to_10": nums,
            "nurse_question": nurse_q,
            "patient_answer": patient_text,
        })
    return rows


def flatten_distribution_by_acuity(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, int]]:
    nested: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        acuity = normalize_key(row.get("ground_truth_acuity"))
        value = normalize_key(row.get(field))
        nested[acuity][value] += 1
    return {str(a): counter_to_sorted_dict(c) for a, c in nested.items()}


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload.setdefault("audit_only", True)
            payload.setdefault("allowed_as_prediction_input", False)
            payload.setdefault("audit_schema_version", AUDIT_SCHEMA_VERSION)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit_dataset(
    raw_data_dir: str,
    output_dir: str,
    limit: Optional[int] = None,
    verbose: bool = True,
    write_legacy_summary: bool = False,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    examples_dir = os.path.join(output_dir, "examples")
    tables_dir = os.path.join(output_dir, "tables")
    ensure_dir(examples_dir)
    ensure_dir(tables_dir)

    json_files = list_json_files(raw_data_dir)
    if limit is not None:
        json_files = json_files[:limit]

    load_errors: List[Dict[str, Any]] = []
    missing_field_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    acuity_mismatch_rows: List[Dict[str, Any]] = []
    history_triage_issue_rows: List[Dict[str, Any]] = []
    pain_anomaly_rows: List[Dict[str, Any]] = []
    vignette_pain_anomaly_rows: List[Dict[str, Any]] = []
    pain_mismatch_rows: List[Dict[str, Any]] = []
    vital_anomaly_rows: List[Dict[str, Any]] = []
    vital_contract_issue_rows: List[Dict[str, Any]] = []
    vital_mismatch_rows: List[Dict[str, Any]] = []
    scale_mention_rows: List[Dict[str, Any]] = []
    system_vital_audit_rows: List[Dict[str, Any]] = []
    raw_label_like_rows: List[Dict[str, Any]] = []

    dataset_counter = Counter()
    case_id_counter = Counter()
    run_uuid_counter = Counter()
    instance_id_counter = Counter()
    dataset_instance_id_counter = Counter()

    acuity_counter = Counter()
    vignette_acuity_counter = Counter()
    pain_counter = Counter()
    pain_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    vignette_pain_counter = Counter()
    vignette_pain_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    arrival_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    gender_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    specialisation_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    chief_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    chief_token_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    persona_patient_by_field: Dict[str, Counter] = defaultdict(Counter)
    persona_nurse_by_field: Dict[str, Counter] = defaultdict(Counter)
    persona_patient_by_acuity_field: Dict[str, Counter] = defaultdict(Counter)
    persona_nurse_by_acuity_field: Dict[str, Counter] = defaultdict(Counter)
    history_stat_values: Dict[str, List[float]] = defaultdict(list)
    history_stat_by_acuity: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    history_triage_value_counter = Counter()
    history_triage_unique_count_counter = Counter()
    history_triage_mode_by_gt: Dict[str, Counter] = defaultdict(Counter)

    vital_numeric_values: Dict[str, List[float]] = defaultdict(list)
    vital_abnormal_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    vital_danger_by_acuity: Dict[str, Counter] = defaultdict(Counter)
    case_any_abnormal_by_acuity = Counter()
    case_any_danger_by_acuity = Counter()

    loaded = 0

    for i, file_path in enumerate(json_files, start=1):
        if verbose and (i == 1 or i % 100 == 0):
            print(f"Loading {i}/{len(json_files)}: {file_path}")

        case, err = load_json(file_path)
        if err or case is None:
            load_errors.append({"file_path": file_path, "error": err})
            continue

        loaded += 1
        missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in case]
        if missing:
            missing_field_rows.append({
                "file_path": file_path,
                "missing_fields": missing,
            })

        dataset = normalize_key(case.get("dataset"))
        case_id = normalize_key(case.get("case_id"))
        run_uuid = normalize_key(case.get("run_uuid"))
        instance_id = make_instance_id(case)
        dataset_instance_id = make_dataset_instance_id(case)

        dataset_counter[dataset] += 1
        case_id_counter[case_id] += 1
        run_uuid_counter[run_uuid] += 1
        instance_id_counter[instance_id] += 1
        dataset_instance_id_counter[dataset_instance_id] += 1

        gt_acuity = get_gt_acuity(case)
        vignette_acuity = get_vignette_acuity(case)
        gt_pain = get_gt_pain(case)
        vignette_pain = get_vignette_pain(case)
        gt_acuity_key = normalize_key(gt_acuity)

        acuity_counter[gt_acuity_key] += 1
        vignette_acuity_counter[normalize_key(vignette_acuity)] += 1
        pain_counter[normalize_key(gt_pain)] += 1
        pain_by_acuity[gt_acuity_key][normalize_key(gt_pain)] += 1
        vignette_pain_counter[normalize_key(vignette_pain)] += 1
        vignette_pain_by_acuity[gt_acuity_key][normalize_key(vignette_pain)] += 1

        if gt_acuity not in ALLOWED_ACUITY_VALUES:
            acuity_mismatch_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "issue": "ground_truth_acuity_not_in_1_to_5",
                "ground_truth_acuity": gt_acuity,
                "vignette_acuity": vignette_acuity,
            })

        if vignette_acuity != gt_acuity:
            acuity_mismatch_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "issue": "vignette_acuity_mismatch_ground_truth",
                "ground_truth_acuity": gt_acuity,
                "vignette_acuity": vignette_acuity,
            })

        gt_pain_validity = classify_pain_value(gt_pain)
        vignette_pain_validity = classify_pain_value(vignette_pain)
        if not gt_pain_validity["usable_for_clinical_reasoning"]:
            pain_anomaly_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "ground_truth_acuity": gt_acuity,
                "ground_truth_pain": gt_pain,
                "issue": "pain_missing_or_out_of_range_0_to_10",
                "value_validity": gt_pain_validity["value_validity"],
                "quarantine_reason": gt_pain_validity["quarantine_reason"],
            })

        vignette = case.get("vignette", {}) if isinstance(case.get("vignette"), dict) else {}
        if not vignette_pain_validity["usable_for_clinical_reasoning"]:
            vignette_pain_anomaly_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "ground_truth_acuity": gt_acuity,
                "vignette_pain": vignette_pain,
                "issue": "vignette_pain_missing_or_out_of_range_0_to_10",
                "value_validity": vignette_pain_validity["value_validity"],
                "quarantine_reason": vignette_pain_validity["quarantine_reason"],
            })
        if gt_pain != vignette_pain:
            pain_mismatch_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "ground_truth_acuity": gt_acuity,
                "ground_truth_pain": gt_pain,
                "vignette_pain": vignette_pain,
                "issue": "vignette_pain_mismatch_ground_truth_pain",
            })
        chief = clean_chiefcomplaint(vignette.get("chiefcomplaint") or case.get("ground_truth", {}).get("chiefcomplaint"))
        arrival = normalize_key(vignette.get("arrival_transport"))
        gender = normalize_key(vignette.get("gender"))
        spec = normalize_key(vignette.get("specialisation"))

        chief_by_acuity[gt_acuity_key][chief or "<EMPTY>"] += 1
        for tok in tokenize_chiefcomplaint(chief):
            chief_token_by_acuity[gt_acuity_key][tok] += 1
        arrival_by_acuity[gt_acuity_key][arrival] += 1
        gender_by_acuity[gt_acuity_key][gender] += 1
        specialisation_by_acuity[gt_acuity_key][spec] += 1

        history = case.get("history")
        history_stats = get_history_text_stats(history)
        for k, v in history_stats.items():
            history_stat_values[k].append(float(v))
            history_stat_by_acuity[gt_acuity_key][k].append(float(v))

        triage_values = extract_history_triage_values(history)
        triage_counter = Counter(triage_values)
        triage_unique_values = sorted(set(triage_values))
        triage_mode = mode_from_values(triage_values)
        history_triage_unique_count_counter[len(triage_unique_values)] += 1
        for v in triage_values:
            history_triage_value_counter[v] += 1
        if triage_mode is not None:
            history_triage_mode_by_gt[gt_acuity_key][triage_mode] += 1

        triage_issue_flags = []
        if triage_values:
            if len(triage_unique_values) > 1:
                triage_issue_flags.append("multiple_history_triage_values")
            if gt_acuity is not None and triage_mode != gt_acuity:
                triage_issue_flags.append("history_triage_mode_mismatch_gt_acuity")
            if gt_acuity is not None and gt_acuity != 3 and 3 in triage_unique_values:
                triage_issue_flags.append("history_triage_contains_3_in_non_gt3_case")
            if any(v not in ALLOWED_ACUITY_VALUES for v in triage_unique_values):
                triage_issue_flags.append("history_triage_value_not_in_1_to_5")
        if triage_issue_flags:
            history_triage_issue_rows.append({
                "instance_id": instance_id,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "file_path": file_path,
                "ground_truth_acuity": gt_acuity,
                "vignette_acuity": vignette_acuity,
                "history_triage_values": triage_values,
                "history_triage_unique_values": triage_unique_values,
                "history_triage_mode": triage_mode,
                "issue_flags": triage_issue_flags,
            })

        # Vitals audit: vignette is the prediction-facing primary source; ground_truth
        # is retained only for consistency checks and audit-only comparisons.
        gt_vitals = case.get("ground_truth", {}).get("vitals", {})
        if not isinstance(gt_vitals, dict):
            gt_vitals = {}

        any_case_abnormal = False
        any_case_danger = False
        primary_vital_validity = {}
        for vital_name in ["temperature", "heartrate", "resprate", "o2sat", "sbp"]:
            value = vignette.get(vital_name)
            validity = classify_vital_measurement(vital_name, value)
            primary_vital_validity[vital_name] = validity
            numeric = validity["numeric_value"]
            if validity["value_plausibility"] != "plausible":
                vital_contract_issue_rows.append({
                    "instance_id": instance_id,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "file_path": file_path,
                    "source": "vignette",
                    "vital_name": vital_name,
                    **validity,
                    "audit_only": True,
                    "allowed_as_prediction_input": False,
                })
                vital_anomaly_rows.append({
                    "instance_id": instance_id,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "file_path": file_path,
                    "ground_truth_acuity": gt_acuity,
                    "source": "vignette",
                    "vital_name": vital_name,
                    "value": numeric,
                    "abnormal": False,
                    "danger_zone": False,
                    "issue": "implausible_data_value",
                    "value_plausibility": validity["value_plausibility"],
                    "quarantine_reason": validity["quarantine_reason"],
                })
                continue
            abnormal, danger, numeric = classify_vital(vital_name, value)
            if numeric is not None:
                vital_numeric_values[vital_name].append(numeric)
            if abnormal:
                any_case_abnormal = True
                vital_abnormal_by_acuity[gt_acuity_key][vital_name] += 1
            if danger:
                any_case_danger = True
                vital_danger_by_acuity[gt_acuity_key][vital_name] += 1
            if abnormal or danger:
                vital_anomaly_rows.append({
                    "instance_id": instance_id,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "file_path": file_path,
                    "ground_truth_acuity": gt_acuity,
                    "source": "vignette",
                    "vital_name": vital_name,
                    "value": numeric,
                    "abnormal": abnormal,
                    "danger_zone": danger,
                    "issue": "clinically_abnormal_or_danger_zone",
                    "value_plausibility": validity["value_plausibility"],
                })


        for vital_name in ["temperature", "heartrate", "resprate", "o2sat", "sbp"]:
            gt_validity = classify_vital_measurement(vital_name, gt_vitals.get(vital_name))
            if gt_validity["value_plausibility"] != "plausible":
                vital_contract_issue_rows.append({
                    "instance_id": instance_id,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "file_path": file_path,
                    "source": "ground_truth",
                    "vital_name": vital_name,
                    **gt_validity,
                    "audit_only": True,
                    "allowed_as_prediction_input": False,
                })

        if any_case_abnormal:
            case_any_abnormal_by_acuity[gt_acuity_key] += 1
        if any_case_danger:
            case_any_danger_by_acuity[gt_acuity_key] += 1

        vital_cmp = compare_vignette_gt_vitals(case)
        for vital_name, cmp_row in vital_cmp.items():
            if cmp_row["matches"] is False:
                vital_mismatch_rows.append({
                    "instance_id": instance_id,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "file_path": file_path,
                    "ground_truth_acuity": gt_acuity,
                    "vital_name": vital_name,
                    "vignette_value": cmp_row["vignette"],
                    "ground_truth_value": cmp_row["ground_truth"],
                })

        system_vital_summary, system_vital_rows = audit_system_vitals(case, vignette)
        system_vital_audit_rows.extend(system_vital_rows)

        history = case.get("history") if isinstance(case.get("history"), list) else []
        for history_index, item in enumerate(history):
            if not isinstance(item, dict) or item.get("actor") not in {"patient", "nurse"}:
                continue
            original_text = normalize_text(item.get("original"))
            utterance_text = normalize_text(item.get("utterance"))
            text_sources = []
            if original_text:
                text_sources.append(("original", original_text))
            if utterance_text and utterance_text != original_text:
                text_sources.append(("utterance", utterance_text))
            elif not original_text and utterance_text:
                text_sources.append(("utterance", utterance_text))
            for source_text_field, text in text_sources:
                for mention in classify_raw_text_label_mentions(text):
                    raw_label_like_rows.append({
                        "instance_id": instance_id,
                        "case_id": case_id,
                        "run_uuid": run_uuid,
                        "file_path": file_path,
                        "history_index": history_index,
                        "turn": item.get("turn"),
                        "actor": item.get("actor"),
                        "source_text_field": source_text_field,
                        "text": text,
                        **mention,
                        "audit_only": True,
                        "allowed_as_prediction_input": False,
                    })

        patient_persona = case.get("patient_persona", {})
        if isinstance(patient_persona, dict):
            for field, value in patient_persona.items():
                if field == "instruction":
                    continue
                v = normalize_key(value)
                persona_patient_by_field[field][v] += 1
                persona_patient_by_acuity_field[f"acuity={gt_acuity_key} | {field}"][v] += 1

        nurse_persona = case.get("nurse_persona", {})
        if isinstance(nurse_persona, dict):
            for field, value in nurse_persona.items():
                if field == "instruction":
                    continue
                v = normalize_key(value)
                persona_nurse_by_field[field][v] += 1
                persona_nurse_by_acuity_field[f"acuity={gt_acuity_key} | {field}"][v] += 1

        scale_rows = audit_scale_mentions(case)
        scale_mention_rows.extend(scale_rows)

        case_rows.append({
            "file_path": file_path,
            "dataset": dataset,
            "case_id": case_id,
            "run_uuid": run_uuid,
            "instance_id": instance_id,
            "dataset_instance_id": dataset_instance_id,
            "ground_truth_acuity": gt_acuity,
            "vignette_acuity": vignette_acuity,
            "ground_truth_pain": gt_pain,
            "vignette_pain": vignette_pain,
            "ground_truth_pain_validity": gt_pain_validity,
            "vignette_pain_validity": vignette_pain_validity,
            "chiefcomplaint": chief,
            "arrival_transport": arrival,
            "gender": gender,
            "specialisation_analysis_only": spec,
            "history_events": history_stats["history_events"],
            "patient_turns": history_stats["patient_turns"],
            "nurse_turns": history_stats["nurse_turns"],
            "system_events": history_stats["system_events"],
            "system_vital_events": history_stats["system_vital_events"],
            "history_triage_entries": history_stats["history_triage_entries"],
            "history_triage_unique_values": triage_unique_values,
            "history_triage_mode": triage_mode,
            "history_triage_mode_matches_gt": (triage_mode == gt_acuity) if triage_mode is not None else None,
            "has_any_abnormal_vital": any_case_abnormal,
            "has_any_danger_zone_vital": any_case_danger,
            "vignette_vital_validity": primary_vital_validity,
            "system_vital_audit": system_vital_summary,
        })

    duplicate_case_rows = [
        {"case_id": k, "count": int(v)}
        for k, v in case_id_counter.items()
        if v > 1
    ]
    duplicate_instance_rows = [
        {"instance_id": k, "count": int(v)}
        for k, v in instance_id_counter.items()
        if v > 1
    ]
    duplicate_dataset_instance_rows = [
        {"dataset_instance_id": k, "count": int(v)}
        for k, v in dataset_instance_id_counter.items()
        if v > 1
    ]

    # Scale question aggregation.
    scale_qtype_counter = Counter(row["question_type"] for row in scale_mention_rows)
    scale_pain0_non_pain_rows = [
        row for row in scale_mention_rows
        if row.get("ground_truth_pain") == 0 and row.get("question_type") in {"non_pain_scale", "multi_target_scale", "ambiguous_scale"}
    ]
    scale_pain0_high_numeric_rows = [
        row for row in scale_mention_rows
        if row.get("ground_truth_pain") == 0 and any(v >= 7 for v in row.get("numeric_values_0_to_10", []))
    ]

    # Case-level GT2/GT1 with normal vitals and GT4/GT5 with abnormal vitals.
    gt_high_normal_vitals_rows = [
        row for row in case_rows
        if row.get("ground_truth_acuity") in {1, 2} and not row.get("has_any_abnormal_vital")
    ]
    gt_low_abnormal_vitals_rows = [
        row for row in case_rows
        if row.get("ground_truth_acuity") in {4, 5} and row.get("has_any_abnormal_vital")
    ]

    id_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_counts": counter_to_sorted_dict(dataset_counter),
        "case_id_unique": len(case_id_counter),
        "case_id_duplicate_groups": len(duplicate_case_rows),
        "case_id_duplicate_total_extra_rows": int(sum(v - 1 for v in case_id_counter.values() if v > 1)),
        "run_uuid_unique": len(run_uuid_counter),
        "run_uuid_duplicate_groups": int(sum(1 for v in run_uuid_counter.values() if v > 1)),
        "instance_id_definition": "case_id__run_uuid",
        "instance_id_unique": len(instance_id_counter),
        "instance_id_duplicate_groups": len(duplicate_instance_rows),
        "dataset_instance_id_definition": "dataset__case_id__run_uuid",
        "dataset_instance_id_unique": len(dataset_instance_id_counter),
        "dataset_instance_id_duplicate_groups": len(duplicate_dataset_instance_rows),
    }

    raw_numeric_pain_ge_7_by_acuity = {
        str(ac): int(sum(
            count for pain, count in c.items()
            if to_float(pain) is not None and to_float(pain) >= 7
        ))
        for ac, c in pain_by_acuity.items()
    }
    valid_pain_7_to_10_by_acuity = {
        str(ac): int(sum(
            count for pain, count in c.items()
            if to_float(pain) is not None
            and 0.0 <= to_float(pain) <= 10.0
            and to_float(pain) >= 7
        ))
        for ac, c in pain_by_acuity.items()
    }
    label_pain_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "ground_truth_acuity_counts": counter_to_sorted_dict(acuity_counter),
        "ground_truth_acuity_unique_values": sorted([k for k in acuity_counter.keys()], key=lambda x: str(x)),
        "allowed_acuity_values": sorted(ALLOWED_ACUITY_VALUES),
        "vignette_acuity_counts": counter_to_sorted_dict(vignette_acuity_counter),
        "vignette_vs_ground_truth_acuity_mismatch_count": len([
            r for r in acuity_mismatch_rows if r.get("issue") == "vignette_acuity_mismatch_ground_truth"
        ]),
        "invalid_ground_truth_acuity_count": len([
            r for r in acuity_mismatch_rows if r.get("issue") == "ground_truth_acuity_not_in_1_to_5"
        ]),
        "ground_truth_pain_counts": counter_to_sorted_dict(pain_counter),
        "vignette_pain_counts": counter_to_sorted_dict(vignette_pain_counter),
        "ground_truth_pain_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in pain_by_acuity.items()},
        "vignette_pain_by_ground_truth_acuity": {str(k): counter_to_sorted_dict(v) for k, v in vignette_pain_by_acuity.items()},
        "pain_missing_or_out_of_range_0_to_10_count": len(pain_anomaly_rows),
        "vignette_pain_missing_or_out_of_range_0_to_10_count": len(vignette_pain_anomaly_rows),
        "vignette_vs_ground_truth_pain_mismatch_count": len(pain_mismatch_rows),
        "raw_numeric_pain_ge_7_by_acuity": raw_numeric_pain_ge_7_by_acuity,
        "valid_pain_7_to_10_by_acuity": valid_pain_7_to_10_by_acuity,
        "valid_pain_7_to_10_total": int(sum(valid_pain_7_to_10_by_acuity.values())),
        "pain_ge_7_by_acuity": valid_pain_7_to_10_by_acuity,
        "scale_question_type_counts": counter_to_sorted_dict(scale_qtype_counter),
        "pain0_non_pain_or_ambiguous_scale_mentions_count": len(scale_pain0_non_pain_rows),
        "pain0_with_scale_numeric_ge7_count": len(scale_pain0_high_numeric_rows),
    }

    history_triage_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "cases_with_any_history_triage": int(sum(1 for row in case_rows if row.get("history_triage_entries", 0) > 0)),
        "total_history_triage_entries": int(sum(row.get("history_triage_entries", 0) for row in case_rows)),
        "history_triage_value_counts": counter_to_sorted_dict(history_triage_value_counter),
        "history_triage_unique_value_count_distribution_per_case": counter_to_sorted_dict(history_triage_unique_count_counter),
        "history_triage_mode_by_ground_truth_acuity": {str(k): counter_to_sorted_dict(v) for k, v in history_triage_mode_by_gt.items()},
        "cases_with_history_triage_issues": len(history_triage_issue_rows),
        "issue_definitions": {
            "multiple_history_triage_values": "A single raw JSON contains more than one distinct history[].triage value.",
            "history_triage_mode_mismatch_gt_acuity": "The most frequent history[].triage value differs from ground_truth.acuity.",
            "history_triage_contains_3_in_non_gt3_case": "A non-GT3 case contains turn-level triage=3, which may pull downstream predictions toward ESI3 if leaked.",
            "history_triage_value_not_in_1_to_5": "A history[].triage value is outside the expected acuity range.",
        },
    }

    vitals_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "primary_source_for_clinical_input_policy": "vignette",
        "ground_truth_source_policy": "audit_only_consistency_reference",
        "data_validity_and_clinical_state_are_separate": True,
        "thresholds_for_audit_only": VITAL_AUDIT_THRESHOLDS,
        "plausibility_ranges_and_units": {
            name: {
                "unit": VITAL_CANONICAL_UNITS[name],
                "min": bounds[0],
                "max": bounds[1],
            }
            for name, bounds in VITAL_PLAUSIBILITY_RANGES.items()
        },
        "numeric_summary_by_vital": {name: describe_numeric(values) for name, values in vital_numeric_values.items()},
        "abnormal_vital_counts_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in vital_abnormal_by_acuity.items()},
        "danger_zone_vital_counts_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in vital_danger_by_acuity.items()},
        "cases_with_any_abnormal_vital_by_acuity": counter_to_sorted_dict(case_any_abnormal_by_acuity),
        "cases_with_any_danger_zone_vital_by_acuity": counter_to_sorted_dict(case_any_danger_by_acuity),
        "gt1_gt2_with_no_abnormal_vitals_count": len(gt_high_normal_vitals_rows),
        "gt4_gt5_with_any_abnormal_vital_count": len(gt_low_abnormal_vitals_rows),
        "vignette_vs_ground_truth_vital_mismatch_count": len(vital_mismatch_rows),
        "vignette_vital_contract_issue_count": len([r for r in vital_contract_issue_rows if r.get("source") == "vignette"]),
        "ground_truth_vital_contract_issue_count": len([r for r in vital_contract_issue_rows if r.get("source") == "ground_truth"]),
        "system_vital_event_count": len(system_vital_audit_rows),
        "system_vital_instance_count": len({r.get("instance_id") for r in system_vital_audit_rows}),
        "system_vital_unique_case_id_count": len({r.get("case_id") for r in system_vital_audit_rows}),
        "system_vital_multiple_distinct_value_instance_field_count": sum(
            1
            for row in case_rows
            for field_summary in (row.get("system_vital_audit", {}).get("fields", {}) or {}).values()
            if field_summary.get("multiple_distinct_values")
        ),
        "system_vital_latest_vs_vignette_mismatch_count": sum(
            1
            for row in case_rows
            for field_summary in (row.get("system_vital_audit", {}).get("fields", {}) or {}).values()
            if field_summary.get("latest_matches_vignette") is False
        ),
    }

    raw_text_leakage_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "mention_count": len(raw_label_like_rows),
        "case_count": len({row.get("instance_id") for row in raw_label_like_rows}),
        "explicit_manual_review_count": sum(1 for row in raw_label_like_rows if row.get("requires_manual_leakage_review")),
        "workflow_context_count": sum(1 for row in raw_label_like_rows if row.get("workflow_context_also_present")),
        "pattern_counts": dict(Counter(row.get("label_type") for row in raw_label_like_rows).most_common()),
        "policy": "Do not blanket-delete ordinary triage workflow language; manually review explicit acuity/level mentions.",
    }

    history_stats_summary = {
        field: describe_numeric(values)
        for field, values in history_stat_values.items()
    }
    history_stats_by_acuity = {
        str(ac): {field: describe_numeric(values) for field, values in field_map.items()}
        for ac, field_map in history_stat_by_acuity.items()
    }

    vignette_persona_audit = {
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "arrival_transport_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in arrival_by_acuity.items()},
        "gender_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in gender_by_acuity.items()},
        "specialisation_by_acuity_analysis_only": {str(k): counter_to_sorted_dict(v) for k, v in specialisation_by_acuity.items()},
        "top_chiefcomplaints_by_acuity": {
            str(k): [{"chiefcomplaint": cc, "count": int(n)} for cc, n in v.most_common(25)]
            for k, v in chief_by_acuity.items()
        },
        "top_chiefcomplaint_tokens_by_acuity": {
            str(k): [{"token": tok, "count": int(n)} for tok, n in v.most_common(50)]
            for k, v in chief_token_by_acuity.items()
        },
        "patient_persona_distribution_by_field": {field: counter_to_sorted_dict(c) for field, c in persona_patient_by_field.items()},
        "nurse_persona_distribution_by_field": {field: counter_to_sorted_dict(c) for field, c in persona_nurse_by_field.items()},
        "history_stats_overall": history_stats_summary,
        "history_stats_by_acuity": history_stats_by_acuity,
        "note": "Persona fields are for robustness / generation-condition audit only. They must not be used as clinical evidence.",
    }

    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_only": True,
        "allowed_as_prediction_input": False,
        "raw_data_is_immutable": True,
        "prediction_modules_must_not_read_this_output": True,
        "raw_data_dir": raw_data_dir,
        "output_dir": output_dir,
        "total_json_files_found": len(json_files),
        "loaded_cases": loaded,
        "load_error_count": len(load_errors),
        "missing_required_field_case_count": len(missing_field_rows),
        "id_audit": id_audit,
        "label_pain_audit": label_pain_audit,
        "history_triage_audit": history_triage_audit,
        "vitals_audit": vitals_audit,
        "raw_text_label_like_audit": raw_text_leakage_audit,
        "vignette_persona_audit_summary": {
            "history_stats_overall": history_stats_summary,
            "arrival_transport_by_acuity": {str(k): counter_to_sorted_dict(v) for k, v in arrival_by_acuity.items()},
            "specialisation_by_acuity_analysis_only": {str(k): counter_to_sorted_dict(v) for k, v in specialisation_by_acuity.items()},
        },
        "interpretation_notes": [
            "Use instance_id=case_id__run_uuid as the pipeline key. case_id alone is not unique enough for this project.",
            "history[].triage, vignette.acuity, and vignette.specialisation are audit/leakage fields and should not be exposed as clinical evidence to downstream agents.",
            "Scale-question audit separates pain-scale answers from fatigue/distress scale answers to avoid false severe_pain triggers.",
            "Vital thresholds in this script are descriptive audit thresholds, not final ESI classification rules.",
            "Vignette values are the prediction-facing structured source; ground_truth values are audit-only references.",
            "Implausible measurements are quarantined in audit findings and must not be consumed by prediction modules.",
            "Raw text label-like mentions require manual leakage review; ordinary workflow phrases are not automatically leakage.",
        ],
    }

    # Main JSON outputs.
    write_json(os.path.join(output_dir, "scan_summary.json"), summary)
    write_json(os.path.join(output_dir, "audit_only_scan_summary.json"), summary)
    write_json(os.path.join(output_dir, "id_audit.json"), id_audit)
    write_json(os.path.join(output_dir, "label_pain_audit.json"), label_pain_audit)
    write_json(os.path.join(output_dir, "history_triage_audit.json"), history_triage_audit)
    write_json(os.path.join(output_dir, "vitals_audit.json"), vitals_audit)
    write_json(os.path.join(output_dir, "vignette_persona_audit.json"), vignette_persona_audit)
    write_json(os.path.join(output_dir, "raw_text_label_like_audit.json"), raw_text_leakage_audit)

    if write_legacy_summary:
        # Compatibility copy matching the original 00 script location.
        ensure_dir(os.path.dirname(LEGACY_SUMMARY_FILE))
        write_json(LEGACY_SUMMARY_FILE, summary)

    # CSV / JSONL tables.
    write_csv(os.path.join(tables_dir, "case_level_audit.csv"), case_rows)
    write_csv(os.path.join(tables_dir, "duplicate_case_ids.csv"), duplicate_case_rows, ["case_id", "count"])
    write_csv(os.path.join(tables_dir, "duplicate_instance_ids.csv"), duplicate_instance_rows, ["instance_id", "count"])
    write_csv(os.path.join(tables_dir, "duplicate_dataset_instance_ids.csv"), duplicate_dataset_instance_rows, ["dataset_instance_id", "count"])
    write_csv(os.path.join(tables_dir, "acuity_counts.csv"), counter_to_rows(acuity_counter, "ground_truth_acuity"))
    write_csv(os.path.join(tables_dir, "pain_counts.csv"), counter_to_rows(pain_counter, "ground_truth_pain"))
    write_csv(os.path.join(tables_dir, "vignette_pain_counts.csv"), counter_to_rows(vignette_pain_counter, "vignette_pain"))
    write_csv(os.path.join(tables_dir, "pain_by_acuity.csv"), nested_counter_to_rows(pain_by_acuity, "ground_truth_acuity", "ground_truth_pain"))
    write_csv(os.path.join(tables_dir, "arrival_transport_by_acuity.csv"), nested_counter_to_rows(arrival_by_acuity, "ground_truth_acuity", "arrival_transport"))
    write_csv(os.path.join(tables_dir, "specialisation_by_acuity_analysis_only.csv"), nested_counter_to_rows(specialisation_by_acuity, "ground_truth_acuity", "specialisation"))

    write_jsonl(os.path.join(examples_dir, "load_errors.jsonl"), load_errors)
    write_jsonl(os.path.join(examples_dir, "missing_required_fields.jsonl"), missing_field_rows)
    write_jsonl(os.path.join(examples_dir, "acuity_mismatches.jsonl"), acuity_mismatch_rows)
    write_jsonl(os.path.join(examples_dir, "history_triage_issues.jsonl"), history_triage_issue_rows)
    write_jsonl(os.path.join(examples_dir, "pain_anomalies.jsonl"), pain_anomaly_rows)
    write_jsonl(os.path.join(examples_dir, "vignette_pain_anomalies.jsonl"), vignette_pain_anomaly_rows)
    write_jsonl(os.path.join(examples_dir, "vignette_gt_pain_mismatches.jsonl"), pain_mismatch_rows)
    write_jsonl(os.path.join(examples_dir, "vital_anomalies.jsonl"), vital_anomaly_rows)
    write_jsonl(os.path.join(examples_dir, "vital_contract_issues.jsonl"), vital_contract_issue_rows)
    write_jsonl(os.path.join(examples_dir, "vital_vignette_gt_mismatches.jsonl"), vital_mismatch_rows)
    write_jsonl(os.path.join(examples_dir, "system_vital_audit.jsonl"), system_vital_audit_rows)
    write_jsonl(os.path.join(examples_dir, "raw_text_label_like_mentions.jsonl"), raw_label_like_rows)
    write_jsonl(os.path.join(examples_dir, "scale_question_mentions.jsonl"), scale_mention_rows)
    write_jsonl(os.path.join(examples_dir, "pain0_non_pain_or_ambiguous_scale_mentions.jsonl"), scale_pain0_non_pain_rows)
    write_jsonl(os.path.join(examples_dir, "pain0_with_scale_numeric_ge7.jsonl"), scale_pain0_high_numeric_rows)
    write_jsonl(os.path.join(examples_dir, "gt1_gt2_with_no_abnormal_vitals.jsonl"), gt_high_normal_vitals_rows)
    write_jsonl(os.path.join(examples_dir, "gt4_gt5_with_any_abnormal_vital.jsonl"), gt_low_abnormal_vitals_rows)

    if verbose:
        print("Done")
        print(f"Found JSON files: {len(json_files)}")
        print(f"Loaded cases: {loaded}")
        print(f"Load errors: {len(load_errors)}")
        print(f"Acuity counts: {counter_to_sorted_dict(acuity_counter)}")
        print(f"Pain missing/out-of-range count: {len(pain_anomaly_rows)}")
        print(f"history[].triage issue cases: {len(history_triage_issue_rows)}")
        print(f"case_id duplicate groups: {len(duplicate_case_rows)}")
        print(f"instance_id duplicate groups: {len(duplicate_instance_rows)}")
        print(f"Audit-only summary: {os.path.join(output_dir, 'audit_only_scan_summary.json')}")
        if write_legacy_summary:
            print(f"Compatibility summary: {LEGACY_SUMMARY_FILE}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended raw dataset audit for SAFE-Triage / TRIBOT.")
    parser.add_argument(
        "--raw-data-dir",
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Directory containing raw JSON transcripts. Default: {DEFAULT_RAW_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for audit outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick smoke testing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress printing.",
    )
    parser.add_argument(
        "--write-legacy-summary-copy",
        action="store_true",
        help="Explicitly write the compatibility outputs/scan_summary.json copy.",
    )
    parser.add_argument(
        "--no-legacy-summary-copy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dataset(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        verbose=not args.quiet,
        write_legacy_summary=args.write_legacy_summary_copy and not args.no_legacy_summary_copy,
    )


if __name__ == "__main__":
    main()
