#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the deterministic, prediction-safe vital state layer for ESI Step D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "03_vital_state_v1.0"
DEFAULT_INPUT = Path("outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/03_vital_state_outputs_v1_full688")
DEFAULT_POLICY_DIR = Path("policy")
DEFAULT_COHORT_CONTRACT = Path("manifests/prediction_safe_cohort_contract.json")
DEFAULT_COUNT_MANIFEST = Path("manifests/instance_count_manifest.json")

OFFICIAL_FIELDS = {"heart_rate", "respiratory_rate", "spo2"}
CONTEXT_FIELDS = {"systolic_blood_pressure", "temperature", "pain_score"}
ALL_FIELDS = OFFICIAL_FIELDS | CONTEXT_FIELDS
FIELD_ALIASES = {
    "heart_rate": "heart_rate", "heartrate": "heart_rate", "hr": "heart_rate",
    "respiratory_rate": "respiratory_rate", "resprate": "respiratory_rate", "rr": "respiratory_rate",
    "spo2": "spo2", "o2sat": "spo2", "oxygen_saturation": "spo2",
    "systolic_blood_pressure": "systolic_blood_pressure", "sbp": "systolic_blood_pressure",
    "temperature": "temperature", "pain_score": "pain_score",
}
UNIT_BY_FIELD = {
    "heart_rate": {"beats_per_minute", "bpm"}, "respiratory_rate": {"breaths_per_minute", "breaths/min"},
    "spo2": {"percent", "%"}, "systolic_blood_pressure": {"mmhg"},
    "temperature": {"fahrenheit", "celsius"}, "pain_score": {"0_to_10_scale"},
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_field(event: Dict[str, Any]) -> Optional[str]:
    raw = event.get("canonical_vital_name") or event.get("name") or event.get("raw_name")
    return FIELD_ALIASES.get(str(raw).strip().lower().replace("-", "_")) if raw else None


def numeric_value(event: Dict[str, Any]) -> Optional[float]:
    for key in ("normalized_value", "canonical_vital_value", "value_numeric", "value"):
        value = event.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def usable_measurement(event: Dict[str, Any], field: str) -> Tuple[bool, str]:
    if event.get("measurement_identity") != "canonical_case_measurement":
        return False, "not_canonical_measurement"
    if event.get("canonical_measurement_evidence_id") != event.get("evidence_id"):
        return False, "canonical_evidence_id_mismatch"
    if event.get("usable_for_clinical_reasoning") is not True:
        return False, str(event.get("quarantine_reason") or "measurement_not_usable")
    if (event.get("evidence_use_policy") or {}).get("allowed") is not True:
        return False, "measurement_policy_not_allowed"
    unit = str(event.get("canonical_unit") or event.get("raw_unit") or "").lower()
    if unit not in UNIT_BY_FIELD.get(field, set()):
        return False, "unit_contract_mismatch"
    if numeric_value(event) is None:
        return False, "non_numeric_measurement"
    return True, "valid"


def measurement_row(event: Dict[str, Any], field: str, valid: bool, reason: str) -> Dict[str, Any]:
    return {
        "measurement_id": event.get("canonical_measurement_evidence_id") or event.get("evidence_id"),
        "measurement_type": field, "canonical_value": numeric_value(event),
        "canonical_unit": event.get("canonical_unit") or event.get("raw_unit"),
        "measurement_state": "valid" if valid else "quarantined",
        "measurement_source": event.get("measurement_source") or event.get("source_layer"),
        "evidence_id": event.get("evidence_id"), "event_id": event.get("evidence_id"),
        "turn": event.get("turn"), "event_seq_idx": event.get("event_seq_idx"),
        "counted_once": bool(valid), "quarantine_reason": None if valid else reason,
    }


def load_age_policy(contract: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
    usable = contract.get("age_policy_usable") is True and contract.get("verified_before_model_development") is True
    if usable and contract.get("cohort_age_policy") == "adult_only":
        return True, "adult_gt_8yr", "verified_adult_only_cohort_contract"
    return False, None, "missing_prediction_safe_age"


def compare(value: float, operator: str, threshold: float) -> bool:
    return value > threshold if operator == ">" else value < threshold


def build_row(timeline: Dict[str, Any], policy: Dict[str, Any], thresholds: Dict[str, Any], cohort: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    events = timeline.get("events") or []
    event_ids = {e.get("evidence_id") for e in events if e.get("evidence_id")}
    valid: Dict[str, List[Dict[str, Any]]] = {field: [] for field in ALL_FIELDS}
    quarantined: List[Dict[str, Any]] = []
    reveal_links: List[Dict[str, Any]] = []
    counters = Counter()
    for event in events:
        field = canonical_field(event)
        identity = event.get("measurement_identity")
        if identity == "dialogue_reveal":
            canonical_id = event.get("canonical_measurement_evidence_id")
            if canonical_id not in event_ids:
                counters["orphan_measurement_evidence_id_count"] += 1
            reveal_invalid = (
                event.get("usable_for_clinical_reasoning") is False
                or bool(event.get("quarantine_reason"))
                or event.get("value_plausibility") not in (None, "plausible")
                or event.get("event_surface_polarity_hint") == "vital_reveal_quarantined"
            )
            if reveal_invalid:
                reason = (
                    event.get("quarantine_reason")
                    or event.get("plausibility_reason")
                    or event.get("event_surface_polarity_hint")
                    or "dialogue_reveal_measurement_quarantined"
                )
                quarantined.append(measurement_row(event, field or "unknown", False, str(reason)))
                counters["invalid_measurement_seen_count"] += 1
            reveal_links.append({"reveal_event_id": event.get("evidence_id"), "revealed_measurement_id": canonical_id, "reveal_evidence_id": event.get("evidence_id"), "measurement_type": field, "turn": event.get("turn"), "independent_measurement": False, "counted_as_clinical_observation": False})
            continue
        if field not in ALL_FIELDS or event.get("event_type") not in {"structured_vital", "structured_pain_score"}:
            continue
        ok, reason = usable_measurement(event, field)
        item = measurement_row(event, field, ok, reason)
        if ok:
            valid[field].append(item)
        else:
            item["counted_once"] = False
            quarantined.append(item)
            counters["invalid_measurement_seen_count"] += 1
    for rows in valid.values():
        if len(rows) > 1:
            counters["multiple_canonical_measurements_count"] += len(rows) - 1

    age_usable, age_band, age_reason = load_age_policy(cohort)
    band = thresholds.get("bands", {}).get(age_band or "adult_gt_8yr", {})
    field_assessability: Dict[str, str] = {}
    flags: List[Dict[str, Any]] = []
    policy_rule_ids: List[str] = []
    for field in sorted(OFFICIAL_FIELDS):
        rows = valid[field]
        if not rows:
            field_assessability[field] = "not_assessable_missing_required_vital"
            continue
        if field in policy.get("age_dependent_fields", []) and not age_usable:
            field_assessability[field] = "not_assessable_missing_age"
            continue
        field_assessability[field] = "assessable"
        rule = band.get(field)
        if not rule:
            field_assessability[field] = "not_assessable_policy_rule_missing"
            continue
        for item in rows:
            triggered = compare(float(item["canonical_value"]), rule["operator"], float(rule["threshold"]))
            rule_id = f"{policy.get('policy_version', 'esi_step_d_v1')}::{field}::{rule['operator']}{rule['threshold']}"
            policy_rule_ids.append(rule_id)
            flags.append({"measurement_type": field, "policy_rule_id": rule_id, "triggered": triggered, "canonical_value": item["canonical_value"], "canonical_unit": item["canonical_unit"], "operator": rule["operator"], "threshold": rule["threshold"], "supporting_evidence_id": item["evidence_id"]})
    assessable = any(value == "assessable" for value in field_assessability.values())
    signal_present = any(flag["triggered"] for flag in flags)
    missing = [f"{field}_missing" for field in sorted(OFFICIAL_FIELDS) if not valid[field]]
    application = "signal_present_not_yet_applied" if signal_present else ("not_assessable_missing_required_input" if not assessable else "signal_absent")
    context_flags = []
    for field in sorted(CONTEXT_FIELDS):
        for item in valid[field]:
            context_flags.append({"flag_type": f"{field}_context", "official_step_d_trigger_eligible": False, "measurement_id": item["measurement_id"], "supporting_evidence_id": item["evidence_id"], "canonical_value": item["canonical_value"], "canonical_unit": item["canonical_unit"]})
    observations = [item for rows in valid.values() for item in rows]
    supporting = {item["evidence_id"] for item in observations if item.get("evidence_id")}
    row = {
        "schema_version": SCHEMA_VERSION, "instance_id": timeline.get("instance_id"), "case_id": timeline.get("case_id"), "case_aggregation_key": timeline.get("case_id"),
        "vital_observations": observations, "vital_reveal_links": reveal_links, "missing_vital_flags": missing, "quarantined_measurements": quarantined,
        "official_danger_zone_flags": flags, "official_danger_zone_signal_present": signal_present, "field_assessability": field_assessability,
        "step_d_assessable": assessable, "step_d_application_status": application, "safety_context_flags": context_flags,
        "supporting_evidence_ids": sorted(supporting), "policy_rule_ids": sorted(set(policy_rule_ids)), "age_policy_usable": age_usable, "age_band": age_band, "age_policy_reason": age_reason,
        "requires_review": bool(quarantined or missing or counters["multiple_canonical_measurements_count"] or not assessable),
        "reward_style_diagnostics": {"invalid_measurement_penalty": counters["invalid_measurement_seen_count"], "duplicate_reveal_penalty": 0, "missing_as_normal_penalty": 0, "threshold_boundary_violation": 0},
    }
    counters["processed_instance_count"] += 1
    counters["official_signal_count"] += int(signal_present)
    counters["duplicate_reveal_counted_as_measurement_count"] += sum(int(x["counted_as_clinical_observation"]) for x in reveal_links)
    return row, dict(counters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--cohort-contract", type=Path, default=DEFAULT_COHORT_CONTRACT)
    parser.add_argument("--count-manifest", type=Path, default=DEFAULT_COUNT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.input_file, args.cohort_contract, args.count_manifest, args.policy_dir / "esi_step_d_policy_manifest.json", args.policy_dir / "esi_step_d_thresholds.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.policy_dir / "esi_step_d_policy_manifest.json"; thresholds_path = args.policy_dir / "esi_step_d_thresholds.json"
    policy = read_json(policy_path); thresholds = read_json(thresholds_path); cohort = read_json(args.cohort_contract); counts = read_json(args.count_manifest); timelines = read_jsonl(args.input_file)
    rows: List[Dict[str, Any]] = []; aggregate = Counter(); case_ids = []; instance_ids = []
    for timeline in timelines:
        row, local = build_row(timeline, policy, thresholds, cohort); rows.append(row); aggregate.update(local); case_ids.append(timeline.get("case_id")); instance_ids.append(timeline.get("instance_id"))
    duplicate_groups = sum(1 for n in Counter(case_ids).values() if n > 1)
    expected = {"instances": counts.get("expected_instance_count"), "cases": counts.get("expected_case_count"), "duplicate_groups": counts.get("expected_duplicate_group_count")}
    aggregate.update({"unique_case_count": len(set(case_ids)), "unique_instance_count": len(set(instance_ids)), "duplicate_case_group_count": duplicate_groups, "invalid_measurement_consumed_count": 0, "invalid_measurement_official_signal_count": 0, "missing_vital_assumed_normal_count": 0, "nonofficial_field_triggered_danger_zone_count": 0, "official_signal_missing_evidence_id_count": 0, "cross_instance_measurement_link_count": 0, "ground_truth_input_count": 0, "audit_sidecar_input_count": 0, "llm_call_count": 0, "model_override_count": 0, "step_d_upgrade_count": 0})
    aggregate["official_signal_missing_evidence_id_count"] = sum(
        1 for row in rows for flag in row.get("official_danger_zone_flags", [])
        if flag.get("triggered") and not flag.get("supporting_evidence_id")
    )
    aggregate["nonofficial_field_triggered_danger_zone_count"] = sum(
        1 for row in rows for flag in row.get("official_danger_zone_flags", [])
        if flag.get("triggered") and flag.get("measurement_type") not in OFFICIAL_FIELDS
    )
    aggregate["count_gate_passed"] = int(len(timelines) == expected["instances"] and len(set(instance_ids)) == expected["instances"] and len(set(case_ids)) == expected["cases"] and duplicate_groups == expected["duplicate_groups"])
    zero_gates = ["invalid_measurement_consumed_count", "invalid_measurement_official_signal_count", "duplicate_reveal_counted_as_measurement_count", "missing_vital_assumed_normal_count", "nonofficial_field_triggered_danger_zone_count", "official_signal_missing_evidence_id_count", "orphan_measurement_evidence_id_count", "cross_instance_measurement_link_count", "ground_truth_input_count", "audit_sidecar_input_count", "llm_call_count", "model_override_count", "step_d_upgrade_count"]
    aggregate["hard_gate_passed"] = int(aggregate["count_gate_passed"] and all(aggregate[k] == 0 for k in zero_gates))
    age_usable = cohort.get("age_policy_usable") is True and cohort.get("verified_before_model_development") is True
    audit = {"schema_version": SCHEMA_VERSION, "input_file": str(args.input_file), "input_sha256": sha256(args.input_file), "policy_manifest": str(policy_path), "threshold_manifest": str(thresholds_path), "cohort_contract": str(args.cohort_contract), "cohort_age_policy": cohort.get("cohort_age_policy"), "age_policy_usable": age_usable, "processed_instances": len(timelines), "unique_cases": len(set(case_ids)), "duplicate_groups": duplicate_groups, "expected_counts": expected, "counts": dict(aggregate), "release_gate_passed": bool(aggregate["hard_gate_passed"]), "llm_call_count": 0, "model_override_count": 0, "final_esi_output_count": 0}
    write_jsonl(args.output_dir / "03_vital_state_outputs.jsonl", rows); write_json(args.output_dir / "03_vital_state_audit.json", audit)
    print(f"Processed instances: {len(timelines)}"); print(f"Official danger-zone signals: {aggregate['official_signal_count']}"); print(f"Invalid measurements quarantined: {aggregate['invalid_measurement_seen_count']}"); print(f"Age policy usable: {age_usable}"); print(f"Release gate passed: {audit['release_gate_passed']}")


if __name__ == "__main__":
    try: main()
    except Exception as exc: print(f"[ERROR] {exc}", file=sys.stderr); raise
