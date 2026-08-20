#!/usr/bin/env python3
"""Build the deterministic stage-03 vital-state fact layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from esi_vital_policy_manifest_v1 import (
    AGE_BANDS,
    MANIFEST_VERSION,
    RULE_IDS,
    SOURCE,
    SPO2_CONTEXT_REQUIREMENT,
    SPO2_OPERATOR,
    SPO2_THRESHOLD,
)


SCHEMA_VERSION = "03_vital_state_facts_v1.0"
EXPECTED_INPUT_SCHEMA = "01_dialogue_to_evidence_cards_v1.0"
VITAL_FIELDS = (
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "systolic_blood_pressure",
)
STEP_D_FIELDS = ("heart_rate", "respiratory_rate", "spo2")
FORBIDDEN_KEYS = {
    "ground_truth",
    "acuity",
    "triage",
    "patient_persona",
    "nurse_persona",
    "specialisation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def allowed_measurement(item: dict[str, Any]) -> bool:
    policy = item.get("evidence_use_policy") or {}
    return (
        item.get("measurement_identity") == "canonical_case_measurement"
        and item.get("usable_for_clinical_reasoning") is True
        and policy.get("allowed") is True
        and item.get("value_plausibility") == "plausible"
        and item.get("normalized_value") is not None
    )


def age_band(age_years: float) -> dict[str, Any] | None:
    for band in AGE_BANDS:
        if age_years >= band["min"] and (band["max"] is None or age_years < band["max"]):
            return band
    return None


def evaluate_age_rule(field: str, value: float, age_years: float | None) -> dict[str, Any]:
    rule = {"field": field, "rule_id": RULE_IDS[field], "operator": ">"}
    if age_years is None:
        return {**rule, "state": "not_assessable", "reason": "missing_prediction_safe_age"}
    band = age_band(age_years)
    if band is None:
        return {**rule, "state": "not_assessable", "reason": "age_outside_manifest"}
    threshold = band["hr_gt" if field == "heart_rate" else "rr_gt"]
    return {
        **rule,
        "state": "threshold_exceeded" if value > threshold else "threshold_not_exceeded",
        "age_band": band["id"],
        "threshold": threshold,
        "value": value,
    }


def evaluate_spo2(value: float) -> dict[str, Any]:
    return {
        "field": "spo2",
        "rule_id": RULE_IDS["spo2"],
        "operator": SPO2_OPERATOR,
        "threshold": SPO2_THRESHOLD,
        "value": value,
        "state": "threshold_exceeded" if value < SPO2_THRESHOLD else "threshold_not_exceeded",
        "context_requirement": SPO2_CONTEXT_REQUIREMENT,
        "context_evaluation": "deferred_to_policy_reconciliation",
    }


def pain_band(value: float) -> str:
    if value == 0:
        return "none"
    if value <= 3:
        return "mild"
    if value <= 6:
        return "moderate"
    return "high_7_to_10"


def forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(forbidden_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return found


def policy_fixtures() -> list[dict[str, Any]]:
    checks = [
        ("spo2_below", evaluate_spo2(91.9)["state"] == "threshold_exceeded"),
        ("spo2_exact", evaluate_spo2(92.0)["state"] == "threshold_not_exceeded"),
        ("adult_hr_exact", evaluate_age_rule("heart_rate", 100.0, 30.0)["state"] == "threshold_not_exceeded"),
        ("adult_hr_above", evaluate_age_rule("heart_rate", 100.1, 30.0)["state"] == "threshold_exceeded"),
        ("adult_rr_exact", evaluate_age_rule("respiratory_rate", 20.0, 30.0)["state"] == "threshold_not_exceeded"),
        ("adult_rr_above", evaluate_age_rule("respiratory_rate", 20.1, 30.0)["state"] == "threshold_exceeded"),
        ("missing_age", evaluate_age_rule("heart_rate", 120.0, None)["state"] == "not_assessable"),
    ]
    return [{"fixture": name, "passed": passed} for name, passed in checks]


def build_row(row: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    instance_id = row["instance_id"]
    measurements = row.get("measurement_records", [])
    quarantined = row.get("quarantined_measurements", [])
    canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reveals: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in measurements:
        identity = item.get("measurement_identity")
        if identity == "canonical_case_measurement":
            canonical[item.get("field")].append(item)
        elif identity == "dialogue_reveal":
            reveals[item.get("canonical_measurement_id")].append(item)

    observations = []
    assessability: dict[str, dict[str, Any]] = {}
    valid_by_field: dict[str, dict[str, Any]] = {}
    quarantined_fields = {item.get("field") for item in quarantined}

    for field in VITAL_FIELDS:
        valid = [item for item in canonical.get(field, []) if allowed_measurement(item)]
        if len(valid) > 1:
            errors.append({"instance_id": instance_id, "error": "duplicate_valid_canonical_measurement", "field": field})
        if not valid:
            state = "quarantined" if field in quarantined_fields else "missing"
            assessability[field] = {"measurement_state": state, "step_d_state": "not_assessable"}
            continue
        item = valid[0]
        valid_by_field[field] = item
        linked_reveals = reveals.get(item["measurement_id"], [])
        observations.append(
            {
                "field": field,
                "value": item["normalized_value"],
                "unit": item.get("canonical_unit"),
                "measurement_id": item["measurement_id"],
                "measurement_identity": "canonical_case_measurement",
                "dialogue_reveal_ids": [reveal["measurement_id"] for reveal in linked_reveals],
                "dialogue_reveal_event_ids": [reveal.get("parent_event_id") for reveal in linked_reveals],
                "independent_measurement_count": 1,
            }
        )
        step_d_state = "assessable" if field == "spo2" else (
            "not_assessable_missing_age" if field in {"heart_rate", "respiratory_rate"} else "not_in_step_d_table"
        )
        assessability[field] = {"measurement_state": "observed_valid", "step_d_state": step_d_state}

    age_value = row.get("prediction_safe_age_years")
    age_years = float(age_value) if isinstance(age_value, (int, float)) else None
    flags = []
    for field in ("heart_rate", "respiratory_rate"):
        item = valid_by_field.get(field)
        flag = evaluate_age_rule(field, float(item["normalized_value"]), age_years) if item else {
            "field": field,
            "rule_id": RULE_IDS[field],
            "state": "not_assessable",
            "reason": "measurement_missing_or_quarantined",
        }
        if item:
            flag["measurement_id"] = item["measurement_id"]
        flags.append(flag)
    spo2 = valid_by_field.get("spo2")
    spo2_flag = evaluate_spo2(float(spo2["normalized_value"])) if spo2 else {
        "field": "spo2",
        "rule_id": RULE_IDS["spo2"],
        "state": "not_assessable",
        "reason": "measurement_missing_or_quarantined",
    }
    if spo2:
        spo2_flag["measurement_id"] = spo2["measurement_id"]
    flags.append(spo2_flag)

    safety_context = []
    for field in ("temperature", "systolic_blood_pressure"):
        item = valid_by_field.get(field)
        if item:
            safety_context.append({
                "flag": f"valid_{field}_context",
                "field": field,
                "value": item["normalized_value"],
                "unit": item.get("canonical_unit"),
                "measurement_id": item["measurement_id"],
                "policy_role": "general_clinical_context_only",
            })
    pain = next((item for item in canonical.get("pain_score", []) if allowed_measurement(item)), None)
    if pain:
        safety_context.append({
            "flag": "valid_pain_score_context",
            "field": "pain_score",
            "value": pain["normalized_value"],
            "unit": pain.get("canonical_unit"),
            "band": pain_band(float(pain["normalized_value"])),
            "measurement_id": pain["measurement_id"],
            "policy_role": "support_only_not_standalone_step_b",
        })

    supporting_ids = [item["measurement_id"] for item in valid_by_field.values()]
    if pain:
        supporting_ids.append(pain["measurement_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_manifest_version": MANIFEST_VERSION,
        "instance_id": instance_id,
        "case_id": row["case_id"],
        "run_uuid": row.get("run_uuid"),
        "age_assessability": "available" if age_years is not None else "missing_prediction_safe_age",
        "vital_observations": observations,
        "field_assessability": assessability,
        "missing_vital_flags": [field for field, state in assessability.items() if state["measurement_state"] == "missing"],
        "quarantined_measurements": quarantined,
        "official_danger_zone_flags": flags,
        "official_danger_zone_signal_present": any(flag["state"] == "threshold_exceeded" for flag in flags),
        "safety_context_flags": safety_context,
        "supporting_evidence_ids": supporting_ids,
        "policy_rule_ids": [RULE_IDS[field] for field in STEP_D_FIELDS],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=project_root / "outputs/01_evidence_cards_v1/evidence_timelines.jsonl")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs/03_vital_state_facts_v1")
    parser.add_argument("--expected-instances", type=int, default=1010)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_hash = sha256_file(args.input)
    rows = []
    errors: list[dict[str, Any]] = []
    input_measurement_ids: set[str] = set()
    invalid_ids: set[str] = set()
    reveal_count = 0
    invalid_reveal_independent = 0
    unlinked_reveal_count = 0
    reveal_value_mismatch_count = 0
    input_schemas = Counter()

    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            input_schemas[source.get("schema_version")] += 1
            canonical_lookup = {
                item.get("measurement_id"): item
                for item in source.get("measurement_records", [])
                if item.get("measurement_identity") == "canonical_case_measurement"
            }
            for item in source.get("measurement_records", []):
                input_measurement_ids.add(item.get("measurement_id"))
                if item.get("measurement_identity") == "dialogue_reveal":
                    reveal_count += 1
                    invalid_reveal_independent += int(item.get("independent_measurement") is not False)
                    linked = canonical_lookup.get(item.get("canonical_measurement_id"))
                    if linked is None:
                        unlinked_reveal_count += 1
                    elif (
                        linked.get("field"), linked.get("normalized_value"), linked.get("canonical_unit")
                    ) != (
                        item.get("field"), item.get("normalized_value"), item.get("canonical_unit")
                    ):
                        reveal_value_mismatch_count += 1
            for item in source.get("quarantined_measurements", []):
                invalid_ids.add(item.get("measurement_id"))
            rows.append(build_row(source, errors))

    output_path = args.output_dir / "03_vital_state_facts.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    seen_ids = [row["instance_id"] for row in rows]
    consumed_ids = {item for row in rows for item in row["supporting_evidence_ids"]}
    unknown_support_ids = consumed_ids - input_measurement_ids
    invalid_consumed = consumed_ids & invalid_ids
    forbidden = [
        {"instance_id": row["instance_id"], "paths": paths}
        for row in rows
        if (paths := forbidden_key_paths(row))
    ]
    fixtures = policy_fixtures()
    if input_schemas != Counter({EXPECTED_INPUT_SCHEMA: len(rows)}):
        errors.append({"error": "unexpected_input_schema", "counts": dict(input_schemas)})
    errors.extend({"error": "forbidden_output_key", **item} for item in forbidden)

    signal_rows = [row for row in rows if row["official_danger_zone_signal_present"]]
    flag_states = Counter(
        (flag["field"], flag["state"])
        for row in rows
        for flag in row["official_danger_zone_flags"]
    )
    assessability_counts = Counter(
        (field, state["step_d_state"])
        for row in rows
        for field, state in row["field_assessability"].items()
    )
    hard_gates = {
        "expected_instance_count": len(rows) == args.expected_instances,
        "unique_instance_ids": len(seen_ids) == len(set(seen_ids)),
        "expected_input_schema": input_schemas == Counter({EXPECTED_INPUT_SCHEMA: len(rows)}),
        "invalid_measurement_consumption_zero": not invalid_consumed,
        "dialogue_reveal_independent_zero": invalid_reveal_independent == 0,
        "dialogue_reveal_links_complete": unlinked_reveal_count == 0,
        "dialogue_reveal_values_match_canonical": reveal_value_mismatch_count == 0,
        "unknown_supporting_evidence_zero": not unknown_support_ids,
        "forbidden_leakage_zero": not forbidden,
        "policy_fixtures_passed": all(item["passed"] for item in fixtures),
        "validation_errors_zero": not errors,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "policy_manifest_version": MANIFEST_VERSION,
        "policy_source": SOURCE,
        "input_file": str(args.input.resolve()),
        "input_file_sha256": input_hash,
        "output_file": str(output_path.resolve()),
        "processed_instance_count": len(rows),
        "unique_instance_count": len(set(seen_ids)),
        "vital_observation_count": sum(len(row["vital_observations"]) for row in rows),
        "dialogue_reveal_count": reveal_count,
        "unlinked_dialogue_reveal_count": unlinked_reveal_count,
        "dialogue_reveal_value_mismatch_count": reveal_value_mismatch_count,
        "quarantined_measurement_count": sum(len(row["quarantined_measurements"]) for row in rows),
        "invalid_measurement_consumption_count": len(invalid_consumed),
        "official_danger_zone_signal_count": len(signal_rows),
        "official_signal_instance_ids": [row["instance_id"] for row in signal_rows],
        "official_flag_state_counts": {f"{field}:{state}": count for (field, state), count in sorted(flag_states.items())},
        "field_assessability_counts": {f"{field}:{state}": count for (field, state), count in sorted(assessability_counts.items())},
        "policy_fixtures": fixtures,
        "hard_gates": hard_gates,
        "release_gate_passed": all(hard_gates.values()),
        "downstream_contract": {
            "step_d_signal_is_not_step_b": True,
            "step_d_signal_is_not_resource_feature": True,
            "04_consumes_vital_context_only": True,
            "05_06_own_policy_reconciliation": True,
        },
    }

    errors_path = args.output_dir / "03_validation_errors.jsonl"
    errors_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in errors), encoding="utf-8")
    invalid_path = args.output_dir / "03_invalid_measurement_audit.jsonl"
    with invalid_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            for item in row["quarantined_measurements"]:
                stream.write(json.dumps({"instance_id": row["instance_id"], **item}, sort_keys=True) + "\n")
    write_json(args.output_dir / "03_build_audit.json", audit)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_hash,
        "artifacts": {},
    }
    for path in (output_path, args.output_dir / "03_build_audit.json", errors_path, invalid_path):
        manifest["artifacts"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(args.output_dir / "ARTIFACT_MANIFEST.json", manifest)

    print(f"Processed: {len(rows)}")
    print(f"Vital observations: {audit['vital_observation_count']}")
    print(f"Quarantined measurements: {audit['quarantined_measurement_count']}")
    print(f"Official danger-zone signals: {audit['official_danger_zone_signal_count']}")
    print(f"Validation errors: {len(errors)}")
    print(f"release_gate_passed: {audit['release_gate_passed']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
