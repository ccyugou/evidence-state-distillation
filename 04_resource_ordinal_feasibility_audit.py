#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit the feasibility of evidence-constrained resource ordinal learning.

This stage does not fit a model or emit an ESI prediction. It keeps prediction
features at instance level and keeps weak supervision at case level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_TIMELINE = Path("outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl")
DEFAULT_CLAIMS = Path("outputs/02_claim_qualification_v3_full688_resumable/02_qualified_claims.jsonl")
DEFAULT_VITALS = Path("outputs/03_vital_state_outputs_v1_full688/03_vital_state_outputs.jsonl")
DEFAULT_RAW_DIR = Path("transcripts/mimic")
DEFAULT_COUNT_MANIFEST = Path("manifests/instance_count_manifest.json")
DEFAULT_OUTPUT = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")

PRESENT_STATUSES = {"confirmed_symptom_only", "confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"}
TRACKED_FIELDS = sorted({
    "active_bleeding", "altered_mental_status", "chest_pain_or_acs_concern", "dyspnea",
    "fever_or_infection", "pain", "severe_pain", "stroke_signs", "suicidal_ideation",
    "syncope", "pregnancy_status_or_context", "toxic_ingestion_or_overdose",
    "gi_bleeding_melena", "respiratory_distress", "dizziness",
})
FORBIDDEN_KEYS = {"ground_truth", "acuity", "triage", "persona", "model", "pairing", "seed", "label", "gold_esi"}

# This is the model contract. Everything else is diagnostic-only.
RESOURCE_MODEL_FEATURE_ALLOWLIST = {
    "claim_present_count", "claim_uncertain_count", "claim_negative_count", "claim_support_only_count",
    "patient_present_claim_count", "structured_present_claim_count", "nurse_observed_present_claim_count",
    "active_problem_domain_count", "independent_active_evidence_count", "valid_pain_score",
    "pain_score_value", "valid_structured_vital_count", "official_danger_signal", "official_danger_signal_count",
    "spo2_danger_signal",
}
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("structured_present_claim_count")
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("claim_field_toxic_ingestion_or_overdose_present")
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("official_danger_signal")
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("official_danger_signal_count")
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("spo2_danger_signal")
RESOURCE_MODEL_FEATURE_ALLOWLIST.update(f"claim_field_{field}_present" for field in TRACKED_FIELDS)
RESOURCE_MODEL_FEATURE_ALLOWLIST.update(f"claim_field_{field}_uncertain" for field in TRACKED_FIELDS)
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("structured_present_claim_count")
RESOURCE_MODEL_FEATURE_ALLOWLIST.discard("claim_field_toxic_ingestion_or_overdose_present")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
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


def norm_label(value: Any) -> Optional[int]:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number in {1, 2, 3, 4, 5} else None


def boolish(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def recursive_forbidden(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_KEYS or any(token in key_text for token in ("ground_truth", "recorded_triage", "raw_history_triage", "final_esi")):
                found.append(key_text)
            found.extend(recursive_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_forbidden(child))
    return found


def raw_label_index(raw_dir: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
    by_instance: Dict[str, int] = {}
    case_values: Dict[str, set] = defaultdict(set)
    file_count = 0
    unparseable = 0
    for path in sorted(raw_dir.glob("*.json")):
        obj = read_json(path)
        file_count += 1
        case_id = str(obj.get("case_id") or "")
        run_uuid = str(obj.get("run_uuid") or path.stem)
        label = norm_label((obj.get("ground_truth") or {}).get("acuity"))
        if not case_id or label is None:
            unparseable += 1
            continue
        by_instance[f"{case_id}__{run_uuid}"] = label
        case_values[case_id].add(label)
    conflicts = {case_id: sorted(values) for case_id, values in case_values.items() if len(values) != 1}
    by_case = {case_id: next(iter(values)) for case_id, values in case_values.items() if len(values) == 1}
    return by_instance, by_case, {
        "raw_file_count": file_count,
        "unparseable_label_count": unparseable,
        "duplicate_case_label_conflict_count": len(conflicts),
        "duplicate_case_label_conflicts": conflicts,
    }


def raw_label_manifest_hash(raw_dir: Path) -> str:
    entries = []
    for path in sorted(raw_dir.glob("*.json")):
        obj = read_json(path)
        case_id = str(obj.get("case_id") or "")
        run_uuid = str(obj.get("run_uuid") or path.stem)
        label = norm_label((obj.get("ground_truth") or {}).get("acuity"))
        entries.append({
            "instance_id": f"{case_id}__{run_uuid}", "case_id": case_id,
            "gold_esi": label, "source_file": path.name, "source_file_sha256": sha256(path),
        })
    payload = "\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def timeline_instance_features(row: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {
        "structured_chief_complaint": 0, "arrival_ambulance": 0, "arrival_walk_in": 0,
        "patient_event_count": 0, "nurse_question_event_count": 0, "nurse_observed_event_count": 0,
        "nurse_process_event_count": 0, "dialogue_reveal_count": 0, "valid_pain_score": 0,
        "pain_score_value": 0, "valid_structured_vital_count": 0,
    }
    for field in TRACKED_FIELDS:
        features[f"timeline_field_{field}"] = 0
    safe = row.get("safe_vignette_context") or {}
    fields = safe.get("fields") or {}
    chief = fields.get("chiefcomplaint") or {}
    if chief.get("value"):
        features["structured_chief_complaint"] = 1
    arrival = str((fields.get("arrival_transport") or {}).get("value") or "").lower().replace("_", " ")
    features["arrival_ambulance"] = int("ambulance" in arrival)
    features["arrival_walk_in"] = int("walk" in arrival)
    pain = fields.get("pain") or {}
    if pain.get("pain_quality", {}).get("usable_for_clinical_reasoning") is True:
        value = numeric(pain.get("value"))
        if value is not None:
            features["valid_pain_score"] = 1
            features["pain_score_value"] = value
    for event in row.get("events") or []:
        event_type = str(event.get("event_type") or "")
        source = str(event.get("source_layer") or "")
        if event_type == "vital" and event.get("measurement_identity") == "dialogue_reveal":
            features["dialogue_reveal_count"] += 1
        if event_type == "structured_vital" and event.get("usable_for_clinical_reasoning") is True:
            features["valid_structured_vital_count"] += 1
        if event_type == "utterance" and source == "patient_reported":
            features["patient_event_count"] += 1
        if event_type == "utterance" and source == "nurse_question":
            features["nurse_question_event_count"] += 1
        if event_type == "utterance" and source in {"nurse_observed_statement", "nurse_mixed_utterance"}:
            features["nurse_observed_event_count"] += 1
        if event_type == "utterance" and source == "nurse_instruction_or_process":
            features["nurse_process_event_count"] += 1
    return features


def claim_instance_features(row: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {
        "claim_present_count": 0, "claim_uncertain_count": 0, "claim_negative_count": 0,
        "claim_support_only_count": 0, "patient_present_claim_count": 0,
        "structured_present_claim_count": 0, "nurse_observed_present_claim_count": 0,
        "step_a_trigger_count": 0, "step_b_trigger_count": 0,
        "active_problem_domain_count": 0, "independent_active_evidence_count": 0,
    }
    for field in TRACKED_FIELDS:
        features[f"claim_field_{field}_present"] = 0
        features[f"claim_field_{field}_uncertain"] = 0
    active_fields = set()
    active_evidence_ids = set()
    for claim in row.get("claims") or []:
        status = str(claim.get("validated_status") or claim.get("status") or "")
        source = str(claim.get("local_source_layer") or claim.get("source_layer") or "")
        field = str(claim.get("field") or "")
        active = status in PRESENT_STATUSES and claim.get("context_only") is not True and claim.get("clinical_assertion_allowed") is True
        if active:
            features["claim_present_count"] += 1
            if source == "patient_reported":
                features["patient_present_claim_count"] += 1
            elif source == "structured_clinical_input":
                features["structured_present_claim_count"] += 1
            elif source == "nurse_observed_statement":
                features["nurse_observed_present_claim_count"] += 1
            if field in TRACKED_FIELDS:
                active_fields.add(field)
            evidence_id = claim.get("evidence_id") or claim.get("positive_evidence_id")
            if evidence_id:
                active_evidence_ids.add(str(evidence_id))
            if status == "confirmed_step_a_trigger":
                features["step_a_trigger_count"] += 1
            if status == "confirmed_step_b_policy_trigger":
                features["step_b_trigger_count"] += 1
        elif status == "uncertain_policy_review":
            features["claim_uncertain_count"] += 1
        elif status == "explicit_negative":
            features["claim_negative_count"] += 1
        elif status == "support_only":
            features["claim_support_only_count"] += 1
        if field in TRACKED_FIELDS and status == "uncertain_policy_review":
            features[f"claim_field_{field}_uncertain"] = 1
    features["active_problem_domain_count"] = len(active_fields)
    features["independent_active_evidence_count"] = len(active_evidence_ids)
    for field in active_fields:
        features[f"claim_field_{field}_present"] = 1
    return features


def vital_instance_features(row: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {
        "official_danger_signal": int(boolish(row.get("official_danger_zone_signal_present"))),
        "official_danger_signal_count": 0, "spo2_danger_signal": 0,
        "step_d_assessable": int(boolish(row.get("step_d_assessable"))),
        "spo2_observed": 0, "heart_rate_observed": 0, "respiratory_rate_observed": 0,
    }
    for obs in row.get("vital_observations") or []:
        if obs.get("measurement_state") != "valid" or obs.get("quarantine_reason"):
            continue
        field = obs.get("measurement_type")
        if field in {"spo2", "heart_rate", "respiratory_rate"}:
            features[f"{field}_observed"] = 1
    for flag in row.get("official_danger_zone_flags") or []:
        if flag.get("triggered"):
            features["official_danger_signal_count"] += 1
            if flag.get("measurement_type") == "spo2":
                features["spo2_danger_signal"] = 1
    return features


def build_instance_features(timeline: Dict[str, Any], claims: Dict[str, Any], vitals: Dict[str, Any]) -> Dict[str, float]:
    result = timeline_instance_features(timeline)
    result.update(claim_instance_features(claims))
    result.update(vital_instance_features(vitals))
    return result


def feature_provenance(timeline: Dict[str, Any], claims: Dict[str, Any], vitals: Dict[str, Any]) -> Dict[str, List[str]]:
    timeline_ids = sorted({str(event.get("evidence_id")) for event in timeline.get("events") or [] if event.get("evidence_id")})
    claim_ids = sorted({str(claim.get("claim_id")) for claim in claims.get("claims") or [] if claim.get("claim_id")})
    claim_evidence = sorted({str(claim.get("evidence_id") or claim.get("positive_evidence_id")) for claim in claims.get("claims") or [] if claim.get("evidence_id") or claim.get("positive_evidence_id")})
    vital_ids = sorted({str(obs.get("evidence_id")) for obs in vitals.get("vital_observations") or [] if obs.get("measurement_state") == "valid" and not obs.get("quarantine_reason") and obs.get("evidence_id")})
    quarantined_ids = sorted({str(item.get("evidence_id") or item.get("measurement_id")) for item in vitals.get("quarantined_measurements") or [] if item.get("evidence_id") or item.get("measurement_id")})
    return {"timeline_evidence_ids": timeline_ids, "claim_ids": claim_ids, "claim_evidence_ids": claim_evidence, "vital_evidence_ids": vital_ids, "quarantined_evidence_ids": quarantined_ids}


def aggregate_case(instance_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    names = sorted({key for row in instance_rows for key in row["features"]})
    boolean_names = {name for name in names if name.startswith("claim_field_") or name.startswith("timeline_field_")}
    boolean_names.update({"structured_chief_complaint", "arrival_ambulance", "arrival_walk_in", "valid_pain_score", "official_danger_signal", "spo2_danger_signal", "step_d_assessable", "spo2_observed", "heart_rate_observed", "respiratory_rate_observed"})
    aggregated: Dict[str, Any] = {"realization_count": len(instance_rows)}
    disagreement = 0
    for name in names:
        values = [float(row["features"].get(name, 0)) for row in instance_rows]
        if name in boolean_names:
            aggregated[f"{name}_any"] = int(any(values))
            disagreement += int(len(set(values)) > 1)
        else:
            aggregated[f"{name}_mean"] = round(sum(values) / len(values), 6)
            aggregated[f"{name}_max"] = max(values)
    aggregated["realization_feature_disagreement_count"] = disagreement
    return aggregated


def assign_grouped_folds(case_targets: Dict[str, int], n_folds: int) -> Dict[str, int]:
    by_target: Dict[int, List[str]] = defaultdict(list)
    for case_id, target in case_targets.items():
        by_target[target].append(case_id)
    fold_counts = [Counter() for _ in range(n_folds)]
    assignments: Dict[str, int] = {}
    for target in sorted(by_target):
        for case_id in sorted(by_target[target]):
            fold = min(range(n_folds), key=lambda idx: (fold_counts[idx][target], sum(fold_counts[idx].values()), idx))
            assignments[case_id] = fold
            fold_counts[fold][target] += 1
    return assignments


def supervision_state(gold_esi: Optional[int], features: Dict[str, Any]) -> Tuple[Optional[int], str, List[str], float]:
    if gold_esi not in {3, 4, 5}:
        return None, "resource_not_supervised_high_acuity", ["gold_esi_not_3_4_5"], 0.0
    reasons: List[str] = []
    if features.get("step_a_trigger_count_max", 0) > 0 or features.get("step_b_trigger_count_max", 0) > 0:
        reasons.append("effective_step_a_or_b_signal")
        return None, "resource_label_policy_conflict", reasons, 0.0
    if features.get("active_problem_domain_count_max", 0) <= 0:
        return None, "resource_evidence_insufficient", ["no_active_current_problem_domain"], 0.0
    partial = []
    if features.get("claim_uncertain_count_max", 0) > 0:
        partial.append("uncertain_claim_present")
    if features.get("claim_support_only_count_max", 0) > 0:
        partial.append("support_only_evidence_present")
    if features.get("realization_feature_disagreement_count", 0) > 0:
        partial.append("realization_feature_disagreement")
    target = {5: 0, 4: 1, 3: 2}[gold_esi]
    if partial:
        return target, "resource_partially_supported", partial, 0.5
    return target, "resource_policy_supported", [], 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-file", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--claims-file", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--vitals-file", type=Path, default=DEFAULT_VITALS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--count-manifest", type=Path, default=DEFAULT_COUNT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [args.timeline_file, args.claims_file, args.vitals_file, args.count_manifest]
    if not args.raw_dir.exists():
        raise FileNotFoundError(args.raw_dir)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timelines, claims, vitals = read_jsonl(args.timeline_file), read_jsonl(args.claims_file), read_jsonl(args.vitals_file)
    counts = read_json(args.count_manifest)
    raw_by_instance, raw_by_case, raw_audit = raw_label_index(args.raw_dir)
    timeline_by_instance = {str(row.get("instance_id")): row for row in timelines}
    claims_by_instance = {str(row.get("instance_id")): row for row in claims}
    vitals_by_instance = {str(row.get("instance_id")): row for row in vitals}
    issues: List[Dict[str, Any]] = []
    instance_items: List[Dict[str, Any]] = []
    case_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_measurement_count = 0
    invalid_measurement_ids_all = set()
    for instance_id, timeline in timeline_by_instance.items():
        case_id = str(timeline.get("case_id") or "")
        if instance_id not in claims_by_instance or instance_id not in vitals_by_instance or instance_id not in raw_by_instance:
            issues.append({"issue": "missing_aligned_input", "instance_id": instance_id, "case_id": case_id})
            continue
        vital_row = vitals_by_instance[instance_id]
        invalid_measurement_ids = {
            str(item.get("evidence_id") or item.get("measurement_id"))
            for item in vital_row.get("quarantined_measurements") or []
            if item.get("evidence_id") or item.get("measurement_id")
        }
        invalid_measurement_ids.update(
            str(obs.get("evidence_id") or obs.get("measurement_id"))
            for obs in vital_row.get("vital_observations") or []
            if (obs.get("measurement_state") != "valid" or obs.get("quarantine_reason")) and (obs.get("evidence_id") or obs.get("measurement_id"))
        )
        invalid_measurement_count += len(invalid_measurement_ids)
        invalid_measurement_ids_all.update(invalid_measurement_ids)
        item = {
            "instance_id": instance_id,
            "case_id": case_id,
            "features": build_instance_features(timeline, claims_by_instance[instance_id], vital_row),
            "provenance": feature_provenance(timeline, claims_by_instance[instance_id], vital_row),
        }
        instance_items.append(item)
        case_instances[case_id].append(item)
    for label, rows in (("timeline", timelines), ("claims", claims), ("vitals", vitals)):
        if len(rows) != len(timelines):
            issues.append({"issue": f"{label}_row_count_mismatch", "rows": len(rows), "expected": len(timelines)})

    case_supervision: List[Dict[str, Any]] = []
    case_targets: Dict[str, int] = {}
    case_diagnostics: Dict[str, Dict[str, Any]] = {}
    for case_id, rows in sorted(case_instances.items()):
        aggregate = aggregate_case(rows)
        gold_values = {raw_by_instance.get(item["instance_id"]) for item in rows}
        gold_values.discard(None)
        gold = next(iter(gold_values)) if len(gold_values) == 1 else None
        if len(gold_values) != 1:
            issues.append({"issue": "case_gold_missing_or_conflicting_after_join", "case_id": case_id, "labels": sorted(gold_values)})
        target, status, reasons, weight = supervision_state(gold, aggregate)
        if target is not None:
            case_targets[case_id] = target
        case_diagnostics[case_id] = aggregate
        case_supervision.append({
            "schema_version": SCHEMA_VERSION, "audit_only": True, "case_id": case_id,
            "instance_ids": sorted(item["instance_id"] for item in rows), "realization_count": len(rows),
            "gold_esi": gold, "weak_resource_bucket": target, "supervision_status": status,
            "supervision_weight": weight, "exclusion_reasons": reasons,
        })

    folds = assign_grouped_folds(case_targets, args.folds)
    for row in case_supervision:
        row["fold_id"] = folds.get(row["case_id"])
    fold_rows = [{
        "schema_version": SCHEMA_VERSION, "audit_only": True, "case_id": case_id,
        "instance_ids": sorted(item["instance_id"] for item in case_instances[case_id]),
        "fold_id": fold, "grouped_by": "case_id", "target_available_in_audit_only": True,
    } for case_id, fold in sorted(folds.items())]

    feature_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    for item in sorted(instance_items, key=lambda value: value["instance_id"]):
        model_features = {name: item["features"].get(name, 0.0) for name in sorted(RESOURCE_MODEL_FEATURE_ALLOWLIST)}
        feature_rows.append({
            "schema_version": SCHEMA_VERSION, "instance_id": item["instance_id"], "case_id": item["case_id"],
            **model_features, "feature_matrix_label_free": True, "feature_provenance": item["provenance"],
        })
        diagnostic_rows.append({
            "schema_version": SCHEMA_VERSION, "instance_id": item["instance_id"], "case_id": item["case_id"],
            "features": item["features"], "diagnostic_only": sorted(set(item["features"]) - RESOURCE_MODEL_FEATURE_ALLOWLIST),
            "feature_provenance": item["provenance"],
        })

    duplicate_rows: List[Dict[str, Any]] = []
    for case_id, rows in sorted(case_instances.items()):
        if len(rows) < 2:
            continue
        names = sorted({key for row in rows for key in row["features"]})
        differing = [name for name in names if len({row["features"].get(name, 0) for row in rows}) > 1]
        duplicate_rows.append({
            "schema_version": SCHEMA_VERSION, "audit_only": True, "case_id": case_id,
            "instance_ids": sorted(row["instance_id"] for row in rows),
            "realization_count": len(rows), "differing_feature_count": len(differing), "differing_features": differing,
        })

    model_feature_names = sorted(RESOURCE_MODEL_FEATURE_ALLOWLIST)
    feature_consumed_evidence_ids = {
        evidence_id
        for item in instance_items
        for evidence_id in item["provenance"].get("vital_evidence_ids", [])
    }
    invalid_feature_ids = invalid_measurement_ids_all & feature_consumed_evidence_ids
    feature_values = {name: [numeric(row.get(name)) for row in feature_rows] for name in model_feature_names}
    finite_value_count = sum(1 for values in feature_values.values() for value in values if value is not None)
    constant_features = [name for name, values in feature_values.items() if len(set(values)) <= 1]
    all_zero_features = [name for name, values in feature_values.items() if values and set(values) == {0.0}]
    label_derived_keys = {"fold_id", "gold_esi", "weak_resource_bucket", "supervision_status", "supervision_weight"}
    matrix_label_derived_hits = sum(1 for row in feature_rows for key in label_derived_keys if key in row)
    forbidden_hits = sum(len(recursive_forbidden(row)) for row in feature_rows)
    status_counts = Counter(row["supervision_status"] for row in case_supervision)
    target_counts = Counter(str(row["weak_resource_bucket"]) for row in case_supervision if row["weak_resource_bucket"] is not None)
    fold_counts = Counter(str(row["fold_id"]) for row in fold_rows)
    training_weight_rows = []
    weight_issues = []
    supervision_by_case = {row["case_id"]: row for row in case_supervision}
    for item in sorted(instance_items, key=lambda value: value["instance_id"]):
        state = supervision_by_case.get(item["case_id"])
        if not state or state.get("weak_resource_bucket") is None:
            continue
        expected_weight = float(state["supervision_weight"]) / float(state["realization_count"])
        training_weight_rows.append({
            "schema_version": SCHEMA_VERSION, "audit_only": True,
            "instance_id": item["instance_id"], "case_id": item["case_id"],
            "fold_id": state["fold_id"], "instance_weight": expected_weight,
        })
    weight_by_case = defaultdict(float)
    for row in training_weight_rows:
        weight_by_case[row["case_id"]] += row["instance_weight"]
    for case_id, state in supervision_by_case.items():
        if state.get("weak_resource_bucket") is not None and abs(weight_by_case[case_id] - float(state["supervision_weight"])) > 1e-9:
            weight_issues.append(case_id)

    fold_feature_rows = []
    eligible_case_ids = set(case_targets)
    for fold in range(args.folds):
        train_rows = [row for row in feature_rows if row["case_id"] in eligible_case_ids and folds.get(row["case_id"]) != fold]
        validation_rows = [row for row in feature_rows if row["case_id"] in eligible_case_ids and folds.get(row["case_id"]) == fold]
        zero_train = [name for name in model_feature_names if len({row.get(name) for row in train_rows}) <= 1]
        validation_only = [name for name in model_feature_names if any(row.get(name) != 0 for row in validation_rows) and all(row.get(name) == 0 for row in train_rows)]
        fold_feature_rows.append({
            "schema_version": SCHEMA_VERSION, "audit_only": True, "fold_id": fold,
            "zero_variance_train_features": zero_train,
            "validation_only_nonzero_features": validation_only,
            "effective_model_feature_count": len(model_feature_names) - len(zero_train),
        })
    sentinel_checks = {
        "32404086_is_policy_conflict": next((row["supervision_status"] == "resource_label_policy_conflict" for row in case_supervision if row["case_id"] == "32404086"), False),
        "38016499_is_high_acuity_excluded": next((row["supervision_status"] == "resource_not_supervised_high_acuity" and row["weak_resource_bucket"] is None for row in case_supervision if row["case_id"] == "38016499"), False),
    }
    hard_gates = {
        "count_gate_passed": len(timelines) == counts.get("expected_instance_count") and len(case_instances) == counts.get("expected_case_count") and len(feature_rows) == len(timelines),
        "case_grouping_passed": all(len(rows) >= 1 for rows in case_instances.values()),
        "duplicate_case_split_passed": all(row["case_id"] in folds for row in duplicate_rows if any(item["case_id"] == row["case_id"] for item in fold_rows)),
        "prediction_input_forbidden_key_gate_passed": forbidden_hits == 0,
        "alignment_gate_passed": not issues,
        "instance_feature_plane_passed": len(feature_rows) == len(timelines),
        "fold_metadata_separated": matrix_label_derived_hits == 0,
        "feature_allowlist_passed": all(set(row) >= set(model_feature_names) for row in feature_rows),
        "feature_values_finite": finite_value_count == len(feature_rows) * len(model_feature_names),
        "known_conflict_sentinels_passed": all(sentinel_checks.values()),
        "invalid_measurement_feature_gate_passed": len(invalid_feature_ids) == 0,
        "training_weight_gate_passed": not weight_issues,
        "fold_local_feature_audit_passed": len(fold_feature_rows) == args.folds,
    }
    audit = {
        "schema_version": SCHEMA_VERSION, "audit_only": True, "purpose": "resource_supervision_feasibility_before_model_fit",
        "inputs": {
            "timeline_file": str(args.timeline_file), "timeline_sha256": sha256(args.timeline_file),
            "claims_file": str(args.claims_file), "claims_sha256": sha256(args.claims_file),
            "vitals_file": str(args.vitals_file), "vitals_sha256": sha256(args.vitals_file), "raw_dir": str(args.raw_dir),
            "raw_label_manifest_sha256": raw_label_manifest_hash(args.raw_dir),
        },
        "expected_counts": {"instances": counts.get("expected_instance_count"), "cases": counts.get("expected_case_count"), "duplicate_groups": counts.get("expected_duplicate_group_count")},
        "observed_counts": {
            "timeline_instances": len(timelines), "claims_instances": len(claims), "vital_instances": len(vitals),
            "instance_feature_rows": len(feature_rows), "supervision_case_rows": len(case_supervision),
            "cases": len(case_instances), "duplicate_groups": len(duplicate_rows),
        },
        "raw_label_audit": raw_audit,
        "supervision_status_counts": dict(status_counts), "weak_resource_bucket_counts": dict(target_counts),
        "fold_counts": dict(fold_counts), "fold_count": args.folds, "eligible_case_count": len(case_targets),
        "model_feature_count": len(model_feature_names), "diagnostic_feature_count": len(diagnostic_rows[0]["features"]) if diagnostic_rows else 0,
        "constant_feature_count": len(constant_features), "all_zero_feature_count": len(all_zero_features),
        "constant_features": constant_features, "all_zero_features": all_zero_features,
        "feature_value_finite_count": finite_value_count, "feature_value_total": len(feature_rows) * len(model_feature_names),
        "invalid_measurement_seen_count": invalid_measurement_count, "invalid_measurement_feature_count": len(invalid_feature_ids),
        "invalid_measurement_ids": sorted(invalid_measurement_ids_all), "feature_consumed_vital_evidence_ids": sorted(feature_consumed_evidence_ids),
        "training_weight_issue_count": len(weight_issues), "fold_local_feature_audit_count": len(fold_feature_rows),
        "label_derived_metadata_in_matrix_count": matrix_label_derived_hits,
        "prediction_input_forbidden_key_hits": forbidden_hits, "missing_alignment_issue_count": len(issues),
        "sentinel_checks": sentinel_checks, "model_fit_performed": False, "hard_gates": hard_gates,
    }
    audit["release_gate_passed"] = all(bool(value) for value in hard_gates.values()) and len(case_targets) > 0 and len(all_zero_features) == 0

    write_json(args.output_dir / "resource_feasibility_audit.json", audit)
    write_json(args.output_dir / "resource_feature_dictionary.json", {
        "schema_version": SCHEMA_VERSION, "label_free": True, "model_feature_count": len(model_feature_names),
        "model_features": {key: {"source": "01_02_03_prediction_safe_outputs"} for key in model_feature_names},
        "diagnostic_features_are_not_model_inputs": True,
        "excluded_from_matrix": sorted(label_derived_keys | {"ground_truth", "acuity", "triage", "gold_esi", "patient_persona", "nurse_persona", "raw_text"}),
    })
    write_jsonl(args.output_dir / "resource_instance_feature_matrix.jsonl", feature_rows)
    write_jsonl(args.output_dir / "resource_feature_diagnostics.jsonl", diagnostic_rows)
    write_jsonl(args.output_dir / "resource_supervision_case_audit.jsonl", case_supervision)
    write_jsonl(args.output_dir / "resource_grouped_fold_manifest.jsonl", fold_rows)
    write_jsonl(args.output_dir / "resource_training_weight_manifest.jsonl", training_weight_rows)
    write_jsonl(args.output_dir / "resource_fold_feature_availability.jsonl", fold_feature_rows)
    write_jsonl(args.output_dir / "resource_duplicate_pair_diagnostics.jsonl", duplicate_rows)
    write_jsonl(args.output_dir / "resource_audit_issues.jsonl", issues)
    print(f"Processed instances: {len(timelines)}")
    print(f"Case groups: {len(case_instances)}")
    print(f"Eligible weak-supervision cases: {len(case_targets)}")
    print(f"Model feature count: {len(model_feature_names)}")
    print("Model fit performed: False")
    print(f"Release gate passed: {audit['release_gate_passed']}")


if __name__ == "__main__":
    main()
