#!/usr/bin/env python3
"""Audit TRIBOT raw transcripts and build leakage-safe grouped split manifests.

The output is audit-only. Prediction modules must read raw transcripts and the
shared pure-code clinical_data_contract, never 00 summaries or manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clinical_data_contract import classify_pain_contract, classify_vital_contract


SCHEMA_VERSION = "00_dataset_engineering_audit_v1.0_case_grouped"
REQUIRED_TOP_LEVEL = {
    "dataset",
    "case_id",
    "run_uuid",
    "vignette",
    "ground_truth",
    "patient_persona",
    "nurse_persona",
    "history",
}
VITAL_FIELDS = ("temperature", "heartrate", "resprate", "o2sat", "sbp")
ALLOWED_ACUITY = {1, 2, 3, 4, 5}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def normalize_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def dialogue_payload(record: dict[str, Any], normalized: bool) -> list[dict[str, Any]]:
    payload = []
    for event in record.get("history", []):
        if not isinstance(event, dict):
            payload.append({"invalid_event": event})
            continue
        text = event.get("original")
        if normalized:
            text = normalize_text(text)
        payload.append(
            {
                key: value
                for key, value in {
                    "turn": event.get("turn"),
                    "actor": event.get("actor"),
                    "event": event.get("event"),
                    "name": event.get("name"),
                    "value": event.get("value"),
                    "original": text,
                }.items()
                if value is not None
            }
        )
    return payload


def dialogue_hash(record: dict[str, Any], normalized: bool = False) -> str:
    payload = json.dumps(
        dialogue_payload(record, normalized),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_numeric(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return left == right


def allocate_splits(case_ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(case_ids, key=lambda case_id: stable_hash(case_id, seed))
    raw = {name: len(ordered) * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = len(ordered) - sum(counts.values())
    for name in sorted(SPLIT_RATIOS, key=lambda key: (-(raw[key] - counts[key]), key))[:remaining]:
        counts[name] += 1
    assignment = {}
    cursor = 0
    for name in ("train", "validation", "test"):
        for case_id in ordered[cursor : cursor + counts[name]]:
            assignment[case_id] = name
        cursor += counts[name]
    return assignment


def build_case_assignments(cases: dict[str, list[dict[str, Any]]], seed: int) -> tuple[dict[str, str], dict[str, int]]:
    by_label = defaultdict(list)
    for case_id, rows in cases.items():
        by_label[int(rows[0]["acuity_audit_only"])].append(case_id)

    split_by_case = {}
    fold_by_case = {}
    fold_load = Counter()
    for acuity in sorted(by_label):
        case_ids = by_label[acuity]
        split_by_case.update(allocate_splits(case_ids, seed + acuity * 101))
        for case_id in sorted(case_ids, key=lambda value: stable_hash(value, seed + acuity * 1009)):
            fold = min(range(5), key=lambda value: (fold_load[value], value))
            fold_by_case[case_id] = fold
            fold_load[fold] += 1
    return split_by_case, fold_by_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("transcripts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/00_dataset_audit_v1"))
    parser.add_argument("--expected-instance-count", type=int, default=1010)
    parser.add_argument("--expected-case-count", type=int, default=541)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    files = sorted(args.input_dir.glob("*.json"))
    load_errors = []
    structural_errors = []
    measurement_rows = []
    instances = []
    seen_run_uuid = set()
    seen_instance_id = set()

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            load_errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        missing = sorted(REQUIRED_TOP_LEVEL - set(record))
        if missing:
            structural_errors.append({"file": path.name, "error": "missing_top_level_keys", "details": missing})
            continue
        dataset = str(record.get("dataset"))
        case_id = str(record.get("case_id"))
        run_uuid = str(record.get("run_uuid"))
        instance_id = f"{case_id}__{run_uuid}"
        dataset_instance_id = f"{dataset}__{case_id}__{run_uuid}"
        errors = []
        try:
            uuid.UUID(run_uuid)
        except (ValueError, AttributeError):
            errors.append("run_uuid_not_uuid")
        if path.stem != run_uuid:
            errors.append("filename_run_uuid_mismatch")
        if run_uuid in seen_run_uuid:
            errors.append("duplicate_run_uuid")
        if instance_id in seen_instance_id:
            errors.append("duplicate_instance_id")
        seen_run_uuid.add(run_uuid)
        seen_instance_id.add(instance_id)

        vignette = record.get("vignette") or {}
        ground_truth = record.get("ground_truth") or {}
        try:
            acuity = int(ground_truth.get("acuity"))
        except (TypeError, ValueError):
            acuity = -1
        if acuity not in ALLOWED_ACUITY:
            errors.append("invalid_ground_truth_acuity")
        if vignette.get("acuity") != ground_truth.get("acuity"):
            errors.append("vignette_ground_truth_acuity_mismatch")
        if vignette.get("chiefcomplaint") != ground_truth.get("chiefcomplaint"):
            errors.append("vignette_ground_truth_chiefcomplaint_mismatch")
        if not compare_numeric(vignette.get("pain"), ground_truth.get("pain")):
            errors.append("vignette_ground_truth_pain_mismatch")
        gt_vitals = ground_truth.get("vitals") or {}
        for field in VITAL_FIELDS:
            if not compare_numeric(vignette.get(field), gt_vitals.get(field)):
                errors.append(f"vignette_ground_truth_vital_mismatch:{field}")

        history = record.get("history")
        if not isinstance(history, list) or not history:
            errors.append("history_missing_or_empty")
            history = []
        previous_turn = -math.inf
        actor_counts = Counter()
        for event_index, event in enumerate(history):
            if not isinstance(event, dict):
                errors.append(f"history_event_not_object:{event_index}")
                continue
            actor = event.get("actor")
            actor_counts[str(actor)] += 1
            if actor not in {"nurse", "patient", "system"}:
                errors.append(f"invalid_actor:{event_index}")
            turn = event.get("turn")
            if not isinstance(turn, int):
                errors.append(f"invalid_turn:{event_index}")
            elif turn < previous_turn:
                errors.append(f"non_monotonic_turn:{event_index}")
            else:
                previous_turn = turn
            if actor in {"nurse", "patient"} and not str(event.get("original") or "").strip():
                errors.append(f"missing_original_utterance:{event_index}")
            if actor == "system" and event.get("event") == "vital":
                contract = classify_vital_contract(str(event.get("name")), event.get("value"))
                if not contract["usable_for_clinical_reasoning"]:
                    measurement_rows.append(
                        {
                            "dataset": dataset,
                            "case_id": case_id,
                            "run_uuid": run_uuid,
                            "instance_id": instance_id,
                            "location": "history_system_vital",
                            "event_index": event_index,
                            **contract,
                            "audit_only": True,
                        }
                    )
        if actor_counts["nurse"] != actor_counts["patient"]:
            errors.append("nurse_patient_utterance_count_mismatch")

        pain_contract = classify_pain_contract(vignette.get("pain"))
        if not pain_contract["usable_for_clinical_reasoning"]:
            measurement_rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "run_uuid": run_uuid,
                    "instance_id": instance_id,
                    "location": "vignette_structured_pain",
                    "field": "pain",
                    **pain_contract,
                    "audit_only": True,
                }
            )
        for field in VITAL_FIELDS:
            contract = classify_vital_contract(field, vignette.get(field))
            if not contract["usable_for_clinical_reasoning"]:
                measurement_rows.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "run_uuid": run_uuid,
                        "instance_id": instance_id,
                        "location": "vignette_structured_vital",
                        "field": field,
                        **contract,
                        "audit_only": True,
                    }
                )

        for error in errors:
            structural_errors.append({"file": path.name, "instance_id": instance_id, "error": error})
        instances.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "run_uuid": run_uuid,
                "instance_id": instance_id,
                "dataset_instance_id": dataset_instance_id,
                "file_name": path.name,
                "file_sha256": sha256_file(path),
                "record": record,
                "acuity_audit_only": acuity,
                "dialogue_hash": dialogue_hash(record),
                "normalized_dialogue_hash": dialogue_hash(record, normalized=True),
            }
        )

    cases = defaultdict(list)
    for row in instances:
        cases[row["case_id"]].append(row)

    case_consistency_errors = []
    for case_id, rows in cases.items():
        for field in ("dataset", "acuity_audit_only"):
            if len({row[field] for row in rows}) != 1:
                case_consistency_errors.append({"case_id": case_id, "error": f"case_{field}_conflict"})
        if len({json.dumps(row["record"].get("vignette"), sort_keys=True) for row in rows}) != 1:
            case_consistency_errors.append({"case_id": case_id, "error": "case_vignette_conflict"})
        if len({json.dumps(row["record"].get("ground_truth"), sort_keys=True) for row in rows}) != 1:
            case_consistency_errors.append({"case_id": case_id, "error": "case_ground_truth_conflict"})

    normalized_groups = defaultdict(list)
    for row in instances:
        normalized_groups[row["normalized_dialogue_hash"]].append(row)
    duplicate_rows = []
    excluded_run_uuids = set()
    canonical_by_hash = {}
    cross_case_duplicate_groups = 0
    for content_hash, rows in normalized_groups.items():
        canonical = min(rows, key=lambda row: row["run_uuid"])
        canonical_by_hash[content_hash] = canonical["run_uuid"]
        if len(rows) <= 1:
            continue
        if len({row["case_id"] for row in rows}) > 1:
            cross_case_duplicate_groups += 1
        for row in sorted(rows, key=lambda value: value["run_uuid"]):
            is_canonical = row["run_uuid"] == canonical["run_uuid"]
            if not is_canonical:
                excluded_run_uuids.add(row["run_uuid"])
            duplicate_rows.append(
                {
                    "normalized_dialogue_hash": content_hash,
                    "case_id": row["case_id"],
                    "run_uuid": row["run_uuid"],
                    "canonical_run_uuid": canonical["run_uuid"],
                    "is_canonical": is_canonical,
                    "same_case_group": len({item["case_id"] for item in rows}) == 1,
                    "audit_only": True,
                }
            )

    split_by_case, fold_by_case = build_case_assignments(cases, args.seed)
    eligible_by_case = Counter(
        row["case_id"] for row in instances if row["run_uuid"] not in excluded_run_uuids
    )
    instance_manifest = []
    for row in sorted(instances, key=lambda value: (value["case_id"], value["run_uuid"])):
        eligible = row["run_uuid"] not in excluded_run_uuids
        instance_manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "run_uuid": row["run_uuid"],
                "instance_id": row["instance_id"],
                "dataset_instance_id": row["dataset_instance_id"],
                "file_name": row["file_name"],
                "file_sha256": row["file_sha256"],
                "holdout_split": split_by_case[row["case_id"]],
                "cv_fold": fold_by_case[row["case_id"]],
                "eligible_for_training": eligible,
                "training_exclusion_reasons": [] if eligible else ["normalized_dialogue_duplicate_within_case"],
                "canonical_dialogue_run_uuid": canonical_by_hash[row["normalized_dialogue_hash"]],
                "case_realization_count_raw": len(cases[row["case_id"]]),
                "case_realization_count_eligible": eligible_by_case[row["case_id"]],
                "case_normalized_sample_weight": 1.0 / eligible_by_case[row["case_id"]] if eligible else 0.0,
                "acuity_audit_only": row["acuity_audit_only"],
                "manifest_is_audit_only": True,
                "prediction_modules_must_not_read": True,
            }
        )

    case_manifest = []
    for case_id, rows in sorted(cases.items()):
        eligible_rows = [row for row in rows if row["run_uuid"] not in excluded_run_uuids]
        case_manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": rows[0]["dataset"],
                "case_id": case_id,
                "holdout_split": split_by_case[case_id],
                "cv_fold": fold_by_case[case_id],
                "acuity_audit_only": rows[0]["acuity_audit_only"],
                "realization_count_raw": len(rows),
                "realization_count_eligible": len(eligible_rows),
                "eligible_run_uuids": sorted(row["run_uuid"] for row in eligible_rows),
                "excluded_duplicate_run_uuids": sorted(
                    row["run_uuid"] for row in rows if row["run_uuid"] in excluded_run_uuids
                ),
                "manifest_is_audit_only": True,
                "prediction_modules_must_not_read": True,
            }
        )

    split_case_sets = {
        split: {row["case_id"] for row in case_manifest if row["holdout_split"] == split}
        for split in SPLIT_RATIOS
    }
    split_leakage = sum(
        len(split_case_sets[left] & split_case_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    fold_case_sets = {
        fold: {row["case_id"] for row in case_manifest if row["cv_fold"] == fold}
        for fold in range(5)
    }
    fold_leakage = sum(
        len(fold_case_sets[left] & fold_case_sets[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )

    validation_errors = load_errors + structural_errors + case_consistency_errors
    count_gate = len(instances) == args.expected_instance_count and len(cases) == args.expected_case_count
    hard_gates = {
        "expected_counts_match": count_gate,
        "load_errors_zero": not load_errors,
        "structural_errors_zero": not structural_errors,
        "case_consistency_errors_zero": not case_consistency_errors,
        "run_uuid_unique": len(seen_run_uuid) == len(instances),
        "instance_id_unique": len(seen_instance_id) == len(instances),
        "cross_case_normalized_dialogue_duplicates_zero": cross_case_duplicate_groups == 0,
        "duplicate_content_excluded_from_training": len(excluded_run_uuids)
        == sum(not row["eligible_for_training"] for row in instance_manifest),
        "holdout_case_overlap_zero": split_leakage == 0,
        "cv_case_overlap_zero": fold_leakage == 0,
        "every_case_has_eligible_realization": all(row["realization_count_eligible"] > 0 for row in case_manifest),
        "case_weights_sum_to_one": all(
            math.isclose(
                sum(
                    row["case_normalized_sample_weight"]
                    for row in instance_manifest
                    if row["case_id"] == case_id
                ),
                1.0,
                abs_tol=1e-9,
            )
            for case_id in cases
        ),
    }
    release_gate = all(hard_gates.values())

    jsonl_dump(args.output_dir / "instance_manifest_audit_only.jsonl", instance_manifest)
    jsonl_dump(args.output_dir / "case_split_manifest_audit_only.jsonl", case_manifest)
    jsonl_dump(args.output_dir / "duplicate_dialogue_audit_only.jsonl", duplicate_rows)
    jsonl_dump(args.output_dir / "measurement_quarantine_audit_only.jsonl", measurement_rows)
    jsonl_dump(args.output_dir / "validation_errors.jsonl", validation_errors)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_only": True,
        "prediction_modules_must_not_read_outputs": True,
        "input_dir": str(args.input_dir.resolve()),
        "input_json_file_count": len(files),
        "loaded_instance_count": len(instances),
        "unique_case_count": len(cases),
        "unique_run_uuid_count": len(seen_run_uuid),
        "unique_instance_id_count": len(seen_instance_id),
        "duplicate_case_group_count": sum(len(rows) > 1 for rows in cases.values()),
        "duplicate_case_extra_realization_count": sum(len(rows) - 1 for rows in cases.values()),
        "acuity_distribution_instance_audit_only": dict(Counter(row["acuity_audit_only"] for row in instances)),
        "acuity_distribution_case_audit_only": dict(Counter(row["acuity_audit_only"] for row in case_manifest)),
        "holdout_case_counts": dict(Counter(row["holdout_split"] for row in case_manifest)),
        "holdout_eligible_instance_counts": dict(
            Counter(row["holdout_split"] for row in instance_manifest if row["eligible_for_training"])
        ),
        "cv_case_counts": dict(Counter(str(row["cv_fold"]) for row in case_manifest)),
        "training_eligible_instance_count": sum(row["eligible_for_training"] for row in instance_manifest),
        "training_excluded_duplicate_instance_count": len(excluded_run_uuids),
        "normalized_duplicate_group_count": sum(len(rows) > 1 for rows in normalized_groups.values()),
        "cross_case_normalized_duplicate_group_count": cross_case_duplicate_groups,
        "measurement_quarantine_record_count": len(measurement_rows),
        "measurement_quarantine_counts_by_location": dict(Counter(row["location"] for row in measurement_rows)),
        "measurement_quarantine_counts_by_reason": dict(Counter(row["quarantine_reason"] for row in measurement_rows)),
        "load_error_count": len(load_errors),
        "structural_error_count": len(structural_errors),
        "case_consistency_error_count": len(case_consistency_errors),
        "validation_error_count": len(validation_errors),
        "hard_gates": hard_gates,
        "release_gate_passed": release_gate,
        "identity_contract": {
            "realization_id": "run_uuid",
            "instance_id": "case_id__run_uuid",
            "group_id": "case_id",
            "primary_training_and_evaluation_unit": "case_id",
            "split_group_key": "case_id",
            "duplicate_weighting": "1 / eligible_realization_count_within_case",
        },
        "measurement_policy": {
            "invalid_measurements_are_field_level_quarantine": True,
            "invalid_measurements_do_not_fail_structurally_valid_rows": True,
            "01_must_reapply_shared_contract": True,
        },
    }
    json_dump(args.output_dir / "dataset_audit.json", summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_gate_passed": release_gate,
        "input_tree_fingerprint": hashlib.sha256(
            "".join(f"{row['file_name']}:{row['file_sha256']}\n" for row in instance_manifest).encode("utf-8")
        ).hexdigest(),
        "script_sha256": sha256_file(Path(__file__)),
        "clinical_data_contract_sha256": sha256_file(Path(__file__).with_name("clinical_data_contract.py")),
        "output_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "dataset_audit.json",
                "instance_manifest_audit_only.jsonl",
                "case_split_manifest_audit_only.jsonl",
                "duplicate_dialogue_audit_only.jsonl",
                "measurement_quarantine_audit_only.jsonl",
                "validation_errors.jsonl",
            )
        },
    }
    json_dump(args.output_dir / "AUDIT_MANIFEST.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not release_gate:
        raise SystemExit("00 release gate failed; inspect validation_errors.jsonl and dataset_audit.json")


if __name__ == "__main__":
    main()
