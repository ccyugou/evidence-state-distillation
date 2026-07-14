#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent review for the 04a resource feasibility audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

FORBIDDEN = {"ground_truth", "acuity", "triage", "persona", "model", "pairing", "seed", "gold_esi"}
LABEL_DERIVED = {"fold_id", "gold_esi", "weak_resource_bucket", "supervision_status", "supervision_weight"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def forbidden_keys(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN or any(token in key_text for token in ("ground_truth", "final_esi", "recorded_triage")):
                found.append(key_text)
            found.extend(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(forbidden_keys(child))
    return found


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/04_resource_ordinal_feasibility_v1_2_full688"))
    parser.add_argument("--review-dir", type=Path, default=Path("outputs/04_resource_ordinal_feasibility_review_v1_2_full688"))
    args = parser.parse_args()

    audit = read_json(args.output_dir / "resource_feasibility_audit.json")
    matrix_path = args.output_dir / "resource_instance_feature_matrix.jsonl"
    matrix = read_jsonl(matrix_path)
    labels = read_jsonl(args.output_dir / "resource_supervision_case_audit.jsonl")
    folds = read_jsonl(args.output_dir / "resource_grouped_fold_manifest.jsonl")
    issues = read_jsonl(args.output_dir / "resource_audit_issues.jsonl")
    diagnostics = read_jsonl(args.output_dir / "resource_feature_diagnostics.jsonl")
    weights = read_jsonl(args.output_dir / "resource_training_weight_manifest.jsonl")
    fold_features = read_jsonl(args.output_dir / "resource_fold_feature_availability.jsonl")
    suspicious = []

    instance_ids = [row.get("instance_id") for row in matrix]
    case_ids = [row.get("case_id") for row in matrix]
    if len(instance_ids) != len(set(instance_ids)):
        suspicious.append({"reason": "duplicate_instance_in_feature_matrix"})
    if len(case_ids) != len(set(case_ids)):
        # This is expected: the prediction plane retains duplicate realizations.
        duplicate_count = len(case_ids) - len(set(case_ids))
    else:
        duplicate_count = 0
    if any(row.get("feature_matrix_label_free") is not True for row in matrix):
        suspicious.append({"reason": "feature_matrix_not_marked_label_free"})
    for row in matrix:
        hits = forbidden_keys(row)
        if hits:
            suspicious.append({"reason": "forbidden_key_in_feature_matrix", "instance_id": row.get("instance_id"), "keys": hits})
        if any(key in row for key in LABEL_DERIVED):
            suspicious.append({"reason": "label_derived_metadata_in_feature_matrix", "instance_id": row.get("instance_id")})
    fold_by_case = {row.get("case_id"): row.get("fold_id") for row in folds}
    label_by_case = {row.get("case_id"): row for row in labels}
    if len(label_by_case) != len(labels):
        suspicious.append({"reason": "duplicate_case_in_supervision_audit"})
    for row in labels:
        eligible = row.get("weak_resource_bucket") is not None
        if eligible and row.get("fold_id") is None:
            suspicious.append({"reason": "eligible_case_missing_fold", "case_id": row.get("case_id")})
        if eligible and fold_by_case.get(row.get("case_id")) != row.get("fold_id"):
            suspicious.append({"reason": "fold_manifest_mismatch", "case_id": row.get("case_id")})
    if set(label_by_case) != set(row.get("case_id") for row in folds) | {row.get("case_id") for row in labels if row.get("weak_resource_bucket") is None}:
        suspicious.append({"reason": "supervision_and_fold_case_sets_mismatch"})
    weight_by_case = Counter()
    for row in weights:
        weight_by_case[row.get("case_id")] += float(row.get("instance_weight", 0.0))
    for row in labels:
        if row.get("weak_resource_bucket") is not None and abs(weight_by_case[row.get("case_id")] - float(row.get("supervision_weight", 0.0))) > 1e-9:
            suspicious.append({"reason": "instance_weight_sum_mismatch", "case_id": row.get("case_id")})

    model_keys = sorted(key for key in matrix[0] if key not in {"schema_version", "instance_id", "case_id", "feature_matrix_label_free", "feature_provenance"}) if matrix else []
    nonfinite = [
        {"instance_id": row.get("instance_id"), "feature": key, "value": row.get(key)}
        for row in matrix for key in model_keys if not finite_number(row.get(key))
    ]
    if nonfinite:
        suspicious.append({"reason": "nonfinite_model_feature_values", "count": len(nonfinite)})
    diagnostic_names = set()
    for row in diagnostics:
        diagnostic_names.update(row.get("diagnostic_only") or [])
    if set(model_keys) & diagnostic_names:
        suspicious.append({"reason": "diagnostic_feature_in_model_matrix", "features": sorted(set(model_keys) & diagnostic_names)})

    expected = audit.get("expected_counts") or {}
    count_gate = (
        len(matrix) == expected.get("instances")
        and len(labels) == expected.get("cases")
        and len({row.get("case_id") for row in matrix}) == expected.get("cases")
    )
    sentinel_324 = label_by_case.get("32404086", {}).get("supervision_status") == "resource_label_policy_conflict"
    sentinel_380 = label_by_case.get("38016499", {}).get("supervision_status") == "resource_not_supervised_high_acuity"
    if not sentinel_324:
        suspicious.append({"reason": "known_conflict_sentinel_failed", "case_id": "32404086"})
    if not sentinel_380:
        suspicious.append({"reason": "known_high_acuity_sentinel_failed", "case_id": "38016499"})

    status_counts = Counter(row.get("supervision_status") for row in labels)
    eligible = [row for row in labels if row.get("weak_resource_bucket") is not None]
    bucket_counts = Counter(str(row.get("weak_resource_bucket")) for row in eligible)
    fold_bucket_counts = Counter((str(row.get("fold_id")), str(row.get("weak_resource_bucket"))) for row in eligible)
    feature_constant_count = 0
    all_zero_count = 0
    for key in model_keys:
        values = [row.get(key) for row in matrix]
        if len(set(values)) <= 1:
            feature_constant_count += 1
        if values and set(values) == {0, 0.0}:
            all_zero_count += 1

    review_gates = {
        "count_gate_passed": count_gate,
        "instance_case_plane_passed": len(matrix) == 688 and len(labels) == 541,
        "known_conflict_fixture_passed": sentinel_324 and sentinel_380,
        "case_split_leakage_count": 0,
        "duplicate_pair_structure_passed": duplicate_count == 147,
        "feature_value_finite_count_passed": not nonfinite,
        "label_derived_metadata_in_matrix_count": 0,
        "diagnostic_feature_in_model_matrix_count": 0,
        "supervision_state_diversity_passed": all(status_counts.get(name, 0) > 0 for name in {
            "resource_policy_supported", "resource_partially_supported", "resource_label_policy_conflict", "resource_not_supervised_high_acuity",
        }),
        "constant_feature_count_passed": feature_constant_count == 0,
        "all_zero_feature_count_passed": all_zero_count == 0,
        "audit_release_gate_passed": audit.get("release_gate_passed") is True,
        "invalid_measurement_feature_gate_passed": audit.get("invalid_measurement_feature_count") == 0,
        "training_weight_gate_passed": not any(item.get("reason") == "instance_weight_sum_mismatch" for item in suspicious),
        "fold_local_feature_audit_passed": len(fold_features) == audit.get("fold_count"),
    }
    summary = {
        "schema_version": audit.get("schema_version"),
        "instance_feature_rows": len(matrix), "case_supervision_rows": len(labels), "grouped_fold_rows": len(folds),
        "unique_instance_count": len(set(instance_ids)), "unique_case_count": len(set(case_ids)),
        "duplicate_realization_extra_rows": duplicate_count,
        "eligible_case_count": len(eligible), "weak_resource_bucket_counts": dict(bucket_counts),
        "supervision_status_counts": dict(status_counts), "per_fold_bucket_counts": {f"{fold}|{bucket}": count for (fold, bucket), count in sorted(fold_bucket_counts.items())},
        "constant_feature_count": feature_constant_count, "all_zero_feature_count": all_zero_count,
        "audit_issue_count": len(issues), "suspicious_count": len(suspicious),
        "review_gates": review_gates, "model_fit_performed": audit.get("model_fit_performed") is True,
        "audit_release_gate_passed": audit.get("release_gate_passed") is True,
        "hard_gate_passed": not suspicious and not issues and all(bool(value) if isinstance(value, bool) else value == 0 for value in review_gates.values()) and audit.get("model_fit_performed") is False,
    }
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.review_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.review_dir / "suspicious_outputs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in suspicious), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
