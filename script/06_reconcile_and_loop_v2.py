#!/usr/bin/env python3
"""Deterministic ESI reconciliation and stage-owned loop planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "06_deterministic_policy_reconciliation_v2.0"
RESOURCE_TO_ESI = {0: 5, 1: 4, 2: 3}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_from_raw(transcripts: Path) -> dict[str, int]:
    labels = {}
    for path in transcripts.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        labels[str(row["case_id"])] = int(row["ground_truth"]["acuity"])
    return labels


def confusion(y_true: list[int], y_pred: list[int], labels: list[int]) -> list[list[int]]:
    index = {value: i for i, value in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        matrix[index[truth]][index[prediction]] += 1
    return matrix


def metrics(y_true: list[int], y_pred: list[int], labels: list[int] | None = None) -> dict[str, Any]:
    labels = labels or [1, 2, 3, 4, 5]
    matrix = confusion(y_true, y_pred, labels)
    recalls, f1s = {}, []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[str(label)] = recall
        f1s.append(f1)
    accuracy = sum(matrix[i][i] for i in range(len(labels))) / len(y_true) if y_true else 0.0
    maoe = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return {
        "case_count": len(y_true),
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "maoe": maoe,
        "class_recall": recalls,
        "confusion_matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--stage02", type=Path, required=True)
    parser.add_argument("--stage03", type=Path, required=True)
    parser.add_argument("--stage04-oof", type=Path, required=True)
    parser.add_argument("--stage05-dir", type=Path, required=True)
    parser.add_argument("--stage05b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-max-prob", type=float, default=0.55)
    parser.add_argument("--resource-margin", type=float, default=0.12)
    args = parser.parse_args()

    labels = labels_from_raw(args.transcripts)
    rows02 = read_jsonl(args.stage02)
    rows03 = read_jsonl(args.stage03)
    oof = {row["case_id"]: row for row in read_jsonl(args.stage04_oof)}
    harness_audit = json.loads((args.stage05_dir / "05_build_audit.json").read_text(encoding="utf-8"))
    attributions = read_jsonl(args.stage05_dir / "05_error_attributions.jsonl")
    gold_audit = json.loads((args.stage05b_dir / "05b_build_audit.json").read_text(encoding="utf-8"))
    gold_reviews = {row["case_id"]: row for row in read_jsonl(args.stage05b_dir / "05b_gold_esi_observability.jsonl")}

    case02: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case03: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows02:
        case02[row["case_id"]].append(row)
    for row in rows03:
        case03[row["case_id"]].append(row)

    predictions = []
    for case_id in sorted(labels):
        step_a = [signal for row in case02[case_id] for signal in row["confirmed_step_a_signals"]]
        step_b = [signal for row in case02[case_id] for signal in row["confirmed_step_b_signals"]]
        step_b_fields = sorted({signal["policy_field"] for signal in step_b})
        uncertain_fields = sorted({
            candidate["policy_field"]
            for row in case02[case_id]
            for candidate in row["policy_candidates"]
            if candidate.get("validated_status") == "uncertain_policy_review"
        })
        danger = any(row["official_danger_zone_signal_present"] for row in case03[case_id])
        resource = oof.get(case_id)
        decision_path, prediction, review_reasons = [], None, []
        if step_a:
            prediction = 1
            decision_path.append("step_a_confirmed_to_esi1")
        elif step_b:
            prediction = 2
            if step_b_fields == ["severe_pain_or_distress"]:
                review_reasons.append("isolated_severe_pain_requires_clinical_review")
                decision_path.append("isolated_severe_pain_step_b_held_for_review")
            else:
                decision_path.append("high_specificity_step_b_confirmed_to_esi2")
        elif resource:
            probabilities = resource["probabilities"]
            ordered = sorted(probabilities, reverse=True)
            if max(probabilities) < args.resource_max_prob:
                review_reasons.append("resource_probability_below_threshold")
            if ordered[0] - ordered[1] < args.resource_margin:
                review_reasons.append("resource_probability_low_margin")
            bucket = resource["threshold_prediction"]
            prediction = RESOURCE_TO_ESI[bucket]
            decision_path.append(f"resource_bucket_{bucket}_to_esi_{prediction}")
            if bucket == 2 and danger:
                if uncertain_fields or review_reasons:
                    review_reasons.append("step_d_signal_present_but_path_requires_review")
                else:
                    prediction = 2
                    decision_path.append("step_d_danger_zone_upgrade_to_esi2")
        else:
            review_reasons.append("resource_model_not_applicable_and_no_step_a_b_signal")

        automatic = prediction is not None and not review_reasons
        predictions.append({
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "gold_esi_audit_only": labels[case_id],
            "prediction": prediction if automatic else None,
            "provisional_prediction_before_review": prediction,
            "automatic_decision": automatic,
            "review_required": not automatic,
            "review_reasons": sorted(set(review_reasons)),
            "decision_path": decision_path,
            "step_a_signal_count": len(step_a),
            "step_b_signal_count": len(step_b),
            "step_b_policy_fields": step_b_fields,
            "uncertain_policy_fields": uncertain_fields,
            "official_danger_zone_signal_present": danger,
            "resource_oof": resource,
            "gold_observability_audit_only": gold_reviews.get(case_id),
        })

    auto = [row for row in predictions if row["automatic_decision"]]
    provisional = [row for row in predictions if row["provisional_prediction_before_review"] is not None]
    auto_metrics = metrics([row["gold_esi_audit_only"] for row in auto], [row["prediction"] for row in auto])
    provisional_metrics = metrics(
        [row["gold_esi_audit_only"] for row in provisional],
        [row["provisional_prediction_before_review"] for row in provisional],
    )
    legacy_auto = [
        row for row in predictions
        if row["automatic_decision"] or row["review_reasons"] == ["isolated_severe_pain_requires_clinical_review"]
    ]
    legacy_metrics = metrics(
        [row["gold_esi_audit_only"] for row in legacy_auto],
        [row["provisional_prediction_before_review"] for row in legacy_auto],
    )
    observable_targets = []
    for row in predictions:
        review = gold_reviews.get(row["case_id"])
        if review and review["gold_support_status"] == "unassessable":
            target, included, source = None, False, "stage05b_unassessable_excluded"
        elif review:
            target = review["proposed_revised_esi"]
            included, source = True, f"stage05b_{review['gold_support_status']}"
        else:
            target, included, source = row["gold_esi_audit_only"], True, "original_gold_no_policy_conflict"
        observable_targets.append({
            "case_id": row["case_id"],
            "original_gold_esi": row["gold_esi_audit_only"],
            "observable_development_target": target,
            "included": included,
            "target_source": source,
            "development_only": True,
        })
    observable_by_case = {row["case_id"]: row for row in observable_targets if row["included"]}
    observable_provisional = [
        (observable_by_case[row["case_id"]]["observable_development_target"], row["provisional_prediction_before_review"])
        for row in predictions
        if row["case_id"] in observable_by_case and row["provisional_prediction_before_review"] is not None
    ]
    observable_auto = [
        (observable_by_case[row["case_id"]]["observable_development_target"], row["prediction"])
        for row in predictions
        if row["case_id"] in observable_by_case and row["prediction"] is not None
    ]
    observable_labels = sorted({target for target, _ in observable_provisional})

    valid_attr = [row["parsed_response"] for row in attributions if row["api_success"] and not row["validator_errors"]]
    patterns: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in valid_attr:
        key = (row["owner_module"], row["primary_error_type"], row["recommended_action"])
        patterns[key].append(row["case_id"])
    loop_actions = []
    for (owner, error_type, action), case_ids in sorted(patterns.items(), key=lambda item: (-len(item[1]), item[0])):
        loop_actions.append({
            "owner_module": owner,
            "error_type": error_type,
            "recommended_action": action,
            "case_count": len(case_ids),
            "case_ids": sorted(case_ids),
            "eligible_for_code_change": owner in {"02", "03", "04"} and len(case_ids) >= 2,
            "required_regression_scope": "owner_fixture_then_full_grouped_oof",
            "stop_condition": "paired_case_level_improvement_without_new_hard_gate_failure",
        })
    gold_pipeline_actions = Counter(row["pipeline_action"] for row in gold_reviews.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "06_final_policy_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_path = args.output_dir / "06_loop_actions.jsonl"
    with action_path.open("w", encoding="utf-8") as stream:
        for row in loop_actions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    observable_path = args.output_dir / "06_observable_supervision_view.jsonl"
    with observable_path.open("w", encoding="utf-8") as stream:
        for row in observable_targets:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    review_counts = Counter(reason for row in predictions for reason in row["review_reasons"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_role": "deterministic_reconciliation_only",
        "total_case_count": len(predictions),
        "automatic_case_count": len(auto),
        "automatic_coverage": len(auto) / len(predictions),
        "review_case_count": len(predictions) - len(auto),
        "review_reason_counts": review_counts,
        "automatic_only_metrics": auto_metrics,
        "provisional_covered_metrics_audit_only": provisional_metrics,
        "legacy_all_step_b_auto_sensitivity_audit_only": legacy_metrics,
        "observable_target_sensitivity_development_only": {
            "eligible_case_count": len(observable_by_case),
            "excluded_unassessable_count": len(predictions) - len(observable_by_case),
            "provisional_coverage": len(observable_provisional) / len(observable_by_case),
            "automatic_coverage": len(observable_auto) / len(observable_by_case),
            "active_labels": observable_labels,
            "provisional_active_class_metrics": metrics(
                [x[0] for x in observable_provisional], [x[1] for x in observable_provisional], observable_labels
            ),
            "automatic_active_class_metrics": metrics(
                [x[0] for x in observable_auto], [x[1] for x in observable_auto], observable_labels
            ),
            "not_a_clinical_gold_result": True,
        },
        "predicted_esi_counts_automatic": Counter(row["prediction"] for row in auto),
        "step_d_upgrade_count": sum("step_d_danger_zone_upgrade_to_esi2" in row["decision_path"] for row in predictions),
        "loop_action_count": len(loop_actions),
        "loop_owner_counts": Counter(row["owner_module"] for row in loop_actions),
        "gold_observability_audit": {
            "reviewed_case_count": len(gold_reviews),
            "gold_support_status_counts": gold_audit["gold_support_status_counts"],
            "observable_policy_path_counts": gold_audit["observable_policy_path_counts"],
            "supervision_action_counts": gold_audit["supervision_action_counts"],
            "pipeline_action_counts": gold_pipeline_actions,
            "automatic_gold_rewrite": False,
        },
        "interpretation": {
            "automatic_only_metrics_are_selective": True,
            "review_cases_are_not_imputed_from_gold": True,
            "stage05_attributions_do_not_modify_predictions": True,
            "stage05b_gold_review_does_not_modify_predictions": True,
            "isolated_severe_pain_requires_review": True,
            "stage04_resource_metric_remains_separate": True,
        },
    }
    (args.output_dir / "06_final_report.json").write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")

    input_paths = {
        "stage02": args.stage02,
        "stage03": args.stage03,
        "stage04_oof": args.stage04_oof,
        "stage05_audit": args.stage05_dir / "05_build_audit.json",
        "stage05_attributions": args.stage05_dir / "05_error_attributions.jsonl",
        "stage05b_audit": args.stage05b_dir / "05b_build_audit.json",
        "stage05b_gold_reviews": args.stage05b_dir / "05b_gold_esi_observability.jsonl",
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "hard_gates": {
            "all_cases_reconciled": len(predictions) == 541,
            "stage05_release_gate_passed": harness_audit["release_gate_passed"],
            "stage05b_release_gate_passed": gold_audit["release_gate_passed"],
            "gold_not_used_for_prediction": True,
            "review_cases_not_force_classified": all(row["prediction"] is None for row in predictions if row["review_required"]),
            "step_d_requires_resource_two_plus": all(
                row["resource_oof"] and row["resource_oof"]["threshold_prediction"] == 2
                for row in predictions
                if "step_d_danger_zone_upgrade_to_esi2" in row["decision_path"]
            ),
            "llm_attribution_does_not_override_policy": True,
            "gold_audit_does_not_override_policy": True,
            "isolated_severe_pain_not_auto_released": all(
                not row["automatic_decision"]
                for row in predictions
                if row["step_b_policy_fields"] == ["severe_pain_or_distress"]
            ),
        },
        "output_sha256": {
            "final_policy_predictions": sha256(prediction_path),
            "loop_actions": sha256(action_path),
            "observable_supervision_view": sha256(observable_path),
            "final_report": sha256(args.output_dir / "06_final_report.json"),
        },
    }
    audit["engineering_gate_passed"] = all(audit["hard_gates"].values())
    audit["clinical_release_ready"] = (
        audit["engineering_gate_passed"]
        and report["automatic_coverage"] >= 0.8
        and auto_metrics["macro_f1"] >= 0.55
        and all(value > 0 for value in auto_metrics["class_recall"].values())
    )
    (args.output_dir / "06_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"report": report, "audit": audit}, indent=2, default=dict))


if __name__ == "__main__":
    main()
