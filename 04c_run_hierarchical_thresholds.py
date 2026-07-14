"""04c Stage B2: model-specific hierarchical threshold strategies.

This runner changes only the decision layer for the frozen Dual and
Conditional boundary models. Thresholds are selected from inner grouped
case-OOF predictions and evaluated on untouched outer cases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


BOUNDARY_PATH = Path(__file__).with_name("04c_run_boundary_specific_ordinal.py")
SPEC = importlib.util.spec_from_file_location("boundary04c", BOUNDARY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import boundary module: {BOUNDARY_PATH}")
BOUNDARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARY)

EXPERIMENTS = BOUNDARY.EXPERIMENTS
RIDGE = BOUNDARY.RIDGE
ORTHO = BOUNDARY.ORTHO
SCHEMA_VERSION = "04c_hierarchical_thresholds_v1"
INPUT_SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_INPUT_DIR = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")
DEFAULT_OUTPUT_DIR = Path("outputs/04c_hierarchical_thresholds_v1")
DEFAULT_PO_T = Path("outputs/04c_resource_ordinal_experiments_v2/case_oof_T_nested_threshold_p05.jsonl")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.concatenate(([0.0, 1.0], unique, midpoints)))


def hierarchical_predict(q1: np.ndarray, q2: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(q1 < t1, 0, np.where(q2 < t2, 1, 2)).astype(int)


def safety_values(targets: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    high = targets == 2
    if not np.any(high):
        return 0.0, 0.0
    return float(np.mean(predictions[high] == 2)), float(np.mean(predictions[high] == 0))


def select_thresholds(
    targets: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    mode: str,
    recall_floor: float,
    severe_under_rate_limit: float,
) -> tuple[float, float, dict[str, Any]]:
    candidates_1 = threshold_candidates(q1)
    candidates_2 = threshold_candidates(q2)
    target_matrix = np.asarray(targets, dtype=int)[:, None]
    n = len(targets)
    actual_counts = np.asarray([np.sum(targets == target) for target in range(3)], dtype=float)
    qwk_weights = np.asarray([[((left - right) / 2.0) ** 2 for right in range(3)] for left in range(3)], dtype=float)

    def evaluate_for_t1(t1: float) -> dict[str, np.ndarray]:
        q1_below = q1[:, None] < t1
        q2_above = q2[:, None] >= candidates_2[None, :]
        predictions = np.where(q1_below, 0, np.where(q2_above, 2, 1))
        f1_values = []
        matrices = []
        for target in range(3):
            true_positive = np.sum((target_matrix == target) & (predictions == target), axis=0).astype(float)
            false_positive = np.sum(predictions == target, axis=0).astype(float) - true_positive
            false_negative = actual_counts[target] - true_positive
            f1_values.append(2.0 * true_positive / np.maximum(2.0 * true_positive + false_positive + false_negative, 1.0))
        macro_f1 = np.mean(np.asarray(f1_values), axis=0)
        for left in range(3):
            matrices.append(np.asarray([np.sum((targets == left)[:, None] & (predictions == right), axis=0) for right in range(3)], dtype=float))
        matrix = np.asarray(matrices)
        predicted_counts = np.sum(matrix, axis=0)
        expected = actual_counts[:, None, None] * predicted_counts[None, :, :] / max(n, 1)
        observed_loss = np.sum(qwk_weights[:, :, None] * matrix, axis=(0, 1)) / max(n, 1)
        expected_loss = np.sum(qwk_weights[:, :, None] * expected, axis=(0, 1)) / max(n, 1)
        qwk = np.where(expected_loss > 1e-12, 1.0 - observed_loss / expected_loss, 1.0)
        maoe = np.mean(np.abs(predictions - target_matrix), axis=0)
        high = targets == 2
        r2_recall = np.sum(predictions[high] == 2, axis=0) / max(np.sum(high), 1)
        severe_under_rate = np.sum(predictions[high] == 0, axis=0) / max(np.sum(high), 1)
        return {"predictions": predictions, "macro_f1": macro_f1, "qwk": qwk, "maoe": maoe, "r2_recall": r2_recall, "severe_under_rate": severe_under_rate}

    best = None
    feasible_count = 0
    all_evaluated: list[tuple[tuple[float, ...], float, float, dict[str, np.ndarray], int]] = []
    for t1 in candidates_1:
        values = evaluate_for_t1(float(t1))
        for index, t2 in enumerate(candidates_2):
            r2_recall = float(values["r2_recall"][index])
            severe_under_rate = float(values["severe_under_rate"][index])
            feasible = r2_recall >= recall_floor and severe_under_rate <= severe_under_rate_limit
            if feasible:
                feasible_count += 1
            key = (float(values["macro_f1"][index]), float(values["qwk"][index]), -float(values["maoe"][index]), -float(t1 + t2))
            all_evaluated.append((key, float(t1), float(t2), values, index))
            if mode == "B2b" and not feasible:
                continue
            if best is None or key > best[0]:
                predictions = values["predictions"][:, index]
                best = (key, float(t1), float(t2), RIDGE.classification_metrics(targets, np.eye(3)[predictions]), r2_recall, severe_under_rate)
    fallback_used = False
    if best is None:
        if mode != "B2b":
            raise RuntimeError("No threshold pair available")
        fallback_used = True
        for _, t1, t2, values, index in all_evaluated:
            predictions = values["predictions"][:, index]
            r2_recall, severe_under_rate = safety_values(targets, predictions)
            metrics = RIDGE.classification_metrics(targets, np.eye(3)[predictions])
            key = (r2_recall, -severe_under_rate, metrics["macro_f1"], -metrics["mean_absolute_ordinal_error"], -float(t1 + t2))
            if best is None or key > best[0]:
                best = (key, t1, t2, metrics, r2_recall, severe_under_rate)
    return best[1], best[2], {
        "selection_mode": mode,
        "inner_selection_metrics": best[3],
        "inner_r2_plus_recall": best[4],
        "inner_r2_plus_to_zero_rate": best[5],
        "recall_floor": recall_floor,
        "severe_under_rate_limit": severe_under_rate_limit,
        "candidate_count": int(len(candidates_1) * len(candidates_2)),
        "feasible_candidate_count": feasible_count,
        "safety_feasible": not fallback_used,
        "safety_fallback_used": fallback_used,
    }


def aggregate_values(rows: list[dict[str, Any]], indices: list[int], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    return BOUNDARY.aggregate_scalar(rows, indices, values)


def inner_oof_boundaries(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    label_by_case: dict[str, dict[str, Any]],
    conditional: bool,
    penalty: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_targets = {rows[index]["case_id"]: EXPERIMENTS.target_of(label_by_case, rows[index]["case_id"]) for index in train_indices}
    counts = np.bincount(np.asarray(list(case_targets.values()), dtype=int), minlength=3)
    fold_count = max(min(3, int(np.min(counts[counts > 0]))), 2)
    assignment = RIDGE.make_group_folds(case_targets, fold_count, seed)
    local_position = {index: position for position, index in enumerate(train_indices)}
    oof_q1 = np.full(len(train_indices), np.nan, dtype=float)
    oof_q2 = np.full(len(train_indices), np.nan, dtype=float)
    for inner_fold in sorted(set(assignment.values())):
        validation_cases = {case_id for case_id, fold in assignment.items() if fold == inner_fold}
        inner_train = [index for index in train_indices if rows[index]["case_id"] not in validation_cases]
        inner_validation = [index for index in train_indices if rows[index]["case_id"] in validation_cases]
        _, _, q1, q2 = BOUNDARY.fit_boundary_models(rows, inner_train, inner_validation, feature_names, weights, targets, penalty, conditional)
        for local, index in enumerate(inner_validation):
            oof_q1[local_position[index]] = q1[local]
            oof_q2[local_position[index]] = q2[local]
    if not np.all(np.isfinite(oof_q1)) or not np.all(np.isfinite(oof_q2)):
        raise RuntimeError("Incomplete inner boundary OOF result")
    case_ids, case_q1 = aggregate_values(rows, train_indices, oof_q1)
    _, case_q2 = aggregate_values(rows, train_indices, oof_q2)
    return np.asarray(case_ids), case_q1, case_q2


def run_variant(
    name: str,
    feature_set: str,
    feature_names: list[str],
    rows: list[dict[str, Any]],
    label_by_case: dict[str, dict[str, Any]],
    fold_by_case: dict[str, int],
    availability_by_fold: dict[int, dict[str, Any]],
    weights: np.ndarray,
    targets: np.ndarray,
    conditional: bool,
    mode: str,
    lambda_grid: list[float],
    recall_floor: float,
    severe_under_rate_limit: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    oof_q1 = np.full(len(rows), np.nan, dtype=float)
    oof_q2 = np.full(len(rows), np.nan, dtype=float)
    oof_prediction = np.full(len(rows), -1, dtype=int)
    fold_records = []
    threshold_records = []
    for fold_id in sorted(set(fold_by_case.values())):
        validation_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] == fold_id]
        train_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] != fold_id]
        selected_lambda, lambda_records = BOUNDARY.select_lambda(rows, train_indices, feature_names, weights, targets, label_by_case, conditional, lambda_grid, 1500 + fold_id)
        inner_case_ids, inner_q1, inner_q2 = inner_oof_boundaries(rows, train_indices, feature_names, weights, targets, label_by_case, conditional, selected_lambda, 1700 + fold_id)
        inner_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in inner_case_ids], dtype=int)
        t1, t2, selection = select_thresholds(inner_targets, inner_q1, inner_q2, mode, recall_floor, severe_under_rate_limit)
        model_bundle, preprocessor, q1, q2 = BOUNDARY.fit_boundary_models(rows, train_indices, validation_indices, feature_names, weights, targets, selected_lambda, conditional)
        frozen_zero = set(availability_by_fold[fold_id].get("zero_variance_train_features", [])).intersection(feature_names)
        if frozen_zero != set(preprocessor["zero_variance_features"]):
            raise RuntimeError(f"{name}: feature availability mismatch in fold {fold_id}")
        oof_q1[validation_indices] = q1
        oof_q2[validation_indices] = q2
        case_ids, case_q1 = aggregate_values(rows, validation_indices, q1)
        _, case_q2 = aggregate_values(rows, validation_indices, q2)
        probabilities, used_q1, used_q2, violations, projection_count = BOUNDARY.reconstruct_probabilities(case_q1, case_q2, conditional)
        predictions = hierarchical_predict(used_q1, used_q2, t1, t2)
        case_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
        case_metrics = EXPERIMENTS.metrics_with_decision(case_targets, predictions, probabilities)
        r2_recall, severe_under_rate = safety_values(case_targets, predictions)
        case_position = {case_id: position for position, case_id in enumerate(case_ids)}
        for local, index in enumerate(validation_indices):
            case_position_value = case_position[rows[index]["case_id"]]
            oof_prediction[index] = predictions[case_position_value]
        threshold_records.append({"fold_id": fold_id, "t1": t1, "t2": t2, **selection})
        fold_records.append({"variant": name, "feature_set": feature_set, "fold_id": fold_id, "conditional": conditional, "threshold_mode": mode, "selected_lambda": selected_lambda, "thresholds": {"t1": t1, "t2": t2}, "threshold_selection": selection, "case_aggregation": "mean_boundary_probability_then_hierarchical_decision", "case_metrics": case_metrics, "outer_r2_plus_recall": r2_recall, "outer_r2_plus_to_zero_rate": severe_under_rate, "raw_monotonic_violation_count": violations if not conditional else None, "conditional_q2_gt_q1_count_diagnostic": violations if conditional else 0, "projection_adjusted_case_count": projection_count, "lambda_candidates": lambda_records})
    if not np.all(np.isfinite(oof_q1)) or np.any(oof_prediction < 0):
        raise RuntimeError(f"{name}: incomplete outer OOF result")
    case_ids = sorted(set(row["case_id"] for row in rows))
    case_q1 = np.asarray([np.mean([oof_q1[index] for index, row in enumerate(rows) if row["case_id"] == case_id]) for case_id in case_ids])
    case_q2 = np.asarray([np.mean([oof_q2[index] for index, row in enumerate(rows) if row["case_id"] == case_id]) for case_id in case_ids])
    probabilities, used_q1, used_q2, violations, projection_count = BOUNDARY.reconstruct_probabilities(case_q1, case_q2, conditional)
    case_predictions = []
    for position, case_id in enumerate(case_ids):
        fold_id = fold_by_case[case_id]
        threshold = next(row for row in threshold_records if row["fold_id"] == fold_id)
        case_predictions.append(int(hierarchical_predict(np.asarray([used_q1[position]]), np.asarray([used_q2[position]]), threshold["t1"], threshold["t2"])[0]))
    case_predictions = np.asarray(case_predictions, dtype=int)
    case_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
    metrics = EXPERIMENTS.metrics_with_decision(case_targets, case_predictions, probabilities)
    r2_recall, severe_under_rate = safety_values(case_targets, case_predictions)
    metrics["outer_r2_plus_recall"] = r2_recall
    metrics["outer_r2_plus_to_zero_rate"] = severe_under_rate
    metrics["raw_monotonic_violation_count"] = violations if not conditional else None
    metrics["conditional_q2_gt_q1_count_diagnostic"] = violations if conditional else 0
    metrics["projection_adjusted_case_count"] = projection_count
    oof_rows = [{"schema_version": SCHEMA_VERSION, "variant": name, "instance_id": row["instance_id"], "case_id": row["case_id"], "fold_id": int(fold_by_case[row["case_id"]]), "target_resource_bucket": int(targets[index]), "q_boundary_1": float(oof_q1[index]), "q_boundary_2": float(oof_q2[index]), "decision_prediction": int(oof_prediction[index])} for index, row in enumerate(rows)]
    case_oof_rows = []
    for position, case_id in enumerate(case_ids):
        threshold = next(row for row in threshold_records if row["fold_id"] == fold_by_case[case_id])
        prediction = int(hierarchical_predict(np.asarray([used_q1[position]]), np.asarray([used_q2[position]]), threshold["t1"], threshold["t2"])[0])
        case_oof_rows.append({"schema_version": SCHEMA_VERSION, "variant": name, "feature_set": feature_set, "case_id": case_id, "fold_id": int(fold_by_case[case_id]), "target_resource_bucket": int(case_targets[position]), "q_boundary_1_raw": float(case_q1[position]), "q_boundary_2_raw": float(case_q2[position]), "q_boundary_1_used": float(used_q1[position]), "q_boundary_2_used": float(used_q2[position]), "threshold_t1": float(threshold["t1"]), "threshold_t2": float(threshold["t2"]), "probability_0": float(probabilities[position, 0]), "probability_1": float(probabilities[position, 1]), "probability_2_plus": float(probabilities[position, 2]), "raw_probability_prediction": int(np.argmax(probabilities[position])), "decision_prediction": prediction, "aggregation": "mean_boundary_probability_then_hierarchical_decision"})
    return oof_rows, case_oof_rows, metrics, fold_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--po-t-file", type=Path, default=DEFAULT_PO_T)
    parser.add_argument("--lambda-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--r2-recall-floor", type=float, default=0.80)
    parser.add_argument("--severe-under-rate-limit", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, label_by_case, fold_by_case, weight_by_instance, availability_by_fold, full_features, audit = EXPERIMENTS.load_inputs(args.input_dir)
    rows = sorted(rows, key=lambda row: row["instance_id"])
    weights = np.asarray([weight_by_instance[row["instance_id"]] for row in rows], dtype=float)
    targets = np.asarray([EXPERIMENTS.target_of(label_by_case, row["case_id"]) for row in rows], dtype=int)
    _, _, dictionary = ORTHO.load_feature_sets(args.input_dir)
    quality_features = sorted(ORTHO.QUALITY_FEATURES)
    clinical_features = [name for name in full_features if name not in ORTHO.QUALITY_FEATURES]
    if len(full_features) != 40 or len(clinical_features) != 23 or len(quality_features) != 17 or set(clinical_features) & set(quality_features) or (set(clinical_features) | set(quality_features)) != set(full_features):
        raise RuntimeError("Frozen feature partition must be 40=23+17 with no overlap")
    lambda_grid = [float(value) for value in args.lambda_grid.split(",") if value.strip()]
    variants = []
    for mode in ("B2a", "B2b"):
        for name, feature_set, feature_names, conditional in (("DUAL_FULL40", "FULL40", full_features, False), ("DUAL_ORTHO23", "ORTHO23", clinical_features, False), ("CONDITIONAL_FULL40", "FULL40", full_features, True), ("CONDITIONAL_ORTHO23", "ORTHO23", clinical_features, True)):
            variants.append((f"{name}_{mode}", feature_set, feature_names, conditional, mode))
    summaries = []
    case_outputs = {}
    for name, feature_set, feature_names, conditional, mode in variants:
        oof, case_oof, metrics, folds = run_variant(name, feature_set, feature_names, rows, label_by_case, fold_by_case, availability_by_fold, weights, targets, conditional, mode, lambda_grid, args.r2_recall_floor, args.severe_under_rate_limit)
        case_outputs[name] = case_oof
        write_jsonl(args.output_dir / f"oof_{name}.jsonl", oof)
        write_jsonl(args.output_dir / f"case_oof_{name}.jsonl", case_oof)
        write_jsonl(args.output_dir / f"folds_{name}.jsonl", folds)
        summaries.append({"variant": name, "feature_set": feature_set, "conditional": conditional, "threshold_mode": mode, "common_cohort": metrics, "thresholds": [{"fold_id": row["fold_id"], "t1": row["thresholds"]["t1"], "t2": row["thresholds"]["t2"], "safety_feasible": row["threshold_selection"]["safety_feasible"], "safety_fallback_used": row["threshold_selection"]["safety_fallback_used"]} for row in folds]})
    po_t_rows = EXPERIMENTS.RIDGE.read_jsonl(args.po_t_file)
    paired = {}
    for name in case_outputs:
        paired[name + "_vs_PO_T"] = EXPERIMENTS.paired_bootstrap(po_t_rows, case_outputs[name], "PO_T_nested_threshold_p05", name)
    write_json(args.output_dir / "paired_bootstrap_vs_PO_T.json", paired)
    write_json(args.output_dir / "run_manifest.json", {"schema_version": SCHEMA_VERSION, "input_schema_version": INPUT_SCHEMA_VERSION, "input_audit_hash": RIDGE.sha256_file(args.input_dir / "resource_feasibility_audit.json"), "po_t_file": str(args.po_t_file), "po_t_hash": RIDGE.sha256_file(args.po_t_file), "case_count": len(fold_by_case), "instance_count": len(rows), "outer_folds": sorted(set(fold_by_case.values())), "lambda_grid": lambda_grid, "feature_sets": {"FULL40": full_features, "ORTHO23": clinical_features}, "quality_features": quality_features, "r2_recall_floor": args.r2_recall_floor, "severe_under_rate_limit": args.severe_under_rate_limit, "partial_weight": 0.5, "class_weighting": "natural", "threshold_selection": "inner_grouped_case_oof", "note": "B2a is unconstrained hierarchical threshold selection; B2b adds safety constraints. Fusion, class weighting, soft labels, and preferences are excluded."})
    write_json(args.output_dir / "experiment_summary.json", {"schema_version": SCHEMA_VERSION, "variants": summaries, "paired_comparisons": list(paired), "note": "Model-specific hierarchical threshold strategies; downstream 05/06 remains blocked pending safety and stability gates."})
    print(f"04c hierarchical threshold experiments complete: {len(variants)} variants, output={args.output_dir}")


if __name__ == "__main__":
    main()
