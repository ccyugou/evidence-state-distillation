"""04c-v2 ablations for the frozen resource ordinal feature plane.

This is a diagnostic runner, not a final ESI model.  Every experiment uses
the same frozen case folds and the same 04a/04b cohort.  Only one factor is
changed at a time: decision thresholds, class weighting, or partial-label
weight.  Thresholds and class weights are selected from outer-training data
only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_PATH = Path(__file__).with_name("04c_train_ridge_resource_ordinal.py")
SPEC = importlib.util.spec_from_file_location("ridge04c", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import baseline module: {BASE_PATH}")
RIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RIDGE)


SCHEMA_VERSION = "04c_resource_ordinal_experiments_v2"
INPUT_SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_INPUT_DIR = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")
DEFAULT_OUTPUT_DIR = Path("outputs/04c_resource_ordinal_experiments_v2")
TARGETS = (0, 1, 2)
EPS = 1e-12


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


def load_inputs(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int], dict[str, float], dict[int, dict[str, Any]], list[str], dict[str, Any]]:
    audit = RIDGE.read_json(input_dir / "resource_feasibility_audit.json")
    dictionary = RIDGE.read_json(input_dir / "resource_feature_dictionary.json")
    matrix = RIDGE.read_jsonl(input_dir / "resource_instance_feature_matrix.jsonl")
    labels = RIDGE.read_jsonl(input_dir / "resource_supervision_case_audit.jsonl")
    folds = RIDGE.read_jsonl(input_dir / "resource_grouped_fold_manifest.jsonl")
    weights = RIDGE.read_jsonl(input_dir / "resource_training_weight_manifest.jsonl")
    availability = RIDGE.read_jsonl(input_dir / "resource_fold_feature_availability.jsonl")
    if audit.get("schema_version") != INPUT_SCHEMA_VERSION or audit.get("release_gate_passed") is not True:
        raise RuntimeError("Frozen 04a audit is not a releasable v1.2 artifact")
    feature_names = list(dictionary["model_features"])
    label_by_case = {row["case_id"]: row for row in labels if row.get("weak_resource_bucket") is not None}
    fold_by_case = {row["case_id"]: int(row["fold_id"]) for row in folds}
    weight_by_instance = {row["instance_id"]: float(row["instance_weight"]) for row in weights}
    availability_by_fold = {int(row["fold_id"]): row for row in availability}
    rows = [row for row in matrix if row["instance_id"] in weight_by_instance]
    if len(rows) != len(weight_by_instance) or len(fold_by_case) != 296:
        raise RuntimeError("Frozen cohort coverage mismatch")
    if any(row.get("feature_matrix_label_free") is not True for row in rows):
        raise RuntimeError("Label-derived feature metadata found")
    if any(name in {"official_danger_signal", "official_danger_signal_count", "spo2_danger_signal"} for name in feature_names):
        raise RuntimeError("Step D feature entered experiment matrix")
    return rows, label_by_case, fold_by_case, weight_by_instance, availability_by_fold, feature_names, audit


def target_of(label_by_case: dict[str, dict[str, Any]], case_id: str) -> int:
    return int(label_by_case[case_id]["weak_resource_bucket"])


def is_partial(label_by_case: dict[str, dict[str, Any]], case_id: str) -> bool:
    return "partial" in str(label_by_case[case_id].get("supervision_status", "")).lower()


def class_multiplier(case_targets: dict[str, int], scheme: str) -> dict[int, float]:
    counts = Counter(case_targets.values())
    if scheme == "natural":
        return {target: 1.0 for target in TARGETS}
    if scheme == "sqrt_inverse":
        raw = {target: math.sqrt(len(case_targets) / max(counts[target], 1)) for target in TARGETS}
    elif scheme.startswith("effective_"):
        beta = float(scheme.split("_")[-1])
        raw = {target: (1.0 - beta) / max(1.0 - beta ** counts[target], EPS) for target in TARGETS}
    else:
        raise ValueError(f"Unknown class weight scheme: {scheme}")
    normalizer = len(case_targets) / sum(raw[target] * counts[target] for target in TARGETS)
    return {target: raw[target] * normalizer for target in TARGETS}


def effective_weights(
    rows: list[dict[str, Any]],
    indices: list[int],
    base_weights: np.ndarray,
    label_by_case: dict[str, dict[str, Any]],
    partial_factor: float,
    scheme: str,
) -> np.ndarray:
    train_cases = {rows[index]["case_id"] for index in indices}
    case_targets = {case_id: target_of(label_by_case, case_id) for case_id in train_cases}
    multipliers = class_multiplier(case_targets, scheme)
    values = base_weights.copy()
    for index in range(len(rows)):
        case_id = rows[index]["case_id"]
        factor = multipliers[target_of(label_by_case, case_id)] if case_id in train_cases else 1.0
        if is_partial(label_by_case, case_id):
            # 04a instance_weight already contains the frozen P05=0.5 label
            # weight.  Convert the requested absolute sensitivity value into
            # a ratio so B0 remains an exact reproduction of the v2 baseline.
            frozen_partial_weight = float(label_by_case[case_id].get("supervision_weight", 0.5))
            factor *= partial_factor / max(frozen_partial_weight, EPS)
        values[index] *= factor
    return values


def select_lambda_weighted(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    base_weights: np.ndarray,
    targets: np.ndarray,
    label_by_case: dict[str, dict[str, Any]],
    partial_factor: float,
    scheme: str,
    grid: list[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    case_targets = {rows[index]["case_id"]: target_of(label_by_case, rows[index]["case_id"]) for index in train_indices}
    fold_count = min(3, min(Counter(case_targets.values()).values()))
    inner_assignment = RIDGE.make_group_folds(case_targets, max(fold_count, 2), seed)
    records = []
    for penalty in grid:
        losses = []
        for inner_fold in sorted(set(inner_assignment.values())):
            validation_cases = {case_id for case_id, fold in inner_assignment.items() if fold == inner_fold}
            inner_train = [index for index in train_indices if rows[index]["case_id"] not in validation_cases]
            inner_validation = [index for index in train_indices if rows[index]["case_id"] in validation_cases]
            train_weights = effective_weights(rows, inner_train, base_weights, label_by_case, partial_factor, scheme)
            fit_train = [index for index in inner_train if train_weights[index] > EPS]
            if not fit_train:
                raise RuntimeError("No positive-weight training rows remain")
            _, _, probability = RIDGE.fit_and_predict(rows, fit_train, inner_validation, feature_names, train_weights, targets, penalty)
            case_ids, case_probabilities, _ = RIDGE.aggregate_case_rows(rows, inner_validation, probability)
            inner_targets = np.asarray([target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
            losses.append(RIDGE.multiclass_log_loss(inner_targets, case_probabilities))
        records.append({"lambda": penalty, "inner_case_log_loss": float(np.mean(losses)), "fold_log_losses": losses})
    chosen = min(records, key=lambda row: (row["inner_case_log_loss"], row["lambda"]))
    return float(chosen["lambda"]), records


def threshold_predict(score: np.ndarray, t01: float, t12: float) -> np.ndarray:
    return np.where(score < t01, 0, np.where(score < t12, 1, 2)).astype(int)


def decision_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    probabilities = np.eye(3)[predictions]
    metrics = RIDGE.classification_metrics(targets, probabilities)
    return metrics


def choose_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    scores = probabilities[:, 1] + 2.0 * probabilities[:, 2]
    unique = np.unique(scores)
    if len(unique) == 1:
        candidates = np.asarray([unique[0] - 1e-6, unique[0] + 1e-6])
    else:
        candidates = np.unique(np.concatenate(([unique[0] - 1e-6], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1e-6])))
    best = None
    for t01 in candidates:
        for t12 in candidates:
            if not t01 < t12:
                continue
            prediction = threshold_predict(scores, float(t01), float(t12))
            metrics = decision_summary(targets, prediction)
            key = (metrics["macro_f1"], metrics["quadratic_weighted_kappa"], -metrics["mean_absolute_ordinal_error"], -float(t01 + t12))
            if best is None or key > best[0]:
                best = (key, float(t01), float(t12), metrics)
    if best is None:
        raise RuntimeError("No ordered threshold pair was available")
    return best[1], best[2], {"inner_selection_metrics": best[3], "candidate_count": len(candidates)}


def inner_oof_for_thresholds(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    base_weights: np.ndarray,
    targets: np.ndarray,
    label_by_case: dict[str, dict[str, Any]],
    partial_factor: float,
    scheme: str,
    penalty: float,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    case_targets = {rows[index]["case_id"]: target_of(label_by_case, rows[index]["case_id"]) for index in train_indices}
    fold_count = min(3, min(Counter(case_targets.values()).values()))
    assignment = RIDGE.make_group_folds(case_targets, max(fold_count, 2), seed)
    probabilities = np.full((len(train_indices), 3), np.nan, dtype=float)
    local_index = {index: position for position, index in enumerate(train_indices)}
    for inner_fold in sorted(set(assignment.values())):
        validation_cases = {case_id for case_id, fold in assignment.items() if fold == inner_fold}
        inner_train = [index for index in train_indices if rows[index]["case_id"] not in validation_cases]
        inner_validation = [index for index in train_indices if rows[index]["case_id"] in validation_cases]
        train_weights = effective_weights(rows, inner_train, base_weights, label_by_case, partial_factor, scheme)
        fit_train = [index for index in inner_train if train_weights[index] > EPS]
        if not fit_train:
            raise RuntimeError("No positive-weight training rows remain")
        _, _, predicted = RIDGE.fit_and_predict(rows, fit_train, inner_validation, feature_names, train_weights, targets, penalty)
        for position, index in enumerate(inner_validation):
            probabilities[local_index[index]] = predicted[position]
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Inner OOF threshold probabilities are incomplete")
    return probabilities, [rows[index]["case_id"] for index in train_indices]


def metrics_with_decision(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = RIDGE.classification_metrics(targets, probabilities)
    decision = decision_summary(targets, predictions)
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "mean_absolute_ordinal_error", "class_metrics", "predicted_class_counts", "resource_under_estimation_count", "resource_over_estimation_count", "confusion_matrix"):
        metrics[key] = decision[key]
    metrics["ordinal_probability_diagnostics"] = RIDGE.ordinal_probability_diagnostics(targets, probabilities)
    return metrics


def paired_bootstrap(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    baseline_name: str = "baseline",
    variant_name: str = "variant",
    repetitions: int = 500,
    seed: int = 20260713,
) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in baseline_rows}
    variant = {row["case_id"]: row for row in variant_rows}
    case_ids = sorted(set(baseline) & set(variant))
    if len(case_ids) != len(baseline) or len(case_ids) != len(variant):
        raise RuntimeError("Paired bootstrap case cohorts do not align")
    targets = np.asarray([baseline[case_id]["target_resource_bucket"] for case_id in case_ids], dtype=int)
    baseline_probabilities = np.asarray([[baseline[case_id]["probability_0"], baseline[case_id]["probability_1"], baseline[case_id]["probability_2_plus"]] for case_id in case_ids])
    variant_probabilities = np.asarray([[variant[case_id]["probability_0"], variant[case_id]["probability_1"], variant[case_id]["probability_2_plus"]] for case_id in case_ids])
    baseline_predictions = np.asarray([baseline[case_id]["decision_prediction"] for case_id in case_ids], dtype=int)
    variant_predictions = np.asarray([variant[case_id]["decision_prediction"] for case_id in case_ids], dtype=int)
    rng = np.random.default_rng(seed)
    deltas = {"macro_f1": [], "quadratic_weighted_kappa": [], "mean_absolute_ordinal_error": [], "log_loss": [], "brier_score": []}
    for _ in range(repetitions):
        sample = rng.integers(0, len(case_ids), size=len(case_ids))
        base_metrics = metrics_with_decision(targets[sample], baseline_predictions[sample], baseline_probabilities[sample])
        variant_metrics = metrics_with_decision(targets[sample], variant_predictions[sample], variant_probabilities[sample])
        for key in deltas:
            deltas[key].append(float(variant_metrics[key] - base_metrics[key]))
    report = {
        "case_count": len(case_ids),
        "repetitions": repetitions,
        "seed": seed,
        "baseline": baseline_name,
        "variant": variant_name,
        "delta_convention": f"{variant_name}_minus_{baseline_name}",
        "metrics": {},
    }
    for key, values in deltas.items():
        array = np.asarray(values, dtype=float)
        report["metrics"][key] = {"mean_delta": float(np.mean(array)), "median_delta": float(np.median(array)), "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))], "probability_delta_positive": float(np.mean(array > 0.0))}
    return report


def run_variant(
    name: str,
    rows: list[dict[str, Any]],
    label_by_case: dict[str, dict[str, Any]],
    fold_by_case: dict[str, int],
    availability_by_fold: dict[int, dict[str, Any]],
    feature_names: list[str],
    base_weights: np.ndarray,
    targets: np.ndarray,
    partial_factor: float,
    scheme: str,
    threshold_mode: bool,
    lambda_grid: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    oof_probability = np.full((len(rows), 3), np.nan, dtype=float)
    oof_prediction = np.full(len(rows), -1, dtype=int)
    oof_fold = np.full(len(rows), -1, dtype=int)
    fold_records = []
    threshold_records = []
    fold_ids = sorted(set(fold_by_case.values()))
    for fold_id in fold_ids:
        validation_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] == fold_id]
        train_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] != fold_id]
        selected_lambda, lambda_records = select_lambda_weighted(rows, train_indices, feature_names, base_weights, targets, label_by_case, partial_factor, scheme, lambda_grid, 700 + fold_id)
        train_weights = effective_weights(rows, train_indices, base_weights, label_by_case, partial_factor, scheme)
        fit_train = [index for index in train_indices if train_weights[index] > EPS]
        if not fit_train:
            raise RuntimeError(f"{name}: no positive-weight training rows remain")
        model, preprocessor, probabilities = RIDGE.fit_and_predict(rows, fit_train, validation_indices, feature_names, train_weights, targets, selected_lambda)
        frozen_zero = set(availability_by_fold[fold_id].get("zero_variance_train_features", []))
        availability_comparable = partial_factor == 0.5
        if availability_comparable and frozen_zero.intersection(feature_names) != set(preprocessor["zero_variance_features"]):
            raise RuntimeError(f"{name}: frozen feature availability mismatch in fold {fold_id}")
        raw_prediction = np.argmax(probabilities, axis=1)
        thresholds = None
        prediction = raw_prediction
        if threshold_mode:
            inner_probabilities, inner_case_ids = inner_oof_for_thresholds(rows, train_indices, feature_names, base_weights, targets, label_by_case, partial_factor, scheme, selected_lambda, 900 + fold_id)
            inner_case_probabilities = []
            inner_case_targets = []
            for case_id in sorted(set(inner_case_ids)):
                positions = [position for position, value in enumerate(inner_case_ids) if value == case_id]
                inner_case_probabilities.append(np.mean(inner_probabilities[positions], axis=0))
                inner_case_targets.append(target_of(label_by_case, case_id))
            t01, t12, selection = choose_thresholds(np.asarray(inner_case_targets), np.asarray(inner_case_probabilities))
            score = probabilities[:, 1] + 2.0 * probabilities[:, 2]
            prediction = threshold_predict(score, t01, t12)
            thresholds = {"t01": t01, "t12": t12, **selection}
            threshold_records.append({"fold_id": fold_id, **thresholds})
        oof_probability[validation_indices] = probabilities
        oof_prediction[validation_indices] = prediction
        oof_fold[validation_indices] = fold_id
        case_ids, case_probabilities, _ = RIDGE.aggregate_case_rows(rows, validation_indices, probabilities)
        case_targets = np.asarray([target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
        if threshold_mode:
            # Case is the evaluation unit: aggregate probabilities first, then
            # apply the fold-frozen thresholds. Instance-level voting would
            # produce a different and invalid decision path for realizations.
            threshold = next(row for row in threshold_records if row["fold_id"] == fold_id)
            case_scores = case_probabilities[:, 1] + 2.0 * case_probabilities[:, 2]
            case_predictions = threshold_predict(case_scores, threshold["t01"], threshold["t12"])
        else:
            case_predictions = np.argmax(case_probabilities, axis=1)
        case_predictions = np.asarray(case_predictions, dtype=int)
        fold_metrics = metrics_with_decision(case_targets, case_predictions, case_probabilities)
        fold_records.append({"variant": name, "fold_id": fold_id, "selected_lambda": selected_lambda, "partial_weight": partial_factor, "class_weight_scheme": scheme, "threshold_mode": threshold_mode, "thresholds": thresholds, "case_aggregation": "mean_probability_then_decision", "case_metrics": fold_metrics, "effective_feature_count": len(preprocessor["active_features"]), "zero_variance_train_features": preprocessor["zero_variance_features"], "frozen_availability_comparable": availability_comparable, "lambda_candidates": lambda_records})
    if not np.all(np.isfinite(oof_probability)) or np.any(oof_prediction < 0):
        raise RuntimeError(f"{name}: incomplete OOF result")
    case_ids = sorted(set(row["case_id"] for row in rows))
    case_probabilities = np.asarray([np.mean(oof_probability[[index for index, row in enumerate(rows) if row["case_id"] == case_id]], axis=0) for case_id in case_ids])
    case_targets = np.asarray([target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
    case_predictions = []
    for case_id in case_ids:
        indices = [index for index, row in enumerate(rows) if row["case_id"] == case_id]
        if threshold_mode:
            fold_id = fold_by_case[case_id]
            threshold = next(row for row in threshold_records if row["fold_id"] == fold_id)
            score = case_probabilities[case_ids.index(case_id), 1] + 2.0 * case_probabilities[case_ids.index(case_id), 2]
            case_predictions.append(int(threshold_predict(np.asarray([score]), threshold["t01"], threshold["t12"])[0]))
        else:
            case_predictions.append(int(np.argmax(case_probabilities[case_ids.index(case_id)])))
    case_predictions = np.asarray(case_predictions, dtype=int)
    metrics = metrics_with_decision(case_targets, case_predictions, case_probabilities)
    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({"schema_version": SCHEMA_VERSION, "variant": name, "instance_id": row["instance_id"], "case_id": row["case_id"], "fold_id": int(oof_fold[index]), "target_resource_bucket": int(targets[index]), "probability_0": float(oof_probability[index, 0]), "probability_1": float(oof_probability[index, 1]), "probability_2_plus": float(oof_probability[index, 2]), "raw_probability_prediction": int(np.argmax(oof_probability[index])), "decision_prediction": int(oof_prediction[index])})
    case_oof_rows = []
    for position, case_id in enumerate(case_ids):
        case_probability = case_probabilities[position]
        raw_prediction = int(np.argmax(case_probability))
        case_oof_rows.append({
            "schema_version": SCHEMA_VERSION,
            "variant": name,
            "case_id": case_id,
            "fold_id": int(fold_by_case[case_id]),
            "target_resource_bucket": int(case_targets[position]),
            "probability_0": float(case_probability[0]),
            "probability_1": float(case_probability[1]),
            "probability_2_plus": float(case_probability[2]),
            "raw_probability_prediction": raw_prediction,
            "decision_prediction": int(case_predictions[position]),
            "aggregation": "equal_mean_across_realizations",
        })
    return oof_rows, case_oof_rows, fold_records, metrics, threshold_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambda-grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, label_by_case, fold_by_case, weight_by_instance, availability_by_fold, feature_names, audit = load_inputs(args.input_dir)
    rows = sorted(rows, key=lambda row: row["instance_id"])
    base_weights = np.asarray([weight_by_instance[row["instance_id"]] for row in rows], dtype=float)
    targets = np.asarray([target_of(label_by_case, row["case_id"]) for row in rows], dtype=int)
    lambda_grid = [float(value) for value in args.lambda_grid.split(",") if value.strip()]
    variants = [
        ("B0_natural_argmax_p05", 0.5, "natural", False),
        ("T_nested_threshold_p05", 0.5, "natural", True),
        ("W_sqrt_inverse_p05", 0.5, "sqrt_inverse", False),
        ("W_effective_beta095_p05", 0.5, "effective_0.95", False),
        ("W_effective_beta099_p05", 0.5, "effective_0.99", False),
        ("P0_natural_argmax", 0.0, "natural", False),
        ("P1_natural_argmax", 1.0, "natural", False),
    ]
    summaries = []
    case_outputs: dict[str, list[dict[str, Any]]] = {}
    fold_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, partial_factor, scheme, threshold_mode in variants:
        oof, case_oof, folds, metrics, thresholds = run_variant(name, rows, label_by_case, fold_by_case, availability_by_fold, feature_names, base_weights, targets, partial_factor, scheme, threshold_mode, lambda_grid)
        case_outputs[name] = case_oof
        fold_outputs[name] = folds
        write_jsonl(args.output_dir / f"oof_{name}.jsonl", oof)
        write_jsonl(args.output_dir / f"case_oof_{name}.jsonl", case_oof)
        write_jsonl(args.output_dir / f"folds_{name}.jsonl", folds)
        if thresholds:
            write_jsonl(args.output_dir / f"thresholds_{name}.jsonl", thresholds)
        supported_rows = [row for row in case_oof if not is_partial(label_by_case, row["case_id"])]
        partial_rows = [row for row in case_oof if is_partial(label_by_case, row["case_id"])]
        def cohort_metrics(cohort: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not cohort:
                return None
            cohort_targets = np.asarray([row["target_resource_bucket"] for row in cohort], dtype=int)
            cohort_probabilities = np.asarray([[row["probability_0"], row["probability_1"], row["probability_2_plus"]] for row in cohort])
            cohort_predictions = np.asarray([row["decision_prediction"] for row in cohort], dtype=int)
            return metrics_with_decision(cohort_targets, cohort_predictions, cohort_probabilities)
        summaries.append({"variant": name, "partial_weight": partial_factor, "class_weight_scheme": scheme, "threshold_mode": threshold_mode, "common_cohort": metrics, "supported_only": cohort_metrics(supported_rows), "partial_only": cohort_metrics(partial_rows)})
    baseline_summary = summaries[0]
    baseline_macro_f1 = baseline_summary["common_cohort"]["macro_f1"]
    bootstrap_reports = {}
    for summary in summaries:
        name = summary["variant"]
        metrics = summary["common_cohort"]
        fold_records = fold_outputs[name]
        recall_by_class = {str(target): sum(record["case_metrics"]["class_metrics"][str(target)]["recall"] > 0.0 for record in fold_records) for target in TARGETS}
        quality_gate = {
            "predicted_class_coverage_3_of_3": len([target for target in TARGETS if metrics["predicted_class_counts"][str(target)] > 0]) == 3,
            "all_class_recall_positive": all(metrics["class_metrics"][str(target)]["recall"] > 0.0 for target in TARGETS),
            "qwk_positive": metrics["quadratic_weighted_kappa"] > 0.0,
            "macro_f1_above_B0": metrics["macro_f1"] > baseline_macro_f1 + 1e-9,
            "balanced_accuracy_above_chance": metrics["balanced_accuracy"] > 1.0 / 3.0,
            "class_0_nonzero_recall_folds_at_least_4": recall_by_class["0"] >= 4,
            "class_1_nonzero_recall_folds_at_least_4": recall_by_class["1"] >= 4,
            "fold_recall_counts": recall_by_class,
        }
        quality_gate["core_quality_gate_passed"] = all(value for key, value in quality_gate.items() if key != "fold_recall_counts")
        summary["quality_gate"] = quality_gate
        if name != baseline_summary["variant"]:
            bootstrap_reports[name] = paired_bootstrap(
                case_outputs[baseline_summary["variant"]],
                case_outputs[name],
                baseline_summary["variant"],
                name,
            )
            summary["paired_bootstrap_vs_B0"] = bootstrap_reports[name]
    write_json(args.output_dir / "paired_bootstrap_vs_B0.json", bootstrap_reports)
    write_json(args.output_dir / "experiment_summary.json", {"schema_version": SCHEMA_VERSION, "input_audit_hash": RIDGE.sha256_file(args.input_dir / "resource_feasibility_audit.json"), "frozen_outer_folds": sorted(set(fold_by_case.values())), "baseline_variant": baseline_summary["variant"], "variants": summaries, "note": "All threshold and class-weight choices are outer-train-only; this is not a final ESI model."})
    print(f"04c experiments complete: {len(variants)} variants, output={args.output_dir}")


if __name__ == "__main__":
    main()
