"""04c Stage B: boundary-specific ordinal resource models.

The runner compares raw dual-cumulative and conditional Ridge models on the
same frozen 04a cohort and case folds. It deliberately excludes threshold
calibration, class weighting, and fusion penalties from this first structural
comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_PATH = Path(__file__).with_name("04c_run_resource_ordinal_experiments.py")
SPEC = importlib.util.spec_from_file_location("experiments04c", EXPERIMENT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import experiment module: {EXPERIMENT_PATH}")
EXPERIMENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENTS)

ORTHO_PATH = Path(__file__).with_name("04c_run_orthogonal_ablation.py")
ORTHO_SPEC = importlib.util.spec_from_file_location("orthogonal04c", ORTHO_PATH)
if ORTHO_SPEC is None or ORTHO_SPEC.loader is None:
    raise ImportError(f"Cannot import orthogonal module: {ORTHO_PATH}")
ORTHO = importlib.util.module_from_spec(ORTHO_SPEC)
ORTHO_SPEC.loader.exec_module(ORTHO)


RIDGE = EXPERIMENTS.RIDGE
SCHEMA_VERSION = "04c_boundary_specific_ordinal_v1"
INPUT_SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_INPUT_DIR = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")
DEFAULT_OUTPUT_DIR = Path("outputs/04c_boundary_specific_ordinal_v1")
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


def binary_objective(params: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float) -> float:
    beta = params[:-1]
    intercept = float(params[-1])
    eta = X @ beta + intercept
    loss = np.logaddexp(0.0, eta) - y * eta
    return float(np.sum(weights * loss) / max(np.sum(weights), EPS) + 0.5 * penalty * np.dot(beta, beta))


def fit_binary_model(X: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float) -> dict[str, Any]:
    initial = np.zeros(X.shape[1] + 1, dtype=float)
    result = RIDGE.minimize(
        binary_objective,
        initial,
        args=(X, y.astype(float), weights, penalty),
        method="L-BFGS-B",
        options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
    )
    if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)) or not result.success:
        raise RuntimeError(f"Binary Ridge optimizer failed: {result.message}")
    return {
        "params": result.x,
        "beta": result.x[:-1],
        "intercept": float(result.x[-1]),
        "objective": float(result.fun),
        "iterations": int(getattr(result, "nit", 0)),
        "optimizer_message": str(result.message),
    }


def binary_probabilities(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return RIDGE.expit(X @ model["beta"] + model["intercept"])


def fit_boundary_models(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    predict_indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    penalty: float,
    conditional: bool,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    preprocessor = RIDGE.fit_preprocessor(rows, train_indices, feature_names, weights)
    X_train = RIDGE.transform_rows(rows, train_indices, preprocessor)
    X_predict = RIDGE.transform_rows(rows, predict_indices, preprocessor)
    train_targets = targets[train_indices]
    train_weights = weights[train_indices]
    y1 = (train_targets >= 1).astype(int)
    y2 = (train_targets == 2).astype(int)
    second_mask = train_targets >= 1 if conditional else np.ones(len(train_targets), dtype=bool)
    if len(np.unique(y1)) < 2 or len(np.unique(y2[second_mask])) < 2:
        raise RuntimeError("Boundary training partition lacks both binary classes")
    model1 = fit_binary_model(X_train, y1, train_weights, penalty)
    model2 = fit_binary_model(X_train[second_mask], y2[second_mask], train_weights[second_mask], penalty)
    return {"head_1": model1, "head_2": model2, "preprocessor": preprocessor}, preprocessor, binary_probabilities(model1, X_predict), binary_probabilities(model2, X_predict)


def aggregate_scalar(rows: list[dict[str, Any]], indices: list[int], values: np.ndarray) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for local, index in enumerate(indices):
        grouped[str(rows[index]["case_id"])].append(float(values[local]))
    case_ids = sorted(grouped)
    return case_ids, np.asarray([np.mean(grouped[case_id]) for case_id in case_ids], dtype=float)


def reconstruct_probabilities(q1: np.ndarray, q2: np.ndarray, conditional: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    raw_q1 = np.clip(q1, 0.0, 1.0)
    raw_q2 = np.clip(q2, 0.0, 1.0)
    violations = raw_q2 > raw_q1 + EPS
    if conditional:
        adjusted_q1, adjusted_q2 = raw_q1, raw_q2
    else:
        midpoint = 0.5 * (raw_q1 + raw_q2)
        adjusted_q1 = np.where(violations, midpoint, raw_q1)
        adjusted_q2 = np.where(violations, midpoint, raw_q2)
    if conditional:
        probabilities = np.column_stack((1.0 - adjusted_q1, adjusted_q1 * (1.0 - adjusted_q2), adjusted_q1 * adjusted_q2))
    else:
        probabilities = np.column_stack((1.0 - adjusted_q1, adjusted_q1 - adjusted_q2, adjusted_q2))
    return probabilities, adjusted_q1, adjusted_q2, int(np.sum(violations)), int(np.sum(np.abs(adjusted_q1 - raw_q1) + np.abs(adjusted_q2 - raw_q2) > EPS))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    positives = int(np.sum(labels == 1))
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(np.sum(precision[ordered == 1]) / positives)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(positive & (predictions == 1)))
    tn = int(np.sum(negative & (predictions == 0)))
    fp = int(np.sum(negative & (predictions == 1)))
    fn = int(np.sum(positive & (predictions == 0)))
    return {
        "count": int(len(labels)),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "roc_auc": RIDGE.binary_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(np.clip(scores, EPS, 1.0)) + (1 - labels) * np.log(np.clip(1.0 - scores, EPS, 1.0)))),
    }


def coefficient_diagnostics(model_bundle: dict[str, Any], feature_names: list[str], top_k: int = 10) -> dict[str, Any]:
    active = model_bundle["preprocessor"]["active_features"]
    vectors = []
    for key in ("head_1", "head_2"):
        vector = np.zeros(len(feature_names), dtype=float)
        for name, coefficient in zip(active, model_bundle[key]["beta"]):
            vector[feature_names.index(name)] = float(coefficient)
        vectors.append(vector)
    left, right = vectors
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    order_left = np.argsort(-np.abs(left))[:top_k]
    order_right = np.argsort(-np.abs(right))[:top_k]
    active_signs = (np.abs(left) > 1e-9) & (np.abs(right) > 1e-9)
    return {
        "coefficient_cosine_similarity": float(np.dot(left, right) / norm) if norm > EPS else None,
        "sign_disagreement_count": int(np.sum(active_signs & (np.sign(left) != np.sign(right)))),
        "top_k": top_k,
        "top_k_feature_overlap": int(len(set(order_left.tolist()) & set(order_right.tolist()))),
        "coefficient_difference_l2": float(np.linalg.norm(left - right)),
        "head_1_coefficients": {name: float(value) for name, value in zip(feature_names, left)},
        "head_2_coefficients": {name: float(value) for name, value in zip(feature_names, right)},
    }


def select_lambda(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    label_by_case: dict[str, dict[str, Any]],
    conditional: bool,
    grid: list[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    case_targets = {rows[index]["case_id"]: EXPERIMENTS.target_of(label_by_case, rows[index]["case_id"]) for index in train_indices}
    class_counts = np.bincount(np.asarray(list(case_targets.values()), dtype=int), minlength=3)
    fold_count = max(min(3, int(np.min(class_counts[class_counts > 0]))), 2)
    assignment = RIDGE.make_group_folds(case_targets, fold_count, seed)
    records = []
    for penalty in grid:
        losses = []
        failed = False
        for inner_fold in sorted(set(assignment.values())):
            validation_cases = {case_id for case_id, fold in assignment.items() if fold == inner_fold}
            inner_train = [index for index in train_indices if rows[index]["case_id"] not in validation_cases]
            inner_validation = [index for index in train_indices if rows[index]["case_id"] in validation_cases]
            try:
                _, _, q1, q2 = fit_boundary_models(rows, inner_train, inner_validation, feature_names, weights, targets, penalty, conditional)
                case_ids, case_q1 = aggregate_scalar(rows, inner_validation, q1)
                _, case_q2 = aggregate_scalar(rows, inner_validation, q2)
                probabilities, _, _, _, _ = reconstruct_probabilities(case_q1, case_q2, conditional)
                inner_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
                losses.append(RIDGE.multiclass_log_loss(inner_targets, probabilities))
            except (RuntimeError, ValueError, FloatingPointError):
                failed = True
                break
        records.append({"lambda": penalty, "mean_case_log_loss": float(np.mean(losses)) if losses and not failed else None, "fold_log_losses": losses, "failed": failed})
    usable = [record for record in records if not record["failed"] and record["mean_case_log_loss"] is not None]
    if not usable:
        raise RuntimeError("All boundary lambda fits failed")
    chosen = min(usable, key=lambda record: (float(record["mean_case_log_loss"]), float(record["lambda"])))
    return float(chosen["lambda"]), records


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
    lambda_grid: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    oof_q1 = np.full(len(rows), np.nan, dtype=float)
    oof_q2 = np.full(len(rows), np.nan, dtype=float)
    fold_records = []
    coefficient_records = []
    for fold_id in sorted(set(fold_by_case.values())):
        validation_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] == fold_id]
        train_indices = [index for index, row in enumerate(rows) if fold_by_case[row["case_id"]] != fold_id]
        selected_lambda, lambda_records = select_lambda(rows, train_indices, feature_names, weights, targets, label_by_case, conditional, lambda_grid, 1100 + fold_id)
        model_bundle, preprocessor, q1, q2 = fit_boundary_models(rows, train_indices, validation_indices, feature_names, weights, targets, selected_lambda, conditional)
        frozen_zero = set(availability_by_fold[fold_id].get("zero_variance_train_features", [])).intersection(feature_names)
        if frozen_zero != set(preprocessor["zero_variance_features"]):
            raise RuntimeError(f"{name}: feature availability mismatch in fold {fold_id}")
        oof_q1[validation_indices] = q1
        oof_q2[validation_indices] = q2
        case_ids, case_q1 = aggregate_scalar(rows, validation_indices, q1)
        _, case_q2 = aggregate_scalar(rows, validation_indices, q2)
        case_probabilities, adjusted_q1, adjusted_q2, violations, projection_count = reconstruct_probabilities(case_q1, case_q2, conditional)
        case_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
        fold_metrics = RIDGE.classification_metrics(case_targets, case_probabilities)
        fold_metrics["ordinal_probability_diagnostics"] = RIDGE.ordinal_probability_diagnostics(case_targets, case_probabilities)
        fold_metrics["boundary_1"] = binary_metrics((case_targets >= 1).astype(int), case_q1)
        boundary2_mask = case_targets >= 1 if conditional else np.ones(len(case_targets), dtype=bool)
        boundary2_labels = (case_targets[boundary2_mask] == 2).astype(int) if conditional else (case_targets >= 2).astype(int)
        fold_metrics["boundary_2"] = binary_metrics(boundary2_labels, case_q2[boundary2_mask])
        fold_metrics["raw_monotonic_violation_count"] = violations if not conditional else None
        fold_metrics["conditional_q2_gt_q1_count_diagnostic"] = violations if conditional else 0
        fold_metrics["projection_adjusted_case_count"] = projection_count
        coefficient_records.append({"fold_id": fold_id, **coefficient_diagnostics(model_bundle, feature_names)})
        fold_records.append({"variant": name, "feature_set": feature_set, "fold_id": fold_id, "conditional": conditional, "selected_lambda": selected_lambda, "case_aggregation": "mean_boundary_probability_then_probability_reconstruction", "case_metrics": fold_metrics, "lambda_candidates": lambda_records})

    if not np.all(np.isfinite(oof_q1)) or not np.all(np.isfinite(oof_q2)):
        raise RuntimeError(f"{name}: incomplete OOF boundary probabilities")
    case_ids = sorted(set(row["case_id"] for row in rows))
    case_q1 = np.asarray([np.mean([oof_q1[index] for index, row in enumerate(rows) if row["case_id"] == case_id]) for case_id in case_ids])
    case_q2 = np.asarray([np.mean([oof_q2[index] for index, row in enumerate(rows) if row["case_id"] == case_id]) for case_id in case_ids])
    case_targets = np.asarray([EXPERIMENTS.target_of(label_by_case, case_id) for case_id in case_ids], dtype=int)
    case_probabilities, adjusted_q1, adjusted_q2, violations, projection_count = reconstruct_probabilities(case_q1, case_q2, conditional)
    metrics = RIDGE.classification_metrics(case_targets, case_probabilities)
    metrics["ordinal_probability_diagnostics"] = RIDGE.ordinal_probability_diagnostics(case_targets, case_probabilities)
    metrics["boundary_1"] = binary_metrics((case_targets >= 1).astype(int), case_q1)
    boundary2_mask = case_targets >= 1 if conditional else np.ones(len(case_targets), dtype=bool)
    boundary2_labels = (case_targets[boundary2_mask] == 2).astype(int) if conditional else (case_targets >= 2).astype(int)
    metrics["boundary_2"] = binary_metrics(boundary2_labels, case_q2[boundary2_mask])
    metrics["raw_monotonic_violation_count"] = violations if not conditional else None
    metrics["conditional_q2_gt_q1_count_diagnostic"] = violations if conditional else 0
    metrics["projection_adjusted_case_count"] = projection_count
    metrics["fold_coefficient_diagnostics"] = coefficient_records

    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({"schema_version": SCHEMA_VERSION, "variant": name, "instance_id": row["instance_id"], "case_id": row["case_id"], "fold_id": int(fold_by_case[row["case_id"]]), "target_resource_bucket": int(targets[index]), "q_boundary_1": float(oof_q1[index]), "q_boundary_2": float(oof_q2[index])})
    case_oof_rows = []
    for position, case_id in enumerate(case_ids):
        probability = case_probabilities[position]
        case_oof_rows.append({"schema_version": SCHEMA_VERSION, "variant": name, "feature_set": feature_set, "case_id": case_id, "fold_id": int(fold_by_case[case_id]), "target_resource_bucket": int(case_targets[position]), "q_boundary_1_raw": float(case_q1[position]), "q_boundary_2_raw": float(case_q2[position]), "q_boundary_1_used": float(adjusted_q1[position]), "q_boundary_2_used": float(adjusted_q2[position]), "probability_0": float(probability[0]), "probability_1": float(probability[1]), "probability_2_plus": float(probability[2]), "raw_probability_prediction": int(np.argmax(probability)), "decision_prediction": int(np.argmax(probability)), "aggregation": "mean_boundary_probability_then_probability_reconstruction"})
    return oof_rows, case_oof_rows, metrics, fold_records


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
    variants = [
        ("DUAL_FULL40", "FULL40", full_features, False),
        ("DUAL_ORTHO23", "ORTHO23", clinical_features, False),
        ("CONDITIONAL_FULL40", "FULL40", full_features, True),
        ("CONDITIONAL_ORTHO23", "ORTHO23", clinical_features, True),
    ]
    summaries = []
    case_outputs = {}
    for name, feature_set, feature_names, conditional in variants:
        oof, case_oof, metrics, folds = run_variant(name, feature_set, feature_names, rows, label_by_case, fold_by_case, availability_by_fold, weights, targets, conditional, lambda_grid)
        case_outputs[name] = case_oof
        write_jsonl(args.output_dir / f"oof_{name}.jsonl", oof)
        write_jsonl(args.output_dir / f"case_oof_{name}.jsonl", case_oof)
        write_jsonl(args.output_dir / f"folds_{name}.jsonl", folds)
        summaries.append({"variant": name, "feature_set": feature_set, "feature_count": len(feature_names), "conditional": conditional, "common_cohort": metrics})
    paired = {
        "DUAL_ORTHO23_vs_DUAL_FULL40": EXPERIMENTS.paired_bootstrap(case_outputs["DUAL_FULL40"], case_outputs["DUAL_ORTHO23"], "DUAL_FULL40", "DUAL_ORTHO23"),
        "CONDITIONAL_ORTHO23_vs_CONDITIONAL_FULL40": EXPERIMENTS.paired_bootstrap(case_outputs["CONDITIONAL_FULL40"], case_outputs["CONDITIONAL_ORTHO23"], "CONDITIONAL_FULL40", "CONDITIONAL_ORTHO23"),
    }
    write_json(args.output_dir / "paired_bootstrap_comparisons.json", paired)
    write_json(args.output_dir / "run_manifest.json", {"schema_version": SCHEMA_VERSION, "input_schema_version": INPUT_SCHEMA_VERSION, "input_audit_hash": RIDGE.sha256_file(args.input_dir / "resource_feasibility_audit.json"), "feature_dictionary_hash": RIDGE.sha256_file(args.input_dir / "resource_feature_dictionary.json"), "case_count": len(fold_by_case), "instance_count": len(rows), "outer_folds": sorted(set(fold_by_case.values())), "feature_sets": {"FULL40": full_features, "ORTHO23": clinical_features}, "quality_features": quality_features, "lambda_grid": lambda_grid, "threshold_calibration": False, "class_weighting": "natural", "partial_weight": 0.5, "note": "Stage B raw structural comparison; 05/06 downstream remains blocked."})
    write_json(args.output_dir / "experiment_summary.json", {"schema_version": SCHEMA_VERSION, "variants": summaries, "paired_comparisons": list(paired), "note": "No threshold calibration, class weighting, or fusion penalty was used in Stage B raw comparison."})
    print(f"04c boundary-specific ordinal experiments complete: {len(variants)} variants, output={args.output_dir}")


if __name__ == "__main__":
    main()
