"""04c: Evidence-Constrained Ridge ordinal resource baseline.

This module is deliberately downstream of the frozen 04a/04b artifacts.  It
does not read raw transcripts, rebuild claims, infer ESI, or apply Step D.
The only learned target is the weak resource state: 0, 1, or 2+.

The model is a proportional-odds ordinal logistic model with a shared Ridge
coefficient vector and ordered thresholds.  Outer evaluation is grouped by
case_id; lambda selection uses only grouped inner folds inside each outer
training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr


SCHEMA_VERSION = "04c_ridge_resource_ordinal_v2"
INPUT_SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_INPUT_DIR = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")
DEFAULT_OUTPUT_DIR = Path("outputs/04c_ridge_resource_ordinal_v2")
DEFAULT_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TARGETS = (0, 1, 2)
EPS = 1e-12


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return expit(value)


def softplus(value: float) -> float:
    return float(np.logaddexp(0.0, value))


def parse_lambdas(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("lambda grid must contain positive finite values")
    return sorted(set(values))


def target_from_label(row: dict[str, Any]) -> int:
    value = row.get("weak_resource_bucket")
    if value is None:
        raise ValueError(f"Eligible supervision row has no weak_resource_bucket: {row.get('case_id')}")
    value = int(value)
    if value not in TARGETS:
        raise ValueError(f"Invalid weak resource bucket {value!r} for {row.get('case_id')}")
    return value


def ordered_parameters(params: np.ndarray, feature_count: int) -> tuple[np.ndarray, float, float]:
    beta = params[:feature_count]
    theta1 = float(params[feature_count])
    theta2 = theta1 + softplus(float(params[feature_count + 1]))
    return beta, theta1, theta2


def ordinal_probabilities(
    params: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    beta, theta1, theta2 = ordered_parameters(params, X.shape[1])
    eta = X @ beta
    p_ge1 = expit(eta - theta1)
    p_ge2 = expit(eta - theta2)
    probabilities = np.column_stack((1.0 - p_ge1, p_ge1 - p_ge2, p_ge2))
    return np.clip(probabilities, EPS, 1.0)


def ordinal_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalty: float,
) -> float:
    probabilities = ordinal_probabilities(params, X)
    log_probability = np.log(probabilities[np.arange(len(y)), y])
    weighted_nll = -float(np.sum(weights * log_probability) / max(np.sum(weights), EPS))
    beta = params[: X.shape[1]]
    return weighted_nll + 0.5 * penalty * float(np.dot(beta, beta))


def fit_ordinal_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalty: float,
) -> dict[str, Any]:
    feature_count = X.shape[1]
    initial = np.zeros(feature_count + 2, dtype=float)
    initial[feature_count] = 0.0
    initial[feature_count + 1] = math.log(math.expm1(1.0))
    result = minimize(
        ordinal_objective,
        initial,
        args=(X, y, weights, penalty),
        method="L-BFGS-B",
        options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Ridge ordinal optimizer returned non-finite parameters")
    if not math.isfinite(float(result.fun)):
        raise RuntimeError("Ridge ordinal optimizer returned a non-finite objective")
    if not result.success:
        raise RuntimeError(f"Ridge ordinal optimizer failed: {result.message}")
    beta, theta1, theta2 = ordered_parameters(result.x, feature_count)
    if not theta1 < theta2:
        raise RuntimeError("Ordinal thresholds are not ordered")
    return {
        "params": result.x,
        "beta": beta,
        "theta1": theta1,
        "theta2": theta2,
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
    }


def case_average_probabilities(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[str(row["case_id"])].append(probability)
    return {case_id: np.mean(values, axis=0) for case_id, values in grouped.items()}


def make_group_folds(case_targets: dict[str, int], fold_count: int, seed: int) -> dict[str, int]:
    """Deterministically distribute cases by target without splitting a case."""
    ordered: list[str] = []
    for target in sorted(TARGETS):
        cases = sorted(case_id for case_id, value in case_targets.items() if value == target)
        if cases:
            offset = (seed + target) % len(cases)
            cases = cases[offset:] + cases[:offset]
            ordered.extend(cases)
    assignments: dict[str, int] = {}
    per_class_position = {target: 0 for target in TARGETS}
    for case_id in ordered:
        target = case_targets[case_id]
        assignments[case_id] = per_class_position[target] % fold_count
        per_class_position[target] += 1
    return assignments


def fit_preprocessor(
    rows: list[dict[str, Any]],
    indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
) -> dict[str, Any]:
    raw = np.asarray([[float(rows[i].get(name, 0.0)) for name in feature_names] for i in indices], dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Non-finite feature value in training partition")
    zero_variance: list[str] = []
    active: list[str] = []
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    binary_features: list[str] = []
    local_weights = np.asarray([weights[i] for i in indices], dtype=float)
    local_weights = np.maximum(local_weights, EPS)
    for column, name in enumerate(feature_names):
        values = raw[:, column]
        if np.all(np.isin(values, [0.0, 1.0])):
            binary_features.append(name)
            variance = float(np.sum(local_weights * (values - np.average(values, weights=local_weights)) ** 2))
            variance /= float(np.sum(local_weights))
            if variance <= EPS:
                zero_variance.append(name)
            else:
                active.append(name)
                means[name] = 0.0
                scales[name] = 1.0
            continue
        mean = float(np.average(values, weights=local_weights))
        variance = float(np.average((values - mean) ** 2, weights=local_weights))
        if variance <= EPS:
            zero_variance.append(name)
            continue
        active.append(name)
        means[name] = mean
        scales[name] = math.sqrt(variance)
    return {
        "feature_names": feature_names,
        "active_features": active,
        "zero_variance_features": sorted(zero_variance),
        "means": means,
        "scales": scales,
        "binary_features": sorted(binary_features),
    }


def transform_rows(rows: list[dict[str, Any]], indices: list[int], preprocessor: dict[str, Any]) -> np.ndarray:
    active = preprocessor["active_features"]
    matrix = np.asarray([[float(rows[i].get(name, 0.0)) for name in active] for i in indices], dtype=float)
    if matrix.size == 0:
        return np.zeros((len(indices), 0), dtype=float)
    for column, name in enumerate(active):
        matrix[:, column] = (matrix[:, column] - preprocessor["means"][name]) / preprocessor["scales"][name]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite transformed feature value")
    return matrix


def fit_and_predict(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    predict_indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    penalty: float,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    preprocessor = fit_preprocessor(rows, train_indices, feature_names, weights)
    X_train = transform_rows(rows, train_indices, preprocessor)
    X_predict = transform_rows(rows, predict_indices, preprocessor)
    model = fit_ordinal_model(X_train, targets[train_indices], weights[train_indices], penalty)
    probabilities = ordinal_probabilities(model["params"], X_predict)
    return model, preprocessor, probabilities


def multiclass_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, EPS, 1.0)
    return float(-np.mean(np.log(probabilities[np.arange(len(targets)), targets])))


def aggregate_case_rows(
    rows: list[dict[str, Any]],
    indices: list[int],
    probabilities: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for local, index in enumerate(indices):
        grouped[str(rows[index]["case_id"])].append(probabilities[local])
    case_ids = sorted(grouped)
    case_probabilities = np.asarray([np.mean(grouped[case_id], axis=0) for case_id in case_ids], dtype=float)
    return case_ids, case_probabilities, np.argmax(case_probabilities, axis=1)


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = [[0 for _ in TARGETS] for _ in TARGETS]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def f1_for_class(targets: np.ndarray, predictions: np.ndarray, target: int) -> tuple[float, float, float]:
    true_positive = int(np.sum((targets == target) & (predictions == target)))
    false_positive = int(np.sum((targets != target) & (predictions == target)))
    false_negative = int(np.sum((targets == target) & (predictions != target)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return precision, recall, f1


def quadratic_kappa(targets: np.ndarray, predictions: np.ndarray) -> float:
    matrix = np.asarray(confusion_matrix(targets, predictions), dtype=float)
    n = float(matrix.sum())
    if n <= 0:
        return 0.0
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    expected = np.outer(actual, predicted) / n
    weights = np.zeros((3, 3), dtype=float)
    for i in TARGETS:
        for j in TARGETS:
            weights[i, j] = ((i - j) / 2.0) ** 2
    observed = float(np.sum(weights * matrix) / n)
    expected_loss = float(np.sum(weights * expected) / n)
    return float(1.0 - observed / expected_loss) if expected_loss > EPS else 1.0


def classification_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    class_metrics = {}
    f1s = []
    recalls = []
    for target in TARGETS:
        precision, recall, f1 = f1_for_class(targets, predictions, target)
        class_metrics[str(target)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(np.sum(targets == target))}
        f1s.append(f1)
        recalls.append(recall)
    return {
        "count": int(len(targets)),
        "accuracy": float(np.mean(predictions == targets)) if len(targets) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "quadratic_weighted_kappa": quadratic_kappa(targets, predictions),
        "mean_absolute_ordinal_error": float(np.mean(np.abs(predictions - targets))) if len(targets) else 0.0,
        "log_loss": multiclass_log_loss(targets, probabilities) if len(targets) else 0.0,
        "brier_score": float(np.mean(np.sum((probabilities - np.eye(3)[targets]) ** 2, axis=1))) if len(targets) else 0.0,
        "class_metrics": class_metrics,
        "predicted_class_counts": {str(target): int(np.sum(predictions == target)) for target in TARGETS},
        "target_class_counts": {str(target): int(np.sum(targets == target)) for target in TARGETS},
        "resource_under_estimation_count": int(np.sum(predictions < targets)),
        "resource_over_estimation_count": int(np.sum(predictions > targets)),
        "confusion_matrix": confusion_matrix(targets, predictions),
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware pairwise AUC without fitting or tuning on the evaluation set."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def ordinal_probability_diagnostics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    expected_score = probabilities[:, 1] + 2.0 * probabilities[:, 2]
    return {
        "mean_expected_resource_score": float(np.mean(expected_score)),
        "mean_expected_score_by_target": {str(target): float(np.mean(expected_score[targets == target])) for target in TARGETS},
        "mean_probability_2_plus_by_target": {str(target): float(np.mean(probabilities[targets == target, 2])) for target in TARGETS},
        "auc_r_ge_1": binary_auc((targets >= 1).astype(int), 1.0 - probabilities[:, 0]),
        "auc_r_ge_2": binary_auc((targets >= 2).astype(int), probabilities[:, 2]),
        "expected_score_spearman": float(spearmanr(targets, expected_score).statistic),
    }


def calibration_report(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predictions == targets).astype(float)
    rows = []
    total_gap = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = (confidence >= low) & ((confidence < high) if index < bins - 1 else (confidence <= high))
        count = int(np.sum(selected))
        if count:
            mean_confidence = float(np.mean(confidence[selected]))
            accuracy = float(np.mean(correct[selected]))
            total_gap += count * abs(mean_confidence - accuracy)
        else:
            mean_confidence = None
            accuracy = None
        rows.append({"bin": index, "lower": low, "upper": high, "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return {"expected_calibration_error": total_gap / max(len(targets), 1), "bins": rows}


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left, EPS, 1.0)
    right = np.clip(right, EPS, 1.0)
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def select_lambda(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    feature_names: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    case_targets: dict[str, int],
    lambda_grid: list[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    train_cases = sorted(case_targets)
    fold_count = min(3, min(Counter(case_targets.values()).values()))
    fold_count = max(fold_count, 2)
    inner_assignment = make_group_folds(case_targets, fold_count, seed)
    records: list[dict[str, Any]] = []
    for penalty in lambda_grid:
        losses: list[float] = []
        failed = False
        for inner_fold in range(fold_count):
            validation_cases = {case_id for case_id in train_cases if inner_assignment[case_id] == inner_fold}
            inner_train = [i for i in train_indices if rows[i]["case_id"] not in validation_cases]
            inner_validation = [i for i in train_indices if rows[i]["case_id"] in validation_cases]
            if not inner_train or not inner_validation:
                failed = True
                break
            try:
                _, _, probabilities = fit_and_predict(rows, inner_train, inner_validation, feature_names, weights, targets, penalty)
                case_ids, case_probabilities, _ = aggregate_case_rows(rows, inner_validation, probabilities)
                inner_targets = np.asarray([case_targets[case_id] for case_id in case_ids], dtype=int)
                losses.append(multiclass_log_loss(inner_targets, case_probabilities))
            except (RuntimeError, ValueError, FloatingPointError):
                failed = True
                break
        record = {"lambda": penalty, "inner_fold_count": fold_count, "mean_case_log_loss": float(np.mean(losses)) if losses and not failed else None, "fold_log_losses": losses, "failed": failed}
        records.append(record)
    usable = [record for record in records if not record["failed"] and record["mean_case_log_loss"] is not None]
    if not usable:
        raise RuntimeError("All inner lambda fits failed")
    chosen = min(usable, key=lambda record: (float(record["mean_case_log_loss"]), float(record["lambda"])))
    return float(chosen["lambda"]), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambda-grid", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--fixed-lambda", type=float, default=None, help="Optional smoke-only lambda; skips inner selection")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = [float(args.fixed_lambda)] if args.fixed_lambda is not None else parse_lambdas(args.lambda_grid)
    if any(value <= 0 or not math.isfinite(value) for value in lambda_grid):
        raise ValueError("fixed lambda must be positive and finite")

    names = {
        "audit": "resource_feasibility_audit.json",
        "dictionary": "resource_feature_dictionary.json",
        "matrix": "resource_instance_feature_matrix.jsonl",
        "labels": "resource_supervision_case_audit.jsonl",
        "folds": "resource_grouped_fold_manifest.jsonl",
        "weights": "resource_training_weight_manifest.jsonl",
        "availability": "resource_fold_feature_availability.jsonl",
    }
    paths = {key: input_dir / value for key, value in names.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen 04a/04b inputs: " + ", ".join(missing))
    audit = read_json(paths["audit"])
    feature_dictionary = read_json(paths["dictionary"])
    matrix_rows = read_jsonl(paths["matrix"])
    label_rows = read_jsonl(paths["labels"])
    fold_rows = read_jsonl(paths["folds"])
    weight_rows = read_jsonl(paths["weights"])
    availability_rows = read_jsonl(paths["availability"])
    feature_names = list(feature_dictionary.get("model_features", {}).keys())
    if not feature_names or len(feature_names) != int(feature_dictionary.get("model_feature_count", -1)):
        raise ValueError("Frozen feature dictionary is missing a consistent model allowlist")
    forbidden_model_features = {"official_danger_signal", "official_danger_signal_count", "spo2_danger_signal", "final_esi", "step_d"}
    forbidden_hits = sorted(name for name in feature_names if any(token in name.lower() for token in forbidden_model_features))

    matrix_by_instance = {row["instance_id"]: row for row in matrix_rows}
    label_by_case = {row["case_id"]: row for row in label_rows if row.get("weak_resource_bucket") is not None}
    fold_by_case = {row["case_id"]: row for row in fold_rows}
    weight_by_instance = {row["instance_id"]: row for row in weight_rows}
    expected_eligible_instances = set(weight_by_instance)
    eligible_cases = set(fold_by_case)
    matrix_instances = set(matrix_by_instance)
    matrix_cases = {row["case_id"] for row in matrix_rows}
    issues: list[str] = []
    if audit.get("schema_version") != INPUT_SCHEMA_VERSION:
        issues.append("input_schema_version_mismatch")
    if audit.get("release_gate_passed") is not True:
        issues.append("frozen_04_release_gate_not_passed")
    if audit.get("model_fit_performed") is not False:
        issues.append("input_artifact_claims_model_fit")
    if audit.get("invalid_measurement_feature_count") != 0:
        issues.append("invalid_measurement_feature_count_nonzero")
    if forbidden_hits:
        issues.append("forbidden_model_features_present")
    if len(matrix_rows) != 688 or len(matrix_instances) != 688:
        issues.append("instance_matrix_count_or_uniqueness_mismatch")
    if len(matrix_cases) != 541:
        issues.append("case_matrix_count_mismatch")
    if not eligible_cases.issubset(set(label_by_case)):
        issues.append("fold_manifest_case_missing_label")
    if not expected_eligible_instances.issubset(matrix_instances):
        issues.append("weight_manifest_instance_missing_matrix")
    if any(row.get("feature_matrix_label_free") is not True for row in matrix_rows):
        issues.append("label_derived_metadata_in_feature_matrix")
    folds = sorted({int(row["fold_id"]) for row in fold_rows})
    if len(availability_rows) != len(folds):
        issues.append("fold_feature_availability_count_mismatch")
    availability_by_fold = {int(row["fold_id"]): row for row in availability_rows}
    if set(availability_by_fold) != set(folds):
        issues.append("fold_feature_availability_id_mismatch")
    if any(row.get("target_available_in_audit_only") is not True for row in fold_rows):
        issues.append("fold_manifest_target_not_audit_only")
    if any(row.get("audit_only") is not True for row in label_rows if row.get("weak_resource_bucket") is not None):
        issues.append("supervision_label_not_audit_only")
    fold_cases = {case_id: int(row["fold_id"]) for case_id, row in fold_by_case.items()}
    if len(fold_cases) != len(eligible_cases):
        issues.append("duplicate_case_fold_assignment")
    if any(weight_by_instance[instance_id]["case_id"] not in eligible_cases for instance_id in expected_eligible_instances):
        issues.append("weight_case_not_in_eligible_fold_manifest")
    if any(int(weight_by_instance[instance_id]["fold_id"]) != fold_cases[weight_by_instance[instance_id]["case_id"]] for instance_id in expected_eligible_instances):
        issues.append("training_weight_fold_mismatch")
    if issues:
        raise RuntimeError("04c preflight failed: " + ", ".join(sorted(set(issues))))

    eligible_rows = [matrix_by_instance[instance_id] for instance_id in sorted(expected_eligible_instances)]
    eligible_ids = [row["instance_id"] for row in eligible_rows]
    row_index = {instance_id: index for index, instance_id in enumerate(eligible_ids)}
    targets = np.asarray([target_from_label(label_by_case[row["case_id"]]) for row in eligible_rows], dtype=int)
    weights = np.asarray([float(weight_by_instance[row["instance_id"]]["instance_weight"]) for row in eligible_rows], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Invalid training weight")
    case_targets = {case_id: target_from_label(label_by_case[case_id]) for case_id in eligible_cases}
    class_counts = Counter(targets.tolist())
    if set(class_counts) != set(TARGETS):
        raise RuntimeError("Eligible training target does not contain all three resource classes")
    if abs(sum(weights) - sum(float(row["supervision_weight"]) for row in label_by_case.values() if row["case_id"] in eligible_cases)) > 1e-8:
        raise RuntimeError("Training weights do not reconcile to case supervision weights")

    outer_assignment = fold_cases
    oof_probabilities = np.full((len(eligible_rows), 3), np.nan, dtype=float)
    oof_fold: dict[str, int] = {}
    fold_metrics: list[dict[str, Any]] = []
    fold_coefficients: list[dict[str, Any]] = []
    lambda_selection: list[dict[str, Any]] = []
    fold_preprocessors: dict[int, dict[str, Any]] = {}
    optimizer_records: list[dict[str, Any]] = []
    for fold_id in folds:
        validation_indices = [index for index, row in enumerate(eligible_rows) if outer_assignment[row["case_id"]] == fold_id]
        train_indices = [index for index, row in enumerate(eligible_rows) if outer_assignment[row["case_id"]] != fold_id]
        train_cases = {eligible_rows[index]["case_id"] for index in train_indices}
        validation_cases = {eligible_rows[index]["case_id"] for index in validation_indices}
        overlap = train_cases & validation_cases
        if overlap:
            raise RuntimeError(f"Case overlap in fold {fold_id}: {sorted(overlap)[:3]}")
        if args.fixed_lambda is None:
            selected_lambda, selection_records = select_lambda(
                eligible_rows,
                train_indices,
                feature_names,
                weights,
                targets,
                {case_id: case_targets[case_id] for case_id in train_cases},
                lambda_grid,
                seed=101 + fold_id,
            )
        else:
            selected_lambda = lambda_grid[0]
            selection_records = [{"lambda": selected_lambda, "inner_fold_count": 0, "mean_case_log_loss": None, "fold_log_losses": [], "failed": False, "selection": "fixed_cli_value"}]
        model, preprocessor, probabilities = fit_and_predict(
            eligible_rows, train_indices, validation_indices, feature_names, weights, targets, selected_lambda
        )
        oof_probabilities[validation_indices] = probabilities
        for index in validation_indices:
            oof_fold[eligible_rows[index]["instance_id"]] = fold_id
        fold_preprocessors[fold_id] = preprocessor
        frozen_availability = availability_by_fold[fold_id]
        frozen_zero_variance = set(frozen_availability.get("zero_variance_train_features", []))
        recomputed_zero_variance = set(preprocessor["zero_variance_features"])
        if frozen_zero_variance != recomputed_zero_variance:
            raise RuntimeError(
                f"Fold {fold_id} feature availability mismatch: "
                f"frozen={sorted(frozen_zero_variance)} recomputed={sorted(recomputed_zero_variance)}"
            )
        validation_only_nonzero = set(frozen_availability.get("validation_only_nonzero_features", []))
        if validation_only_nonzero & set(preprocessor["active_features"]):
            raise RuntimeError(f"Fold {fold_id} uses validation-only feature(s)")
        lambda_selection.append({"fold_id": fold_id, "selected_lambda": selected_lambda, "candidates": selection_records})
        optimizer_records.append({"fold_id": fold_id, "success": model["optimizer_success"], "status": model["optimizer_status"], "iterations": model["iterations"], "message": model["optimizer_message"]})
        case_ids, case_probabilities, _ = aggregate_case_rows(eligible_rows, validation_indices, probabilities)
        case_targets_array = np.asarray([case_targets[case_id] for case_id in case_ids], dtype=int)
        fold_metrics.append({"fold_id": fold_id, "train_case_count": len(train_cases), "validation_case_count": len(validation_cases), "train_instance_count": len(train_indices), "validation_instance_count": len(validation_indices), "case_metrics": classification_metrics(case_targets_array, case_probabilities), "selected_lambda": selected_lambda, "zero_variance_train_features": preprocessor["zero_variance_features"], "effective_feature_count": len(preprocessor["active_features"])})
        for name in feature_names:
            coefficient = 0.0
            estimable = name in preprocessor["active_features"]
            if estimable:
                coefficient = float(model["beta"][preprocessor["active_features"].index(name)])
            fold_coefficients.append({"fold_id": fold_id, "feature": name, "coefficient": coefficient, "estimable": estimable, "lambda": selected_lambda, "preprocess_scale": preprocessor["scales"].get(name), "preprocess_center": preprocessor["means"].get(name)})
    if not np.all(np.isfinite(oof_probabilities)):
        raise RuntimeError("OOF probability matrix contains non-finite values")
    probability_sums = oof_probabilities.sum(axis=1)
    if np.any(np.abs(probability_sums - 1.0) > 1e-8):
        raise RuntimeError("OOF probabilities do not sum to one")

    case_ids, case_probabilities, case_predictions = aggregate_case_rows(eligible_rows, list(range(len(eligible_rows))), oof_probabilities)
    case_targets_array = np.asarray([case_targets[case_id] for case_id in case_ids], dtype=int)
    instance_predictions = np.argmax(oof_probabilities, axis=1)
    instance_metrics = classification_metrics(targets, oof_probabilities)
    case_metrics = classification_metrics(case_targets_array, case_probabilities)
    calibration = calibration_report(case_targets_array, case_probabilities)
    majority_target = Counter(case_targets_array.tolist()).most_common(1)[0][0]
    majority_probabilities = np.zeros((len(case_targets_array), 3), dtype=float)
    majority_probabilities[:, majority_target] = 1.0
    case_probability_diagnostics = ordinal_probability_diagnostics(case_targets_array, case_probabilities)
    case_metrics["ordinal_probability_diagnostics"] = case_probability_diagnostics
    prior = np.bincount(case_targets_array, minlength=3).astype(float)
    prior /= prior.sum()
    prior_probabilities = np.tile(prior, (len(case_targets_array), 1))
    comparison = {
        "model": {**case_metrics, "ordinal_probability_diagnostics": case_probability_diagnostics},
        "majority_class_baseline": classification_metrics(case_targets_array, majority_probabilities),
        "empirical_prior_probability_baseline": {
            **classification_metrics(case_targets_array, prior_probabilities),
            "prior_probability": prior.tolist(),
        },
        "model_name": "weighted_proportional_odds_ridge",
        "target_definition": "0=0 resources, 1=1 resource, 2=2+ resources",
    }
    predicted_class_coverage = sum(1 for target in TARGETS if case_metrics["predicted_class_counts"][str(target)] > 0)
    model_recall_values = [case_metrics["class_metrics"][str(target)]["recall"] for target in TARGETS]
    model_quality = {
        "predicted_class_coverage": predicted_class_coverage,
        "class_prediction_collapse": predicted_class_coverage < len(TARGETS),
        "beats_empirical_prior_macro_f1": case_metrics["macro_f1"] > comparison["empirical_prior_probability_baseline"]["macro_f1"] + 1e-9,
        "beats_empirical_prior_balanced_accuracy": case_metrics["balanced_accuracy"] > comparison["empirical_prior_probability_baseline"]["balanced_accuracy"] + 1e-9,
        "all_class_recalls_positive": all(value > 0.0 for value in model_recall_values),
        "quadratic_weighted_kappa_positive": case_metrics["quadratic_weighted_kappa"] > 0.0,
        "cumulative_auc_above_chance": (case_probability_diagnostics["auc_r_ge_1"] or 0.0) > 0.5 and (case_probability_diagnostics["auc_r_ge_2"] or 0.0) > 0.5,
        "model_quality_gate_passed": (
            predicted_class_coverage == len(TARGETS)
            and case_metrics["macro_f1"] > comparison["empirical_prior_probability_baseline"]["macro_f1"] + 1e-9
            and case_metrics["quadratic_weighted_kappa"] > 0.0
            and all(value > 0.0 for value in model_recall_values)
            and (case_probability_diagnostics["auc_r_ge_1"] or 0.0) > 0.5
            and (case_probability_diagnostics["auc_r_ge_2"] or 0.0) > 0.5
        ),
        "downstream_model_ready": False,
        "interpretation": "A structural baseline may be retained for diagnosis, but class collapse blocks downstream resource-path use.",
    }
    model_quality["downstream_model_ready"] = bool(model_quality["model_quality_gate_passed"])
    comparison["model_quality"] = model_quality

    duplicate_rows: list[dict[str, Any]] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eligible_rows):
        grouped_indices[row["case_id"]].append(index)
    for case_id, indices in sorted(grouped_indices.items()):
        if len(indices) <= 1:
            continue
        mean_probability = np.mean(oof_probabilities[indices], axis=0)
        js_values = [js_divergence(oof_probabilities[index], mean_probability) for index in indices]
        duplicate_rows.append({"case_id": case_id, "instance_ids": [eligible_rows[index]["instance_id"] for index in indices], "fold_ids": sorted({oof_fold[eligible_rows[index]["instance_id"]] for index in indices}), "realization_count": len(indices), "mean_js_to_case_probability": float(np.mean(js_values)), "mean_absolute_probability_difference": float(np.mean(np.abs(oof_probabilities[indices] - mean_probability))), "instance_prediction_agreement": float(np.mean(instance_predictions[indices] == int(np.argmax(mean_probability)))), "case_probability": mean_probability.tolist()})

    feature_stability: dict[str, Any] = {}
    for name in feature_names:
        values = [row["coefficient"] for row in fold_coefficients if row["feature"] == name and row["estimable"]]
        signs = [np.sign(value) for value in values if abs(value) > 1e-10]
        feature_stability[name] = {"estimable_fold_count": len(values), "coefficient_mean": float(np.mean(values)) if values else None, "coefficient_sd": float(np.std(values)) if values else None, "nonzero_sign_agreement": float(max(Counter(signs).values()) / len(signs)) if signs else None, "not_estimable_fold_count": len(folds) - len(values)}

    instance_oof_rows = []
    for index, row in enumerate(eligible_rows):
        instance_oof_rows.append({"schema_version": SCHEMA_VERSION, "instance_id": row["instance_id"], "case_id": row["case_id"], "fold_id": oof_fold[row["instance_id"]], "target_resource_bucket": int(targets[index]), "instance_weight": float(weights[index]), "probability_0": float(oof_probabilities[index, 0]), "probability_1": float(oof_probabilities[index, 1]), "probability_2_plus": float(oof_probabilities[index, 2]), "predicted_resource_bucket": int(instance_predictions[index]), "feature_set_label_free": True})
    case_oof_rows = []
    for case_id, probability, prediction in zip(case_ids, case_probabilities, case_predictions):
        case_oof_rows.append({"schema_version": SCHEMA_VERSION, "case_id": case_id, "instance_ids": [row["instance_id"] for row in eligible_rows if row["case_id"] == case_id], "fold_id": outer_assignment[case_id], "target_resource_bucket": int(case_targets[case_id]), "probability_0": float(probability[0]), "probability_1": float(probability[1]), "probability_2_plus": float(probability[2]), "predicted_resource_bucket": int(prediction), "aggregation": "equal_mean_across_realizations"})

    error_rows = []
    review_rows = []
    for row in case_oof_rows:
        probability = np.asarray([row["probability_0"], row["probability_1"], row["probability_2_plus"]])
        if row["predicted_resource_bucket"] != row["target_resource_bucket"]:
            error_rows.append({**row, "error_type": "resource_bucket_mismatch", "ordinal_error": row["predicted_resource_bucket"] - row["target_resource_bucket"], "confidence": float(np.max(probability))})
        entropy = float(-np.sum(probability * np.log(np.clip(probability, EPS, 1.0))))
        if row["predicted_resource_bucket"] != row["target_resource_bucket"] or entropy > 0.95:
            review_rows.append({**row, "review_reason": "oof_error" if row["predicted_resource_bucket"] != row["target_resource_bucket"] else "high_entropy_boundary", "entropy": entropy})

    input_hashes = {key: sha256_file(path) for key, path in paths.items()}
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "input_hashes": input_hashes,
        "frozen_04_audit_inputs": audit.get("inputs", {}),
        "model_name": "weighted_proportional_odds_ridge",
        "target": "weak_resource_bucket {0,1,2_plus}",
        "model_feature_count": len(feature_names),
        "model_features": feature_names,
        "forbidden_model_features": sorted(forbidden_hits),
        "outer_fold_count": len(folds),
        "inner_lambda_selection": args.fixed_lambda is None,
        "lambda_grid": lambda_grid,
        "selected_lambdas": {str(row["fold_id"]): row["selected_lambda"] for row in lambda_selection},
        "processed_instance_count": len(matrix_rows),
        "eligible_instance_count": len(eligible_rows),
        "eligible_case_count": len(case_ids),
        "target_class_counts_instance": {str(target): int(class_counts.get(target, 0)) for target in TARGETS},
        "target_class_counts_case": {str(target): int(np.sum(case_targets_array == target)) for target in TARGETS},
        "preflight": {
            "train_validation_case_overlap": 0,
            "duplicate_realization_cross_fold": 0,
            "oof_eligible_case_coverage": len(case_ids) == len(eligible_cases),
            "oof_eligible_instance_coverage": len(eligible_rows) == len(expected_eligible_instances),
            "probability_nonfinite_count": 0,
            "probability_sum_violation_count": 0,
            "threshold_order_violation_count": 0,
            "training_weight_mismatch_count": 0,
            "validation_used_for_preprocessing": 0,
            "validation_used_for_feature_selection": 0,
            "final_esi_output_count": 0,
            "step_d_application_count": 0,
            "step_d_model_feature_count": 0,
            "invalid_measurement_feature_count": audit.get("invalid_measurement_feature_count", 0),
            "optimizer_failure_count": int(sum(not row["success"] for row in optimizer_records)),
            "objective_nonfinite_count": 0,
        },
        "optimizer_records": optimizer_records,
        "optimizer_failure_count": int(sum(not row["success"] for row in optimizer_records)),
        "objective_nonfinite_count": 0,
        "training_weight_sum": float(np.sum(weights)),
        "supervision_weight_sum": float(sum(float(row["supervision_weight"]) for row in label_by_case.values() if row["case_id"] in eligible_cases)),
        "release_gates": {
            "all_preflight_checks_passed": True,
            "model_fit_performed": True,
            "final_esi_generated": False,
            "artifact_release_gate_passed": True,
            "model_quality_gate_passed": model_quality["model_quality_gate_passed"],
            "downstream_model_ready": model_quality["downstream_model_ready"],
            "release_gate_passed": True,
            "release_gate_scope": "artifact_integrity_and_reproducibility_only",
        },
        "model_quality": model_quality,
        "note": "This is a resource ordinal baseline, not a clinical ESI predictor. Step D is diagnostic-only and never applied here.",
    }
    write_json(output_dir / "ridge_training_audit.json", audit_payload)
    write_json(output_dir / "ridge_model_comparison.json", comparison)
    write_jsonl(output_dir / "ridge_fold_metrics.jsonl", fold_metrics)
    write_jsonl(output_dir / "ridge_fold_coefficients.jsonl", fold_coefficients)
    write_json(output_dir / "ridge_feature_stability.json", {"schema_version": SCHEMA_VERSION, "features": feature_stability})
    write_jsonl(output_dir / "ridge_instance_oof_predictions.jsonl", instance_oof_rows)
    write_jsonl(output_dir / "ridge_case_oof_predictions.jsonl", case_oof_rows)
    write_json(output_dir / "ridge_case_metrics.json", case_metrics)
    write_json(output_dir / "ridge_confusion_matrix.json", {"schema_version": SCHEMA_VERSION, "labels": ["0", "1", "2_plus"], "matrix": case_metrics["confusion_matrix"]})
    write_json(output_dir / "ridge_calibration.json", calibration)
    write_jsonl(output_dir / "ridge_duplicate_consistency.jsonl", duplicate_rows)
    write_jsonl(output_dir / "ridge_error_cases.jsonl", error_rows)
    write_jsonl(output_dir / "ridge_review_queue.jsonl", review_rows)
    write_json(output_dir / "ridge_lambda_selection.json", {"schema_version": SCHEMA_VERSION, "folds": lambda_selection})
    print(f"04c complete: eligible cases={len(case_ids)}, eligible instances={len(eligible_rows)}, folds={len(folds)}")
    print(f"Output directory: {output_dir}")
    print(f"Case OOF accuracy={case_metrics['accuracy']:.4f}, macro_f1={case_metrics['macro_f1']:.4f}, log_loss={case_metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()
