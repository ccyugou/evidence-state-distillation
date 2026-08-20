#!/usr/bin/env python3
"""Grouped nested-CV training for fused conditional ordinal resource learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


SCHEMA_VERSION = "04_el_fafcor_v2.0"
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0)
FUSION_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_weight_vector(y: np.ndarray, mode: str, beta: float) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    if mode == "natural":
        by_class = np.ones(3)
    elif mode == "sqrt_inverse":
        by_class = 1.0 / np.sqrt(counts)
    else:
        by_class = (1.0 - beta) / (1.0 - np.power(beta, counts))
    by_class /= np.average(by_class, weights=counts)
    return by_class[y]


class Transform:
    def fit(self, x: np.ndarray, names: list[str]) -> "Transform":
        self.log_mask = np.array([
            name.startswith(("trajectory__", "modifier__", "quality__", "interaction__")) and np.max(x[:, j]) > 1
            for j, name in enumerate(names)
        ])
        z = x.copy()
        z[:, self.log_mask] = np.log1p(z[:, self.log_mask])
        self.scale_mask = np.array([len(np.unique(z[:, j])) > 2 for j in range(z.shape[1])])
        self.mean = np.where(self.scale_mask, z.mean(0), 0.0)
        self.std = np.where(self.scale_mask, z.std(0), 1.0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[:, self.log_mask] = np.log1p(z[:, self.log_mask])
        return (z - self.mean) / self.std


class FusedConditional:
    def __init__(self, ridge: float, fusion: float, ordinal_smoothing: float = 0.0):
        self.ridge, self.fusion = ridge, fusion
        self.ordinal_smoothing = ordinal_smoothing

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> "FusedConditional":
        n, p = x.shape
        mask2 = y >= 1
        alpha = self.ordinal_smoothing
        targets = np.eye(3)[y] * (1.0 - alpha)
        targets[y == 0, 1] += alpha
        targets[y == 1, 0] += alpha / 2
        targets[y == 1, 2] += alpha / 2
        targets[y == 2, 1] += alpha
        y1 = 1.0 - targets[:, 0]
        y2 = targets[mask2, 2] / targets[mask2, 1:].sum(1)
        w1, w2 = w / w.sum(), w[mask2] / w[mask2].sum()

        def objective(v):
            a1, a2 = v[:2]
            b1, b2 = v[2:2+p], v[2+p:]
            e1, e2 = a1 + x @ b1, a2 + x[mask2] @ b2
            loss = np.sum(w1 * (np.logaddexp(0, e1) - y1 * e1))
            loss += np.sum(w2 * (np.logaddexp(0, e2) - y2 * e2))
            loss += self.ridge * (b1 @ b1 + b2 @ b2) + self.fusion * ((b1 - b2) @ (b1 - b2))
            g1, g2 = w1 * (expit(e1) - y1), w2 * (expit(e2) - y2)
            da1, da2 = g1.sum(), g2.sum()
            db1 = x.T @ g1 + 2 * self.ridge * b1 + 2 * self.fusion * (b1 - b2)
            db2 = x[mask2].T @ g2 + 2 * self.ridge * b2 + 2 * self.fusion * (b2 - b1)
            return loss, np.r_[da1, da2, db1, db2]

        init = np.zeros(2 + 2 * p)
        init[0] = np.log(np.clip(y1.mean(), 1e-3, 1-1e-3) / np.clip(1-y1.mean(), 1e-3, 1))
        init[1] = np.log(np.clip(y2.mean(), 1e-3, 1-1e-3) / np.clip(1-y2.mean(), 1e-3, 1))
        result = minimize(objective, init, method="L-BFGS-B", jac=True, options={"maxiter": 800, "ftol": 1e-10})
        self.converged = bool(result.success)
        self.intercepts = result.x[:2]
        self.coef1, self.coef2 = result.x[2:2+p], result.x[2+p:]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q1 = expit(self.intercepts[0] + x @ self.coef1)
        q2 = expit(self.intercepts[1] + x @ self.coef2)
        probs = np.c_[1-q1, q1*(1-q2), q1*q2]
        return q1, q2, probs


def aggregate_case(y, groups, q1, q2):
    unique = np.unique(groups)
    cy, cq1, cq2 = [], [], []
    for group in unique:
        mask = groups == group
        cy.append(y[mask][0]); cq1.append(q1[mask].mean()); cq2.append(q2[mask].mean())
    cq1, cq2 = np.array(cq1), np.array(cq2)
    return np.array(cy), unique, cq1, cq2, np.c_[1-cq1, cq1*(1-cq2), cq1*cq2]


def fit_platt(q: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(q, 1e-6, 1-1e-6) / np.clip(1-q, 1e-6, 1))

    def objective(v):
        eta = v[0] + v[1] * x
        loss = np.mean(np.logaddexp(0, eta) - y * eta) + 1e-4 * (v[1] - 1) ** 2
        residual = expit(eta) - y
        grad = np.array([residual.mean(), np.mean(residual * x) + 2e-4 * (v[1] - 1)])
        return loss, grad

    result = minimize(objective, np.array([0.0, 1.0]), method="L-BFGS-B", jac=True)
    return float(result.x[0]), float(result.x[1])


def fit_calibration(y, groups, q1, q2, mode):
    if mode == "none":
        return ((0.0, 1.0), (0.0, 1.0))
    cy, _, cq1, cq2, _ = aggregate_case(y, groups, q1, q2)
    first = fit_platt(cq1, (cy >= 1).astype(float))
    second = fit_platt(cq2[cy >= 1], (cy[cy >= 1] >= 2).astype(float))
    return first, second


def apply_calibration(q, parameters):
    a, b = parameters
    logit = np.log(np.clip(q, 1e-6, 1-1e-6) / np.clip(1-q, 1e-6, 1))
    return expit(a + b * logit)


def classify(q1, q2, thresholds):
    t1, t2 = thresholds
    return np.where(q1 < t1, 0, np.where(q2 < t2, 1, 2))


def score_tuple(y, pred):
    macro = f1_score(y, pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y, pred, weights="quadratic")
    maoe = np.mean(np.abs(y-pred))
    return macro, 0.0 if np.isnan(qwk) else qwk, -maoe


def choose_thresholds(y, groups, q1, q2):
    cy, _, cq1, cq2, _ = aggregate_case(y, groups, q1, q2)
    best = None
    quantiles = np.linspace(0.02, 0.98, 21)
    candidates1 = np.unique(np.r_[0.01, np.quantile(cq1, quantiles), 0.5, 0.99])
    candidates2 = np.unique(np.r_[0.01, np.quantile(cq2, quantiles), 0.5, 0.99])
    for t1 in candidates1:
        for t2 in candidates2:
            pred = classify(cq1, cq2, (t1, t2))
            candidate = (*score_tuple(cy, pred), -abs(t1-.5)-abs(t2-.5), float(t1), float(t2))
            if best is None or candidate > best:
                best = candidate
    return best[-2], best[-1]


def fit_predict(x_train, y_train, w_train, x_valid, ridge, fusion, class_weight, effective_beta, ordinal_smoothing):
    transform = Transform().fit(x_train, FEATURE_NAMES)
    train_weight = w_train * class_weight_vector(y_train, class_weight, effective_beta)
    model = FusedConditional(ridge, fusion, ordinal_smoothing).fit(transform.apply(x_train), y_train, train_weight)
    return model, transform, model.predict(transform.apply(x_valid))


def tune_outer(x, y, w, groups, train_idx, ridge_grid, fusion_grid, class_weight, effective_beta, calibration_mode, ordinal_smoothing):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    best = None
    for ridge in ridge_grid:
        for fusion in fusion_grid:
            q1 = np.zeros(len(train_idx)); q2 = np.zeros(len(train_idx))
            local_y, local_g = y[train_idx], groups[train_idx]
            for inner_train, inner_valid in inner.split(x[train_idx], local_y, local_g):
                _, _, pred = fit_predict(
                    x[train_idx][inner_train], local_y[inner_train], w[train_idx][inner_train],
                    x[train_idx][inner_valid], ridge, fusion, class_weight, effective_beta, ordinal_smoothing,
                )
                q1[inner_valid], q2[inner_valid] = pred[0], pred[1]
            calibration = fit_calibration(local_y, local_g, q1, q2, calibration_mode)
            q1 = apply_calibration(q1, calibration[0])
            q2 = apply_calibration(q2, calibration[1])
            thresholds = choose_thresholds(local_y, local_g, q1, q2)
            cy, _, cq1, cq2, _ = aggregate_case(local_y, local_g, q1, q2)
            metric = score_tuple(cy, classify(cq1, cq2, thresholds))
            candidate = (*metric, -ridge-fusion, ridge, fusion, thresholds, calibration)
            if best is None or candidate > best:
                best = candidate
    return best[-4], best[-3], best[-2], best[-1]


def metrics(y, probs, pred):
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    expected = probs @ np.arange(3)
    rho = 0.0 if np.ptp(expected) < 1e-12 else spearmanr(y, expected).statistic
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "maoe": float(np.mean(np.abs(y-pred))),
        "class_recall": {str(i): recalls[i] for i in range(3)},
        "predicted_class_counts": {str(k): v for k, v in Counter(pred).items()},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "auc_r_ge_1": roc_auc_score(y >= 1, probs[:, 1] + probs[:, 2]),
        "auc_r_ge_2": roc_auc_score(y >= 2, probs[:, 2]),
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
        "brier_multiclass": float(np.mean(np.sum((probs-np.eye(3)[y])**2, axis=1))),
        "spearman_expected_score": 0.0 if np.isnan(rho) else float(rho),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=RIDGE_GRID)
    parser.add_argument("--fusion-grid", nargs="+", type=float, default=FUSION_GRID)
    parser.add_argument("--experiment", default="fused_conditional")
    parser.add_argument("--realization-weighting", choices=("inverse_case", "equal"), default="inverse_case")
    parser.add_argument("--partial-weight", type=float, default=0.5)
    parser.add_argument("--class-weight", choices=("natural", "sqrt_inverse", "effective"), default="natural")
    parser.add_argument("--effective-beta", type=float, default=0.99)
    parser.add_argument("--probability-calibration", choices=("none", "platt"), default="none")
    parser.add_argument("--ordinal-smoothing", type=float, choices=(0.0, 0.05, 0.1), default=0.0)
    args = parser.parse_args()
    feature_rows = [json.loads(line) for line in args.features.open(encoding="utf-8")]
    supervision = {row["instance_id"]: row["supervision"] for row in map(json.loads, args.supervision.open(encoding="utf-8"))}
    rows = [{**row, "supervision": supervision[row["instance_id"]]} for row in feature_rows]
    global FEATURE_NAMES
    candidate_names = sorted(rows[0]["features"])
    FEATURE_NAMES = [name for name in candidate_names if len({row["features"][name] for row in rows}) > 1]
    x = np.array([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([row["supervision"]["resource_bucket"] for row in rows], dtype=int)
    groups = np.array([row["case_id"] for row in rows])
    quality_weight = np.array([
        args.partial_weight if row["supervision"]["quality"] == "partial_label_policy_conflict" else 1.0
        for row in rows
    ])
    realization_weight = np.array([
        row["supervision"]["inverse_realization_weight"] if args.realization_weighting == "inverse_case" else 1.0
        for row in rows
    ])
    w = quality_weight * realization_weight

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_q1, oof_q2, oof_probs = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros((len(rows), 3))
    raw_oof_q1, raw_oof_q2 = np.zeros(len(rows)), np.zeros(len(rows))
    fold_records, fold_models, case_explanations = [], [], {}
    for fold, (train_idx, valid_idx) in enumerate(outer.split(x, y, groups), 1):
        ridge, fusion, thresholds, calibration = tune_outer(
            x, y, w, groups, train_idx, args.ridge_grid, args.fusion_grid,
            args.class_weight, args.effective_beta, args.probability_calibration,
            args.ordinal_smoothing,
        )
        model, transform, raw_pred = fit_predict(
            x[train_idx], y[train_idx], w[train_idx], x[valid_idx], ridge, fusion,
            args.class_weight, args.effective_beta, args.ordinal_smoothing,
        )
        raw_oof_q1[valid_idx], raw_oof_q2[valid_idx] = raw_pred[0], raw_pred[1]
        calibrated_q1 = apply_calibration(raw_pred[0], calibration[0])
        calibrated_q2 = apply_calibration(raw_pred[1], calibration[1])
        pred = (calibrated_q1, calibrated_q2, np.c_[1-calibrated_q1, calibrated_q1*(1-calibrated_q2), calibrated_q1*calibrated_q2])
        oof_q1[valid_idx], oof_q2[valid_idx], oof_probs[valid_idx] = pred
        cy, _, cq1, cq2, cp = aggregate_case(y[valid_idx], groups[valid_idx], pred[0], pred[1])
        fold_metric = metrics(cy, cp, classify(cq1, cq2, thresholds))
        fold_records.append({"fold": fold, "ridge": ridge, "fusion": fusion, "thresholds": thresholds, "calibration": calibration, "case_count": len(cy), "metrics": fold_metric})
        fold_models.append((model, transform, valid_idx, thresholds, calibration))
        transformed = transform.apply(x[valid_idx])
        for case_id in np.unique(groups[valid_idx]):
            local = np.flatnonzero(groups[valid_idx] == case_id)
            mean_x = transformed[local].mean(0)
            top1 = np.argsort(np.abs(mean_x * model.coef1))[-8:][::-1]
            top2 = np.argsort(np.abs(mean_x * model.coef2))[-8:][::-1]
            evidence = {}
            source_rows = [rows[valid_idx[j]] for j in local]
            for feature_idx in set(top1) | set(top2):
                name = FEATURE_NAMES[feature_idx]
                evidence[name] = sorted({eid for row in source_rows for eid in row["feature_evidence_map"].get(name, [])})
            case_explanations[case_id] = {
                "boundary_r_ge_1": [{"feature": FEATURE_NAMES[j], "contribution": mean_x[j] * model.coef1[j], "evidence_ids": evidence[FEATURE_NAMES[j]]} for j in top1],
                "boundary_r_ge_2_given_ge_1": [{"feature": FEATURE_NAMES[j], "contribution": mean_x[j] * model.coef2[j], "evidence_ids": evidence[FEATURE_NAMES[j]]} for j in top2],
            }
        print(f"fold={fold} ridge={ridge} fusion={fusion} thresholds={thresholds} macro_f1={fold_metric['macro_f1']:.3f}")

    case_y, case_ids, case_q1, case_q2, case_probs = aggregate_case(y, groups, oof_q1, oof_q2)
    case_pred = np.zeros_like(case_y)
    case_fold = {}
    for record, (_, _, valid_idx, thresholds, _) in zip(fold_records, fold_models):
        valid_cases = set(groups[valid_idx])
        for j, case_id in enumerate(case_ids):
            if case_id in valid_cases:
                case_pred[j] = classify(np.array([case_q1[j]]), np.array([case_q2[j]]), thresholds)[0]
                case_fold[case_id] = record["fold"]

    primary = metrics(case_y, case_probs, case_pred)
    _, _, _, _, raw_case_probs = aggregate_case(y, groups, raw_oof_q1, raw_oof_q2)
    uncalibrated = metrics(case_y, raw_case_probs, np.argmax(raw_case_probs, axis=1))
    raw_pred = np.argmax(case_probs, axis=1)
    raw = metrics(case_y, case_probs, raw_pred)
    majority = np.full_like(case_y, Counter(case_y).most_common(1)[0][0])
    majority_metrics = metrics(case_y, np.tile(np.bincount(case_y, minlength=3)/len(case_y), (len(case_y), 1)), majority)

    chosen_pairs = Counter((r, f) for r, f, _ in [(x["ridge"], x["fusion"], x["thresholds"]) for x in fold_records])
    final_ridge, final_fusion = chosen_pairs.most_common(1)[0][0]
    final_thresholds = tuple(np.median([x["thresholds"] for x in fold_records], axis=0))
    final_calibration = tuple(tuple(np.median([x["calibration"][head] for x in fold_records], axis=0)) for head in range(2))
    final_transform = Transform().fit(x, FEATURE_NAMES)
    final_weight = w * class_weight_vector(y, args.class_weight, args.effective_beta)
    final_model = FusedConditional(final_ridge, final_fusion, args.ordinal_smoothing).fit(final_transform.apply(x), y, final_weight)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "04_oof_case_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, case_id in enumerate(case_ids):
            f.write(json.dumps({
                "case_id": case_id, "fold": case_fold[case_id], "resource_bucket": int(case_y[i]),
                "q_r_ge_1": case_q1[i], "q_r_ge_2_given_ge_1": case_q2[i],
                "probabilities": case_probs[i].tolist(), "raw_prediction": int(raw_pred[i]),
                "threshold_prediction": int(case_pred[i]), "explanation": case_explanations[case_id],
            }) + "\n")

    model_artifact = {
        "schema_version": SCHEMA_VERSION, "feature_names": FEATURE_NAMES,
        "ridge": final_ridge, "fusion": final_fusion, "thresholds": final_thresholds,
        "probability_calibration": args.probability_calibration, "calibration_parameters": final_calibration,
        "ordinal_smoothing": args.ordinal_smoothing,
        "intercepts": final_model.intercepts.tolist(), "boundary_1_coefficients": final_model.coef1.tolist(),
        "boundary_2_coefficients": final_model.coef2.tolist(), "transform": {
            "log_mask": final_transform.log_mask.tolist(), "scale_mask": final_transform.scale_mask.tolist(),
            "mean": final_transform.mean.tolist(), "std": final_transform.std.tolist(),
        },
    }
    (args.output_dir / "04_model.json").write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")
    r1_nonzero_folds = int(sum(x["metrics"]["class_recall"]["1"] > 0 for x in fold_records))
    report = {
        "schema_version": SCHEMA_VERSION, "experiment": args.experiment,
        "training_configuration": {
            "realization_weighting": args.realization_weighting,
            "partial_weight": args.partial_weight,
            "class_weight": args.class_weight,
            "effective_beta": args.effective_beta if args.class_weight == "effective" else None,
            "probability_calibration": args.probability_calibration,
            "ordinal_smoothing": args.ordinal_smoothing,
        },
        "realization_count": len(rows), "case_count": len(case_ids),
        "all_realizations_used": len(rows) == 732, "grouped_case_folds": True,
        "input_sha256": {"features": sha256(args.features), "supervision": sha256(args.supervision)},
        "primary_threshold_metrics": primary, "raw_argmax_metrics": raw,
        "uncalibrated_argmax_metrics": uncalibrated,
        "majority_baseline_metrics": majority_metrics, "folds": fold_records,
        "r1_nonzero_recall_fold_count": r1_nonzero_folds,
        "active_feature_count": len(FEATURE_NAMES),
        "zero_variance_features_excluded": sorted(set(candidate_names) - set(FEATURE_NAMES)),
        "final_hyperparameters": {"ridge": final_ridge, "fusion": final_fusion, "thresholds": final_thresholds},
        "downstream_model_ready": bool(
            len(primary["predicted_class_counts"]) == 3 and primary["class_recall"]["0"] > 0
            and primary["class_recall"]["1"] > 0 and primary["qwk"] > 0
            and primary["macro_f1"] > majority_metrics["macro_f1"]
            and primary["balanced_accuracy"] > 1/3
            and r1_nonzero_folds >= 4
            and primary["log_loss"] <= majority_metrics["log_loss"]
        ),
    }
    report_path = args.output_dir / "04_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": args.experiment,
        "input_sha256": report["input_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": {
            "model": sha256(args.output_dir / "04_model.json"),
            "oof_case_predictions": sha256(args.output_dir / "04_oof_case_predictions.jsonl"),
            "training_report": sha256(report_path),
        },
    }
    (args.output_dir / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
