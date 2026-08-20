#!/usr/bin/env python3
"""Nonlinear ceiling probe for the frozen stage-04 feature plane."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier


PARAMETERS = (
    {"max_depth": 2, "min_child_weight": 5, "learning_rate": 0.05},
    {"max_depth": 3, "min_child_weight": 5, "learning_rate": 0.05},
    {"max_depth": 2, "min_child_weight": 15, "learning_rate": 0.08},
    {"max_depth": 3, "min_child_weight": 15, "learning_rate": 0.08},
)


def aggregate(y, groups, probabilities):
    unique = np.unique(groups)
    labels, probs = [], []
    for group in unique:
        mask = groups == group
        labels.append(y[mask][0]); probs.append(probabilities[mask].mean(0))
    probs = np.array(probs)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.array(labels), unique, probs


def sqrt_weights(y):
    counts = np.bincount(y, minlength=3).astype(float)
    by_class = 1.0 / np.sqrt(counts)
    by_class /= np.average(by_class, weights=counts)
    return by_class[y]


def model(params):
    return XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=300,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
        reg_alpha=0.1, n_jobs=4, random_state=2026, eval_metric="mlogloss",
        **params,
    )


def metrics(y, probs):
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(1)
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "maoe": float(np.mean(np.abs(y-pred))),
        "class_recall": {str(i): recalls[i] for i in range(3)},
        "predicted_class_counts": {str(k): v for k, v in Counter(pred).items()},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "auc_r_ge_1": roc_auc_score(y >= 1, probs[:, 1:].sum(1)),
        "auc_r_ge_2": roc_auc_score(y >= 2, probs[:, 2]),
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("grouped_case", "random_realization"), default="grouped_case")
    parser.add_argument("--cohort", choices=("all", "supported_only"), default="all")
    args = parser.parse_args()

    feature_rows = [json.loads(x) for x in args.features.open(encoding="utf-8")]
    supervision = {x["instance_id"]: x["supervision"] for x in map(json.loads, args.supervision.open(encoding="utf-8"))}
    if args.cohort == "supported_only":
        feature_rows = [row for row in feature_rows if supervision[row["instance_id"]]["quality"] == "policy_supported_proxy"]
    names = sorted(feature_rows[0]["features"])
    names = [name for name in names if len({row["features"][name] for row in feature_rows}) > 1]
    x = np.array([[row["features"][name] for name in names] for row in feature_rows])
    y = np.array([supervision[row["instance_id"]]["resource_bucket"] for row in feature_rows])
    groups = np.array([row["case_id"] for row in feature_rows])

    grouped = args.split_mode == "grouped_case"
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026) if grouped else StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros((len(y), 3)); fold_rows = []
    outer_splits = outer.split(x, y, groups) if grouped else outer.split(x, y)
    for fold, (train, valid) in enumerate(outer_splits, 1):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) if grouped else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best = None
        for params in PARAMETERS:
            inner_oof = np.zeros((len(train), 3))
            inner_splits = inner.split(x[train], y[train], groups[train]) if grouped else inner.split(x[train], y[train])
            for inner_train, inner_valid in inner_splits:
                estimator = model(params)
                estimator.fit(x[train][inner_train], y[train][inner_train], sample_weight=sqrt_weights(y[train][inner_train]))
                inner_oof[inner_valid] = estimator.predict_proba(x[train][inner_valid])
            if grouped:
                iy, _, ip = aggregate(y[train], groups[train], inner_oof)
            else:
                iy, ip = y[train], inner_oof
            score = f1_score(iy, ip.argmax(1), average="macro", zero_division=0)
            candidate = (score, -params["max_depth"], -params["min_child_weight"], params)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        estimator = model(best[-1])
        estimator.fit(x[train], y[train], sample_weight=sqrt_weights(y[train]))
        oof[valid] = estimator.predict_proba(x[valid])
        if grouped:
            vy, _, vp = aggregate(y[valid], groups[valid], oof[valid])
        else:
            vy, vp = y[valid], oof[valid]
        fold_rows.append({"fold": fold, "parameters": best[-1], "metrics": metrics(vy, vp)})
        print(f"fold={fold} parameters={best[-1]} macro_f1={fold_rows[-1]['metrics']['macro_f1']:.3f}")

    if grouped:
        case_y, case_ids, case_probs = aggregate(y, groups, oof)
    else:
        case_y, case_ids, case_probs = y, np.array([row["instance_id"] for row in feature_rows]), oof
    report = {
        "schema_version": "04_nonlinear_ceiling_probe_v1.0",
        "model": "XGBoost",
        "role": "nonlinear_upper_bound_probe_not_primary_model",
        "realization_count": len(y), "case_count": len(np.unique(groups)),
        "realization_weighting": "equal", "class_weight": "sqrt_inverse",
        "cohort": args.cohort,
        "split_mode": args.split_mode,
        "evaluation_unit": "case" if grouped else "realization",
        "same_case_cross_fold_leakage_possible": not grouped,
        "grouped_nested_cv": grouped, "metrics": metrics(case_y, case_probs), "folds": fold_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "04_nonlinear_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "04_nonlinear_probe_oof.jsonl").open("w", encoding="utf-8") as f:
        for case_id, label, probs in zip(case_ids, case_y, case_probs):
            f.write(json.dumps({"case_id": case_id, "resource_bucket": int(label), "probabilities": probs.tolist(), "prediction": int(np.argmax(probs))}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
