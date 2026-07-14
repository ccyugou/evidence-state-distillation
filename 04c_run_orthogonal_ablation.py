"""04c Stage A: orthogonal feature ablation for resource ordinal learning.

This diagnostic keeps the frozen 04a cohort, case folds, partial weight, and
inner selection protocol fixed. It tests whether supervision-quality features
act as shortcuts for the clinical resource target. The quality-only model is a
probe, not a deployable clinical decision model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


BASE_PATH = Path(__file__).with_name("04c_run_resource_ordinal_experiments.py")
SPEC = importlib.util.spec_from_file_location("experiments04c", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot import experiment module: {BASE_PATH}")
EXPERIMENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENTS)


SCHEMA_VERSION = "04c_orthogonal_ablation_v1"
INPUT_SCHEMA_VERSION = "04_resource_ordinal_feasibility_v1.2"
DEFAULT_INPUT_DIR = Path("outputs/04_resource_ordinal_feasibility_v1_2_full688")
DEFAULT_OUTPUT_DIR = Path("outputs/04c_orthogonal_ablation_v1")
QUALITY_FEATURES = frozenset(
    {
        "claim_support_only_count",
        "claim_uncertain_count",
        *{
            name
            for name in (
                "claim_field_active_bleeding_uncertain",
                "claim_field_altered_mental_status_uncertain",
                "claim_field_chest_pain_or_acs_concern_uncertain",
                "claim_field_dizziness_uncertain",
                "claim_field_dyspnea_uncertain",
                "claim_field_fever_or_infection_uncertain",
                "claim_field_gi_bleeding_melena_uncertain",
                "claim_field_pain_uncertain",
                "claim_field_pregnancy_status_or_context_uncertain",
                "claim_field_respiratory_distress_uncertain",
                "claim_field_severe_pain_uncertain",
                "claim_field_stroke_signs_uncertain",
                "claim_field_suicidal_ideation_uncertain",
                "claim_field_syncope_uncertain",
                "claim_field_toxic_ingestion_or_overdose_uncertain",
            )
        },
    }
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_feature_sets(input_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    audit = EXPERIMENTS.RIDGE.read_json(input_dir / "resource_feasibility_audit.json")
    dictionary = EXPERIMENTS.RIDGE.read_json(input_dir / "resource_feature_dictionary.json")
    if audit.get("schema_version") != INPUT_SCHEMA_VERSION or audit.get("release_gate_passed") is not True:
        raise RuntimeError("Frozen 04a audit is not a releasable v1.2 artifact")
    feature_names = list(dictionary["model_features"])
    missing = sorted(QUALITY_FEATURES - set(feature_names))
    if missing or len(QUALITY_FEATURES) != 17:
        raise RuntimeError(f"Quality feature contract mismatch: missing={missing}")
    return audit, feature_names, dictionary


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

    audit, full_features, dictionary = load_feature_sets(args.input_dir)
    (
        rows,
        label_by_case,
        fold_by_case,
        weight_by_instance,
        availability_by_fold,
        _,
        _,
    ) = EXPERIMENTS.load_inputs(args.input_dir)
    rows = sorted(rows, key=lambda row: row["instance_id"])
    base_weights = np.asarray([weight_by_instance[row["instance_id"]] for row in rows], dtype=float)
    targets = np.asarray([EXPERIMENTS.target_of(label_by_case, row["case_id"]) for row in rows], dtype=int)
    lambda_grid = [float(value) for value in args.lambda_grid.split(",") if value.strip()]

    clinical_features = [name for name in full_features if name not in QUALITY_FEATURES]
    quality_features = [name for name in full_features if name in QUALITY_FEATURES]
    if len(full_features) != 40 or len(clinical_features) != 23 or len(quality_features) != 17:
        raise RuntimeError("Frozen feature partition must be 40=23+17")
    if set(clinical_features) & set(quality_features):
        raise RuntimeError("Clinical and quality feature sets overlap")
    if set(clinical_features) | set(quality_features) != set(full_features):
        raise RuntimeError("Clinical and quality feature sets do not cover FULL40")
    feature_sets = {
        "FULL40": full_features,
        "ORTHO23": clinical_features,
        "QUALITY17_ONLY": quality_features,
    }
    variants = [
        ("FULL40_B0_natural_argmax_p05", "FULL40", False),
        ("FULL40_T_nested_threshold_p05", "FULL40", True),
        ("ORTHO23_B0_natural_argmax_p05", "ORTHO23", False),
        ("ORTHO23_T_nested_threshold_p05", "ORTHO23", True),
        ("QUALITY17_ONLY_B0_natural_argmax_p05", "QUALITY17_ONLY", False),
    ]

    summaries: list[dict[str, Any]] = []
    case_outputs: dict[str, list[dict[str, Any]]] = {}
    fold_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, feature_set, threshold_mode in variants:
        feature_names = feature_sets[feature_set]
        oof, case_oof, folds, metrics, thresholds = EXPERIMENTS.run_variant(
            name,
            rows,
            label_by_case,
            fold_by_case,
            availability_by_fold,
            feature_names,
            base_weights,
            targets,
            0.5,
            "natural",
            threshold_mode,
            lambda_grid,
        )
        case_outputs[name] = case_oof
        fold_outputs[name] = folds
        write_jsonl(args.output_dir / f"oof_{name}.jsonl", oof)
        write_jsonl(args.output_dir / f"case_oof_{name}.jsonl", case_oof)
        write_jsonl(args.output_dir / f"folds_{name}.jsonl", folds)
        if thresholds:
            write_jsonl(args.output_dir / f"thresholds_{name}.jsonl", thresholds)
        summaries.append(
            {
                "variant": name,
                "feature_set": feature_set,
                "feature_count": len(feature_names),
                "removed_quality_features": sorted(QUALITY_FEATURES) if feature_set == "ORTHO23" else [],
                "partial_weight": 0.5,
                "class_weight_scheme": "natural",
                "threshold_mode": threshold_mode,
                "common_cohort": metrics,
            }
        )

    full_b0 = "FULL40_B0_natural_argmax_p05"
    full_t = "FULL40_T_nested_threshold_p05"
    ortho_b0 = "ORTHO23_B0_natural_argmax_p05"
    ortho_t = "ORTHO23_T_nested_threshold_p05"
    paired = {
        "FULL40_T_vs_FULL40_B0": EXPERIMENTS.paired_bootstrap(case_outputs[full_b0], case_outputs[full_t], full_b0, full_t),
        "ORTHO23_B0_vs_FULL40_B0": EXPERIMENTS.paired_bootstrap(case_outputs[full_b0], case_outputs[ortho_b0], full_b0, ortho_b0),
        "ORTHO23_T_vs_FULL40_T": EXPERIMENTS.paired_bootstrap(case_outputs[full_t], case_outputs[ortho_t], full_t, ortho_t),
    }
    write_json(args.output_dir / "paired_bootstrap_comparisons.json", paired)
    write_json(
        args.output_dir / "ablation_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "input_audit_hash": EXPERIMENTS.RIDGE.sha256_file(args.input_dir / "resource_feasibility_audit.json"),
            "feature_dictionary_hash": EXPERIMENTS.RIDGE.sha256_file(args.input_dir / "resource_feature_dictionary.json"),
            "outer_folds": sorted(set(fold_by_case.values())),
            "case_count": len(fold_by_case),
            "instance_count": len(rows),
            "quality_feature_count": len(QUALITY_FEATURES),
            "quality_features": sorted(QUALITY_FEATURES),
            "clinical_feature_count": len(clinical_features),
            "note": "QUALITY17_ONLY is a diagnostic proxy probe, not a clinical decision model.",
        },
    )
    write_json(
        args.output_dir / "experiment_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "input_audit_hash": EXPERIMENTS.RIDGE.sha256_file(args.input_dir / "resource_feasibility_audit.json"),
            "variants": summaries,
            "paired_comparisons": list(paired),
            "note": "All feature-set comparisons use the same frozen case folds, natural class weighting, P05 partial weight, and outer-train-only threshold selection.",
        },
    )
    print(f"04c orthogonal ablation complete: {len(variants)} variants, output={args.output_dir}")


if __name__ == "__main__":
    main()
