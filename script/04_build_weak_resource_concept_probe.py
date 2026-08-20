#!/usr/bin/env python3
"""Compile prediction-safe features into an audit-only weak resource concept bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESOURCE_CONCEPTS = {
    "laboratory": (
        "concept__fever__active", "concept__systemic_infection_symptom__active",
        "concept__diabetes_context__active", "concept__nausea__active",
        "claim__vomiting__active", "claim__active_bleeding__active",
    ),
    "basic_diagnostic": (
        "claim__chest_pain__active", "claim__chest_pressure__active",
        "claim__dyspnea__active", "concept__cough__active",
        "concept__musculoskeletal_pain__active", "concept__swelling__active",
        "concept__back_pain__active", "claim__fall_or_trauma__active",
        "claim__laceration_or_wound__active",
    ),
    "advanced_imaging": (
        "claim__abdominal_pain__active", "claim__altered_mental_status__active",
        "claim__headache__active", "claim__fall_or_trauma__active",
        "policy__focal_neurologic_risk__active", "policy__high_risk_headache__active",
        "policy__pregnancy_high_risk__active",
    ),
    "iv_fluids": (
        "claim__vomiting__active", "concept__nausea__active",
        "claim__active_bleeding__active", "concept__perfusion_appearance__active",
        "concept__systemic_infection_symptom__active",
    ),
    "parenteral_or_nebulized_medication": (
        "policy__severe_pain_or_distress__active", "policy__respiratory_high_risk__active",
        "claim__dyspnea__active", "claim__vomiting__active", "concept__nausea__active",
    ),
    "specialty_consult": (
        "policy__organ_or_limb_threat__active", "policy__pregnancy_high_risk__active",
        "policy__focal_neurologic_risk__active", "policy__acute_coronary_risk__active",
        "policy__active_self_or_other_harm__active", "policy__high_risk_trauma__active",
        "policy__recent_seizure_high_risk__active",
    ),
    "simple_procedure": (
        "claim__laceration_or_wound__active", "claim__active_bleeding__active",
        "concept__inflammatory_skin_finding__active", "concept__swelling__active",
    ),
    "complex_procedure": (
        "policy__organ_or_limb_threat__active", "policy__critical_airway_or_gas_exchange__active",
        "policy__critical_perfusion_or_hemorrhage__active", "policy__high_risk_trauma__active",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.features.open(encoding="utf-8")]
    output = []
    prevalence = {name: 0 for name in RESOURCE_CONCEPTS}
    for row in rows:
        features, evidence_map = {}, {}
        for concept, sources in RESOURCE_CONCEPTS.items():
            active_sources = [name for name in sources if row["features"].get(name, 0) > 0]
            feature_name = f"resource_concept__{concept}__indicated"
            features[feature_name] = float(bool(active_sources))
            prevalence[concept] += bool(active_sources)
            evidence_map[feature_name] = sorted({
                evidence_id for name in active_sources
                for evidence_id in row["feature_evidence_map"].get(name, [])
            })
        features["resource_concept__distinct_count"] = float(sum(features.values()))
        features["resource_concept__resource_units"] = float(
            sum(value * (2 if "complex_procedure" in name else 1) for name, value in features.items()
            if name.endswith("__indicated"))
        )
        output.append({
            "schema_version": "04_weak_resource_concept_probe_v1.0",
            "case_id": row["case_id"], "instance_id": row["instance_id"],
            "features": features, "feature_evidence_map": evidence_map,
            "audit_only_weak_concepts": True,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "04_weak_resource_concept_features.jsonl").open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")
    report = {
        "schema_version": "04_weak_resource_concept_probe_v1.0",
        "realization_count": len(rows), "concept_count": len(RESOURCE_CONCEPTS),
        "concept_prevalence": prevalence,
        "actual_resource_labels_available": False,
        "role": "feasibility_probe_not_validated_resource_concept_model",
    }
    (args.output_dir / "04_weak_resource_concept_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
