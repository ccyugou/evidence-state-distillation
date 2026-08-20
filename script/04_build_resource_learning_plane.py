#!/usr/bin/env python3
"""Compile 02/03 artifacts into the stage-04 resource-learning plane."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from esi_policy_manifest_v5 import MANIFEST_VERSION, POLICIES


SCHEMA_VERSION = "04_resource_learning_plane_v2.2_nonredundant_concepts"
ACTIVE_STATES = {"present", "present_persistent", "present_worsening", "present_with_uncertainty"}
UNCERTAIN_STATES = {"uncertain", "present_with_uncertainty", "earlier_uncertain"}
SELECTED_ATOMS = (
    "dyspnea", "dyspnea_at_rest", "chest_pain", "chest_pressure", "active_bleeding",
    "vaginal_bleeding", "abdominal_pain", "pelvic_pain", "focal_weakness",
    "focal_numbness", "altered_mental_status", "functional_limitation", "vomiting",
    "pregnancy_status", "fall_or_trauma", "laceration_or_wound", "headache",
    "pain_mention", "anticoagulant_use", "syncope",
)
CONCEPT_PARENTS = (
    "swelling", "musculoskeletal_pain", "dizziness", "nausea", "fever",
    "diabetes_context", "hypertension_context", "back_pain", "cough",
    "inflammatory_skin_finding", "fatigue", "upper_airway_symptom",
    "perfusion_appearance", "systemic_infection_symptom",
)
MODIFIERS = ("severe_language", "functional_limitation", "persistent", "worsening", "partial_resolution")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["instance_id"]: row for row in map(json.loads, f)}


def add_feature(features: dict, evidence_map: dict, name: str, value: float, evidence_ids=()) -> None:
    features[name] = float(value)
    if evidence_ids:
        evidence_map[name] = sorted(set(evidence_ids))


def raw_labels(transcripts: Path) -> dict[str, dict]:
    labels = {}
    for path in transcripts.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        instance_id = f"{row['case_id']}__{row['run_uuid']}"
        labels[instance_id] = {
            "case_id": str(row["case_id"]),
            "acuity": int(row["ground_truth"]["acuity"]),
        }
    return labels


def compile_row(row02: dict, row03: dict, label: dict, realization_count: int) -> dict:
    features, evidence_map = {}, {}
    trajectories = [t for t in row02["atom_trajectories"] if t["subject"] == "patient"]
    atom_by_id = defaultdict(list)
    for trajectory in trajectories:
        atom_by_id[trajectory["clinical_atom_id"]].append(trajectory)

    active_atoms = {
        atom_id for atom_id, items in atom_by_id.items()
        if any(item["effective_state"] in ACTIVE_STATES for item in items)
    }
    uncertain_atoms = {
        atom_id for atom_id, items in atom_by_id.items()
        if any(item["effective_state"] in UNCERTAIN_STATES for item in items)
    }

    active_policy_count = 0
    for field, policy in POLICIES.items():
        active = sorted(active_atoms & policy["anchors"])
        uncertain = sorted(uncertain_atoms & policy["anchors"])
        active_policy_count += bool(active)
        active_ids = [eid for atom in active for t in atom_by_id[atom] for eid in t["evidence_card_ids"]]
        uncertain_ids = [eid for atom in uncertain for t in atom_by_id[atom] for eid in t["evidence_card_ids"]]
        add_feature(features, evidence_map, f"policy__{field}__active", bool(active), active_ids)
        add_feature(features, evidence_map, f"policy__{field}__uncertain", bool(uncertain), uncertain_ids)

    for atom_id in SELECTED_ATOMS:
        items = atom_by_id.get(atom_id, [])
        active = [t for t in items if t["effective_state"] in ACTIVE_STATES]
        ids = [eid for t in active for eid in t["evidence_card_ids"]]
        add_feature(features, evidence_map, f"claim__{atom_id}__active", bool(active), ids)

    for parent_id in CONCEPT_PARENTS:
        active = [t for t in trajectories if t["concept_parent_id"] == parent_id and t["effective_state"] in ACTIVE_STATES]
        ids = [eid for t in active for eid in t["evidence_card_ids"]]
        add_feature(features, evidence_map, f"concept__{parent_id}__active", bool(active), ids)

    transitions = [x for t in trajectories for x in t["transitions"]]
    reason_counts = Counter(x["transition_reason"] for x in transitions)
    state_counts = Counter(t["effective_state"] for t in trajectories)
    trajectory_features = {
        "trajectory__worsening_count": reason_counts["worsening_update"],
        "trajectory__persistence_count": reason_counts["persistence_update"],
        "trajectory__resolution_count": reason_counts["current_resolution_update"],
        "trajectory__partial_resolution_count": reason_counts["partial_or_unverified_resolution"],
        "trajectory__recurrence_count": reason_counts["new_onset_or_recurrence"],
        "trajectory__scope_uncertainty_count": reason_counts["uncertainty_does_not_erase_active_state"],
        "trajectory__current_active_atom_count": len(active_atoms),
        "trajectory__current_uncertain_atom_count": len(uncertain_atoms),
        "trajectory__historical_atom_count": state_counts["historical"],
        "trajectory__multi_update_atom_count": sum(len(t["transitions"]) > 1 for t in trajectories),
    }
    for name, value in trajectory_features.items():
        add_feature(features, evidence_map, name, value)

    modifier_ids = defaultdict(set)
    patient_ids, nurse_ids, independent_ids = set(), set(), set()
    validated_counts = Counter()
    for candidate in row02["policy_candidates"]:
        validated_counts[candidate.get("validated_status")] += 1
        for evidence in candidate.get("evidence_bundle", []):
            evidence_id = evidence.get("evidence_id")
            if evidence.get("independent_evidence") and evidence_id:
                independent_ids.add(evidence_id)
            if evidence.get("source_subtype", "").startswith("patient") and evidence_id:
                patient_ids.add(evidence_id)
            if evidence.get("source_subtype") == "nurse_direct_observation" and evidence_id:
                nurse_ids.add(evidence_id)
            for modifier in evidence.get("semantic_modifiers", []):
                modifier_ids[modifier].add(evidence_id)
        for modifier in candidate.get("semantic_modifiers", []):
            modifier_ids[modifier].update(candidate.get("evidence_ids", []))

    for modifier in MODIFIERS:
        add_feature(features, evidence_map, f"modifier__{modifier}__count", len(modifier_ids[modifier]), modifier_ids[modifier])

    evidence_groups = {eid for t in trajectories for eid in t["evidence_card_ids"]}
    quality = {
        "quality__active_policy_count": active_policy_count,
        "quality__confirmed_symptom_count": validated_counts["confirmed_symptom_only"],
        "quality__uncertain_policy_count": validated_counts["uncertain_policy_review"],
        "quality__explicit_negative_count": len(row02["explicit_negatives"]),
        "quality__uncertain_evidence_count": len(row02["uncertain_evidence"]),
        "quality__independent_evidence_count": len(independent_ids),
        "quality__patient_endorsed_count": len(patient_ids),
        "quality__nurse_observed_count": len(nurse_ids),
        "quality__evidence_group_count": len(evidence_groups),
        "quality__single_source_only": len(independent_ids) <= 1,
    }
    for name, value in quality.items():
        add_feature(features, evidence_map, name, value)

    contexts = {x["field"]: x for x in row03["safety_context_flags"]}
    pain = contexts.get("pain_score")
    temperature = contexts.get("temperature")
    sbp = contexts.get("systolic_blood_pressure")
    add_feature(features, evidence_map, "vital__valid_pain_context", pain is not None, [pain["measurement_id"]] if pain else [])
    add_feature(features, evidence_map, "vital__pain_score_value", pain.get("value", 0) if pain else 0, [pain["measurement_id"]] if pain else [])
    add_feature(features, evidence_map, "vital__valid_temperature_context", temperature is not None, [temperature["measurement_id"]] if temperature else [])
    add_feature(features, evidence_map, "vital__temperature_value", temperature.get("value", 0) if temperature else 0, [temperature["measurement_id"]] if temperature else [])
    add_feature(features, evidence_map, "vital__valid_sbp_context", sbp is not None, [sbp["measurement_id"]] if sbp else [])
    add_feature(features, evidence_map, "vital__sbp_value", sbp.get("value", 0) if sbp else 0, [sbp["measurement_id"]] if sbp else [])
    add_feature(features, evidence_map, "vital__non_step_d_context_count", sum(x is not None for x in (pain, temperature, sbp)))

    def f(name: str) -> float:
        return features.get(name, 0.0)

    interactions = {
        "interaction__respiratory_x_worsening": f("policy__respiratory_high_risk__active") * f("trajectory__worsening_count"),
        "interaction__dyspnea_rest_x_worsening": f("claim__dyspnea_at_rest__active") * f("trajectory__worsening_count"),
        "interaction__severe_language_x_functional_limitation": f("modifier__severe_language__count") * f("claim__functional_limitation__active"),
        "interaction__bleeding_x_anticoagulant": f("claim__active_bleeding__active") * f("claim__anticoagulant_use__active"),
        "interaction__pregnancy_x_abdominal_pain": f("claim__pregnancy_status__active") * f("claim__abdominal_pain__active"),
        "interaction__pregnancy_x_vaginal_bleeding": f("claim__pregnancy_status__active") * f("claim__vaginal_bleeding__active"),
        "interaction__neurologic_x_functional_limitation": max(f("claim__focal_weakness__active"), f("claim__focal_numbness__active")) * f("claim__functional_limitation__active"),
        "interaction__multi_policy_x_worsening": (active_policy_count > 1) * f("trajectory__worsening_count"),
        "interaction__uncertain_x_single_source": (quality["quality__uncertain_policy_count"] > 0) * quality["quality__single_source_only"],
        "interaction__persistent_x_severe_language": f("trajectory__persistence_count") * f("modifier__severe_language__count"),
    }
    for name, value in interactions.items():
        add_feature(features, evidence_map, name, value)

    acuity = label["acuity"]
    resource_bucket = {5: 0, 4: 1, 3: 2}[acuity]
    policy_conflict = bool(row02["confirmed_step_a_signals"] or row02["confirmed_step_b_signals"])
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": row02["instance_id"],
        "case_id": row02["case_id"],
        "features": features,
        "feature_evidence_map": evidence_map,
        "policy_only": {
            "confirmed_step_a_signal_count": len(row02["confirmed_step_a_signals"]),
            "confirmed_step_b_signal_count": len(row02["confirmed_step_b_signals"]),
            "official_danger_zone_signal_present": row03["official_danger_zone_signal_present"],
        },
        "supervision": {
            "resource_bucket": resource_bucket,
            "source_acuity": acuity,
            "label_source": "esi_3_4_5_resource_proxy",
            "quality": "partial_label_policy_conflict" if policy_conflict else "policy_supported_proxy",
            "quality_weight": 0.5 if policy_conflict else 1.0,
            "inverse_realization_weight": 1.0 / realization_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--stage02", type=Path, required=True)
    parser.add_argument("--stage03", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = raw_labels(args.transcripts)
    rows02, rows03 = read_jsonl(args.stage02), read_jsonl(args.stage03)
    eligible_ids = sorted(iid for iid, x in labels.items() if x["acuity"] in {3, 4, 5})
    case_counts = Counter(labels[iid]["case_id"] for iid in eligible_ids)
    compiled = [compile_row(rows02[iid], rows03[iid], labels[iid], case_counts[labels[iid]["case_id"]]) for iid in eligible_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_output = args.output_dir / "04_prediction_safe_features.jsonl"
    supervision_output = args.output_dir / "04_resource_supervision.jsonl"
    with feature_output.open("w", encoding="utf-8") as f:
        for row in compiled:
            f.write(json.dumps({k: row[k] for k in ("schema_version", "instance_id", "case_id", "features", "feature_evidence_map", "policy_only")}, ensure_ascii=False, sort_keys=True) + "\n")
    with supervision_output.open("w", encoding="utf-8") as f:
        for row in compiled:
            f.write(json.dumps({"instance_id": row["instance_id"], "case_id": row["case_id"], "supervision": row["supervision"]}, ensure_ascii=False, sort_keys=True) + "\n")

    feature_names = sorted(compiled[0]["features"])
    forbidden = ("acuity", "triage", "ground_truth", "official_danger", "step_a", "step_b")
    forbidden_features = [name for name in feature_names if any(token in name.lower() for token in forbidden)]
    invalid_ids = {x["measurement_id"] for row in rows03.values() for x in row["quarantined_measurements"]}
    consumed_ids = {eid for row in compiled for ids in row["feature_evidence_map"].values() for eid in ids}
    audit = {
        "schema_version": SCHEMA_VERSION,
        "processed_realization_count": len(compiled),
        "independent_case_count": len(case_counts),
        "resource_bucket_realization_counts": Counter(row["supervision"]["resource_bucket"] for row in compiled),
        "resource_bucket_case_counts": Counter(next(row["supervision"]["resource_bucket"] for row in compiled if row["case_id"] == case_id) for case_id in case_counts),
        "supervision_quality_counts": Counter(row["supervision"]["quality"] for row in compiled),
        "realizations_per_case": dict(sorted(Counter(case_counts.values()).items())),
        "feature_count": len(feature_names),
        "feature_groups": Counter(name.split("__", 1)[0] for name in feature_names),
        "forbidden_model_feature_names": forbidden_features,
        "quarantined_measurement_count": len(invalid_ids),
        "invalid_measurement_feature_intersection_count": len(invalid_ids & consumed_ids),
        "input_sha256": {"stage02": sha256(args.stage02), "stage03": sha256(args.stage03)},
        "output_sha256": {"prediction_safe_features": sha256(feature_output), "resource_supervision": sha256(supervision_output)},
        "hard_gates": {
            "all_eligible_realizations_used": len(compiled) == 732,
            "expected_case_count": len(case_counts) == 341,
            "instance_alignment_complete": set(eligible_ids) <= rows02.keys() and set(eligible_ids) <= rows03.keys(),
            "forbidden_model_features_zero": not forbidden_features,
            "invalid_measurement_consumption_zero": not (invalid_ids & consumed_ids),
            "policy_only_not_in_features": all(not name.startswith("policy_only") for name in feature_names),
            "label_feature_plane_physically_separated": True,
        },
    }
    audit["release_gate_passed"] = all(audit["hard_gates"].values())
    (args.output_dir / "04_feature_schema.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "feature_names": feature_names}, indent=2), encoding="utf-8")
    (args.output_dir / "04_build_audit.json").write_text(json.dumps(audit, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=dict))


if __name__ == "__main__":
    main()
