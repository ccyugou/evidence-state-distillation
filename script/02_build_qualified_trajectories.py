#!/usr/bin/env python3
"""Stage 02: evidence-locked trajectories and ESI Step A/B qualification.

Dry-run mode performs no model calls. It preserves stage-01 clinical atoms,
reduces current episode state, and emits only policy candidates for later
bounded-model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clinical_concept_registry import CLINICAL_ATOMS, REGISTRY_VERSION
from esi_policy_manifest_v5 import MANIFEST_VERSION, POLICIES, POLICY_ATOMS


SCHEMA_VERSION = "02_evidence_locked_policy_trajectories_v1.1_state_time_guard"
EXPECTED_INPUT_SCHEMA = "01_dialogue_to_evidence_cards_v1.0"
CONTEXT_SOURCES = {
    "nurse_question",
    "nurse_process_or_instruction",
    "nurse_restatement_context",
    "nurse_context_statement",
    "nurse_subtype_ambiguous",
}
CURRENT_TIMES = {"current", "recent", "current_or_unspecified", "earlier_in_episode"}
ACTIVE_STATES = {"present", "present_persistent", "present_worsening", "present_improving", "present_with_uncertainty"}
PAIN_ATOMS = {
    "pain_present", "abdominal_pain", "flank_pain", "back_pain", "neck_pain",
    "pelvic_pain", "testicular_or_scrotal_pain", "limb_or_joint_pain",
    "eye_pain_or_visual_change", "headache",
}
NEAR_FALL_RE = re.compile(r"\b(?:almost|nearly)\s+(?:fall|fell)\b|\b(?:might|may|could)\s+fall\b|\bfeel(?:ing)?\s+like[^.?!;]{0,30}\bfall\b", re.I)
HEADACHE_MEDICATION_INDICATION_RE = re.compile(
    r"\b(?:took|taking|take|gave|given)\b[\s\S]{0,120}?\b(?:for|because of)\b[\s\S]{0,25}?\bheadache\b",
    re.I,
)
CURRENT_HEADACHE_RE = re.compile(
    r"\b(?:have|having|got|with)\s+(?:a\s+)?(?:bad|severe|mild|strong)?\s*headache\b|"
    r"\bheadache\s+(?:right\s+)?now\b|\bmy\s+head\s+(?:hurts|is\s+pounding)\b",
    re.I,
)
CONDITIONAL_LIMITATION_WITH_CURRENT_RELIEF_RE = re.compile(
    r"\bright\s+now\b[\s\S]{0,80}?\b(?:fine|nothing\s+hurting|no\s+pain)\b"
    r"[\s\S]{0,100}?\bwhen\b[\s\S]{0,40}?\b(?:hurt|hurts|pain)\b[\s\S]{0,60}?\b(?:cannot|can't)\b",
    re.I,
)
NOT_RESOLVED_RE = re.compile(
    r"\b(?:hasn['’]?t|has not|haven['’]?t|have not|isn['’]?t|is not|wasn['’]?t|was not|"
    r"doesn['’]?t|does not|didn['’]?t|did not|won['’]?t|will not|wouldn['’]?t|would not)"
    r"\b[^.?!;]{0,25}\b(?:stop(?:ped)?|resolv(?:e|ed))\b",
    re.I,
)
PARTIAL_RESOLUTION_RE = re.compile(r"\b(?:mostly|almost|seems? to have|appears? to have)\s+(?:stopped|resolved)\b|\b(?:stopped|resolved)\s+for now\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_card(card: dict[str, Any]) -> str:
    if card.get("context_only") or card.get("source_subtype") in CONTEXT_SOURCES:
        return "context_only"
    if not card.get("clinical_assertion_allowed"):
        return "not_clinically_assertable"
    if card.get("support_only"):
        return "support_only"
    if card.get("temporality") in {"historical", "chronic_or_background"}:
        return "history_or_background"
    if card.get("subject") != "patient" and card.get("source_subtype") != "nurse_direct_observation":
        return "non_patient_context"
    if card.get("assertion") == "absent" and card.get("negative_evidence_allowed"):
        return "explicit_negative"
    if card.get("assertion") == "uncertain":
        return "uncertain"
    if card.get("assertion") == "present" and card.get("positive_trigger_allowed"):
        return "positive_candidate"
    return "support_only"


def normalized_modifiers(card: dict[str, Any], parent_text: str) -> set[str]:
    modifiers = set(card.get("semantic_modifiers") or [])
    if "resolved_language" in modifiers and NOT_RESOLVED_RE.search(parent_text):
        modifiers.discard("resolved_language")
        modifiers.add("persistent")
    elif "resolved_language" in modifiers and PARTIAL_RESOLUTION_RE.search(parent_text):
        modifiers.discard("resolved_language")
        modifiers.add("partial_resolution")
    return modifiers


def state_after(previous: str, card: dict[str, Any], parent_text: str) -> tuple[str, str]:
    assertion = card.get("assertion")
    temporality = card.get("temporality")
    modifiers = normalized_modifiers(card, parent_text)
    if temporality in {"historical", "chronic_or_background"}:
        if previous != "unknown":
            return previous, "historical_context_does_not_override_current_state"
        return "historical", "historical_context_retained"
    if temporality == "earlier_in_episode":
        if previous not in {"unknown", "historical", "earlier_present", "earlier_absent", "earlier_uncertain"}:
            return previous, "earlier_episode_evidence_does_not_override_current_state"
        return {
            "present": ("earlier_present", "earlier_episode_positive_retained"),
            "absent": ("earlier_absent", "earlier_episode_negative_retained"),
            "uncertain": ("earlier_uncertain", "earlier_episode_uncertainty_retained"),
        }.get(assertion, (previous, "no_state_change"))
    if "partial_resolution" in modifiers:
        return "present_with_uncertainty", "partial_or_unverified_resolution"
    if "resolved_language" in modifiers:
        if previous in ACTIVE_STATES or previous == "earlier_present":
            return "currently_absent_after_prior_presence", "current_resolution_update"
        return "absent", "current_resolution_without_prior_active_state"
    if assertion == "uncertain":
        if previous in ACTIVE_STATES:
            return "present_with_uncertainty", "uncertainty_does_not_erase_active_state"
        return "uncertain", "current_state_uncertain"
    if assertion == "absent":
        if previous in ACTIVE_STATES:
            return "currently_absent_after_prior_presence", "later_target_local_denial"
        return "absent", "current_target_local_denial"
    if assertion == "present":
        if "worsening" in modifiers:
            return "present_worsening", "worsening_update"
        if "improving" in modifiers:
            return "present_improving", "improving_update"
        if "persistent" in modifiers:
            return "present_persistent", "persistence_update"
        if previous in {"absent", "currently_absent_after_prior_presence"}:
            return "present", "new_onset_or_recurrence"
        return "present", "positive_update"
    return previous, "no_state_change"


def build_trajectories(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    event_order = {event["parent_event_id"]: event["event_seq_idx"] for event in row["parent_events"]}
    event_text = {event["parent_event_id"]: event["original"] for event in row["parent_events"]}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    routed: list[dict[str, Any]] = []
    route_counts: Counter = Counter()
    for card in row["evidence_cards"]:
        route = route_card(card)
        route_counts[route] += 1
        routed.append({
            "evidence_card_id": card["evidence_card_id"],
            "clinical_atom_id": card["clinical_atom_id"],
            "route": route,
        })
        if route in {"context_only", "not_clinically_assertable", "non_patient_context", "support_only"}:
            continue
        episode = card.get("episode_id") or stable_id(
            "episode", row["instance_id"], card.get("subject"), card["concept_parent_id"]
        )
        grouped[(str(card.get("subject")), episode, card["clinical_atom_id"])].append(card)

    trajectories = []
    conflicts = []
    for (subject, episode_id, atom_id), cards in grouped.items():
        cards.sort(key=lambda card: (event_order.get(card["parent_event_id"], -1), card["char_start"], card["evidence_card_id"]))
        state = "unknown"
        transitions = []
        exact_scope: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for card in cards:
            old_state = state
            state, reason = state_after(state, card, event_text[card["parent_event_id"]])
            transitions.append({
                "from_state": old_state,
                "to_state": state,
                "transition_reason": reason,
                "evidence_card_id": card["evidence_card_id"],
                "clinical_atom_id": card["clinical_atom_id"],
                "assertion": card["assertion"],
                "temporality": card["temporality"],
                "turn_order": event_order.get(card["parent_event_id"], -1),
            })
            if card["assertion"] in {"present", "absent"} and card["temporality"] in CURRENT_TIMES:
                exact_scope[(card["parent_event_id"], card["clinical_atom_id"], card["temporality"])].add(card["assertion"])
        bad_scopes = [key for key, assertions in exact_scope.items() if assertions == {"present", "absent"}]
        trajectory_id = stable_id("trajectory", row["instance_id"], subject, episode_id, atom_id)
        if bad_scopes:
            conflicts.append({
                "instance_id": row["instance_id"],
                "trajectory_id": trajectory_id,
                "conflict_type": "same_event_same_atom_present_absent",
                "scopes": [
                    {"parent_event_id": key[0], "clinical_atom_id": key[1], "temporality": key[2]}
                    for key in bad_scopes
                ],
            })
        trajectories.append({
            "trajectory_id": trajectory_id,
            "episode_id": episode_id,
            "subject": subject,
            "concept_parent_id": cards[0]["concept_parent_id"],
            "clinical_atom_id": atom_id,
            "effective_state": "unresolved_conflict" if bad_scopes else state,
            "evidence_card_ids": [card["evidence_card_id"] for card in cards],
            "transitions": transitions,
        })
    return trajectories, routed, route_counts


def policy_evaluation(
    policy_id: str,
    policy: dict[str, Any],
    active_atoms: set[str],
    modifiers_by_atom: dict[str, set[str]],
    atom_evidence: dict[str, set[str]],
    pain_score_ids: list[str],
    current_positive_atoms: set[str],
    observed_current_atoms: set[str],
) -> dict[str, Any] | None:
    anchors = active_atoms & policy["anchors"]
    modifiers = active_atoms & policy["modifiers"]
    semantic_modifiers = {modifier for atom in active_atoms for modifier in modifiers_by_atom.get(atom, set())}
    mode = policy["mode"]
    if not anchors and not (mode == "respiratory" and modifiers):
        return None
    eligible = False
    reason = ""
    missing: list[str] = []

    if mode == "anchor_any":
        eligible = bool(anchors)
        reason = "current_positive_anchor"
    elif mode == "current_anchor":
        eligible = bool(anchors & current_positive_atoms)
        reason = "current_or_recent_positive_anchor"
        if anchors and not eligible:
            missing.append("current_or_recent_anchor")
    elif mode == "direct_observation":
        eligible = bool(anchors & observed_current_atoms)
        reason = "active_event_requires_direct_observation"
        if anchors and not eligible:
            missing.append("direct_current_observation")
    elif mode == "critical_airway":
        current_anchors = anchors & current_positive_atoms
        strong = current_anchors & {"stridor", "cyanosis", "tripod_position"}
        breath_limited = current_anchors & {"gasping", "short_sentence_speech"}
        eligible = bool(strong or (breath_limited and ({"dyspnea", "dyspnea_at_rest"} & current_positive_atoms)))
        reason = "critical_signal_requires_policy_review"
        if anchors and not eligible:
            missing.append("gas_exchange_context")
    elif mode == "anchor_plus_modifier":
        eligible = bool(anchors and modifiers)
        reason = "anchor_and_modifier_present"
        if anchors and not modifiers:
            missing.append("risk_modifier")
    elif mode == "context_plus_modifier":
        eligible = bool(anchors and modifiers)
        reason = "clinical_context_and_risk_feature_present"
        if anchors and not modifiers:
            missing.append("risk_feature")
    elif mode == "respiratory":
        current_anchors = anchors & current_positive_atoms
        current_modifiers = modifiers & current_positive_atoms
        eligible = bool(
            "dyspnea_at_rest" in current_anchors
            or ({"dyspnea", "dyspnea_exertional"} & current_anchors and current_modifiers)
            or ({"dyspnea", "dyspnea_exertional"} & current_anchors and "worsening" in semantic_modifiers)
            or current_modifiers & {"stridor", "cyanosis", "tripod_position"}
        )
        reason = "respiratory_distress_or_scope_escalation"
        if anchors and not eligible:
            missing.append("distress_or_deterioration_feature")
    elif mode == "bleeding":
        eligible = bool(anchors and (modifiers or {"worsening", "persistent", "severe_language"} & semantic_modifiers))
        reason = "bleeding_with_risk_modifier"
        if anchors and not eligible:
            missing.append("severity_or_instability_context")
    elif mode == "pain_context":
        severe_context = bool(
            pain_score_ids
            or modifiers
            or {"severe_language", "functional_limitation", "treatment_failure"} & semantic_modifiers
        )
        eligible = bool(anchors & current_positive_atoms and severe_context)
        reason = "pain_anchor_with_severity_context"
        if anchors and not severe_context:
            missing.append("severity_context")
    elif mode == "headache":
        eligible = "thunderclap_headache" in anchors or bool("headache" in anchors and modifiers)
        reason = "thunderclap_or_headache_with_red_flag"
        if anchors and not eligible:
            missing.append("headache_red_flag")

    evidence_ids = sorted({evidence_id for atom in anchors | modifiers for evidence_id in atom_evidence.get(atom, set())})
    if policy_id == "severe_pain_or_distress":
        evidence_ids = sorted(set(evidence_ids) | set(pain_score_ids))
    return {
        "policy_candidate_id": stable_id("policy_candidate", policy_id, *evidence_ids),
        "policy_field": policy_id,
        "policy_step": policy["step"],
        "policy_rule_id": policy["rule_id"],
        "anchor_atom_ids": sorted(anchors),
        "modifier_atom_ids": sorted(modifiers),
        "semantic_modifiers": sorted(semantic_modifiers),
        "evidence_ids": evidence_ids,
        "missing_requirements": missing,
        "deterministic_route": "needs_llm_policy_qualification" if eligible else "confirmed_symptom_or_context_only",
        "route_reason": reason,
        "dry_run": True,
        "automatic_positive_claim": False,
    }


def build_instance(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Counter]:
    trajectories, routed, route_counts = build_trajectories(row)
    active_atoms: set[str] = set()
    current_positive_atoms: set[str] = set()
    observed_current_atoms: set[str] = set()
    atom_evidence: dict[str, set[str]] = defaultdict(set)
    modifiers_by_atom: dict[str, set[str]] = defaultdict(set)
    cards_by_id = {card["evidence_card_id"]: card for card in row["evidence_cards"]}
    events_by_id = {event["parent_event_id"]: event for event in row["parent_events"]}
    for trajectory in trajectories:
        if trajectory["effective_state"] not in ACTIVE_STATES:
            continue
        atom = trajectory["clinical_atom_id"]
        if atom == "fall_or_trauma":
            fall_texts = [
                events_by_id[cards_by_id[evidence_id]["parent_event_id"]]["original"]
                for evidence_id in trajectory["evidence_card_ids"]
            ]
            if fall_texts and all(NEAR_FALL_RE.search(text) for text in fall_texts):
                continue
        if atom == "headache":
            headache_texts = [
                events_by_id[cards_by_id[evidence_id]["parent_event_id"]]["original"]
                for evidence_id in trajectory["evidence_card_ids"]
            ]
            if headache_texts and all(
                HEADACHE_MEDICATION_INDICATION_RE.search(text) and not CURRENT_HEADACHE_RE.search(text)
                for text in headache_texts
            ):
                continue
        if atom == "functional_limitation":
            limitation_texts = [
                events_by_id[cards_by_id[evidence_id]["parent_event_id"]]["original"]
                for evidence_id in trajectory["evidence_card_ids"]
            ]
            if limitation_texts and all(CONDITIONAL_LIMITATION_WITH_CURRENT_RELIEF_RE.search(text) for text in limitation_texts):
                continue
        active_atoms.add(atom)
        if any(item["transition_reason"] == "persistence_update" for item in trajectory["transitions"]):
            modifiers_by_atom[atom].add("persistent")
        for evidence_id in trajectory["evidence_card_ids"]:
            card = cards_by_id[evidence_id]
            atom_evidence[atom].add(evidence_id)
            parent_text = events_by_id[card["parent_event_id"]]["original"]
            modifiers_by_atom[atom].update(normalized_modifiers(card, parent_text))
            if card.get("assertion") == "present" and card.get("temporality") in {"current", "recent", "current_or_unspecified"}:
                current_positive_atoms.add(atom)
                if card.get("source_subtype") == "nurse_direct_observation":
                    observed_current_atoms.add(atom)

    pain_score_ids = []
    for measurement in row.get("measurement_records", []):
        policy = measurement.get("evidence_use_policy") or {}
        if (
            measurement.get("field") == "pain_score"
            and measurement.get("measurement_identity") == "canonical_case_measurement"
            and measurement.get("usable_for_clinical_reasoning") is True
            and policy.get("allowed") is True
            and float(measurement.get("normalized_value")) >= 7
        ):
            pain_score_ids.append(measurement["measurement_id"])

    policy_candidates = []
    for policy_id, policy in POLICIES.items():
        candidate = policy_evaluation(
            policy_id, policy, active_atoms, modifiers_by_atom, atom_evidence, pain_score_ids,
            current_positive_atoms, observed_current_atoms,
        )
        if candidate:
            candidate["instance_id"] = row["instance_id"]
            candidate["case_id"] = row["case_id"]
            policy_candidates.append(candidate)

    measurements_by_id = {measurement["measurement_id"]: measurement for measurement in row.get("measurement_records", [])}
    trajectory_by_evidence = {
        evidence_id: trajectory["trajectory_id"]
        for trajectory in trajectories
        for evidence_id in trajectory["evidence_card_ids"]
    }
    for candidate in policy_candidates:
        bundle = []
        seen_card_scopes: set[tuple[str, str, str, str]] = set()
        for evidence_id in candidate["evidence_ids"]:
            card = cards_by_id.get(evidence_id)
            if card is not None:
                event = events_by_id[card["parent_event_id"]]
                scope_key = (card["parent_event_id"], card["clinical_atom_id"], card["assertion"], card["temporality"])
                if scope_key in seen_card_scopes:
                    continue
                seen_card_scopes.add(scope_key)
                question_id = (card.get("qa_context") or {}).get("question_event_id")
                bundle.append({
                    "evidence_id": evidence_id,
                    "evidence_type": "clinical_atom",
                    "trajectory_id": trajectory_by_evidence.get(evidence_id),
                    "clinical_atom_id": card["clinical_atom_id"],
                    "span_text": card["span_text"],
                    "assertion": card["assertion"],
                    "temporality": card["temporality"],
                    "subject": card["subject"],
                    "source_subtype": card["source_subtype"],
                    "anatomy_site": card.get("anatomy_site"),
                    "laterality": card.get("laterality"),
                    "scope_resolution": card.get("scope_resolution"),
                    "independent_evidence": card.get("independent_evidence"),
                    "positive_trigger_allowed": card.get("positive_trigger_allowed"),
                    "semantic_modifiers": sorted(normalized_modifiers(card, event["original"])),
                    "parent_event_id": card["parent_event_id"],
                    "turn_order": event["event_seq_idx"],
                    "parent_text": event["original"],
                    "question_text": events_by_id.get(question_id, {}).get("original"),
                    "qwen_resolution": card.get("qwen_resolution"),
                    "nli_verification": card.get("nli_verification"),
                })
                continue
            measurement = measurements_by_id[evidence_id]
            bundle.append({
                "evidence_id": evidence_id,
                "evidence_type": "measurement_support",
                "field": measurement["field"],
                "normalized_value": measurement["normalized_value"],
                "canonical_unit": measurement["canonical_unit"],
                "support_only": True,
            })
        candidate["evidence_bundle"] = bundle

    explicit_negatives = [item for item in routed if item["route"] == "explicit_negative"]
    uncertain_evidence = [item for item in routed if item["route"] == "uncertain"]
    output = {
        "schema_version": SCHEMA_VERSION,
        "input_schema_version": row["schema_version"],
        "manifest_version": MANIFEST_VERSION,
        "registry_version": row["registry_version"],
        "case_id": row["case_id"],
        "instance_id": row["instance_id"],
        "atom_trajectories": trajectories,
        "policy_candidates": policy_candidates,
        "explicit_negatives": explicit_negatives,
        "uncertain_evidence": uncertain_evidence,
        "confirmed_step_a_signals": [],
        "confirmed_step_b_signals": [],
        "dry_run": True,
    }
    conflicts = [
        {
            "instance_id": row["instance_id"],
            "trajectory_id": trajectory["trajectory_id"],
            "conflict_type": "unresolved_trajectory_conflict",
        }
        for trajectory in trajectories
        if trajectory["effective_state"] == "unresolved_conflict"
    ]
    return output, policy_candidates, conflicts, route_counts


def forbidden_locations(value: Any, path: str = "") -> list[str]:
    forbidden = {"ground_truth", "acuity", "triage", "patient_persona", "nurse_persona", "specialisation"}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key.lower() in forbidden:
                found.append(child_path)
            found.extend(forbidden_locations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_locations(child, f"{path}[{index}]"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-instance-count", type=int, default=1010)
    args = parser.parse_args()

    source_rows = read_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    flat_candidates = []
    conflicts = []
    validation_errors = []
    route_counts: Counter = Counter()
    invalid_ids = {
        measurement["measurement_id"]
        for row in source_rows
        for measurement in row.get("quarantined_measurements", [])
    }

    for row in source_rows:
        if row.get("schema_version") != EXPECTED_INPUT_SCHEMA:
            validation_errors.append({"instance_id": row.get("instance_id"), "type": "unexpected_input_schema"})
            continue
        output, candidates, row_conflicts, counts = build_instance(row)
        outputs.append(output)
        flat_candidates.extend(candidates)
        conflicts.extend(row_conflicts)
        route_counts.update(counts)
        for location in forbidden_locations(output):
            validation_errors.append({"instance_id": row["instance_id"], "type": "forbidden_output_key", "location": location})

    consumed_ids = {
        evidence_id
        for candidate in flat_candidates
        for evidence_id in candidate["evidence_ids"]
    }
    unknown_policy_atoms = sorted(POLICY_ATOMS - set(CLINICAL_ATOMS) - {"pain_present"})
    question_positive = sum(
        1
        for row in source_rows
        for card in row["evidence_cards"]
        if card.get("source_subtype") in CONTEXT_SOURCES and route_card(card) == "positive_candidate"
    )
    support_only_triggers = sum(
        1
        for row in source_rows
        for card in row["evidence_cards"]
        if card.get("support_only") and route_card(card) == "positive_candidate"
    )
    empty_evidence_candidates = sum(not candidate["evidence_ids"] for candidate in flat_candidates)
    context_only_policy_triggers = sum(
        1
        for candidate in flat_candidates
        if candidate["deterministic_route"] == "needs_llm_policy_qualification"
        and candidate["policy_field"] == "pregnancy_high_risk"
        and not candidate["modifier_atom_ids"]
    )
    hard_gates = {
        "expected_instance_count": len(outputs) == args.expected_instance_count,
        "unique_instance_ids": len({row["instance_id"] for row in outputs}) == len(outputs),
        "validation_errors_zero": not validation_errors,
        "forbidden_leakage_zero": not any(error["type"] == "forbidden_output_key" for error in validation_errors),
        "unknown_policy_atoms_zero": not unknown_policy_atoms,
        "invalid_measurement_consumption_zero": not (invalid_ids & consumed_ids),
        "question_process_positive_zero": question_positive == 0,
        "support_only_standalone_trigger_zero": support_only_triggers == 0,
        "unresolved_exact_scope_conflict_zero": not conflicts,
        "candidate_empty_evidence_zero": empty_evidence_candidates == 0,
        "pregnancy_context_standalone_trigger_zero": context_only_policy_triggers == 0,
        "dry_run_confirmed_signal_zero": all(not row["confirmed_step_a_signals"] and not row["confirmed_step_b_signals"] for row in outputs),
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "input_schema_version": EXPECTED_INPUT_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "registry_version": REGISTRY_VERSION,
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "processed_instance_count": len(outputs),
        "unique_case_count": len({row["case_id"] for row in outputs}),
        "trajectory_count": sum(len(row["atom_trajectories"]) for row in outputs),
        "route_counts": dict(route_counts),
        "policy_candidate_count": len(flat_candidates),
        "llm_eligible_candidate_count": sum(candidate["deterministic_route"] == "needs_llm_policy_qualification" for candidate in flat_candidates),
        "symptom_or_context_only_count": sum(candidate["deterministic_route"] == "confirmed_symptom_or_context_only" for candidate in flat_candidates),
        "policy_field_counts": dict(Counter(candidate["policy_field"] for candidate in flat_candidates)),
        "llm_eligible_policy_field_counts": dict(Counter(candidate["policy_field"] for candidate in flat_candidates if candidate["deterministic_route"] == "needs_llm_policy_qualification")),
        "explicit_negative_count": sum(len(row["explicit_negatives"]) for row in outputs),
        "uncertain_evidence_count": sum(len(row["uncertain_evidence"]) for row in outputs),
        "invalid_measurement_seen_count": len(invalid_ids),
        "invalid_measurement_consumed_count": len(invalid_ids & consumed_ids),
        "conflict_count": len(conflicts),
        "validation_error_count": len(validation_errors),
        "hard_gates": hard_gates,
        "development_gate_passed": all(hard_gates.values()),
        "release_gate_passed": all(hard_gates.values()),
        "dry_run": True,
    }

    write_jsonl(args.output_dir / "02_qualified_trajectories.jsonl", outputs)
    write_jsonl(args.output_dir / "02_policy_candidates.jsonl", flat_candidates)
    write_jsonl(args.output_dir / "02_conflict_audit.jsonl", conflicts)
    write_jsonl(args.output_dir / "validation_errors.jsonl", validation_errors)
    (args.output_dir / "02_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    artifact_names = (
        "02_qualified_trajectories.jsonl", "02_policy_candidates.jsonl",
        "02_conflict_audit.jsonl", "validation_errors.jsonl", "02_build_audit.json",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "source_sha256": {
            "02_build_qualified_trajectories.py": sha256(Path(__file__)),
            "esi_policy_manifest_v5.py": sha256(Path(__file__).with_name("esi_policy_manifest_v5.py")),
            "clinical_concept_registry.py": sha256(Path(__file__).with_name("clinical_concept_registry.py")),
        },
        "artifacts": {name: sha256(args.output_dir / name) for name in artifact_names},
    }
    (args.output_dir / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
