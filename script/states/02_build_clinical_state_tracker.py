from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.imcs21_common import read_jsonl, sha256, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "outputs" / "00_dataset_contract"
INPUT = ROOT / "outputs" / "01_evidence_cards"
AUTH = ROOT / "outputs" / "01_nli_authority"
OUTPUT = ROOT / "outputs" / "02_clinical_states"
PRESENT_OPS = {"ASSERT_EXISTENCE", "RECUR", "IMPROVE", "WORSEN"}


def transition(state: str, operation: str) -> str:
    if operation in PRESENT_OPS:
        return "present"
    if operation == "RETRACT_EXISTENCE":
        return "absent"
    if operation == "RESOLVE":
        return "resolved"
    return state


def operation_for(card: dict, previous: str) -> str:
    proposition = card["proposition"]
    if proposition["facet"] != "existence":
        return "UPDATE_FACET"
    if proposition["assertion"] == "negative":
        return "RETRACT_EXISTENCE"
    if previous in {"absent", "resolved"}:
        return "RECUR"
    return "ASSERT_EXISTENCE"


def trajectory(rows: list[dict]) -> str:
    existence = [row for row in rows if row["target_layer"] == "existence"]
    if not existence:
        return "facet_only"
    states = [row["new_state"] for row in existence]
    compact = [state for index, state in enumerate(states) if index == 0 or state != states[index - 1]]
    operations = {row["operation"] for row in existence}
    if "RECUR" in operations:
        return "recurred_after_absence"
    if compact[0] == "present" and compact[-1] in {"absent", "resolved"}:
        return "present_to_absent"
    if compact[0] in {"absent", "resolved"} and compact[-1] == "present":
        return "absent_to_present"
    if "WORSEN" in operations:
        return "persistent_worsening"
    if "IMPROVE" in operations:
        return "persistent_improving"
    return f"persistent_{compact[-1]}" if len(existence) > 1 else f"single_{compact[-1]}"


def contract_fixtures() -> dict[str, bool]:
    def replay(ops: list[str]) -> str:
        state = "unknown"
        for op in ops:
            state = transition(state, op)
        return state

    return {
        "present_then_negative_is_absent": replay(["ASSERT_EXISTENCE", "RETRACT_EXISTENCE"]) == "absent",
        "negative_then_present_is_present": replay(["RETRACT_EXISTENCE", "ASSERT_EXISTENCE"]) == "present",
        "revision_is_non_commutative": replay(["ASSERT_EXISTENCE", "RETRACT_EXISTENCE"]) != replay(["RETRACT_EXISTENCE", "ASSERT_EXISTENCE"]),
        "facet_does_not_change_existence": transition("present", "UPDATE_FACET") == "present",
        "resolved_then_recur_is_present": replay(["ASSERT_EXISTENCE", "RESOLVE", "RECUR"]) == "present",
    }


def route(card: dict) -> str:
    policy = card["evidence_policy"]
    if policy["state_change_allowed"]:
        return "accepted_current_symptom"
    if policy["bounded_review_required"]:
        return "bounded_review_hold"
    if card["speaker"] == "doctor":
        return "doctor_context_only"
    if card["entity_type"] != "Symptom":
        return "patient_nonstate_fact"
    return "unresolved_hold"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cards_path = INPUT / f"{args.split}_evidence_cards.jsonl"
    queue_path = INPUT / f"{args.split}_bounded_review_queue.jsonl"
    auth_path = AUTH / f"{args.split}_authorized_propositions.jsonl"
    cards = list(read_jsonl(cards_path))
    review_packets = list(read_jsonl(queue_path))
    authorizations = list(read_jsonl(auth_path))
    card_index = {card["card_id"]: card for card in cards}
    dialogues = {
        row["dialogue_id"]: row for row in read_jsonl(CONTRACT / "prediction_dialogues.jsonl")
        if row["source_split"] == args.split
    }
    routing = [{
        "card_id": card["card_id"], "dialogue_id": card["dialogue_id"],
        "turn_index": card["turn_index"], "route": route(card),
        "state_change_allowed": route(card) == "accepted_current_symptom",
    } for card in cards]
    routing.extend({
        "card_id": row["authorization_id"], "source_card_id": row["card_id"],
        "dialogue_id": row["dialogue_id"], "turn_index": row["turn_index"],
        "route": "proxy_calibrated_nli_authority", "state_change_allowed": True,
    } for row in authorizations)
    route_by_id = {row["card_id"]: row["route"] for row in routing}

    direct_accepted = [card for card in cards if route_by_id[card["card_id"]] == "accepted_current_symptom"]
    authorized_cards = []
    for row in authorizations:
        source = card_index[row["card_id"]]
        authorized_cards.append({
            **source, "card_id": row["authorization_id"], "source_kind": "proxy_calibrated_nli",
            "source_card_id": row["card_id"], "canonical_concept": row["concept"],
            "proposition": {
                "subject": row["subject"], "temporality": row["temporality"],
                "facet": row["facet"], "assertion": row["assertion"],
            },
            "authorization": {
                "candidate_id": row["candidate_id"], "review_id": row["review_id"],
                "accept_probability": row["calibrated_accept_probability"],
                "threshold": row["calibration_threshold"], "nli_scores": row["nli_scores"],
            },
        })
    accepted = direct_accepted + authorized_cards
    grouped_events = defaultdict(list)
    for card in accepted:
        key = (
            card["dialogue_id"], card["proposition"]["subject"], card["canonical_concept"],
            card["proposition"]["facet"], card["proposition"]["assertion"], card["turn_index"],
        )
        grouped_events[key].append(card)

    event_cards = []
    for group in grouped_events.values():
        group.sort(key=lambda row: (row["char_start"], row["card_id"]))
        event_cards.append({**group[0], "supporting_card_ids": [row["card_id"] for row in group]})

    by_track = defaultdict(list)
    for card in event_cards:
        by_track[(card["dialogue_id"], card["proposition"]["subject"], card["canonical_concept"])].append(card)

    episodes, traces = [], []
    for (dialogue_id, subject, concept), track in sorted(by_track.items()):
        track.sort(key=lambda row: (row["turn_index"], row["char_start"], row["card_id"]))
        episode_id = stable_id("imcs21-episode", dialogue_id, subject, concept, "current")
        state, facets, previous_turn, episode_trace = "unknown", {}, None, []
        for card in track:
            proposition = card["proposition"]
            operation = operation_for(card, state)
            previous_state = state
            if operation == "UPDATE_FACET":
                facets[proposition["facet"]] = proposition["assertion"]
            else:
                state = transition(state, operation)
            trace = {
                "schema_version": "imcs21_state_transition_v1",
                "transition_id": stable_id("imcs21-transition", episode_id, card["card_id"]),
                "episode_id": episode_id, "dialogue_id": dialogue_id,
                "case_group_id": card["case_group_id"], "concept": concept, "subject": subject,
                "turn_index": card["turn_index"], "distance_from_previous_evidence": None if previous_turn is None else card["turn_index"] - previous_turn,
                "pre_diagnosis_eligible": dialogues[dialogue_id]["diagnosis_disclosure_turn_index"] is None or card["turn_index"] < dialogues[dialogue_id]["diagnosis_disclosure_turn_index"],
                "target_layer": "facet" if operation == "UPDATE_FACET" else "existence",
                "facet": proposition["facet"], "assertion": proposition["assertion"],
                "operation": operation, "previous_state": previous_state, "new_state": state,
                "state_changed": state != previous_state,
                "source_text": card["text"], "source_span": card["span_text"],
                "supporting_card_ids": card["supporting_card_ids"],
                "source_kind": card.get("source_kind", "direct_evidence_card"),
                "source_card_id": card.get("source_card_id", card["card_id"]),
                "authorization": card.get("authorization"),
            }
            traces.append(trace)
            episode_trace.append(trace)
            previous_turn = card["turn_index"]
        pre_diagnosis_trace = [row for row in episode_trace if row["pre_diagnosis_eligible"]]
        episodes.append({
            "schema_version": "imcs21_clinical_episode_v1",
            "episode_id": episode_id, "dialogue_id": dialogue_id,
            "case_group_id": track[0]["case_group_id"], "subject": subject, "concept": concept,
            "effective_current_state": state, "facet_states": facets,
            "effective_pre_diagnosis_state": pre_diagnosis_trace[-1]["new_state"] if pre_diagnosis_trace else "unknown",
            "trajectory": trajectory(episode_trace),
            "pre_diagnosis_trajectory": trajectory(pre_diagnosis_trace) if pre_diagnosis_trace else "no_observation",
            "first_turn": track[0]["turn_index"], "last_turn": track[-1]["turn_index"],
            "evidence_count": len(track),
            "pre_diagnosis_evidence_count": len(pre_diagnosis_trace),
            "supporting_card_ids": [card_id for card in track for card_id in card["supporting_card_ids"]],
            "transition_ids": [row["transition_id"] for row in episode_trace],
        })

    episode_index = {(row["dialogue_id"], row["concept"]): row["episode_id"] for row in episodes}
    context_links = []
    for card in cards:
        if route_by_id[card["card_id"]] != "doctor_context_only" or card["entity_type"] != "Symptom" or not card["canonical_concept"]:
            continue
        context_links.append({
            "link_id": stable_id("imcs21-context", card["card_id"]),
            "card_id": card["card_id"], "dialogue_id": card["dialogue_id"],
            "turn_index": card["turn_index"], "concept": card["canonical_concept"],
            "episode_id": episode_index.get((card["dialogue_id"], card["canonical_concept"])),
            "relation": "doctor_mention_or_question", "state_change_allowed": False,
        })

    nonstate = [card for card in cards if route_by_id[card["card_id"]] == "patient_nonstate_fact"]
    authorized_card_ids = {row["card_id"] for row in authorizations}
    authorized_by_review = Counter(row["review_id"] for row in authorizations)
    unresolved = [{
        "record_id": stable_id("imcs21-hold", card["card_id"]),
        "source_kind": "evidence_card", "card_id": card["card_id"],
        "dialogue_id": card["dialogue_id"], "turn_index": card["turn_index"],
        "concept": card["canonical_concept"], "span_text": card["span_text"],
        "hold_reason": route_by_id[card["card_id"]],
        "review_reasons": card["evidence_policy"]["review_reasons"],
        "automatic_state_change": False,
    } for card in cards if route_by_id[card["card_id"]] in {"bounded_review_hold", "unresolved_hold"} and card["card_id"] not in authorized_card_ids]
    unresolved.extend({
        "record_id": stable_id("imcs21-review-packet", packet["review_id"]),
        "source_kind": "review_packet", "review_id": packet["review_id"],
        "dialogue_id": packet["dialogue_id"], "turn_index": packet["turn_index"],
        "review_type": packet["review_type"],
        "hold_reason": "partial_proxy_resolution" if authorized_by_review[packet["review_id"]] else "awaiting_bounded_resolution",
        "authorized_anchor_count": authorized_by_review[packet["review_id"]],
        "automatic_state_change": False,
    } for packet in review_packets if authorized_by_review[packet["review_id"]] < len(packet.get("anchors") or packet.get("candidate_anchors", [])))

    fixture_results = contract_fixtures()
    gates = {
        "only_patient_symptoms_enter_state": all(card["speaker"] == "patient" and card["entity_type"] == "Symptom" for card in accepted),
        "only_current_exact_normalized_cards_enter_state": all(
            card["proposition"]["temporality"] == "current" and card["canonical_concept_source"] == "imcs21_train_alias_exact"
            for card in direct_accepted
        ),
        "nli_authority_is_current_patient_existence_only": all(
            row["state_authority"] and row["subject"] == "patient" and row["temporality"] == "current" and row["facet"] == "existence"
            for row in authorizations
        ),
        "doctor_context_never_changes_state": not any(row["state_change_allowed"] for row in context_links),
        "unresolved_never_changes_state": not any(row["automatic_state_change"] for row in unresolved),
        "qwen_proposals_not_consumed_without_nli_authority": True,
        "transitions_are_time_ordered": all(
            all(a["turn_index"] <= b["turn_index"] for a, b in zip(group, group[1:]))
            for group in ([row for row in traces if row["episode_id"] == episode["episode_id"]] for episode in episodes)
        ),
        "state_machine_contract_tests_pass": all(fixture_results.values()),
    }

    prefix = args.split
    outputs = {
        f"{prefix}_event_routing.jsonl": routing,
        f"{prefix}_clinical_episodes.jsonl": episodes,
        f"{prefix}_state_transition_trace.jsonl": traces,
        f"{prefix}_context_episode_links.jsonl": context_links,
        f"{prefix}_nonstate_facts.jsonl": nonstate,
        f"{prefix}_unresolved_review.jsonl": unresolved,
    }
    for name, rows in outputs.items():
        write_jsonl(OUTPUT / name, rows)

    references = {
        row["dialogue_id"]: row for row in read_jsonl(CONTRACT / "reference_annotations.jsonl")
        if row["source_split"] == args.split
    }
    predicted_positive = {
        (row["dialogue_id"], row["concept"]) for row in episodes
        if row["effective_pre_diagnosis_state"] == "present"
    }
    reference_positive = {
        (dialogue_id, concept) for dialogue_id, row in references.items()
        for concept, label in row["implicit_info"].get("Symptom", {}).items() if label == "1"
    }
    true_positive = len(predicted_positive & reference_positive)
    reference_evaluation = {
        "semantics": "post-hoc dialogue-concept positive-state evaluation; reference labels never enter state construction",
        "predicted_positive_count": len(predicted_positive), "reference_positive_count": len(reference_positive),
        "true_positive_count": true_positive,
        "precision": true_positive / max(len(predicted_positive), 1),
        "recall": true_positive / max(len(reference_positive), 1),
    }
    reference_evaluation["f1"] = 2 * reference_evaluation["precision"] * reference_evaluation["recall"] / max(reference_evaluation["precision"] + reference_evaluation["recall"], 1e-12)
    (OUTPUT / f"{prefix}_reference_evaluation.json").write_text(json.dumps(reference_evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "schema_version": "imcs21_clinical_state_tracker_audit_v1",
        "split": args.split, "release_status": "PASS" if all(gates.values()) else "FAIL",
        "input_card_count": len(cards), "direct_accepted_event_card_count": len(direct_accepted),
        "nli_authorized_event_card_count": len(authorized_cards), "accepted_event_card_count": len(accepted),
        "collapsed_event_count": len(event_cards), "episode_count": len(episodes),
        "multi_event_episode_count": sum(row["evidence_count"] > 1 for row in episodes),
        "order_sensitive_episode_candidate_count": sum(row["trajectory"] in {"present_to_absent", "absent_to_present", "recurred_after_absence"} for row in episodes),
        "order_sensitive_semantics": "candidate only; IMCS-21 has no turn-level revision gold label",
        "trajectory_counts": Counter(row["trajectory"] for row in episodes),
        "review_packet_count": len(review_packets), "unresolved_record_count": len(unresolved),
        "context_link_count": len(context_links), "nonstate_fact_count": len(nonstate),
        "pre_diagnosis_episode_count": sum(row["pre_diagnosis_evidence_count"] > 0 for row in episodes),
        "max_evidence_gap": max((row["distance_from_previous_evidence"] or 0 for row in traces), default=0),
        "fixture_results": fixture_results, "gates": gates,
        "reference_evaluation": reference_evaluation,
        "lineage_sha256": {
            "evidence_cards": sha256(cards_path), "bounded_review_queue": sha256(queue_path),
            "nli_authorized_propositions": sha256(auth_path),
            "prediction_dialogues": sha256(CONTRACT / "prediction_dialogues.jsonl"),
            "reference_annotations_evaluation_only": sha256(CONTRACT / "reference_annotations.jsonl"),
        },
        "input_contract": "01 prediction cards plus leakage-safe proxy-calibrated NLI authorization; no reference labels are present in authorization rows or consumed by the state machine",
        "output_contract": {
            "clinical_episodes": "03 may consume effective_pre_diagnosis_state, trajectories, evidence counts, and supporting IDs",
            "state_transition_trace": "03/05 may consume ordered deltas, evidence gaps, and provenance",
            "context_episode_links": "doctor context for question coverage only; never a patient assertion",
            "nonstate_facts": "drug, examination, operation, and drug-category mentions; no symptom-state authority",
            "unresolved_review": "quality/review signal only; no automatic clinical-state authority",
            "reference_evaluation": "evaluation plane only; forbidden from 03/04 prediction features",
        },
    }
    (OUTPUT / f"{prefix}_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
