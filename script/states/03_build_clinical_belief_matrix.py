from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.imcs21_common import read_jsonl, sha256, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "outputs" / "00_dataset_contract"
STATES = ROOT / "outputs" / "02_clinical_states"
OUTPUT = ROOT / "outputs" / "03_clinical_belief_matrix"


def action_family(dialogue_act: str) -> str:
    if dialogue_act == "Request-Symptom":
        return "ASK_SYMPTOM"
    if dialogue_act.startswith("Request-"):
        return "ASK_OTHER"
    if dialogue_act == "Diagnose":
        return "DIAGNOSE"
    if dialogue_act in {
        "Inform-Symptom", "Inform-Etiology", "Inform-Existing_Examination_and_Treatment",
        "Inform-Basic_Information",
    }:
        return "CLINICAL_INFORM"
    if dialogue_act in {"Inform-Medical_Advice", "Inform-Drug_Recommendation", "Inform-Precautions"}:
        return "RECOMMEND"
    return "OTHER"


def snapshot_row(dialogue: dict, turn_index: int, states: dict, holds: list, last_evidence: int | None,
                 source_counts: Counter, transition_ids: list[str], episodes: dict) -> dict:
    active = []
    for episode_id, state in sorted(states.items()):
        episode = episodes[episode_id]
        active.append({
            "episode_id": episode_id,
            "concept": episode["concept"],
            "existence_state": state["existence_state"],
            "last_update_turn": state["last_update_turn"],
            "evidence_count": state["evidence_count"],
            "source_kinds": sorted(state["source_kinds"]),
        })
    evidence_holds = [row for row in holds if row["source_kind"] == "evidence_card"]
    held_cards = {row.get("card_id") for row in evidence_holds if row.get("card_id")}
    held_concepts = {row.get("concept") for row in evidence_holds if row.get("concept")}
    quality = {
        "held_card_count": len(held_cards),
        "unresolved_record_count": len(holds),
        "review_packet_count": sum(row["source_kind"] == "review_packet" for row in holds),
        "held_concept_count": len(held_concepts),
        "review_required": bool(holds),
    }
    payload = [(row["episode_id"], row["existence_state"], row["last_update_turn"]) for row in active]
    return {
        "schema_version": "imcs21_turn_state_snapshot_v1",
        "snapshot_id": stable_id(dialogue["dialogue_id"], turn_index, payload, quality),
        "dialogue_id": dialogue["dialogue_id"],
        "case_group_id": dialogue["case_group_id"],
        "source_split": dialogue["source_split"],
        "turn_index": turn_index,
        "effective_episode_states": active,
        "active_state_count": sum(row["existence_state"] == "present" for row in active),
        "distance_since_last_state_evidence": None if last_evidence is None else turn_index - last_evidence,
        "quality_state": quality,
        "cumulative_direct_transition_count": source_counts["direct_evidence_card"],
        "cumulative_nli_transition_count": source_counts["proxy_calibrated_nli"],
        "cumulative_transition_ids": list(transition_ids),
        "state_hash": stable_id(payload, quality),
    }


def build_split(split: str, dialogues_all: list[dict], references_all: list[dict]) -> dict:
    dialogues = [row for row in dialogues_all if row["source_split"] == split]
    references = {row["dialogue_id"]: row for row in references_all if row["source_split"] == split}
    episodes_path = STATES / f"{split}_clinical_episodes.jsonl"
    traces_path = STATES / f"{split}_state_transition_trace.jsonl"
    nonstate_path = STATES / f"{split}_nonstate_facts.jsonl"
    unresolved_path = STATES / f"{split}_unresolved_review.jsonl"
    audit02_path = STATES / f"{split}_build_audit.json"
    episodes = list(read_jsonl(episodes_path))
    traces = list(read_jsonl(traces_path))
    nonstate = list(read_jsonl(nonstate_path))
    unresolved = list(read_jsonl(unresolved_path))
    audit02 = json.loads(audit02_path.read_text(encoding="utf-8"))

    episode_lookup = {row["episode_id"]: row for row in episodes}
    episodes_by_dialogue, traces_by_dialogue = defaultdict(list), defaultdict(list)
    nonstate_by_dialogue, unresolved_by_dialogue = defaultdict(list), defaultdict(list)
    for row in episodes:
        episodes_by_dialogue[row["dialogue_id"]].append(row)
    for row in traces:
        traces_by_dialogue[row["dialogue_id"]].append(row)
    for row in nonstate:
        nonstate_by_dialogue[row["dialogue_id"]].append(row)
    for row in unresolved:
        unresolved_by_dialogue[row["dialogue_id"]].append(row)

    matrices, snapshots, coordinates, units, supervision = [], [], [], [], []
    snapshot_lookup = {}
    for dialogue in dialogues:
        dialogue_id = dialogue["dialogue_id"]
        by_turn, holds_by_turn = defaultdict(list), defaultdict(list)
        for row in traces_by_dialogue[dialogue_id]:
            by_turn[row["turn_index"]].append(row)
        for row in unresolved_by_dialogue[dialogue_id]:
            holds_by_turn[row["turn_index"]].append(row)

        states, cumulative_holds, source_counts, transition_ids = {}, [], Counter(), []
        last_evidence = None
        initial = snapshot_row(dialogue, -1, states, cumulative_holds, last_evidence, source_counts,
                               transition_ids, episode_lookup)
        snapshots.append(initial)
        snapshot_lookup[(dialogue_id, -1)] = initial
        for turn in dialogue["turns"]:
            turn_index = turn["turn_index"]
            for trace in sorted(by_turn[turn_index], key=lambda row: row["transition_id"]):
                state = states.setdefault(trace["episode_id"], {
                    "existence_state": "unknown", "last_update_turn": None,
                    "evidence_count": 0, "source_kinds": set(),
                })
                state["existence_state"] = trace["new_state"]
                state["last_update_turn"] = turn_index
                state["evidence_count"] += 1
                state["source_kinds"].add(trace["source_kind"])
                source_counts[trace["source_kind"]] += 1
                transition_ids.append(trace["transition_id"])
                last_evidence = turn_index
            cumulative_holds.extend(holds_by_turn[turn_index])
            snapshot = snapshot_row(dialogue, turn_index, states, cumulative_holds, last_evidence,
                                    source_counts, transition_ids, episode_lookup)
            snapshots.append(snapshot)
            snapshot_lookup[(dialogue_id, turn_index)] = snapshot

        final_snapshot = snapshot_lookup[(dialogue_id, dialogue["turns"][-1]["turn_index"])]
        matrices.append({
            "schema_version": "imcs21_clinical_belief_matrix_v1",
            "dialogue_id": dialogue_id,
            "case_group_id": dialogue["case_group_id"],
            "source_split": split,
            "final_snapshot_id": final_snapshot["snapshot_id"],
            "effective_episode_states": final_snapshot["effective_episode_states"],
            "quality_state": final_snapshot["quality_state"],
            "nonstate_facts": [{
                "card_id": row["card_id"], "turn_index": row["turn_index"],
                "concept": row.get("canonical_concept"), "entity_type": row.get("entity_type"),
                "proposition": row.get("proposition"),
            } for row in nonstate_by_dialogue[dialogue_id]],
        })

        reference = references[dialogue_id]
        annotation_by_turn = {row["turn_id"]: row for row in reference["turn_annotations"]}
        turns = dialogue["turns"]
        for turn in turns:
            if turn["speaker"] != "doctor":
                continue
            target_index = turn["turn_index"]
            prior_turns = [row for row in turns if row["turn_index"] < target_index][-6:]
            input_snapshot = snapshot_lookup[(dialogue_id, target_index - 1)]
            unit_id = stable_id(dialogue_id, target_index, "doctor_action")
            units.append({
                "schema_version": "imcs21_doctor_action_prediction_unit_v1",
                "unit_id": unit_id,
                "dialogue_id": dialogue_id,
                "case_group_id": dialogue["case_group_id"],
                "source_split": split,
                "target_turn_index": target_index,
                "input_snapshot_id": input_snapshot["snapshot_id"],
                "input_snapshot_turn_index": input_snapshot["turn_index"],
                "input_snapshot": input_snapshot,
                "prior_context": [{"turn_index": row["turn_index"], "speaker": row["speaker"], "text": row["text"]}
                                  for row in prior_turns],
                "prior_context_text": "\n".join(f"{row['speaker']}: {row['text']}" for row in prior_turns),
            })
            annotation = annotation_by_turn[turn["turn_id"]]
            act = annotation["dialogue_act"]
            supervision.append({
                "schema_version": "imcs21_doctor_action_supervision_v1",
                "unit_id": unit_id,
                "dialogue_id": dialogue_id,
                "case_group_id": dialogue["case_group_id"],
                "source_split": split,
                "target_turn_index": target_index,
                "dialogue_act": act,
                "action_family": action_family(act),
                "request_gate": "REQUEST" if act.startswith("Request-") else "NON_REQUEST",
                "request_symptom_targets": annotation["symptom_norm"] if act == "Request-Symptom" else [],
                "request_symptom_target_available": bool(annotation["symptom_norm"]) if act == "Request-Symptom" else False,
            })

        for episode in episodes_by_dialogue[dialogue_id]:
            episode_traces = sorted((row for row in traces_by_dialogue[dialogue_id]
                                     if row["episode_id"] == episode["episode_id"]), key=lambda row: row["turn_index"])
            gaps = [row["distance_from_previous_evidence"] for row in episode_traces
                    if row["distance_from_previous_evidence"] is not None]
            coordinates.append({
                "schema_version": "imcs21_propagation_coordinate_v1",
                "episode_id": episode["episode_id"],
                "dialogue_id": dialogue_id,
                "case_group_id": dialogue["case_group_id"],
                "concept": episode["concept"],
                "first_turn": episode["first_turn"],
                "last_turn": episode["last_turn"],
                "evidence_count": episode["evidence_count"],
                "max_evidence_gap": max(gaps, default=0),
                "state_change_count": sum(row["state_changed"] for row in episode_traces),
                "trajectory": episode["trajectory"],
                "source_kinds": sorted({row["source_kind"] for row in episode_traces}),
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "matrix": OUTPUT / f"{split}_clinical_belief_matrix.jsonl",
        "snapshots": OUTPUT / f"{split}_turn_state_snapshots.jsonl",
        "coordinates": OUTPUT / f"{split}_propagation_coordinates.jsonl",
        "units": OUTPUT / f"{split}_doctor_action_prediction_units.jsonl",
        "supervision": OUTPUT / f"{split}_doctor_action_supervision.jsonl",
    }
    for name, rows in (("matrix", matrices), ("snapshots", snapshots), ("coordinates", coordinates),
                       ("units", units), ("supervision", supervision)):
        write_jsonl(paths[name], rows)

    supervision_ids = {row["unit_id"] for row in supervision}
    prediction_ids = {row["unit_id"] for row in units}
    no_time_leak = all(row["input_snapshot_turn_index"] < row["target_turn_index"] for row in units)
    no_label_keys = all(not ({"dialogue_act", "action_family", "request_gate", "request_symptom_targets"}
                             & set(row)) for row in units)
    unit_snapshots = [row["input_snapshot"] for row in units]
    action_counts = Counter(row["dialogue_act"] for row in supervision)
    family_counts = Counter(row["action_family"] for row in supervision)
    target_rows = [row for row in supervision if row["dialogue_act"] == "Request-Symptom"]
    audit = {
        "schema_version": "imcs21_clinical_belief_matrix_audit_v1",
        "split": split,
        "release_status": "PASS" if audit02["release_status"] == "PASS" and prediction_ids == supervision_ids
                                        and no_time_leak and no_label_keys else "FAIL",
        "dialogue_count": len(dialogues),
        "matrix_count": len(matrices),
        "snapshot_count": len(snapshots),
        "episode_count": len(episodes),
        "propagation_coordinate_count": len(coordinates),
        "doctor_action_unit_count": len(units),
        "unresolved_record_count": len(unresolved),
        "doctor_action_counts": dict(sorted(action_counts.items())),
        "action_family_counts": dict(sorted(family_counts.items())),
        "request_symptom_unit_count": len(target_rows),
        "request_symptom_target_available_count": sum(row["request_symptom_target_available"] for row in target_rows),
        "request_symptom_target_coverage": sum(row["request_symptom_target_available"] for row in target_rows) / max(len(target_rows), 1),
        "prediction_snapshot_empty_count": sum(row["active_state_count"] == 0 for row in unit_snapshots),
        "prediction_snapshot_empty_rate": sum(row["active_state_count"] == 0 for row in unit_snapshots) / max(len(unit_snapshots), 1),
        "mean_active_state_count": sum(row["active_state_count"] for row in unit_snapshots) / max(len(unit_snapshots), 1),
        "nli_transition_count": sum(row["source_kind"] == "proxy_calibrated_nli" for row in traces),
        "gates": {
            "upstream_02_pass": audit02["release_status"] == "PASS",
            "one_prediction_unit_per_supervision_row": prediction_ids == supervision_ids,
            "snapshot_precedes_target_turn": no_time_leak,
            "prediction_plane_has_no_supervision_keys": no_label_keys,
            "all_dialogues_have_reference_supervision": len(references) == len(dialogues),
        },
        "lineage_sha256": {
            "prediction_dialogues": sha256(CONTRACT / "prediction_dialogues.jsonl"),
            "reference_annotations_supervision_only": sha256(CONTRACT / "reference_annotations.jsonl"),
            "02_audit": sha256(audit02_path),
            "02_episodes": sha256(episodes_path),
            "02_traces": sha256(traces_path),
            "02_nonstate": sha256(nonstate_path),
            "02_unresolved": sha256(unresolved_path),
        },
        "prediction_contract": "all state and text inputs end before the target doctor turn; supervision is exported separately",
    }
    (OUTPUT / f"{split}_build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="all")
    args = parser.parse_args()
    dialogues = list(read_jsonl(CONTRACT / "prediction_dialogues.jsonl"))
    references = list(read_jsonl(CONTRACT / "reference_annotations.jsonl"))
    splits = ("train", "dev", "test") if args.split == "all" else (args.split,)
    summary = {split: build_split(split, dialogues, references) for split in splits}
    print(json.dumps({split: {key: value for key, value in audit.items()
                             if key in {"release_status", "doctor_action_unit_count",
                                        "prediction_snapshot_empty_rate", "request_symptom_target_coverage"}}
                      for split, audit in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
