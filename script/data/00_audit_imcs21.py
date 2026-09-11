from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


FULL_SPLITS = ("train", "dev", "test")
REQUIRED_FILES = ("train.json", "dev.json", "test.json", "test_input.json", "symptom_norm.csv")
REPORT_FIELDS = ("主诉", "现病史", "辅助检查", "既往史", "诊断", "建议")
VALID_SPEAKERS = {"医生", "患者"}
VALID_SYMPTOM_TYPES = {"0", "1", "2"}
VALID_DIALOGUE_ACTS = {
    "Request-Symptom",
    "Inform-Symptom",
    "Request-Etiology",
    "Inform-Etiology",
    "Request-Basic_Information",
    "Inform-Basic_Information",
    "Request-Existing_Examination_and_Treatment",
    "Inform-Existing_Examination_and_Treatment",
    "Request-Drug_Recommendation",
    "Inform-Drug_Recommendation",
    "Request-Medical_Advice",
    "Inform-Medical_Advice",
    "Request-Precautions",
    "Inform-Precautions",
    "Diagnose",
    "Other",
}
REVISION_LANGUAGE_RE = re.compile(
    r"其实|不对|说错|改口|后来|现在|目前|以前|之前|原来|已经|"
    r"好了|好转|缓解|减轻|消失|加重|越来越|更严重|又(?:开始|出现|犯|烧|咳|吐|拉)"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "00_dataset_contract")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1):
        pos = q * (len(ordered) - 1)
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        value = ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
        result[str(q)] = round(value, 3)
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_symptom_vocabulary(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return {line.strip() for line in lines[1:] if line.strip()}


def prediction_payload(sample: dict) -> dict:
    return {
        "self_report": sample["self_report"],
        "dialogue": [
            {"speaker": turn["speaker"], "sentence": turn["sentence"]}
            for turn in sample["dialogue"]
        ],
    }


def normalized_payload(sample: dict) -> dict:
    return {
        "self_report": normalize_text(sample["self_report"]),
        "dialogue": [
            {"speaker": turn["speaker"], "sentence": normalize_text(turn["sentence"])}
            for turn in sample["dialogue"]
        ],
    }


def collapse_track(events: list[dict]) -> list[dict]:
    collapsed = []
    for event in events:
        if not collapsed or collapsed[-1]["symptom_type"] != event["symptom_type"]:
            collapsed.append(event)
    return collapsed


def transition_rows(dialogue_id: str, tracks: dict[str, list[dict]], scope: str) -> list[dict]:
    rows = []
    for concept, events in tracks.items():
        collapsed = collapse_track(events)
        if len(collapsed) < 2:
            continue
        transitions = [
            {
                "from": left["symptom_type"],
                "to": right["symptom_type"],
                "turn_gap": right["turn_index"] - left["turn_index"],
            }
            for left, right in zip(collapsed, collapsed[1:])
        ]
        rows.append({
            "dialogue_id": dialogue_id,
            "scope": scope,
            "concept": concept,
            "events": events,
            "collapsed_states": [event["symptom_type"] for event in collapsed],
            "transitions": transitions,
            "contains_binary_reversal": {"0", "1"}.issubset({event["symptom_type"] for event in collapsed}),
            "max_turn_gap": max(item["turn_gap"] for item in transitions),
            "reference_semantics": "candidate_only; mention-level labels require contextual review",
        })
    return rows


def canonical_rows(split: str, source_id: str, sample: dict, payload_hash: str, normalized_hash: str) -> tuple[dict, dict]:
    dialogue_id = f"imcs21_{split}_{source_id}"
    first_diagnose = next(
        (i for i, turn in enumerate(sample["dialogue"]) if turn["dialogue_act"] == "Diagnose"),
        None,
    )
    prediction = {
        "schema_version": "imcs21_prediction_dialogue_v1",
        "dialogue_id": dialogue_id,
        "case_group_id": f"imcs21_case_{payload_hash[:20]}",
        "source_split": split,
        "source_id": source_id,
        "payload_hash": payload_hash,
        "normalized_payload_hash": normalized_hash,
        "self_report": sample["self_report"],
        "turns": [
            {
                "turn_id": f"{dialogue_id}_t{i:03d}",
                "turn_index": i,
                "source_sentence_id": turn["sentence_id"],
                "speaker": "doctor" if turn["speaker"] == "医生" else "patient",
                "text": turn["sentence"],
            }
            for i, turn in enumerate(sample["dialogue"])
        ],
        "diagnosis_disclosure_turn_index": first_diagnose,
        "prediction_contract": {
            "allowed": ["self_report", "turns.speaker", "turns.text", "turns.turn_index"],
            "forbidden": ["diagnosis", "explicit_info", "implicit_info", "report", "dialogue labels"],
        },
    }
    reference = {
        "schema_version": "imcs21_reference_annotations_v1",
        "dialogue_id": dialogue_id,
        "case_group_id": prediction["case_group_id"],
        "source_split": split,
        "source_id": source_id,
        "diagnosis": sample["diagnosis"],
        "explicit_info": sample["explicit_info"],
        "implicit_info": sample["implicit_info"],
        "turn_annotations": [
            {
                "turn_id": f"{dialogue_id}_t{i:03d}",
                "dialogue_act": turn["dialogue_act"],
                "BIO_label": turn["BIO_label"],
                "symptom_norm": turn["symptom_norm"],
                "symptom_type": turn["symptom_type"],
            }
            for i, turn in enumerate(sample["dialogue"])
        ],
        "reports": sample["report"],
        "reference_contract": "evaluation_and_supervision_only; never expose to online evidence extraction",
    }
    return prediction, reference


def audit_split(split: str, data: object, vocabulary: set[str]):
    errors, predictions, references = [], [], []
    revision_candidates, revision_language_candidates = [], []
    diagnosis_counts = Counter()
    speaker_counts = Counter()
    act_counts = Counter()
    speaker_act_counts = Counter()
    bio_type_counts = Counter()
    symptom_type_counts = Counter()
    symptom_type_by_speaker = Counter()
    symptom_vocab = Counter()
    action_targets = Counter()
    turn_counts, turn_lengths, self_report_lengths = [], [], []
    implicit_counts, explicit_counts = [], []
    bio_alignment_errors = 0
    symptom_alignment_errors = 0
    out_of_vocab_mentions = Counter()
    short_qa_answer_count = 0
    request_symptom_turns = 0
    request_without_target = 0
    first_diagnose_turns = []
    diagnosis_disclosed_count = 0
    same_turn_status_conflicts = 0
    patient_revision_count = 0
    all_revision_count = 0
    repeated_patient_track_count = 0
    repeated_patient_gap4_count = 0
    repeated_patient_track_with_revision_language_count = 0
    explicit_reference_vocab = set()
    implicit_reference_vocab = set()

    if not isinstance(data, dict):
        return ({}, [], [], [], [{"split": split, "error": "top_level_not_object"}])

    for source_id, sample in data.items():
        dialogue_id = f"imcs21_{split}_{source_id}"
        required = {"diagnosis", "self_report", "explicit_info", "dialogue", "report", "implicit_info"}
        missing = sorted(required - set(sample)) if isinstance(sample, dict) else sorted(required)
        if missing:
            errors.append({"dialogue_id": dialogue_id, "error": "missing_sample_fields", "fields": missing})
            continue

        diagnosis_counts[sample["diagnosis"]] += 1
        self_report_lengths.append(len(sample["self_report"]))
        turn_counts.append(len(sample["dialogue"]))
        explicit_symptoms = sample["explicit_info"].get("Symptom", [])
        implicit_symptoms = sample["implicit_info"].get("Symptom", {})
        explicit_reference_vocab.update(explicit_symptoms)
        implicit_reference_vocab.update(implicit_symptoms)
        explicit_counts.append(len(explicit_symptoms))
        implicit_counts.append(len(implicit_symptoms))

        if len(sample["report"]) != 2 or any(set(report) != set(REPORT_FIELDS) for report in sample["report"]):
            errors.append({"dialogue_id": dialogue_id, "error": "invalid_report_contract"})
        if not isinstance(implicit_symptoms, dict) or any(str(value) not in VALID_SYMPTOM_TYPES for value in implicit_symptoms.values()):
            errors.append({"dialogue_id": dialogue_id, "error": "invalid_implicit_symptom_contract"})

        all_tracks = defaultdict(list)
        patient_tracks = defaultdict(list)
        sentence_ids = []
        first_diagnose = None
        for turn_index, turn in enumerate(sample["dialogue"]):
            expected = {"sentence_id", "speaker", "sentence", "dialogue_act", "BIO_label", "symptom_norm", "symptom_type"}
            if set(turn) != expected:
                errors.append({
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "error": "unexpected_turn_schema",
                    "fields": sorted(turn),
                })
                continue
            sentence_ids.append(turn["sentence_id"])
            speaker_counts[turn["speaker"]] += 1
            act_counts[turn["dialogue_act"]] += 1
            speaker_act_counts[f"{turn['speaker']}|{turn['dialogue_act']}"] += 1
            turn_lengths.append(len(turn["sentence"]))
            if turn["speaker"] not in VALID_SPEAKERS:
                errors.append({"dialogue_id": dialogue_id, "turn_index": turn_index, "error": "invalid_speaker"})
            if turn["dialogue_act"] not in VALID_DIALOGUE_ACTS:
                errors.append({"dialogue_id": dialogue_id, "turn_index": turn_index, "error": "invalid_dialogue_act"})

            bio = turn["BIO_label"].split()
            if len(bio) != len(turn["sentence"]):
                bio_alignment_errors += 1
            for label in bio:
                if label != "O":
                    bio_type_counts[label.split("-", 1)[-1]] += int(label.startswith("B-"))
            if len(turn["symptom_norm"]) != len(turn["symptom_type"]):
                symptom_alignment_errors += 1
                continue

            per_turn = defaultdict(set)
            for concept, status in zip(turn["symptom_norm"], turn["symptom_type"]):
                status = str(status)
                symptom_vocab[concept] += 1
                symptom_type_counts[status] += 1
                symptom_type_by_speaker[f"{turn['speaker']}|{status}"] += 1
                per_turn[concept].add(status)
                if concept not in vocabulary:
                    out_of_vocab_mentions[concept] += 1
                event = {
                    "turn_index": turn_index,
                    "sentence_id": turn["sentence_id"],
                    "speaker": turn["speaker"],
                    "dialogue_act": turn["dialogue_act"],
                    "symptom_type": status,
                    "text": turn["sentence"],
                }
                all_tracks[concept].append(event)
                if turn["speaker"] == "患者":
                    patient_tracks[concept].append(event)
            same_turn_status_conflicts += sum(len(states) > 1 for states in per_turn.values())

            if turn["dialogue_act"] == "Request-Symptom":
                request_symptom_turns += 1
                targets = list(dict.fromkeys(turn["symptom_norm"]))
                if not targets:
                    request_without_target += 1
                action_targets.update(targets)
            if turn["dialogue_act"] == "Diagnose" and first_diagnose is None:
                first_diagnose = turn_index

            if turn["speaker"] == "患者" and len(turn["sentence"].strip()) <= 8 and not turn["symptom_norm"] and turn_index:
                previous = sample["dialogue"][turn_index - 1]
                if previous["speaker"] == "医生" and previous["dialogue_act"] == "Request-Symptom":
                    short_qa_answer_count += 1

        if len(sentence_ids) != len(set(sentence_ids)):
            errors.append({"dialogue_id": dialogue_id, "error": "duplicate_sentence_id"})
        if first_diagnose is not None:
            first_diagnose_turns.append(first_diagnose)
            diagnosis_disclosed_count += 1

        all_rows = transition_rows(dialogue_id, all_tracks, "all_speakers")
        patient_rows = transition_rows(dialogue_id, patient_tracks, "patient_only")
        all_revision_count += len(all_rows)
        patient_revision_count += len(patient_rows)
        revision_candidates.extend(all_rows)
        revision_candidates.extend(patient_rows)
        for concept, events in patient_tracks.items():
            distinct_turns = sorted({event["turn_index"] for event in events})
            if len(distinct_turns) < 2:
                continue
            repeated_patient_track_count += 1
            gap = distinct_turns[-1] - distinct_turns[0]
            repeated_patient_gap4_count += int(gap >= 4)
            cue_events = [event for event in events if REVISION_LANGUAGE_RE.search(event["text"])]
            if cue_events:
                repeated_patient_track_with_revision_language_count += 1
                revision_language_candidates.append({
                    "dialogue_id": dialogue_id,
                    "concept": concept,
                    "events": events,
                    "cue_turn_indices": [event["turn_index"] for event in cue_events],
                    "turn_span": gap,
                    "observed_reference_states": sorted({event["symptom_type"] for event in events}),
                    "review_contract": "raw-language candidate; requires proposition-level contextual resolution",
                })

        payload = prediction_payload(sample)
        payload_hash = stable_hash(payload)
        normalized_hash = stable_hash(normalized_payload(sample))
        prediction, reference = canonical_rows(split, source_id, sample, payload_hash, normalized_hash)
        predictions.append(prediction)
        references.append(reference)

    summary = {
        "case_count": len(predictions),
        "diagnosis_counts": dict(diagnosis_counts),
        "speaker_counts": dict(speaker_counts),
        "dialogue_act_counts": dict(act_counts),
        "speaker_dialogue_act_counts": dict(speaker_act_counts),
        "turn_count_quantiles": quantiles(turn_counts),
        "turn_length_quantiles": quantiles(turn_lengths),
        "self_report_length_quantiles": quantiles(self_report_lengths),
        "explicit_symptom_count_quantiles": quantiles(explicit_counts),
        "implicit_symptom_count_quantiles": quantiles(implicit_counts),
        "implicit_matrix_density_mean": round(statistics.mean(implicit_counts) / max(len(vocabulary), 1), 6),
        "bio_entity_counts": dict(bio_type_counts),
        "bio_character_alignment_error_count": bio_alignment_errors,
        "symptom_norm_type_alignment_error_count": symptom_alignment_errors,
        "symptom_mention_count": sum(symptom_vocab.values()),
        "symptom_vocabulary_size": len(symptom_vocab),
        "explicit_reference_vocabulary_size": len(explicit_reference_vocab),
        "implicit_reference_vocabulary_size": len(implicit_reference_vocab),
        "full_reference_vocabulary_out_of_dictionary": sorted(
            (explicit_reference_vocab | implicit_reference_vocab) - vocabulary
        ),
        "symptom_type_counts": dict(symptom_type_counts),
        "symptom_type_by_speaker": dict(symptom_type_by_speaker),
        "out_of_dictionary_mention_count": sum(out_of_vocab_mentions.values()),
        "out_of_dictionary_concepts": dict(out_of_vocab_mentions.most_common()),
        "same_turn_same_concept_status_conflict_count": same_turn_status_conflicts,
        "request_symptom_turn_count": request_symptom_turns,
        "request_symptom_without_normalized_target_count": request_without_target,
        "request_target_vocabulary_size": len(action_targets),
        "top_request_targets": dict(action_targets.most_common(30)),
        "short_elliptical_qa_answer_count": short_qa_answer_count,
        "diagnosis_disclosed_in_dialogue_count": diagnosis_disclosed_count,
        "first_diagnose_turn_quantiles": quantiles(first_diagnose_turns),
        "all_speaker_transition_track_count": all_revision_count,
        "patient_only_transition_track_count": patient_revision_count,
        "repeated_patient_concept_track_count": repeated_patient_track_count,
        "repeated_patient_concept_gap4_count": repeated_patient_gap4_count,
        "repeated_patient_track_with_revision_language_count": repeated_patient_track_with_revision_language_count,
        "error_count": len(errors),
    }
    return summary, predictions, references, revision_candidates, revision_language_candidates, errors


def duplicate_groups(predictions: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in predictions:
        groups[row[key]].append({
            "dialogue_id": row["dialogue_id"],
            "source_split": row["source_split"],
            "source_id": row["source_id"],
        })
    return [
        {
            "hash_type": key,
            "hash": value,
            "members": members,
            "cross_split": len({member["source_split"] for member in members}) > 1,
        }
        for value, members in groups.items()
        if len(members) > 1
    ]


def compare_test_projection(test: dict, test_input: dict) -> dict:
    test_ids, input_ids = set(test), set(test_input)
    mismatches = []
    for source_id in sorted(test_ids & input_ids):
        labeled = test[source_id]
        unlabeled = test_input[source_id]
        labeled_raw = {
            "self_report": labeled["self_report"],
            "explicit_info": labeled["explicit_info"],
            "dialogue": [
                {"sentence_id": turn["sentence_id"], "speaker": turn["speaker"], "sentence": turn["sentence"]}
                for turn in labeled["dialogue"]
            ],
        }
        if labeled_raw != unlabeled:
            mismatches.append(source_id)
    return {
        "labeled_test_count": len(test_ids),
        "test_input_count": len(input_ids),
        "missing_from_test_input": sorted(test_ids - input_ids),
        "extra_in_test_input": sorted(input_ids - test_ids),
        "raw_projection_mismatch_count": len(mismatches),
        "raw_projection_mismatch_ids": mismatches[:50],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_FILES if not (args.data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    vocabulary = read_symptom_vocabulary(args.data_dir / "symptom_norm.csv")
    datasets = {split: load_json(args.data_dir / f"{split}.json") for split in FULL_SPLITS}
    test_input = load_json(args.data_dir / "test_input.json")

    summaries = {}
    predictions, references, revisions, revision_language, errors = [], [], [], [], []
    for split in FULL_SPLITS:
        summary, split_predictions, split_references, split_revisions, split_revision_language, split_errors = audit_split(
            split, datasets[split], vocabulary
        )
        summaries[split] = summary
        predictions.extend(split_predictions)
        references.extend(split_references)
        revisions.extend(split_revisions)
        revision_language.extend(split_revision_language)
        errors.extend(split_errors)

    exact_duplicates = duplicate_groups(predictions, "payload_hash")
    normalized_duplicates = duplicate_groups(predictions, "normalized_payload_hash")
    duplicates = exact_duplicates + normalized_duplicates
    cross_split_exact = sum(row["cross_split"] for row in exact_duplicates)
    cross_split_normalized = sum(row["cross_split"] for row in normalized_duplicates)
    projection = compare_test_projection(datasets["test"], test_input)
    patient_revisions = [row for row in revisions if row["scope"] == "patient_only"]
    binary_patient_reversals = [row for row in patient_revisions if row["contains_binary_reversal"]]

    file_manifest = {
        name: {
            "size_bytes": (args.data_dir / name).stat().st_size,
            "sha256": sha256_file(args.data_dir / name),
        }
        for name in REQUIRED_FILES
    }
    audit = {
        "schema_version": "imcs21_dataset_contract_audit_v1",
        "source": "IMCS-21 official GitHub JSON release",
        "file_manifest": file_manifest,
        "symptom_dictionary_size": len(vocabulary),
        "splits": summaries,
        "totals": {
            "case_count": len(predictions),
            "turn_count": sum(sum(item["speaker_counts"].values()) for item in summaries.values()),
            "reference_error_count": len(errors),
            "exact_duplicate_group_count": len(exact_duplicates),
            "normalized_duplicate_group_count": len(normalized_duplicates),
            "cross_split_exact_duplicate_group_count": cross_split_exact,
            "cross_split_normalized_duplicate_group_count": cross_split_normalized,
            "revision_candidate_count": len(revisions),
            "patient_only_revision_candidate_count": len(patient_revisions),
            "patient_only_binary_reversal_candidate_count": len(binary_patient_reversals),
            "raw_revision_language_candidate_count": len(revision_language),
        },
        "test_projection_audit": projection,
        "critical_semantic_contracts": {
            "dialogue_symptom_labels": "reference annotations, not speaker assertions",
            "doctor_question_labels": "must not create positive or negative online evidence",
            "local_implicit_info": "not present in this downloaded release",
            "explicit_info": "annotation-derived reference; excluded from prediction plane",
            "diagnosis_and_reports": "targets only; forbidden from online inputs",
            "natural_revision_candidates": "weak candidates requiring contextual validation",
            "symptom_type_temporal_scope": "corpus-level patient-symptom relation, not a dynamic per-turn state target",
        },
        "release_gates": {
            "required_files_present": True,
            "full_split_schema_valid": len(errors) == 0,
            "test_input_matches_labeled_test_raw_projection": projection["raw_projection_mismatch_count"] == 0
            and not projection["missing_from_test_input"]
            and not projection["extra_in_test_input"],
            "cross_split_exact_payload_isolation": cross_split_exact == 0,
            "cross_split_normalized_payload_isolation": cross_split_normalized == 0,
            "prediction_reference_planes_separated": True,
        },
    }
    audit["release_status"] = "PASS" if all(audit["release_gates"].values()) else "REVIEW"

    write_json(args.output_dir / "audit_summary.json", audit)
    write_json(args.output_dir / "file_manifest.json", file_manifest)
    write_jsonl(args.output_dir / "prediction_dialogues.jsonl", predictions)
    write_jsonl(args.output_dir / "reference_annotations.jsonl", references)
    write_jsonl(args.output_dir / "natural_revision_candidates.jsonl", revisions)
    write_jsonl(args.output_dir / "raw_revision_language_candidates.jsonl", revision_language)
    write_jsonl(args.output_dir / "duplicate_groups.jsonl", duplicates)
    write_jsonl(args.output_dir / "audit_errors.jsonl", errors)
    print(json.dumps({
        "release_status": audit["release_status"],
        "cases": len(predictions),
        "turns": audit["totals"]["turn_count"],
        "errors": len(errors),
        "cross_split_exact_duplicates": cross_split_exact,
        "cross_split_normalized_duplicates": cross_split_normalized,
        "patient_revision_candidates": len(patient_revisions),
        "patient_binary_reversal_candidates": len(binary_patient_reversals),
        "raw_revision_language_candidates": len(revision_language),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
