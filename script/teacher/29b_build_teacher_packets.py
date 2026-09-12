from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scripts.common.imcs21_common import bio_spans, read_jsonl, sha256, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("IMCS21_DATA_DIR", ROOT / "data"))
OUTPUT = Path(os.environ.get(
    "EVIDENCE_STATE_TEACHER_OUTPUT",
    ROOT / "outputs" / "teacher_distillation",
))
SPLITS = ("train", "dev", "test")


def symptom_spans(turn: dict) -> list[dict]:
    labels = turn.get("BIO_label", "").split()
    if len(labels) != len(turn["sentence"]):
        return []
    return [x for x in bio_spans(turn["sentence"], labels) if x["entity_type"] == "Symptom"]


def target_surface(turn: dict) -> str | None:
    spans = symptom_spans(turn)
    return spans[0]["span_text"] if len(spans) == 1 else None


def previous_mentions(turns: list[dict], stop: int, concept: str) -> list[dict]:
    result = []
    for turn in turns[:stop]:
        if turn["speaker"] != "患者":
            continue
        for name, value in zip(turn.get("symptom_norm") or [], turn.get("symptom_type") or []):
            if name == concept:
                result.append({
                    "sentence_id": turn["sentence_id"], "text": turn["sentence"],
                    "official_status": {"0": "absent", "1": "present", "2": "uncertain"}.get(str(value), "unknown"),
                })
    return result[-3:]


def used_test_cases(paths: list[Path]) -> set[str]:
    used = set()
    for path in paths:
        for row in read_jsonl(path):
            source = row.get("source", {})
            case_id = source.get("source_case_id") or source.get("question_case_id")
            if case_id:
                used.add(str(case_id))
    return used


def build(split: str, excluded: set[str]) -> list[dict]:
    data = json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))
    rows = []
    for case_id, case in data.items():
        if case_id in excluded:
            continue
        turns = case["dialogue"]
        for index in range(len(turns) - 1):
            question, answer = turns[index:index + 2]
            concepts = question.get("symptom_norm") or []
            if not (
                question["speaker"] == "医生" and question["dialogue_act"] == "Request-Symptom"
                and answer["speaker"] == "患者" and answer["dialogue_act"] == "Inform-Symptom"
                and len(concepts) == 1
            ):
                continue
            surface = target_surface(question)
            if not surface:
                continue
            concept = concepts[0]
            rows.append({
                "schema_version": "imcs21_semantic_teacher_packet_v1",
                "packet_id": stable_id("semantic-packet", split, case_id, question["sentence_id"], answer["sentence_id"]),
                "source_split": split,
                "case_group_id": case_id,
                "question_sentence_id": question["sentence_id"],
                "answer_sentence_id": answer["sentence_id"],
                "input": {
                    "target_concept": concept,
                    "target_surface": surface,
                    "doctor_question": question["sentence"],
                    "patient_answer": answer["sentence"],
                    "previous_target_mentions": previous_mentions(turns, index, concept),
                    "local_context": [
                        {"speaker": x["speaker"], "text": x["sentence"], "sentence_id": x["sentence_id"]}
                        for x in turns[max(0, index - 4):index + 2]
                    ],
                },
                "official_auxiliary": {
                    "answer_symptom_norm": answer.get("symptom_norm") or [],
                    "answer_symptom_type": answer.get("symptom_type") or [],
                    "role": "dataset_annotation_diagnostic_not_operation_gold",
                },
                "authority": "NO_STATE_AUTHORITY_BEFORE_TEACHER_CONSENSUS",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-test-cases",
        action="append",
        default=[],
        type=Path,
        help="JSONL artifact containing previously viewed test case IDs; repeat as needed.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    excluded = used_test_cases(args.exclude_test_cases)
    partitions = {
        "teacher_train": build("train", set()),
        "teacher_dev": build("dev", set()),
        "blind_locked": build("test", excluded),
    }
    for name, rows in partitions.items():
        write_jsonl(OUTPUT / f"{name}_packets.jsonl", rows)
    protocol = {
        "schema_version": "imcs21_semantic_distillation_protocol_v1",
        "status": "BLIND_LOCKED_BEFORE_TEACHER_OR_STUDENT_RUNS",
        "research_claim": "Context-Sufficiency-Aware Candidate-Coupled State Revision with a Distribution-Aware Write Gate",
        "packet_counts": {name: len(rows) for name, rows in partitions.items()},
        "case_counts": {name: len({x["case_group_id"] for x in rows}) for name, rows in partitions.items()},
        "excluded_previously_viewed_test_case_count": len(excluded),
        "blind_policy": "Do not inspect text, tune prompts, select models, or calibrate thresholds on blind_locked.",
        "teacher_policy": "Qwen3.5 two-order proposal plus independent DeepSeek adjudication; disagreement becomes HOLD.",
        "student_policy": "Train on teacher_train, tune/calibrate on teacher_dev, run blind_locked once after freeze.",
        "forbidden": [
            "Lexical rules that assign facet, stance, temporality, subject, sufficiency, or operation.",
            "Using official symptom status as operation gold.",
            "Using blind labels or errors for prompt, architecture, or threshold changes.",
        ],
        "source_hashes": {split: sha256(DATA / f"{split}.json") for split in SPLITS},
        "blind_packet_hash": sha256(OUTPUT / "blind_locked_packets.jsonl"),
        "blind_case_commitment": stable_id(sorted({x["case_group_id"] for x in partitions["blind_locked"]}), length=64),
        "natural_distribution": {
            name: dict(Counter(x["input"]["target_concept"] for x in rows).most_common(10))
            for name, rows in partitions.items() if name != "blind_locked"
        },
        "claim_boundary": "Teacher labels are proxy semantics; the locked test measures teacher-to-student transfer, not clinical correctness.",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in protocol.items() if k not in {"natural_distribution", "blind_case_commitment"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
