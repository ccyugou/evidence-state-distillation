from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from script.common.imcs21_common import ENTITY_TYPES, bio_spans, entity_prf, read_jsonl, sha256, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "outputs" / "00_dataset_contract"
MODEL = ROOT / "resources" / "models" / "imcs21-roberta-ner"
OUTPUT = ROOT / "outputs" / "01_evidence_cards"
NEGATION = re.compile(r"没有|沒有|没(?:有)?|沒(?:有)?|无|無|未|不(?:是|会|再|怎么|太)?|否认|否認")
HISTORY = re.compile(r"以前|之前|既往|过去|小时候|曾经|原来|上次|此前|先前|早些时候|前段时间|前阵子|去年|昨天|昨晚|前天|上周|上个月|前几天|前(?:\d+|[一二两三四五六七八九十]+)天|(?:\d+|[一二两三四五六七八九十]+)(?:个)?(?:月|星期|周|年)(?:前|的时候|时候)|\d+月底|有过(?:一|1)?次")
UNCERTAIN = re.compile(r"可能|好像|似乎|大概|怀疑|不确定|不知道|说不清|也许|想想|担心|就怕|怕(?:得|是)?|像(?:是)?|应该|恐怕")
REVISION = re.compile(r"其实|不对|说错|改口|后来|现在|目前|已经|好了|好转|缓解|减轻|减少|变少|消失|加重|越来越|又")
DISJUNCTION = re.compile(r"还是|或者|要么|不是.+就是")
QUESTION = re.compile(r"吗|呢|吧|是否|有没有|会不会|怎么|什么|为何|为什么|\?|？")
FACET_CUE = re.compile(r"严重|厉害|轻微|一点|偶尔|有时|一直|阵发|次数|频繁|每次|每天|白天|晚上|夜里|持续|多久|几天|小时|影响")
SUBJECT_CUE = re.compile(r"爸爸|妈妈|父母|爸妈|宝爸|宝妈|父亲|母亲|爷爷|奶奶|姥姥|姥爷|外公|外婆|舅舅|姑姑|阿姨|叔叔|哥哥|姐姐|弟弟|妹妹|家人|家长|大人|他爸|她爸|他妈|她妈")
ATTRIBUTED_DIAGNOSIS = re.compile(r"医生说|大夫说|医院.{0,8}(?:说|诊断)|诊断为|考虑(?:是|为)")
TREATMENT_INDICATION = re.compile(r"(?:治疗|治).{0,8}(?:药|的)")
SHORT_ANSWER = re.compile(r"(?:是|有|没有|没|不是|不|嗯|对|是的|有的|没有的|不清楚|不知道)[。！!？?]?$")


def load_dialogues(split: str, limit: int | None) -> list[dict]:
    rows = [row for row in read_jsonl(CONTRACT / "prediction_dialogues.jsonl") if row["source_split"] == split]
    return rows[:limit] if limit else rows


def gold_symptom_aliases() -> tuple[dict[str, str], dict[str, list[str]]]:
    train = json.loads((ROOT / "train.json").read_text(encoding="utf-8"))
    votes = defaultdict(Counter)
    for sample in train.values():
        for turn in sample["dialogue"]:
            spans = [row for row in bio_spans(turn["sentence"], turn["BIO_label"].split()) if row["entity_type"] == "Symptom"]
            if len(spans) != len(turn["symptom_norm"]):
                continue
            for span, concept in zip(spans, turn["symptom_norm"]):
                votes[span["span_text"].strip()][concept] += 1
    exact, ambiguous = {}, {}
    for surface, counts in votes.items():
        ordered = counts.most_common()
        if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
            exact[surface] = ordered[0][0]
        else:
            ambiguous[surface] = [concept for concept, count in ordered if count == ordered[0][1]]
    return exact, ambiguous


def char_labels(
    text: str, offsets: list[tuple[int, int]], token_ids: list[int], token_scores: list[float], id2label: dict[int, str]
) -> tuple[list[str], list[float]]:
    labels = ["O"] * len(text)
    scores = [0.0] * len(text)
    for (start, end), label_id, score in zip(offsets, token_ids, token_scores):
        if end <= start or start >= len(text):
            continue
        label = id2label[int(label_id)]
        if label == "O":
            continue
        prefix, kind = label.split("-", 1)
        for index in range(start, min(end, len(text))):
            labels[index] = f"{prefix if index == start else 'I'}-{kind}"
            scores[index] = score
    for index, label in enumerate(labels):
        if label.startswith("I-") and (index == 0 or labels[index - 1][2:] != label[2:]):
            labels[index] = "B-" + label[2:]
    return labels, scores


def infer(rows: list[dict], tokenizer, model, batch_size: int) -> list[tuple[list[str], list[float]]]:
    predictions = []
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch], padding=True, truncation=True, max_length=512,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        probabilities = logits.softmax(-1)
        token_scores, token_ids = probabilities.max(-1)
        for row, row_offsets, row_ids, row_scores in zip(batch, offsets, token_ids.cpu().tolist(), token_scores.cpu().tolist()):
            predictions.append(char_labels(row["text"], row_offsets, row_ids, row_scores, model.config.id2label))
    return predictions


def candidate_suggestions(surface: str, aliases: dict[str, str], limit: int = 3) -> list[dict]:
    scored = sorted(
        ((SequenceMatcher(None, surface, alias).ratio(), alias, concept) for alias, concept in aliases.items()),
        reverse=True,
    )[:limit]
    return [{"surface": alias, "concept": concept, "similarity": round(score, 4)} for score, alias, concept in scored]


def review_reasons(text: str, speaker: str, entity_type: str, normalized: bool, confidence: float) -> list[str]:
    reasons = []
    if speaker == "doctor":
        reasons.append("doctor_context_only")
    if entity_type == "Symptom" and not normalized:
        reasons.append("unresolved_normalization")
    if confidence < 0.8:
        reasons.append("low_ner_confidence")
    if NEGATION.search(text):
        reasons.append("negation_scope")
    if HISTORY.search(text):
        reasons.append("temporality_scope")
    if UNCERTAIN.search(text):
        reasons.append("uncertain_proposition")
    if REVISION.search(text):
        reasons.append("revision_or_current_scope")
    if DISJUNCTION.search(text):
        reasons.append("disjunction_or_hypothesis")
    if speaker == "patient" and QUESTION.search(text):
        reasons.append("patient_question_or_hypothesis")
    if FACET_CUE.search(text):
        reasons.append("facet_resolution_required")
    if speaker == "patient" and SUBJECT_CUE.search(text):
        reasons.append("subject_resolution_required")
    if speaker == "patient" and ATTRIBUTED_DIAGNOSIS.search(text):
        reasons.append("attributed_diagnosis_requires_facet")
    if speaker == "patient" and TREATMENT_INDICATION.search(text):
        reasons.append("treatment_indication_scope")
    return reasons


def compile_split(split: str, limit: int | None, batch_size: int) -> tuple[dict, Path]:
    dialogues = load_dialogues(split, limit)
    exact_aliases, ambiguous_aliases = gold_symptom_aliases()
    print(f"loaded split={split} dialogues={len(dialogues)} aliases={len(exact_aliases)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print(f"loaded model device={next(model.parameters()).device}", flush=True)

    units = []
    for dialogue in dialogues:
        units.append({
            "dialogue_id": dialogue["dialogue_id"], "case_group_id": dialogue["case_group_id"],
            "turn_index": -1, "turn_id": f"{dialogue['dialogue_id']}_self_report",
            "speaker": "patient", "text": dialogue["self_report"], "source": "self_report",
        })
        units.extend({
            "dialogue_id": dialogue["dialogue_id"], "case_group_id": dialogue["case_group_id"],
            "turn_index": turn["turn_index"], "turn_id": turn["turn_id"],
            "speaker": turn["speaker"], "text": turn["text"], "source": "dialogue_turn",
        } for turn in dialogue["turns"])
    predictions = infer(units, tokenizer, model, batch_size)
    print(f"inferred units={len(units)}", flush=True)

    cards, bounded, normalization_queue = [], [], []
    suggestion_cache = {}
    cards_by_turn = defaultdict(list)
    unit_by_dialogue_turn = {(row["dialogue_id"], row["turn_index"]): row for row in units}
    for unit, (labels, scores) in zip(units, predictions):
        for span_index, span in enumerate(bio_spans(unit["text"], labels)):
            surface = span["span_text"].strip()
            confidence = sum(scores[span["char_start"]:span["char_end"]]) / max(span["char_end"] - span["char_start"], 1)
            concept = exact_aliases.get(surface) if span["entity_type"] == "Symptom" else surface
            normalized = concept is not None
            reasons = review_reasons(unit["text"], unit["speaker"], span["entity_type"], normalized, confidence)
            state_allowed = (
                unit["speaker"] == "patient" and span["entity_type"] == "Symptom"
                and normalized and not reasons and unit["source"] == "dialogue_turn"
            )
            card_id = stable_id("card", unit["turn_id"], span["char_start"], span["char_end"], span["entity_type"])
            card = {
                "schema_version": "imcs21_evidence_card_v1",
                "card_id": card_id, "dialogue_id": unit["dialogue_id"], "case_group_id": unit["case_group_id"],
                "turn_id": unit["turn_id"], "turn_index": unit["turn_index"], "speaker": unit["speaker"],
                "source": unit["source"], "text": unit["text"], **span,
                "extractor_confidence": round(confidence, 6),
                "canonical_concept": concept,
                "canonical_concept_source": "imcs21_train_alias_exact" if span["entity_type"] == "Symptom" and normalized else "surface_form",
                "proposition": {
                    "subject": "patient" if unit["speaker"] == "patient" else "unspecified_in_doctor_context",
                    "facet": "existence" if state_allowed else "unresolved",
                    "assertion": "positive" if state_allowed else "unresolved",
                    "temporality": "current" if state_allowed else "unresolved",
                },
                "evidence_policy": {
                    "context_only": unit["speaker"] == "doctor",
                    "state_change_allowed": state_allowed,
                    "bounded_review_required": bool(reasons) and unit["speaker"] == "patient",
                    "review_reasons": reasons,
                },
            }
            cards.append(card)
            cards_by_turn[(unit["dialogue_id"], unit["turn_index"])].append(card)
            if span["entity_type"] == "Symptom" and not normalized:
                suggestion_cache.setdefault(surface, candidate_suggestions(surface, exact_aliases))
                normalization_queue.append({
                    "card_id": card_id, "dialogue_id": unit["dialogue_id"], "turn_index": unit["turn_index"],
                    "surface": surface, "suggestions": suggestion_cache[surface],
                    "automatic_state_change": False,
                })
            if reasons and unit["speaker"] == "patient":
                bounded.append({
                    "review_id": stable_id("review", card_id), "review_type": "faceted_proposition",
                    "card_id": card_id, "dialogue_id": unit["dialogue_id"], "turn_index": unit["turn_index"],
                    "anchor_span": surface, "canonical_concept": concept, "entity_type": span["entity_type"],
                    "context": [
                        unit_by_dialogue_turn[(unit["dialogue_id"], index)]
                        for index in range(max(-1, unit["turn_index"] - 1), unit["turn_index"] + 1)
                        if (unit["dialogue_id"], index) in unit_by_dialogue_turn
                    ],
                    "requested_fields": ["facet", "assertion", "temporality", "subject", "revision_operation"],
                    "reasons": reasons, "automatic_state_change": False,
                })

    grouped_reviews = defaultdict(list)
    for row in bounded:
        grouped_reviews[(row["dialogue_id"], row["turn_index"])].append(row)
    bounded_candidate_count = len(bounded)
    bounded = [{
        "review_id": stable_id("turn-review", dialogue_id, turn_index),
        "review_type": "faceted_proposition_packet", "dialogue_id": dialogue_id, "turn_index": turn_index,
        "anchors": [
            {"card_id": row["card_id"], "span": row["anchor_span"], "canonical_concept": row["canonical_concept"], "entity_type": row["entity_type"]}
            for row in group
        ],
        "context": group[0]["context"], "requested_fields": group[0]["requested_fields"],
        "reasons": sorted({reason for row in group for reason in row["reasons"]}),
        "automatic_state_change": False,
    } for (dialogue_id, turn_index), group in grouped_reviews.items()]

    for dialogue in dialogues:
        turns = dialogue["turns"]
        for index, turn in enumerate(turns[1:], 1):
            previous = turns[index - 1]
            if turn["speaker"] != "patient" or previous["speaker"] != "doctor" or not SHORT_ANSWER.fullmatch(turn["text"].strip()):
                continue
            anchors = [card for card in cards_by_turn[(dialogue["dialogue_id"], previous["turn_index"])] if card["entity_type"] == "Symptom"]
            if not anchors:
                continue
            bounded.append({
                "review_id": stable_id("qa-review", turn["turn_id"]), "review_type": "qa_ellipsis",
                "dialogue_id": dialogue["dialogue_id"], "turn_index": turn["turn_index"],
                "answer_text": turn["text"],
                "candidate_anchors": [{"card_id": card["card_id"], "surface": card["span_text"], "canonical_concept": card["canonical_concept"]} for card in anchors],
                "context": [previous, turn],
                "requested_fields": ["target_concept", "facet", "assertion", "temporality", "subject"],
                "reasons": ["qa_ellipsis_requires_joint_context"], "automatic_state_change": False,
            })

    prefix = split if not limit else f"{split}_smoke{limit}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cards_path = OUTPUT / f"{prefix}_evidence_cards.jsonl"
    review_groups = {
        "state_authorized": [card for card in cards if card["evidence_policy"]["state_change_allowed"]],
        "doctor_context": [card for card in cards if card["speaker"] == "doctor"],
        "bounded_patient": [card for card in cards if card["evidence_policy"]["bounded_review_required"]],
        "unresolved_normalization": [card for card in cards if "unresolved_normalization" in card["evidence_policy"]["review_reasons"]],
    }
    medical_review = [
        {"review_stratum": name, **card}
        for name, group in review_groups.items()
        for card in sorted(group, key=lambda row: row["card_id"])[:25]
    ]
    write_jsonl(cards_path, cards)
    write_jsonl(OUTPUT / f"{prefix}_bounded_review_queue.jsonl", bounded)
    write_jsonl(OUTPUT / f"{prefix}_normalization_review_queue.jsonl", normalization_queue)
    write_jsonl(OUTPUT / f"{prefix}_medical_review_sample.jsonl", medical_review)
    card_ids = [card["card_id"] for card in cards]
    gates = {
        "unique_card_ids": len(card_ids) == len(set(card_ids)),
        "all_spans_match_source_text": all(card["text"][card["char_start"]:card["char_end"]] == card["span_text"] for card in cards),
        "doctor_never_changes_state": not any(card["speaker"] == "doctor" and card["evidence_policy"]["state_change_allowed"] for card in cards),
        "review_queue_never_changes_state": not any(row["automatic_state_change"] for row in bounded),
    }
    audit = {
        "schema_version": "imcs21_evidence_compiler_audit_v1", "split": split,
        "dialogue_count": len(dialogues), "inference_unit_count": len(units), "card_count": len(cards),
        "card_type_counts": dict(Counter(card["entity_type"] for card in cards)),
        "state_authorized_count": sum(card["evidence_policy"]["state_change_allowed"] for card in cards),
        "doctor_state_authorized_count": sum(card["speaker"] == "doctor" and card["evidence_policy"]["state_change_allowed"] for card in cards),
        "bounded_review_packet_count": len(bounded), "bounded_review_candidate_count": bounded_candidate_count,
        "normalization_review_count": len(normalization_queue),
        "ambiguous_train_alias_count": len(ambiguous_aliases),
        "engineering_gates": gates,
        "generation_contract": "prediction_dialogues_only; no turn labels, diagnosis, implicit_info, or reports consumed",
        "lineage_sha256": {"prediction_dialogues": sha256(CONTRACT / "prediction_dialogues.jsonl"), "model": sha256(MODEL / "model.safetensors")},
    }
    (OUTPUT / f"{prefix}_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit, cards_path


def evaluate(split: str, cards_path: Path, prefix: str, limit: int | None) -> dict:
    raw = json.loads((ROOT / f"{split}.json").read_text(encoding="utf-8"))
    selected_dialogues = {row["dialogue_id"] for row in load_dialogues(split, limit)}
    predicted_by_turn = defaultdict(list)
    for card in read_jsonl(cards_path):
        if card["source"] == "dialogue_turn":
            predicted_by_turn[(card["dialogue_id"], card["turn_index"])].append(card)
    gold_labels, predicted_labels = [], []
    normalization_total = normalization_correct = 0
    for source_id, sample in raw.items():
        dialogue_id = f"imcs21_{split}_{source_id}"
        if dialogue_id not in selected_dialogues:
            continue
        for turn_index, turn in enumerate(sample["dialogue"]):
            gold = turn["BIO_label"].split()
            predicted = ["O"] * len(turn["sentence"])
            for card in predicted_by_turn[(dialogue_id, turn_index)]:
                for index in range(card["char_start"], card["char_end"]):
                    predicted[index] = ("B-" if index == card["char_start"] else "I-") + card["entity_type"]
            gold_labels.append(gold)
            predicted_labels.append(predicted)
            gold_symptoms = [row for row in bio_spans(turn["sentence"], gold) if row["entity_type"] == "Symptom"]
            if len(gold_symptoms) != len(turn["symptom_norm"]):
                continue
            gold_map = {(row["char_start"], row["char_end"]): concept for row, concept in zip(gold_symptoms, turn["symptom_norm"])}
            for card in predicted_by_turn[(dialogue_id, turn_index)]:
                key = (card["char_start"], card["char_end"])
                if card["entity_type"] == "Symptom" and key in gold_map and card["canonical_concept"]:
                    normalization_total += 1
                    normalization_correct += card["canonical_concept"] == gold_map[key]
    metrics = entity_prf(gold_labels, predicted_labels)
    metrics["normalization_accuracy_on_exact_true_positive"] = normalization_correct / max(normalization_total, 1)
    metrics["normalization_evaluable_count"] = normalization_total
    (OUTPUT / f"{prefix}_reference_evaluation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--limit-dialogues", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    prefix = args.split if not args.limit_dialogues else f"{args.split}_smoke{args.limit_dialogues}"
    audit, cards_path = compile_split(args.split, args.limit_dialogues, args.batch_size)
    metrics = evaluate(args.split, cards_path, prefix, args.limit_dialogues)
    audit["reference_evaluation"] = metrics
    audit["release_status"] = "PASS" if all(audit["engineering_gates"].values()) and metrics["f1"] >= 0.75 and metrics["normalization_accuracy_on_exact_true_positive"] >= 0.98 else "FAIL"
    (OUTPUT / f"{prefix}_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
