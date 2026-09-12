from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.common.imcs21_common import read_jsonl, sha256, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "outputs" / "01_evidence_cards"
CONTRACT = ROOT / "outputs" / "00_dataset_contract"
MODEL = Path(os.environ.get(
    "NLI_MODEL_PATH",
    ROOT / "resources" / "models" / "mdeberta-v3-base-mnli-xnli",
))
OUTPUT = ROOT / "outputs" / "01_nli_authority"
FACETS = ("existence", "severity", "duration", "frequency", "course", "trigger", "treatment_response")
FEATURES = (
    "positive_entail", "negative_entail", "historical_entail", "family_entail",
    "selected_entail", "selected_contradiction", "assertion_margin", "current_margin",
    "qa_ellipsis", "negation_trigger", "revision_trigger",
) + tuple(f"facet_{name}" for name in FACETS)
BLOCKING_REASONS = {
    "family_or_other_subject", "historical_context", "uncertain_or_hypothetical",
    "unresolved_normalization", "low_ner_confidence", "attributed_diagnosis",
    "treatment_indication_context",
}
SOURCE_HOLD_RE = re.compile(
    r"如果.{0,8}(?:是|有)|是不是|是否|这是.{0,16}还是|可能|应该|怀疑|"
    r"我想.{0,8}(?:可能|是)|以前|之前得过|上个月|[一二三四五六七八九十\d]+个月之前|"
    r"(?:大夫|医生|医院).{0,8}(?:说|诊断)|想吐|要吐"
)


def source_hold(text: str, span: str) -> bool:
    if SOURCE_HOLD_RE.search(text):
        return True
    token = re.escape(span)
    return bool(
        re.search(rf"前[一二三四五六七八九十\d]+(?:多)?天.{{0,10}}(?:有一次)?{token}", text)
        or re.search(rf"{token}.{{0,8}}(?:好了|已好|痊愈|恢复)", text)
    )


def high_value(packet: dict) -> bool:
    anchors = packet.get("anchors") or packet.get("candidate_anchors", [])
    return bool(anchors) and all(
        row.get("entity_type", "Symptom") == "Symptom" and row.get("canonical_concept") for row in anchors
    ) and not (set(packet["reasons"]) & BLOCKING_REASONS)


def facet_hint(text: str, reasons: list[str]) -> str:
    patterns = (
        ("course", r"好转|缓解|减轻|加重|越来越|又|复发|消失"),
        ("treatment_response", r"吃药|用药|治疗|输液|雾化|没效|无效|有效"),
        ("duration", r"持续|多久|几天|小时|从.+开始"),
        ("frequency", r"偶尔|有时|一直|每天|每次|次数|频繁|白天|晚上|夜里"),
        ("severity", r"严重|厉害|轻微|一点|影响|睡不着"),
        ("trigger", r"一.+就|之后|以后|因为|运动|吃完"),
    )
    for facet, pattern in patterns:
        if re.search(pattern, text):
            return facet
    return "existence" if "facet_resolution_required" not in reasons else "severity"


def reference_label(reference: dict, card: dict, concept: str) -> tuple[str | None, str | None]:
    for turn in reference["turn_annotations"]:
        if turn["turn_id"] != card["turn_id"]:
            continue
        labels = [label for name, label in zip(turn["symptom_norm"], turn["symptom_type"]) if name == concept]
        if labels:
            return min(labels, key=lambda label: {"1": 0, "0": 1, "2": 2}[label]), "turn_symptom_type"
    label = reference["implicit_info"].get("Symptom", {}).get(concept)
    return (label, "dialogue_implicit_info") if label is not None else (None, None)


def build_records(split: str) -> list[dict]:
    card_index = {row["card_id"]: row for row in read_jsonl(CARDS / f"{split}_evidence_cards.jsonl")}
    references = {
        row["dialogue_id"]: row for row in read_jsonl(CONTRACT / "reference_annotations.jsonl")
        if row["source_split"] == split
    }
    records = []
    for packet in read_jsonl(CARDS / f"{split}_bounded_review_queue.jsonl"):
        if not high_value(packet):
            continue
        context = "\n".join(f"{row['speaker']}: {row['text']}" for row in packet["context"])
        anchors = packet.get("anchors") or packet.get("candidate_anchors", [])
        for anchor in anchors:
            card = card_index[anchor["card_id"]]
            if card["speaker"] != "patient" or card["entity_type"] != "Symptom" or card["proposition"]["subject"] != "patient":
                continue
            if set(card["evidence_policy"]["review_reasons"]) & BLOCKING_REASONS:
                continue
            if source_hold(card["text"], card["span_text"]):
                continue
            concept = anchor["canonical_concept"]
            label, label_source = reference_label(references[packet["dialogue_id"]], card, concept)
            records.append({
                "candidate_id": stable_id("nli-candidate", packet["review_id"], card["card_id"]),
                "review_id": packet["review_id"], "review_type": packet["review_type"],
                "dialogue_id": packet["dialogue_id"], "case_group_id": card["case_group_id"],
                "turn_index": packet["turn_index"], "card_id": card["card_id"],
                "concept": concept, "premise": context, "source_text": card["text"],
                "source_span": card["span_text"], "facet_hint": facet_hint(context, packet["reasons"]),
                "reasons": packet["reasons"], "reference_label": label, "reference_source": label_source,
            })
    return records


def hypotheses(concept: str) -> list[str]:
    return [
        f"当前患儿有{concept}。",
        f"当前患儿没有{concept}。",
        f"患儿过去有过{concept}，这不是当前状态。",
        f"{concept}描述的是患儿家属，而不是患儿本人。",
    ]


def score_nli(records: list[dict], tokenizer, model, batch_size: int) -> None:
    pairs = [(row["premise"], hypothesis) for row in records for hypothesis in hypotheses(row["concept"])]
    probabilities = []
    device = next(model.parameters()).device
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        encoded = tokenizer([x[0] for x in batch], [x[1] for x in batch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        probabilities.extend(logits.softmax(-1).cpu().tolist())
    for index, row in enumerate(records):
        scores = probabilities[index * 4:(index + 1) * 4]
        row["nli"] = {
            name: {"entailment": score[0], "neutral": score[1], "contradiction": score[2]}
            for name, score in zip(("positive", "negative", "historical", "family"), scores)
        }


def vector(row: dict) -> list[float]:
    nli = row["nli"]
    selected = "positive" if nli["positive"]["entailment"] >= nli["negative"]["entailment"] else "negative"
    opposite = "negative" if selected == "positive" else "positive"
    values = [
        nli["positive"]["entailment"], nli["negative"]["entailment"],
        nli["historical"]["entailment"], nli["family"]["entailment"],
        nli[selected]["entailment"], nli[selected]["contradiction"],
        nli[selected]["entailment"] - nli[opposite]["entailment"],
        nli[selected]["entailment"] - max(nli["historical"]["entailment"], nli["family"]["entailment"]),
        float(row["review_type"] == "qa_ellipsis"),
        float("negation_scope" in row["reasons"]), float("revision_or_current_scope" in row["reasons"]),
    ]
    return values + [float(row["facet_hint"] == facet) for facet in FACETS]


def selected_assertion(row: dict) -> str:
    return "positive" if row["nli"]["positive"]["entailment"] >= row["nli"]["negative"]["entailment"] else "negative"


def calibration_target(row: dict) -> int | None:
    if row["reference_label"] not in {"0", "1", "2"}:
        return None
    expected = {"0": "negative", "1": "positive", "2": "abstain"}[row["reference_label"]]
    return int(selected_assertion(row) == expected)


def metrics(rows: list[dict], probabilities: np.ndarray, threshold: float) -> dict:
    targets = np.array([calibration_target(row) for row in rows])
    mask = np.array([target is not None for target in targets])
    targets = targets[mask].astype(int)
    predicted = probabilities[mask] >= threshold
    tp = int(((targets == 1) & predicted).sum())
    fp = int(((targets == 0) & predicted).sum())
    fn = int(((targets == 1) & ~predicted).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "support": int(mask.sum()), "authorized": int(predicted.sum()), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def choose_threshold(rows: list[dict], probabilities: np.ndarray, precision_floor: float) -> tuple[float, dict]:
    choices = []
    for threshold in np.linspace(0.05, 0.99, 189):
        result = metrics(rows, probabilities, float(threshold))
        if result["authorized"] and result["precision"] >= precision_floor:
            choices.append((result["recall"], result["f1"], -threshold, threshold, result))
    if not choices:
        return 1.0, metrics(rows, probabilities, 1.0)
    _, _, _, threshold, result = max(choices)
    return float(threshold), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--precision-floor", type=float, default=0.90)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    records = {split: build_records(split) for split in ("train", "dev", "test")}
    for split in records:
        score_nli(records[split], tokenizer, model, args.batch_size)

    train_rows = [row for row in records["train"] if calibration_target(row) is not None]
    train_x = np.array([vector(row) for row in train_rows])
    train_y = np.array([calibration_target(row) for row in train_rows])
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    classifier.fit(scaler.transform(train_x), train_y)

    probabilities = {
        split: classifier.predict_proba(scaler.transform(np.array([vector(row) for row in rows])))[:, 1]
        for split, rows in records.items()
    }
    row_positions = {row["candidate_id"]: index for index, row in enumerate(records["train"])}
    groups = np.array([row["case_group_id"] for row in train_rows])
    for train_index, valid_index in GroupKFold(5).split(train_x, train_y, groups):
        fold_scaler = StandardScaler().fit(train_x[train_index])
        fold_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        fold_model.fit(fold_scaler.transform(train_x[train_index]), train_y[train_index])
        fold_probabilities = fold_model.predict_proba(fold_scaler.transform(train_x[valid_index]))[:, 1]
        for row_index, probability in zip(valid_index, fold_probabilities):
            probabilities["train"][row_positions[train_rows[row_index]["candidate_id"]]] = probability
    threshold, dev_metrics = choose_threshold(records["dev"], probabilities["dev"], args.precision_floor)
    split_metrics = {split: metrics(rows, probabilities[split], threshold) for split, rows in records.items()}

    for split, rows in records.items():
        authorized = []
        scored = []
        for row, probability in zip(rows, probabilities[split]):
            assertion = selected_assertion(row)
            decision = "AUTHORIZED_EXISTENCE" if probability >= threshold else "ABSTAIN_REVIEW"
            scored.append({
                **{key: value for key, value in row.items() if not key.startswith("reference_")},
                "selected_assertion": assertion, "calibrated_accept_probability": float(probability),
                "calibration_threshold": threshold, "decision": decision,
                "facet_authority": False, "state_authority": decision == "AUTHORIZED_EXISTENCE",
            })
            if decision == "AUTHORIZED_EXISTENCE":
                authorized.append({
                    "schema_version": "imcs21_nli_authorized_proposition_v1",
                    "authorization_id": stable_id("nli-authority", row["candidate_id"]),
                    "candidate_id": row["candidate_id"], "review_id": row["review_id"],
                    "dialogue_id": row["dialogue_id"], "case_group_id": row["case_group_id"],
                    "turn_index": row["turn_index"], "card_id": row["card_id"], "concept": row["concept"],
                    "subject": "patient", "temporality": "current", "facet": "existence",
                    "assertion": assertion,
                    "operation": "ASSERT_EXISTENCE" if assertion == "positive" else "RETRACT_EXISTENCE",
                    "source_text": row["source_text"], "context_text": row["premise"],
                    "source_span": row["source_span"],
                    "nli_scores": row["nli"], "calibrated_accept_probability": float(probability),
                    "calibration_threshold": threshold, "state_authority": True,
                    "authority_scope": "proxy-calibrated current existence only; no facet or clinical-gold claim",
                })
        write_jsonl(OUTPUT / f"{split}_nli_scored_candidates.jsonl", scored)
        write_jsonl(OUTPUT / f"{split}_authorized_propositions.jsonl", authorized)
        eval_rows = [{
            "candidate_id": row["candidate_id"], "reference_label": row["reference_label"],
            "reference_source": row["reference_source"], "selected_assertion": selected_assertion(row),
            "calibrated_accept_probability": float(probability),
        } for row, probability in zip(rows, probabilities[split])]
        write_jsonl(OUTPUT / f"{split}_proxy_reference_evaluation.jsonl", eval_rows)

    artifact = {
        "schema_version": "imcs21_facet_aware_nli_calibrator_v1",
        "model": str(MODEL), "feature_names": FEATURES,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "precision_floor": args.precision_floor, "selected_threshold": threshold,
        "train_prediction_contract": "5-fold case-grouped OOF for proxy-labeled rows; final train model only for unlabeled rows",
        "dev_threshold_metrics": dev_metrics, "split_metrics": split_metrics,
        "record_counts": {split: len(rows) for split, rows in records.items()},
        "authorization_contract": {
            "existence": "proxy-calibrated authorization",
            "facet": "NLI/Qwen support only; no automatic existence reversal",
            "historical_family_uncertain": "abstain/context",
            "turn_revision_gold": "unavailable; not claimed",
        },
        "lineage_sha256": {
            "train_cards": sha256(CARDS / "train_evidence_cards.jsonl"),
            "dev_cards": sha256(CARDS / "dev_evidence_cards.jsonl"),
            "test_cards": sha256(CARDS / "test_evidence_cards.jsonl"),
            "references_calibration_only": sha256(CONTRACT / "reference_annotations.jsonl"),
        },
    }
    (OUTPUT / "nli_calibrator.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
