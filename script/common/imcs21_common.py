from __future__ import annotations

import hashlib
import json
from pathlib import Path


ENTITY_TYPES = ("Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation")
BIO_LABELS = ("O",) + tuple(f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: object, length: int = 24) -> str:
    value = "|".join(json.dumps(part, ensure_ascii=False, sort_keys=True) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bio_spans(text: str, labels: list[str]) -> list[dict]:
    spans = []
    start = 0
    while start < len(labels):
        label = labels[start]
        if not label.startswith("B-"):
            start += 1
            continue
        kind = label[2:]
        end = start + 1
        while end < len(labels) and labels[end] == f"I-{kind}":
            end += 1
        spans.append({
            "char_start": start,
            "char_end": end,
            "span_text": text[start:end],
            "entity_type": kind,
        })
        start = end
    return spans


def entity_prf(gold_sequences: list[list[str]], predicted_sequences: list[list[str]]) -> dict:
    gold = set()
    predicted = set()
    by_type = {}
    for sample_index, (gold_labels, predicted_labels) in enumerate(zip(gold_sequences, predicted_sequences)):
        for row in bio_spans(" " * len(gold_labels), gold_labels):
            gold.add((sample_index, row["char_start"], row["char_end"], row["entity_type"]))
        for row in bio_spans(" " * len(predicted_labels), predicted_labels):
            predicted.add((sample_index, row["char_start"], row["char_end"], row["entity_type"]))
    for kind in ENTITY_TYPES:
        kind_gold = {row for row in gold if row[-1] == kind}
        kind_predicted = {row for row in predicted if row[-1] == kind}
        true_positive = len(kind_gold & kind_predicted)
        precision = true_positive / max(len(kind_predicted), 1)
        recall = true_positive / max(len(kind_gold), 1)
        by_type[kind] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "support": len(kind_gold),
        }
    true_positive = len(gold & predicted)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(gold), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "by_type": by_type,
    }
