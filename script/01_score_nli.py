#!/usr/bin/env python3
"""Score stage-01 controlled hypotheses with a local NLI model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_label(value: str) -> str:
    lowered = value.lower()
    if "entail" in lowered:
        return "entailment"
    if "contra" in lowered:
        return "contradiction"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    id2label = {int(index): normalize_label(label) for index, label in model.config.id2label.items()}

    directions = []
    for row in rows:
        directions.append((row["pair_id"], "chosen", row["premise"], row["chosen_hypothesis"], None))
        for index, hypothesis in enumerate(row.get("contrast_hypotheses", [])):
            directions.append((row["pair_id"], "contrast", row["premise"], hypothesis, index))

    score_map: dict[tuple[str, str, int | None], dict[str, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(directions), args.batch_size):
            batch = directions[start:start + args.batch_size]
            encoded = tokenizer(
                [item[2] for item in batch],
                [item[3] for item in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            probabilities = model(**encoded).logits.softmax(dim=-1).cpu().tolist()
            for item, vector in zip(batch, probabilities):
                score_map[(item[0], item[1], item[4])] = {
                    id2label[index]: float(value) for index, value in enumerate(vector)
                }
            print(f"NLI scoring {min(start + args.batch_size, len(directions))}/{len(directions)}", flush=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            result = dict(row)
            result["chosen_scores"] = score_map[(row["pair_id"], "chosen", None)]
            result["contrast_scores"] = [
                {"hypothesis": hypothesis, "scores": score_map[(row["pair_id"], "contrast", index)]}
                for index, hypothesis in enumerate(row.get("contrast_hypotheses", []))
            ]
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

