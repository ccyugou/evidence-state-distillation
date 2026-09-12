from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "30_student_distillation_v1"
MODEL = os.environ.get("QWEN3_MODEL_PATH", str(ROOT / "resources" / "models" / "Qwen3-4B"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    splits = {name: read_jsonl(DATA / f"{name}_silver_sft.jsonl") for name in ("train", "dev")}
    report: dict = {"model": MODEL, "splits": {}}
    packet_sets = {name: {x["packet_id"] for x in rows} for name, rows in splits.items()}
    case_sets = {name: {x["case_group_id"] for x in rows} for name, rows in splits.items()}
    input_hashes = {name: {digest(x["input"]) for x in rows} for name, rows in splits.items()}
    for name, rows in splits.items():
        lengths, assistant_lengths, prefix_mismatch = [], [], 0
        labels = {field: Counter() for field in ("target_binding", "subject", "temporality", "existence_effect", "facet_effect")}
        for row in rows:
            prefix = tokenizer.apply_chat_template(
                row["messages"][:2], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"]
            full = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )["input_ids"]
            prefix_mismatch += full[:len(prefix)] != prefix
            lengths.append(len(full))
            assistant_lengths.append(len(full) - len(prefix))
            for field in labels:
                labels[field][row["target"][field]] += 1
        report["splits"][name] = {
            "rows": len(rows), "case_groups": len(case_sets[name]),
            "duplicate_packet_ids": len(rows) - len(packet_sets[name]),
            "duplicate_inputs": len(rows) - len(input_hashes[name]),
            "prefix_mismatch_count": prefix_mismatch,
            "tokens": {k: float(v) for k, v in zip(
                ("min", "p50", "p95", "p99", "max"),
                np.percentile(lengths, (0, 50, 95, 99, 100)),
            )},
            "assistant_tokens": {k: float(v) for k, v in zip(
                ("min", "p50", "p95", "p99", "max"),
                np.percentile(assistant_lengths, (0, 50, 95, 99, 100)),
            )},
            "over_2048": sum(x > 2048 for x in lengths),
            "label_counts": {field: dict(counts) for field, counts in labels.items()},
        }
    report["cross_split"] = {
        "packet_overlap": len(packet_sets["train"] & packet_sets["dev"]),
        "case_group_overlap": len(case_sets["train"] & case_sets["dev"]),
        "exact_input_overlap": len(input_hashes["train"] & input_hashes["dev"]),
    }
    report["pass"] = (
        all(report["splits"][x]["prefix_mismatch_count"] == 0 for x in splits)
        and all(report["splits"][x]["over_2048"] == 0 for x in splits)
        and not any(report["cross_split"].values())
        and all(report["splits"][x]["duplicate_inputs"] == 0 for x in splits)
    )
    path = DATA / "student_token_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
