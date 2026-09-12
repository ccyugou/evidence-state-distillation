from __future__ import annotations

import hashlib
import json
from collections import Counter

from scripts.common.semantic_distillation_common import OUTPUT, read_jsonl


CANONICAL = (
    "teacher_train_adaptive_large_teacher_final.jsonl",
    "teacher_dev_adaptive_large_teacher_final.jsonl",
    "teacher_train_large_teacher_gate.json",
    "large_teacher_gate_amendment.json",
    "blind_locked_packets.jsonl",
    "protocol.json",
)


def digest(name: str) -> str:
    return hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()


def main() -> None:
    gates = {
        "train": json.loads((OUTPUT / "teacher_train_large_teacher_gate.json").read_text(encoding="utf-8")),
        "dev": json.loads((OUTPUT / "large_teacher_gate_amendment.json").read_text(encoding="utf-8")),
    }
    finals = {
        split: read_jsonl(OUTPUT / f"teacher_{split}_adaptive_large_teacher_final.jsonl")
        for split in ("train", "dev")
    }
    blind_outputs = [p.name for p in OUTPUT.glob("blind_locked_*") if p.name != "blind_locked_packets.jsonl"]
    routes = {split: dict(Counter(x["label_status"] for x in rows)) for split, rows in finals.items()}
    summary = {
        "schema_version": "imcs21_large_teacher_stage_v1",
        "status": "LARGE_TEACHER_LAYER_FROZEN" if all(x["gate_pass"] for x in gates.values()) and not blind_outputs else "BLOCKED",
        "canonical_artifacts": {name: {"sha256": digest(name)} for name in CANONICAL},
        "packet_counts": {split: len(rows) for split, rows in finals.items()},
        "routes": routes,
        "silver_coverage": {split: gates[split]["silver_coverage"] for split in gates},
        "hold_rate": {split: gates[split]["hold_rate"] for split in gates},
        "state_authority_rate": {split: gates[split]["state_authority_rate"] for split in gates},
        "train_dev_absolute_deltas": {
            key: abs(gates["train"][key] - gates["dev"][key])
            for key in ("silver_coverage", "hold_rate", "state_authority_rate")
        },
        "authorized_silver_contract_invalid_rate": {
            split: gates[split]["authorized_silver_contract_invalid_rate"] for split in gates
        },
        "blind_teacher_outputs": blind_outputs,
        "blind_status": "LOCKED_AND_UNTOUCHED" if not blind_outputs else "VIOLATION",
        "teacher_definition": "Qwen3.5 open-semantic proposals plus independent and meta DeepSeek adjudication; unresolved cases HOLD.",
        "claim_boundary": "Proxy/silver semantic supervision, not expert clinical gold.",
        "small_model_status": "DEFERRED_UNTIL_MODEL_SELECTION",
        "next_stage": "Choose/download the small model, distill on Train silver, calibrate on Dev, then run blind once after freeze.",
    }
    target = OUTPUT / "large_teacher_stage_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
