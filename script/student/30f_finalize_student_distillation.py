from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get(
    "EVIDENCE_STATE_STUDENT_OUTPUT",
    ROOT / "outputs" / "student_distillation",
))
ADAPTER = OUTPUT / "qwen3_4b_qlora_v1" / "adapter"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifacts = [
        OUTPUT / "train_silver_sft.jsonl", OUTPUT / "dev_silver_sft.jsonl",
        OUTPUT / "preflight_audit.json", OUTPUT / "student_token_audit.json",
        OUTPUT / "dev_qwen3_4b_zero_shot_v2_metrics.json",
        OUTPUT / "dev_qwen3_4b_qlora_v1_metrics.json",
        OUTPUT / "qwen3_4b_distillation_comparison.json",
        OUTPUT / "qwen3_4b_qlora_v1" / "train_summary.json",
        ADAPTER / "adapter_config.json", ADAPTER / "adapter_model.safetensors",
    ]
    scripts = [ROOT / "scripts" / "student" / name for name in (
        "30a_audit_and_build_student_data.py", "30b_run_qwen3_4b_zero_shot.py",
        "30c_audit_student_tokens.py", "30d_train_qwen3_4b_qlora.py",
        "30e_compare_student_distillation.py",
    )]
    scripts.extend([
        ROOT / "scripts" / "common" / "student_distillation_common.py",
        ROOT / "scripts" / "common" / "semantic_distillation_common.py",
    ])
    preflight = json.loads((OUTPUT / "preflight_audit.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "student_token_audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((OUTPUT / "qwen3_4b_distillation_comparison.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "imcs21_qwen3_4b_student_stage_v1",
        "status": "FROZEN_DEV_PASS" if preflight["status"] == "PASS" and audit["pass"] else "BLOCKED",
        "base_model": os.environ.get("QWEN3_MODEL_PATH", "Qwen/Qwen3-4B"),
        "method": "QLoRA NF4, rank 16, alpha 32, two epochs, assistant-only silver supervision",
        "train_rows": audit["splits"]["train"]["rows"],
        "dev_rows": audit["splits"]["dev"]["rows"],
        "result": comparison,
        "blind_hash": preflight["blind_hash"],
        "blind_status": "LOCKED_NOT_RUN",
        "canonical_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts},
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        "claim_boundary": [
            "Supports stable transfer of common structured semantic operations from proxy/silver large-teacher consensus to Qwen3-4B on isolated Dev.",
            "Does not establish clinician-gold accuracy, rare-operation generalization, or Blind performance.",
        ],
    }
    (OUTPUT / "student_distillation_stage_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("status", "train_rows", "dev_rows", "blind_hash", "blind_status", "claim_boundary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
