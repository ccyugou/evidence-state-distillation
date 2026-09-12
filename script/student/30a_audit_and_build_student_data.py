from __future__ import annotations

import hashlib
import json
from collections import Counter

from scripts.common.semantic_distillation_common import authority_scopes, validate_annotation
from scripts.common.student_distillation_common import OUTPUT, STUDENT_FIELDS, TEACHER_OUTPUT, read_jsonl, student_input, student_target, system_prompt, write_jsonl


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_digest(row: dict) -> str:
    payload = json.dumps(row["input"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    frozen = json.loads((TEACHER_OUTPUT / "large_teacher_stage_summary.json").read_text(encoding="utf-8"))
    finals = {
        split: read_jsonl(TEACHER_OUTPUT / f"teacher_{split}_adaptive_large_teacher_final.jsonl")
        for split in ("train", "dev")
    }
    packets = {
        split: read_jsonl(TEACHER_OUTPUT / f"teacher_{split}_packets.jsonl")
        for split in ("train", "dev")
    }
    blind = read_jsonl(TEACHER_OUTPUT / "blind_locked_packets.jsonl")
    failures = []
    for name, item in frozen["canonical_artifacts"].items():
        if digest(TEACHER_OUTPUT / name) != item["sha256"]:
            failures.append(f"hash:{name}")

    group_sets = {
        "train": {x["case_group_id"] for x in packets["train"]},
        "dev": {x["case_group_id"] for x in packets["dev"]},
        "blind": {x["case_group_id"] for x in blind},
    }
    for a, b in (("train", "dev"), ("train", "blind"), ("dev", "blind")):
        if group_sets[a] & group_sets[b]:
            failures.append(f"group_overlap:{a}:{b}")

    built_rows, unresolved_rows = {}, {}
    for split in ("train", "dev"):
        packet_ids = {x["packet_id"] for x in packets[split]}
        final_ids = {x["packet_id"] for x in finals[split]}
        if len(final_ids) != len(finals[split]) or packet_ids != final_ids:
            failures.append(f"packet_contract:{split}")
        silver, unresolved = [], []
        for row in finals[split]:
            action = row["teacher_action"]
            errors = validate_annotation(action, {"packet_id": row["packet_id"], "input": row["input"]})
            if errors:
                failures.append(f"teacher_contract:{split}:{row['packet_id']}")
            expected = authority_scopes(action, row["label_status"])
            if expected != (row["existence_write_scope"], row["facet_write_scope"]):
                failures.append(f"authority_scope:{split}:{row['packet_id']}")
            base = {
                "packet_id": row["packet_id"], "case_group_id": row["case_group_id"],
                "input": student_input(row["input"]), "label_status": row["label_status"],
            }
            if row["label_status"] == "HOLD_META_DISAGREEMENT":
                unresolved.append(base)
            else:
                target = student_target(action)
                silver.append({
                    **base, "target": target,
                    "messages": [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": json.dumps(base["input"], ensure_ascii=False)},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
                    ],
                })
        built_rows[split], unresolved_rows[split] = silver, unresolved

    dev_hashes = {input_digest(x) for x in built_rows["dev"]}
    seen, clean_train = set(), []
    removed_cross, removed_within = 0, 0
    for row in built_rows["train"]:
        key = input_digest(row)
        if key in dev_hashes:
            removed_cross += 1
        elif key in seen:
            removed_within += 1
        else:
            seen.add(key)
            clean_train.append(row)
    built_rows["train"] = clean_train

    built, distributions = {}, {}
    for split in ("train", "dev"):
        silver, unresolved = built_rows[split], unresolved_rows[split]
        write_jsonl(OUTPUT / f"{split}_silver_sft.jsonl", silver)
        write_jsonl(OUTPUT / f"{split}_unresolved.jsonl", unresolved)
        built[split] = {"silver": len(silver), "unresolved": len(unresolved)}
        distributions[split] = {
            field: dict(Counter(json.dumps(x["target"][field], ensure_ascii=False, sort_keys=True) for x in silver))
            for field in STUDENT_FIELDS[:6]
        }

    audit = {
        "schema_version": "imcs21_student_distillation_preflight_v1",
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "built": built,
        "case_groups": {k: len(v) for k, v in group_sets.items()},
        "student_target_fields": list(STUDENT_FIELDS),
        "excluded_teacher_fields": ["question_facet", "answer_stance", "context_requirement", "confidence", "packet_id"],
        "decontamination": {"train_removed_cross_dev": removed_cross, "train_removed_within": removed_within},
        "label_distributions": distributions,
        "blind_hash": digest(TEACHER_OUTPUT / "blind_locked_packets.jsonl"),
        "blind_policy": "No text inspection, inference, tuning, or calibration before student and gate freeze.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "preflight_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "failures", "built", "case_groups", "student_target_fields", "blind_hash")}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
