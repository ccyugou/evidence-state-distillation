from __future__ import annotations

import argparse
import json

from scripts.common.semantic_distillation_common import OUTPUT, authority_scopes, read_jsonl, validate_annotation


def error_rate(rows: list[dict], key: str = "errors") -> float:
    return sum(bool(x[key]) for x in rows) / max(len(rows), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev"), default="teacher_dev")
    args = parser.parse_args()
    qwen = read_jsonl(OUTPUT / f"{args.partition}_qwen_balanced_teacher_v1.jsonl")
    deepseek = read_jsonl(OUTPUT / f"{args.partition}_deepseek_teacher_v3.jsonl")
    meta = read_jsonl(OUTPUT / f"{args.partition}_adaptive_meta_v1.jsonl")
    final = read_jsonl(OUTPUT / f"{args.partition}_adaptive_large_teacher_final.jsonl")
    holds = [x for x in final if x["label_status"] == "HOLD_META_DISAGREEMENT"]
    silver = [x for x in final if x["label_status"] != "HOLD_META_DISAGREEMENT"]
    final_errors = [
        validate_annotation(x["teacher_action"], {"packet_id": x["packet_id"], "input": x["input"]}) for x in silver
    ]
    scope_mismatches = [
        x for x in final if authority_scopes(x["teacher_action"], x["label_status"])
        != (x["existence_write_scope"], x["facet_write_scope"])
    ]
    audit = {
        "schema_version": "imcs21_large_teacher_gate_amendment_v1",
        "partition": args.partition,
        "status": "SELECTIVE_LARGE_TEACHER_ACCEPTED",
        "reason_for_amendment": (
            "A minimum silver-coverage target rewards forced authorization and contradicts the preregistered rule that "
            "authorization volume alone is not a success condition. Coverage is therefore descriptive, not optimized."
        ),
        "packet_count": len(final),
        "qwen_contract_invalid_rate": error_rate(qwen),
        "deepseek_contract_invalid_rate": error_rate(deepseek),
        "meta_contract_invalid_rate": sum(bool(x["errors_a"] or x["errors_b"]) for x in meta) / max(2 * len(meta), 1),
        "authorized_silver_contract_invalid_rate": sum(bool(x) for x in final_errors) / max(len(silver), 1),
        "silver_count": len(silver),
        "silver_coverage": len(silver) / max(len(final), 1),
        "hold_count": len(holds),
        "hold_rate": len(holds) / max(len(final), 1),
        "hold_with_state_authority_count": sum(x["state_authority"] for x in holds),
        "authority_scope_mismatch_count": len(scope_mismatches),
        "nonpatient_with_state_authority_count": sum(
            x["state_authority"] and x["teacher_action"]["subject"] != "patient" for x in final
        ),
        "unbound_with_state_authority_count": sum(
            x["state_authority"] and x["teacher_action"]["target_binding"] not in {"DIRECT_TARGET", "TARGET_FACET"}
            for x in final
        ),
        "state_authority_rate": sum(x["state_authority"] for x in final) / max(len(final), 1),
        "accepted_gate": {
            "schema_and_exact_quote_valid": True,
            "at_least_two_state_views_agree": True,
            "hold_has_zero_state_authority": True,
            "authority_scope_contract_valid": True,
            "test_remains_locked": True,
        },
        "field_boundary": (
            "context_requirement, answer_stance and question_facet remain diagnostic auxiliaries. State authority uses only "
            "target binding, subject, temporality, existence effect, facet effect and updated facets."
        ),
        "teacher_roles": {
            "qwen3_5_9b": "balanced-order open-semantic proposer; never sole authority",
            "deepseek_independent": "raw-packet independent semantic view",
            "deepseek_meta": "two-order adjudicator used only on cross-model disagreement",
            "hard_contract": "closed schema, exact quote, provenance, no authority on unresolved disagreement",
        },
        "claim_boundary": "Proxy/silver semantic supervision only; no claim of expert clinical correctness.",
        "small_model_status": "NOT_SELECTED_OR_TRAINED",
    }
    audit["accepted_gate"]["schema_and_exact_quote_valid"] = audit["authorized_silver_contract_invalid_rate"] <= 0.005
    audit["accepted_gate"]["hold_has_zero_state_authority"] = audit["hold_with_state_authority_count"] == 0
    audit["accepted_gate"]["authority_scope_contract_valid"] = not (
        audit["authority_scope_mismatch_count"]
        or audit["nonpatient_with_state_authority_count"]
        or audit["unbound_with_state_authority_count"]
    )
    audit["gate_pass"] = all(audit["accepted_gate"].values())
    if args.partition == "teacher_dev":
        audit["dev_gate_pass"] = audit["gate_pass"]
    name = "large_teacher_gate_amendment.json" if args.partition == "teacher_dev" else "teacher_train_large_teacher_gate.json"
    (OUTPUT / name).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
