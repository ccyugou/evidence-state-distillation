from __future__ import annotations

import argparse
import json
from collections import Counter

from scripts.common.semantic_distillation_common import (
    OUTPUT, authority_scopes, compact_packet, deepseek_chat, read_jsonl, system_prompt,
    validate_annotation, write_jsonl,
)


STATE_FIELDS = ("target_binding", "subject", "temporality", "existence_effect", "facet_effect")
META = """
你是最终语义教师。两个Qwen候选可能错误，必须回到raw_packet独立裁决，不做多数投票。
重点检查侧面否定、历史条件、主体和省略回答；输出封闭JSON，不解释。
"""


def signature(annotation: dict) -> tuple:
    return tuple(annotation.get(x) for x in STATE_FIELDS) + (tuple(annotation.get("updated_facets", [])),)


def meta_request(rows: list[tuple], reverse: bool) -> dict[str, dict]:
    items = []
    for packet, pair in rows:
        proposals = [pair["annotation_a"], pair["annotation_b"]]
        if reverse:
            proposals.reverse()
        items.append({"raw_packet": compact_packet(packet), "untrusted_qwen_proposals": proposals})
    result = deepseek_chat(system_prompt(reverse) + META, {"items": items})
    return {str(x.get("packet_id")): x for x in result.get("items", [])}


def hold() -> dict:
    return {
        "target_binding": "UNSUPPORTED", "question_facet": "unclear", "answer_stance": "uncertain",
        "subject": "unknown", "temporality": "unknown", "context_requirement": "INSUFFICIENT",
        "existence_effect": "HOLD", "facet_effect": "HOLD", "updated_facets": [],
        "question_quote": "", "answer_quote": "", "confidence": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    packets = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_packets.jsonl")}
    qwen = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_qwen_balanced_teacher_v1.jsonl")}
    deepseek = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_deepseek_teacher_v3.jsonl")}
    pairs = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_qwen_adaptive_pair_v1.jsonl")}
    old_meta_path = OUTPUT / f"{args.partition}_deepseek_meta_teacher_v1.jsonl"
    old_meta = {x["packet_id"]: x for x in read_jsonl(old_meta_path)} if old_meta_path.exists() else {}
    meta_target = OUTPUT / f"{args.partition}_adaptive_meta_v1.jsonl"
    meta_rows = read_jsonl(meta_target) if meta_target.exists() else []
    meta_done = {x["packet_id"] for x in meta_rows}
    for packet_id, pair in pairs.items():
        if packet_id in meta_done:
            continue
        if packet_id in old_meta:
            meta_rows.append(old_meta[packet_id]); meta_done.add(packet_id)
    write_jsonl(meta_target, meta_rows)
    pending = [(packets[packet_id], pair) for packet_id, pair in pairs.items() if packet_id not in meta_done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        a, b = meta_request(batch, False), meta_request(batch, True)
        for packet, _ in batch:
            meta_rows.append({
                "packet_id": packet["packet_id"], "source_split": packet["source_split"],
                "case_group_id": packet["case_group_id"], "meta_a": a.get(packet["packet_id"], {}),
                "meta_b": b.get(packet["packet_id"], {}),
                "errors_a": validate_annotation(a.get(packet["packet_id"], {}), packet),
                "errors_b": validate_annotation(b.get(packet["packet_id"], {}), packet),
            })
        write_jsonl(meta_target, meta_rows)
        print(f"adaptive_meta={len(meta_rows)}/{len(pairs)}", flush=True)

    if args.checkpoint_only:
        print(json.dumps({"partition": args.partition, "meta_checkpoint": len(meta_rows)}, ensure_ascii=False))
        return

    meta = {x["packet_id"]: x for x in meta_rows}
    final = []
    for packet_id in sorted(qwen.keys() & deepseek.keys()):
        packet, qa, da = packets[packet_id], qwen[packet_id]["annotation"], deepseek[packet_id]["annotation"]
        if not qwen[packet_id]["errors"] and not deepseek[packet_id]["errors"] and signature(qa) == signature(da):
            action, status, sources = qa, "DIRECT_CROSS_MODEL_AGREEMENT", ["qwen", "deepseek"]
        else:
            row = meta[packet_id]
            candidates = []
            if not row["errors_a"]: candidates.append(("meta_a", row["meta_a"]))
            if not row["errors_b"]: candidates.append(("meta_b", row["meta_b"]))
            if not deepseek[packet_id]["errors"]: candidates.append(("deepseek_independent", da))
            counts = Counter(signature(x) for _, x in candidates)
            winning = counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= 2 else None
            selected = next(((name, x) for name, x in candidates if winning and signature(x) == winning), None)
            action = selected[1] if selected else hold()
            sources = [name for name, x in candidates if winning and signature(x) == winning]
            status = "ADAPTIVE_META_MAJORITY" if selected else "HOLD_META_DISAGREEMENT"
        existence_scope, facet_scope = authority_scopes(action, status)
        authority = existence_scope != "NONE" or facet_scope != "NONE"
        action = {**action, "packet_id": packet_id}
        final.append({
            "schema_version": "imcs21_adaptive_large_teacher_v1", "packet_id": packet_id,
            "source_split": packet["source_split"], "case_group_id": packet["case_group_id"],
            "input": compact_packet(packet), "teacher_action": action, "label_status": status,
            "agreement_sources": sources, "existence_write_scope": existence_scope,
            "facet_write_scope": facet_scope, "state_authority": authority,
        })
    write_jsonl(OUTPUT / f"{args.partition}_adaptive_large_teacher_final.jsonl", final)
    routes = Counter(x["label_status"] for x in final)
    audit = {
        "partition": args.partition, "packet_count": len(final), "routes": dict(routes),
        "final_silver_coverage": 1 - routes["HOLD_META_DISAGREEMENT"] / max(len(final), 1),
        "state_authority_rate": sum(x["state_authority"] for x in final) / max(len(final), 1),
        "claim_boundary": "Adaptive multi-view large-model proxy semantics; not clinical gold.",
    }
    audit["checkpoint_pass"] = audit["final_silver_coverage"] >= 0.90
    (OUTPUT / f"{args.partition}_adaptive_large_teacher_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
