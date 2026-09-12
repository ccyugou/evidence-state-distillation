from __future__ import annotations

import argparse

from scripts.common.semantic_distillation_common import OUTPUT, read_jsonl, write_jsonl
from importlib.util import module_from_spec, spec_from_file_location


STATE_FIELDS = ("target_binding", "subject", "temporality", "existence_effect", "facet_effect")


def signature(annotation: dict) -> tuple:
    return tuple(annotation.get(x) for x in STATE_FIELDS) + (tuple(annotation.get("updated_facets", [])),)


def load_qwen_module():
    path = __file__.replace("29k_adaptive_second_qwen.py", "29c_qwen_teacher_semantic_parser.py")
    spec = spec_from_file_location("qwen_teacher_parser", path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    qwen_module = load_qwen_module()
    packets = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_packets.jsonl")}
    first = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_qwen_balanced_teacher_v1.jsonl")}
    deepseek = {x["packet_id"]: x for x in read_jsonl(OUTPUT / f"{args.partition}_deepseek_teacher_v3.jsonl")}
    double_path = OUTPUT / f"{args.partition}_qwen_teacher_v3.jsonl"
    double = {x["packet_id"]: x for x in read_jsonl(double_path)} if double_path.exists() else {}
    target = OUTPUT / f"{args.partition}_qwen_adaptive_pair_v1.jsonl"
    existing = read_jsonl(target) if target.exists() else []
    done = {x["packet_id"] for x in existing}
    disputes = [packet_id for packet_id in sorted(first.keys() & deepseek.keys()) if (
        first[packet_id]["errors"] or deepseek[packet_id]["errors"]
        or signature(first[packet_id]["annotation"]) != signature(deepseek[packet_id]["annotation"])
    )]
    pending = [packet_id for packet_id in disputes if packet_id not in done]
    for reverse in (False, True):
        ordered = [x for x in pending if (first[x]["candidate_order"] != "REVERSED") == reverse]
        for start in range(0, len(ordered), args.batch_size):
            batch_ids = ordered[start:start + args.batch_size]
            generated = [x for x in batch_ids if x not in double]
            generated_items = qwen_module.parse_items(qwen_module.request([packets[x] for x in generated], reverse)) if generated else {}
            for packet_id in batch_ids:
                packet, initial = packets[packet_id], first[packet_id]
                if packet_id in double:
                    key = "annotation_b" if reverse else "annotation_a"
                    error_key = "errors_b" if reverse else "errors_a"
                    second, errors = double[packet_id][key], double[packet_id][error_key]
                else:
                    second = generated_items.get(packet_id, {})
                    errors = qwen_module.validate_annotation(second, packet)
                existing.append({
                    "packet_id": packet_id, "source_split": packet["source_split"], "case_group_id": packet["case_group_id"],
                    "annotation_a": initial["annotation"], "errors_a": initial["errors"],
                    "annotation_b": second, "errors_b": errors,
                })
            write_jsonl(target, existing)
            print(f"adaptive_qwen={len(existing)}/{len(disputes)}", flush=True)


if __name__ == "__main__":
    main()
