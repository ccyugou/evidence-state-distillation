from __future__ import annotations

import argparse
import hashlib

from scripts.common.semantic_distillation_common import OUTPUT, read_jsonl, write_jsonl
from importlib.util import module_from_spec, spec_from_file_location


def load_qwen_module():
    path = __file__.replace("29i_qwen_balanced_proposer.py", "29c_qwen_teacher_semantic_parser.py")
    spec = spec_from_file_location("qwen_teacher_parser", path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


def reverse_order(packet_id: str) -> bool:
    return int(hashlib.sha256(packet_id.encode()).hexdigest()[-1], 16) % 2 == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    qwen_module = load_qwen_module()
    packets = read_jsonl(OUTPUT / f"{args.partition}_packets.jsonl")
    if args.limit:
        packets = packets[:args.limit]
    target = OUTPUT / f"{args.partition}_qwen_balanced_teacher_v1.jsonl"
    existing = read_jsonl(target) if target.exists() else []
    done = {x["packet_id"] for x in existing}

    double_path = OUTPUT / f"{args.partition}_qwen_teacher_v3.jsonl"
    if double_path.exists():
        double = {x["packet_id"]: x for x in read_jsonl(double_path)}
        for packet in packets:
            packet_id = packet["packet_id"]
            if packet_id in done or packet_id not in double:
                continue
            reverse = reverse_order(packet_id)
            key = "annotation_b" if reverse else "annotation_a"
            error_key = "errors_b" if reverse else "errors_a"
            existing.append({
                "packet_id": packet_id, "source_split": packet["source_split"],
                "case_group_id": packet["case_group_id"], "candidate_order": "REVERSED" if reverse else "FORWARD",
                "annotation": double[packet_id][key], "errors": double[packet_id][error_key], "source": "SEEDED_FROM_DOUBLE_PASS",
            })
            done.add(packet_id)
        write_jsonl(target, existing)

    pending = [x for x in packets if x["packet_id"] not in done]
    for reverse in (False, True):
        ordered = [x for x in pending if reverse_order(x["packet_id"]) == reverse]
        for start in range(0, len(ordered), args.batch_size):
            batch = ordered[start:start + args.batch_size]
            annotations = qwen_module.parse_items(qwen_module.request(batch, reverse))
            for packet in batch:
                annotation = annotations.get(packet["packet_id"], {})
                existing.append({
                    "packet_id": packet["packet_id"], "source_split": packet["source_split"],
                    "case_group_id": packet["case_group_id"], "candidate_order": "REVERSED" if reverse else "FORWARD",
                    "annotation": annotation, "errors": qwen_module.validate_annotation(annotation, packet), "source": "BALANCED_SINGLE_PASS",
                })
            write_jsonl(target, existing)
            print(f"qwen_balanced={len(existing)}/{len(packets)}", flush=True)


if __name__ == "__main__":
    main()
