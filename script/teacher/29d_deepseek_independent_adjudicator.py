from __future__ import annotations

import argparse

from scripts.common.semantic_distillation_common import (
    OUTPUT, compact_packet, deepseek_chat, parse_items, read_jsonl, system_prompt,
    validate_annotation, write_jsonl,
)


def request(rows: list[dict]) -> dict:
    return deepseek_chat(system_prompt(True), {"items": [compact_packet(x) for x in rows]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev", "blind_locked"), required=True)
    parser.add_argument("--allow-blind-after-freeze", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.partition == "blind_locked" and not args.allow_blind_after_freeze:
        raise SystemExit("blind_locked requires --allow-blind-after-freeze after model and gate freeze")

    source = OUTPUT / f"{args.partition}_packets.jsonl"
    target = OUTPUT / f"{args.partition}_deepseek_teacher_v3.jsonl"
    rows = read_jsonl(source)[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    existing = read_jsonl(target) if target.exists() and args.offset == 0 else []
    done = {x["packet_id"] for x in existing}
    pending = [x for x in rows if x["packet_id"] not in done]

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        items = parse_items(request(batch))
        for row in batch:
            annotation = items.get(row["packet_id"], {})
            existing.append({
                "packet_id": row["packet_id"],
                "source_split": row["source_split"],
                "case_group_id": row["case_group_id"],
                "annotation": annotation,
                "errors": validate_annotation(annotation, row),
            })
        write_jsonl(target, existing)
        print(f"deepseek={len(existing)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
