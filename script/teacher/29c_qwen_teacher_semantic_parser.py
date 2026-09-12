from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from scripts.common.semantic_distillation_common import (
    OUTPUT, compact_packet, parse_items, qwen_response_format, read_jsonl, system_prompt,
    validate_annotation, write_jsonl,
)


ENDPOINT = os.environ.get("QWEN35_API_URL", "http://127.0.0.1:9098/v1/chat/completions")
MODEL = os.environ.get("QWEN35_MODEL_NAME", "Qwen3.5-9B-W4A16")


def request(rows: list[dict], reverse: bool) -> dict:
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": qwen_response_format(),
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system_prompt(reverse)},
            {"role": "user", "content": json.dumps({"items": [compact_packet(x) for x in rows]}, ensure_ascii=False)},
        ],
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json", "Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8")) from error
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("teacher_train", "teacher_dev", "blind_locked"), required=True)
    parser.add_argument("--allow-blind-after-freeze", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.partition == "blind_locked" and not args.allow_blind_after_freeze:
        raise SystemExit("blind_locked requires --allow-blind-after-freeze after model and gate freeze")

    source = OUTPUT / f"{args.partition}_packets.jsonl"
    target = OUTPUT / f"{args.partition}_qwen_teacher_v3.jsonl"
    rows = read_jsonl(source)[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    existing = read_jsonl(target) if target.exists() and args.offset == 0 else []
    done = {x["packet_id"] for x in existing}
    pending = [x for x in rows if x["packet_id"] not in done]

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        first = parse_items(request(batch, False))
        second = parse_items(request(batch, True))
        for row in batch:
            packet_id = row["packet_id"]
            a, b = first.get(packet_id, {}), second.get(packet_id, {})
            errors_a, errors_b = validate_annotation(a, row), validate_annotation(b, row)
            existing.append({
                "packet_id": packet_id,
                "source_split": row["source_split"],
                "case_group_id": row["case_group_id"],
                "annotation_a": a,
                "annotation_b": b,
                "errors_a": errors_a,
                "errors_b": errors_b,
                "order_stable": not errors_a and not errors_b and all(a[x] == b[x] for x in (
                    "target_binding", "question_facet", "answer_stance", "subject", "temporality",
                    "context_requirement", "existence_effect", "facet_effect",
                )) and a.get("updated_facets") == b.get("updated_facets"),
            })
        write_jsonl(target, existing)
        print(f"qwen={len(existing)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
