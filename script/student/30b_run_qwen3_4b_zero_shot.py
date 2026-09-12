from __future__ import annotations

import argparse
import functools
import json
import os
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from scripts.common.semantic_distillation_common import BINDINGS, EXISTENCE_EFFECTS, FACETS, FACET_EFFECTS, SUBJECTS, TEMPORALITIES
from scripts.common.student_distillation_common import OUTPUT, STUDENT_FIELDS, read_jsonl, response_schema, system_prompt, write_jsonl


ENDPOINT = os.environ.get("QWEN3_API_URL", "http://127.0.0.1:8014/v1/chat/completions")
ENUMS = {
    "target_binding": BINDINGS, "subject": SUBJECTS, "temporality": TEMPORALITIES,
    "existence_effect": EXISTENCE_EFFECTS, "facet_effect": FACET_EFFECTS,
}


def request(row: dict, model: str) -> dict:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": response_schema(),
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": json.dumps(row["input"], ensure_ascii=False)},
        ],
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=180) as response:
        content = json.loads(response.read())["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid structured response: {content!r}") from error


def errors(pred: dict, row: dict) -> list[str]:
    found = [field for field, values in ENUMS.items() if pred.get(field) not in values]
    if not isinstance(pred.get("updated_facets"), list) or any(x not in FACETS[:-1] for x in pred.get("updated_facets", [])):
        found.append("updated_facets")
    for field, source in (("question_quote", "doctor_question"), ("answer_quote", "patient_answer")):
        quote = pred.get(field)
        if not isinstance(quote, str) or quote not in row["input"][source]:
            found.append(field)
    return found


def macro_f1(gold: list, pred: list) -> float:
    labels = set(gold) | set(pred)
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, pred))
        fp = sum(a != label and b == label for a, b in zip(gold, pred))
        fn = sum(a == label and b != label for a, b in zip(gold, pred))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(scores) / max(len(scores), 1)


def evaluate(rows: list[dict]) -> dict:
    valid = [x for x in rows if not x["errors"]]
    fields = tuple(ENUMS)
    metrics = {
        "count": len(rows), "contract_valid_rate": len(valid) / max(len(rows), 1),
        "full_state_exact": sum(all(x["prediction"][f] == x["target"][f] for f in fields)
                                and set(x["prediction"]["updated_facets"]) == set(x["target"]["updated_facets"])
                                for x in valid) / max(len(rows), 1),
        "facet_set_exact": sum(set(x["prediction"]["updated_facets"]) == set(x["target"]["updated_facets"])
                               for x in valid) / max(len(rows), 1),
        "quote_exact": sum(all(x["prediction"][f] == x["target"][f] for f in ("question_quote", "answer_quote"))
                           for x in valid) / max(len(rows), 1),
    }
    metrics["field_accuracy"] = {
        f: sum(x["prediction"][f] == x["target"][f] for x in valid) / max(len(rows), 1) for f in fields
    }
    metrics["field_macro_f1"] = {
        f: macro_f1([x["target"][f] for x in valid], [x["prediction"][f] for x in valid]) for f in fields
    }
    metrics["per_label_accuracy"] = {
        field: {
            label: {
                "count": sum(x["target"][field] == label for x in valid),
                "accuracy": sum(x["target"][field] == label and x["prediction"][field] == label for x in valid)
                / max(sum(x["target"][field] == label for x in valid), 1),
            }
            for label in sorted({x["target"][field] for x in valid})
        }
        for field in fields
    }
    metrics["error_counts"] = dict(Counter(e for x in rows for e in x["errors"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--tag", default="zero_shot_v2")
    args = parser.parse_args()
    source = read_jsonl(OUTPUT / "dev_silver_sft.jsonl")
    if args.limit:
        source = source[:args.limit]
    target_path = OUTPUT / f"dev_qwen3_4b_{args.tag}.jsonl"
    existing = read_jsonl(target_path)
    done = {x["packet_id"] for x in existing}
    pending = [x for x in source if x["packet_id"] not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(pending), 20):
            batch = pending[start:start + 20]
            predictions = list(pool.map(functools.partial(request, model=args.model), batch))
            for row, pred in zip(batch, predictions):
                existing.append({
                    "packet_id": row["packet_id"], "case_group_id": row["case_group_id"],
                    "target": row["target"], "prediction": pred, "errors": errors(pred, row),
                })
            write_jsonl(target_path, existing)
            print(f"zero_shot={len(existing)}/{len(source)}", flush=True)
    selected = [x for x in existing if x["packet_id"] in {r["packet_id"] for r in source}]
    metrics = evaluate(selected)
    (OUTPUT / f"dev_qwen3_4b_{args.tag}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
