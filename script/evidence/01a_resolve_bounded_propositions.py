from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from scripts.common.imcs21_common import read_jsonl, stable_id, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "01_evidence_cards"
OUTPUT = ROOT / "outputs" / "01_faceted_propositions"
ENDPOINT = os.environ.get("QWEN35_API_URL", "http://127.0.0.1:9098/v1/chat/completions")
MODEL = os.environ.get("QWEN35_MODEL_NAME", "Qwen3.5-9B-W4A16")
FACETS = {"existence", "severity", "duration", "frequency", "onset", "course", "trigger", "location", "radiation", "associated_feature", "treatment_response", "diagnostic_status", "other"}
ASSERTIONS = {"positive", "negative", "uncertain", "not_applicable"}
TEMPORALITIES = {"current", "historical", "mixed", "unknown"}
SUBJECTS = {"patient", "family", "other", "unknown"}
OPERATIONS = {"ASSERT_EXISTENCE", "RETRACT_EXISTENCE", "UPDATE_FACET", "RESOLVE", "RECUR", "IMPROVE", "WORSEN", "HOLD"}
BLOCKING_REASONS = {
    "temporality_scope", "subject_resolution_required", "disjunction_or_hypothesis",
    "patient_question_or_hypothesis", "attributed_diagnosis_requires_facet",
    "treatment_indication_scope", "uncertain_proposition", "unresolved_normalization",
    "low_ner_confidence",
}


SYSTEM = """你是医疗对话命题解析器，不做诊断。只能解释给定anchor，不得新增、删除或改名医学概念。
联合阅读相邻医生问题与患者回答。医生提问本身不证明症状阳性；患者问某症状是否相关也不自动证明该症状存在。
必须区分：否定症状存在、否定症状某个侧面、既往状态、当前状态、家属状态、患者状态、改善、恶化、缓解、复发。
输出单个JSON对象：{"propositions":[...] }。每个anchor恰好一项，字段固定为：
card_id, facet, facet_value, assertion, temporality, subject, revision_operation, evidence_quote, confidence。
facet只能取 existence,severity,duration,frequency,onset,course,trigger,location,radiation,associated_feature,treatment_response,diagnostic_status,other。
assertion只能取 positive,negative,uncertain,not_applicable；temporality只能取 current,historical,mixed,unknown；subject只能取 patient,family,other,unknown。
revision_operation只能取 ASSERT_EXISTENCE,RETRACT_EXISTENCE,UPDATE_FACET,RESOLVE,RECUR,IMPROVE,WORSEN,HOLD。
不能从原文严格支持时用HOLD，confidence为0到1。不要输出解释或Markdown。"""


def packet_payload(packet: dict, cards: dict[str, dict]) -> dict:
    anchors = packet.get("anchors") or packet.get("candidate_anchors", [])
    return {
        "review_type": packet["review_type"],
        "context": [{"speaker": row["speaker"], "text": row["text"], "turn_index": row["turn_index"]} for row in packet["context"]],
        "patient_answer": packet.get("answer_text"),
        "anchors": [{
            **anchor,
            "char_start": cards.get(anchor["card_id"], {}).get("char_start"),
            "char_end": cards.get(anchor["card_id"], {}).get("char_end"),
            "source_text": cards.get(anchor["card_id"], {}).get("text"),
        } for anchor in anchors],
        "trigger_reasons": packet["reasons"],
    }


def request_qwen(packet: dict, cards: dict[str, dict]) -> tuple[dict, str]:
    anchor_count = len(packet.get("anchors") or packet.get("candidate_anchors", []))
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(packet_payload(packet, cards), ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": min(1400, 220 + 180 * anchor_count),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read().decode()
    message = json.loads(raw)["choices"][0]["message"]
    content = message.get("content") or ""
    return json.loads(content), content


def validate(packet: dict, parsed: dict) -> tuple[list[dict], list[str]]:
    anchors = packet.get("anchors") or packet.get("candidate_anchors", [])
    allowed = {row["card_id"] for row in anchors}
    errors, valid, seen = [], [], set()
    source_text = "\n".join(item["text"] for item in packet["context"])
    for row in parsed.get("propositions", []):
        card_id = row.get("card_id")
        if card_id not in allowed or card_id in seen:
            errors.append("invalid_or_duplicate_card_id")
            continue
        seen.add(card_id)
        checks = (
            row.get("facet") in FACETS, row.get("assertion") in ASSERTIONS,
            row.get("temporality") in TEMPORALITIES, row.get("subject") in SUBJECTS,
            row.get("revision_operation") in OPERATIONS,
            isinstance(row.get("evidence_quote"), str),
            row.get("evidence_quote", "") in source_text,
            isinstance(row.get("confidence"), (int, float)) and 0 <= row["confidence"] <= 1,
        )
        if not all(checks):
            errors.append(f"invalid_schema:{card_id}")
            continue
        valid.append(row)
    missing = allowed - seen
    if missing:
        errors.append(f"missing_anchor_count:{len(missing)}")
    return valid, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--high-value-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    card_index = {row["card_id"]: row for row in read_jsonl(INPUT / f"{args.split}_evidence_cards.jsonl")}
    queue = list(read_jsonl(INPUT / f"{args.split}_bounded_review_queue.jsonl"))
    if args.high_value_only:
        queue = [packet for packet in queue if (
            (packet.get("anchors") or packet.get("candidate_anchors", []))
            and all(anchor.get("entity_type", "Symptom") == "Symptom" and anchor.get("canonical_concept") for anchor in (packet.get("anchors") or packet.get("candidate_anchors", [])))
            and not (set(packet["reasons"]) & BLOCKING_REASONS)
        )]
    queue = queue[args.offset:args.offset + args.limit if args.limit else None]
    suffix = f"_{args.offset}_{args.limit}" if args.limit else ""
    output_path = OUTPUT / f"{args.split}_qwen_proposals{suffix}.jsonl"
    results = []
    for index, packet in enumerate(queue, 1):
        parsed, raw, error = {}, "", None
        try:
            parsed, raw = request_qwen(packet, card_index)
            proposals, errors = validate(packet, parsed)
        except Exception as exc:
            proposals, errors, error = [], [], str(exc)
        results.append({
            "proposal_id": stable_id("qwen-proposal", packet["review_id"]),
            "review_id": packet["review_id"], "dialogue_id": packet["dialogue_id"],
            "turn_index": packet["turn_index"], "propositions": proposals,
            "validator_errors": errors, "request_error": error,
            "proposal_status": "PROPOSED_ONLY_NO_STATE_AUTHORITY",
            "raw_response": raw,
        })
        if index % 10 == 0:
            print(f"processed={index}/{len(queue)}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)
    write_jsonl(output_path, results)
    audit = {
        "schema_version": "imcs21_qwen_faceted_proposals_v1", "split": args.split,
        "packet_count": len(results), "valid_packet_count": sum(bool(row["propositions"]) and not row["validator_errors"] for row in results),
        "request_error_count": sum(row["request_error"] is not None for row in results),
        "validator_error_count": sum(bool(row["validator_errors"]) for row in results),
        "high_value_only": args.high_value_only,
        "state_authority": False, "output": str(output_path),
    }
    (OUTPUT / f"{args.split}_qwen_audit{suffix}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
