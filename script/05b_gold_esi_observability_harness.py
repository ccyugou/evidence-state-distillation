#!/usr/bin/env python3
"""Evidence-only Gold ESI observability audit for policy-conflict cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SCHEMA_VERSION = "05b_gold_esi_observability_v1.0"
SUPPORT = {"supported", "contradicted", "unassessable"}
PATHS = {"step_a", "step_b", "resource_2plus", "resource_1", "resource_0", "indeterminate"}
ACTIONS = {"retain_original_gold", "exclude_unobservable", "exclude_policy_conflict", "manual_clinician_review"}
PIPELINE = {"none", "patch_01", "patch_02", "patch_03"}
PATH_ESI = {"step_a": 1, "step_b": 2, "resource_2plus": 3, "resource_1": 4, "resource_0": 5}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("missing_json_object")
    return json.loads(text[start:end + 1])


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "evidence_id", "turn_order", "speaker", "span_text", "clinical_atom_id",
        "assertion", "temporality", "subject", "scope_resolution", "source_subtype",
        "positive_trigger_allowed", "independent_evidence", "semantic_modifiers",
    )
    result = {key: card.get(key) for key in keep if key in card}
    result["evidence_id"] = card["evidence_card_id"]
    return result


def compact_02(row: dict[str, Any]) -> dict[str, Any]:
    trajectories = []
    for item in row.get("atom_trajectories", []):
        trajectories.append({
            key: item.get(key)
            for key in (
                "trajectory_id", "clinical_atom_id", "policy_field", "subject", "episode_key",
                "effective_state", "trajectory_state", "evidence_card_ids",
            )
            if key in item
        })
    return {
        "instance_id": row["instance_id"],
        "atom_trajectories": trajectories,
        "step_a": [
            {"policy_field": x["policy_field"], "evidence_ids": x.get("evidence_ids", [])}
            for x in row.get("confirmed_step_a_signals", [])
        ],
        "step_b": [
            {
                "policy_field": x["policy_field"],
                "evidence_ids": x.get("evidence_ids", []),
                "validated_status": x.get("validated_status"),
                "llm_rationale": x.get("llm_rationale"),
            }
            for x in row.get("confirmed_step_b_signals", [])
        ],
        "explicit_negatives": row.get("explicit_negatives", []),
        "uncertain_evidence": row.get("uncertain_evidence", []),
    }


def validate(result: dict[str, Any], audit_id: str, gold: int, allowed_ids: set[str]) -> list[str]:
    errors = []
    if result.get("audit_id") != audit_id:
        errors.append("audit_id_mismatch")
    if result.get("gold_support_status") not in SUPPORT:
        errors.append("invalid_gold_support_status")
    path = result.get("observable_policy_path")
    if path not in PATHS:
        errors.append("invalid_observable_policy_path")
    if result.get("supervision_action") not in ACTIONS:
        errors.append("invalid_supervision_action")
    if result.get("pipeline_action") not in PIPELINE:
        errors.append("invalid_pipeline_action")
    revised = result.get("proposed_revised_esi")
    expected = PATH_ESI.get(path)
    if revised != expected:
        errors.append("path_esi_mismatch")
    if result.get("gold_support_status") == "supported" and revised != gold:
        errors.append("supported_gold_changed")
    if result.get("gold_support_status") == "supported" and result.get("supervision_action") != "retain_original_gold":
        errors.append("supported_gold_not_retained")
    if result.get("gold_support_status") == "contradicted" and (revised is None or revised == gold):
        errors.append("contradicted_gold_without_different_path")
    if result.get("gold_support_status") == "contradicted" and result.get("supervision_action") not in {"exclude_policy_conflict", "manual_clinician_review"}:
        errors.append("contradicted_gold_wrong_action")
    if result.get("gold_support_status") == "unassessable" and (path != "indeterminate" or revised is not None):
        errors.append("unassessable_gold_has_forced_path")
    cited = result.get("cited_evidence_ids", [])
    if not isinstance(cited, list) or set(cited) - allowed_ids:
        errors.append("hallucinated_evidence_id")
    return errors


def normalize_semantic_contract(result: dict[str, Any], gold: int) -> list[str]:
    changes = []
    revised = result.get("proposed_revised_esi")
    path = result.get("observable_policy_path")
    if revised == gold and PATH_ESI.get(path) == gold and result.get("gold_support_status") != "supported":
        result["gold_support_status"] = "supported"
        result["supervision_action"] = "retain_original_gold"
        changes.append("same_path_reclassified_supported")
    elif path == "indeterminate" and revised is None and result.get("gold_support_status") == "contradicted":
        result["gold_support_status"] = "unassessable"
        if result.get("supervision_action") not in {"exclude_unobservable", "manual_clinician_review"}:
            result["supervision_action"] = "exclude_unobservable"
        changes.append("indeterminate_reclassified_unassessable")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--stage01", type=Path, required=True)
    parser.add_argument("--stage02", type=Path, required=True)
    parser.add_argument("--stage03", type=Path, required=True)
    parser.add_argument("--stage05-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    key = os.environ["DEEPSEEK_API_KEY"]
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
    retries = int(os.getenv("MAX_RETRY", "2"))

    traces = {
        row["case_id"]: row
        for row in read_jsonl(args.stage05_dir / "05_case_error_traces.jsonl")
        if any(issue.startswith("gold_esi") for issue in row["observed_issues"])
    }
    selected = sorted(traces)[:args.limit]
    raw_by_instance = {}
    for path in args.transcripts.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        instance_id = f"{row['case_id']}__{row['run_uuid']}"
        if str(row["case_id"]) in traces:
            raw_by_instance[instance_id] = row

    cards_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.stage01):
        if row["case_id"] in traces:
            cards_by_instance[row["instance_id"]].append(row)
    rows02: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.stage02):
        if row["case_id"] in traces:
            rows02[row["case_id"]].append(row)
    rows03 = {row["instance_id"]: row for row in read_jsonl(args.stage03)}

    system = """You are the evidence-only Gold ESI observability adjudicator for TRIBOT.
The dialogue is deidentified. Do not infer facts that are not stated and do not use model predictions.
ESI policy summary: Step A requires an immediate life-saving intervention, not merely an abnormal symptom or vital. Step B covers a supported high-risk situation, new confusion/lethargy/disorientation, or clinically severe pain/distress. Only after A and B are not met, ESI 3/4/5 correspond to 2+, 1, or 0 distinct expected resources. Never infer an exact resource count from a diagnosis or symptom alone.
Decide whether the original Gold ESI is observable from the supplied dialogue and evidence chain. If an exact different policy path is directly supported, report it. If expected resources or life-saving intervention need are not observable, use indeterminate and null revised ESI.
pipeline_action means a clear extraction/state/measurement defect supported by raw text, not merely missing source information.
Return one JSON object with exactly: audit_id, gold_support_status, observable_policy_path, proposed_revised_esi, supervision_action, pipeline_action, cited_evidence_ids, rationale.
gold_support_status: supported, contradicted, unassessable.
observable_policy_path: step_a, step_b, resource_2plus, resource_1, resource_0, indeterminate.
supervision_action: retain_original_gold, exclude_unobservable, exclude_policy_conflict, manual_clinician_review.
pipeline_action: none, patch_01, patch_02, patch_03."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "05b_response_cache.jsonl"
    cached = {row["case_id"]: row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    private_map, results = [], []
    with httpx.Client(timeout=180.0) as client, cache_path.open("a", encoding="utf-8") as cache:
        for index, case_id in enumerate(selected, 1):
            trace = traces[case_id]
            instance_id = trace["representative_instance_id"]
            raw = raw_by_instance[instance_id]
            audit_id = f"GOLD_AUDIT_{index:04d}"
            cards = [compact_card(x) for x in cards_by_instance[instance_id]]
            allowed_ids = {x["evidence_id"] for x in cards}
            payload = {
                "audit_id": audit_id,
                "original_gold_esi": trace["gold_esi_audit_only"],
                "observed_contract_issue": [x for x in trace["observed_issues"] if x.startswith("gold_esi")],
                "dialogue": [
                    {"turn": x.get("turn", i), "speaker": x.get("actor"), "text": x.get("utterance")}
                    for i, x in enumerate(raw["history"])
                ],
                "evidence_cards": cards,
                "qualified_state": [compact_02(x) for x in rows02[case_id]],
                "vital_state": rows03[instance_id],
                "allowed_evidence_ids": sorted(allowed_ids),
            }
            prompt_hash = hashlib.sha256((system + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
            prior = cached.get(case_id)
            prior_valid = prior and prior.get("parsed_response") and not validate(
                prior["parsed_response"], audit_id, trace["gold_esi_audit_only"], allowed_ids
            )
            if prior and prior.get("prompt_sha256") == prompt_hash and prior_valid:
                record = prior
            else:
                last_error = ""
                messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                for attempt in range(1, retries + 1):
                    try:
                        response = client.post(
                            f"{base}/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={
                                "model": model,
                                "messages": messages,
                                "response_format": {"type": "json_object"},
                                "temperature": 0,
                                "max_tokens": max_tokens,
                            },
                        )
                        response.raise_for_status()
                        parsed = parse_json(response.json()["choices"][0]["message"]["content"])
                        parsed["cited_evidence_ids"] = [x for x in parsed.get("cited_evidence_ids", []) if x in allowed_ids]
                        changes = normalize_semantic_contract(parsed, trace["gold_esi_audit_only"])
                        errors = validate(parsed, audit_id, trace["gold_esi_audit_only"], allowed_ids)
                        record = {"case_id": case_id, "prompt_sha256": prompt_hash, "attempt_count": attempt, "api_success": not errors, "validator_errors": errors, "normalization_changes": changes, "parsed_response": parsed}
                        if not errors:
                            break
                        last_error = ",".join(errors)
                        messages.extend([
                            {"role": "assistant", "content": json.dumps(parsed)},
                            {"role": "user", "content": f"Correct these contract errors and return the full JSON object: {last_error}"},
                        ])
                    except Exception as exc:
                        last_error = str(exc)
                else:
                    record = {"case_id": case_id, "prompt_sha256": prompt_hash, "attempt_count": retries, "api_success": False, "validator_errors": [last_error], "parsed_response": None}
                cache.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                cache.flush()
            private_map.append({"audit_id": audit_id, "case_id": case_id, "representative_instance_id": instance_id})
            results.append(record)
            print(f"Gold ESI audit {index}/{len(selected)} case={case_id} success={record['api_success']}", flush=True)

    valid = [row["parsed_response"] | {"case_id": row["case_id"]} for row in results if row["api_success"]]
    output_path = args.output_dir / "05b_gold_esi_observability.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in valid:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "05b_private_id_map.jsonl").open("w", encoding="utf-8") as stream:
        for row in private_map:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "selected_case_count": len(selected),
        "valid_case_count": len(valid),
        "gold_support_status_counts": Counter(x["gold_support_status"] for x in valid),
        "observable_policy_path_counts": Counter(x["observable_policy_path"] for x in valid),
        "supervision_action_counts": Counter(x["supervision_action"] for x in valid),
        "pipeline_action_counts": Counter(x["pipeline_action"] for x in valid),
        "proposed_revised_esi_counts": Counter(str(x["proposed_revised_esi"]) for x in valid),
        "semantic_normalization_count": sum(len(row.get("normalization_changes", [])) for row in results if row["api_success"]),
        "input_sha256": {
            "stage01": sha256(args.stage01), "stage02": sha256(args.stage02), "stage03": sha256(args.stage03),
            "stage05_traces": sha256(args.stage05_dir / "05_case_error_traces.jsonl"),
        },
        "output_sha256": {
            "gold_esi_observability": sha256(output_path),
            "private_id_map": sha256(args.output_dir / "05b_private_id_map.jsonl"),
            "response_cache": sha256(cache_path),
        },
        "hard_gates": {
            "all_selected_valid": len(valid) == len(selected),
            "original_gold_preserved": True,
            "stage04_predictions_not_supplied": True,
            "resource_count_not_inferred_without_observation": True,
        },
    }
    audit["release_gate_passed"] = all(audit["hard_gates"].values())
    (args.output_dir / "05b_build_audit.json").write_text(json.dumps(audit, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=dict))


if __name__ == "__main__":
    main()
