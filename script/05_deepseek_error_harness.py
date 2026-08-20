#!/usr/bin/env python3
"""Evidence-locked DeepSeek harness for stage-01 to stage-04 error attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx


SCHEMA_VERSION = "05_deepseek_error_harness_v1.0"
ERROR_TYPES = {
    "upstream_recall_gap",
    "upstream_semantic_state_error",
    "policy_composition_error",
    "llm_qualification_error",
    "measurement_contract_error",
    "resource_feature_error",
    "resource_decision_boundary_error",
    "proxy_label_policy_mismatch",
    "insufficient_observable_evidence",
    "no_material_pipeline_error",
}
OWNERS = {"01", "02", "03", "04", "supervision", "none"}
ACTIONS = {
    "patch_01_fixture",
    "patch_02_policy_or_trajectory",
    "patch_03_measurement_contract",
    "rebuild_04_features_or_model",
    "review_resource_proxy_label",
    "retain_for_manual_review",
    "no_change",
}


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Malformed JSON boundaries")
    return json.loads(text[start : end + 1])


def raw_cases(transcripts: Path) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    labels: dict[str, int] = {}
    instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(transcripts.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(row["case_id"])
        instance_id = f"{case_id}__{row['run_uuid']}"
        labels[case_id] = int(row["ground_truth"]["acuity"])
        dialogue = [
            {
                "turn": item.get("turn"),
                "actor": item.get("actor"),
                "text": item.get("original") or item.get("utterance"),
                "event": item.get("event"),
                "name": item.get("name"),
                "value": item.get("value"),
            }
            for item in row.get("history", [])
            if item.get("actor") in {"patient", "nurse", "system"}
        ]
        instances[case_id].append({"instance_id": instance_id, "dialogue": dialogue})
    return labels, instances


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def compact_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": row["instance_id"],
        "step_a": row["confirmed_step_a_signals"],
        "step_b": row["confirmed_step_b_signals"],
        "uncertain_policy_fields": sorted({
            item["policy_field"]
            for item in row["policy_candidates"]
            if item.get("validated_status") == "uncertain_policy_review"
        }),
        "atom_states": [
            {
                "atom": item["clinical_atom_id"],
                "subject": item["subject"],
                "state": item["effective_state"],
                "evidence_ids": item["evidence_card_ids"],
            }
            for item in row["atom_trajectories"]
            if item["subject"] == "patient"
        ],
    }


def choose_instance(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        return (
            20 * len(row["confirmed_step_a_signals"])
            + 10 * len(row["confirmed_step_b_signals"])
            + len(row["policy_candidates"])
            + len(row["atom_trajectories"]),
            len(row["uncertain_evidence"]),
            row["instance_id"],
        )

    return max(case_rows, key=score)


def compile_traces(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    labels, raw_by_case = raw_cases(args.transcripts)
    cards = {row["evidence_card_id"]: row for row in read_jsonl(args.stage01_cards)}
    rows02 = read_jsonl(args.stage02)
    rows03 = {row["instance_id"]: row for row in read_jsonl(args.stage03)}
    features = read_jsonl(args.stage04_features)
    supervision = read_jsonl(args.stage04_supervision)
    oof = {row["case_id"]: row for row in read_jsonl(args.stage04_oof)}

    row02_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quality_by_case: dict[str, set[str]] = defaultdict(set)
    for row in rows02:
        row02_by_case[row["case_id"]].append(row)
    for row in features:
        feature_by_case[row["case_id"]].append(row)
    for row in supervision:
        quality_by_case[row["case_id"]].add(row["supervision"]["quality"])

    traces = []
    prompt_payloads = {}
    for case_id in sorted(labels):
        gold_esi = labels[case_id]
        case02 = row02_by_case[case_id]
        representative = choose_instance(case02)
        rep_id = representative["instance_id"]
        step_a_count = sum(len(row["confirmed_step_a_signals"]) for row in case02)
        step_b_count = sum(len(row["confirmed_step_b_signals"]) for row in case02)
        resource = oof.get(case_id)
        resource_error = bool(resource and resource["threshold_prediction"] != resource["resource_bucket"])
        issues = []
        if gold_esi == 1 and not step_a_count:
            issues.append("gold_esi1_without_step_a")
        if gold_esi == 2 and not step_b_count:
            issues.append("gold_esi2_without_step_b")
        if gold_esi >= 3 and (step_a_count or step_b_count):
            issues.append("gold_esi3_5_with_step_a_or_b")
        if resource_error:
            issues.append("resource_bucket_oof_error")
        low_margin = False
        if resource:
            ordered = sorted(resource["probabilities"], reverse=True)
            low_margin = ordered[0] - ordered[1] < args.low_margin
            if low_margin:
                issues.append("resource_low_margin")
        selected = bool(set(issues) - {"resource_low_margin"}) or (low_margin and resource_error)

        trace = {
            "schema_version": SCHEMA_VERSION,
            "audit_only": True,
            "case_id": case_id,
            "gold_esi_audit_only": gold_esi,
            "realization_count": len(case02),
            "representative_instance_id": rep_id,
            "step_a_signal_count": step_a_count,
            "step_b_signal_count": step_b_count,
            "resource_oof": resource,
            "supervision_quality": sorted(quality_by_case.get(case_id, set())),
            "observed_issues": issues,
            "selected_for_deepseek": selected,
        }
        traces.append(trace)
        if not selected:
            continue

        selected_ids = set()
        for row in case02:
            for signal in row["confirmed_step_a_signals"] + row["confirmed_step_b_signals"]:
                selected_ids.update(signal.get("evidence_ids", []))
            for candidate in row["policy_candidates"]:
                selected_ids.update(candidate.get("evidence_ids", []))
        if resource:
            for boundary in resource["explanation"].values():
                for item in boundary:
                    selected_ids.update(item.get("evidence_ids", []))
        selected_cards = []
        for evidence_id in sorted(selected_ids):
            card = cards.get(evidence_id)
            if card:
                selected_cards.append({
                    "evidence_id": evidence_id,
                    "span_text": card["span_text"],
                    "atom": card["clinical_atom_id"],
                    "assertion": card["assertion"],
                    "temporality": card["temporality"],
                    "subject": card["subject"],
                    "scope": card["scope_resolution"],
                    "source": card["source_subtype"],
                })

        raw_instance = next(row for row in raw_by_case[case_id] if row["instance_id"] == rep_id)
        feature_rows = feature_by_case.get(case_id, [])
        feature_names = sorted({name for row in feature_rows for name, value in row["features"].items() if value})
        prompt_payloads[case_id] = {
            "case_id": case_id,
            "gold_esi_audit_only": gold_esi,
            "observed_issues": issues,
            "representative_raw_dialogue": raw_instance["dialogue"],
            "all_realization_state_summaries": [compact_trajectory(row) for row in case02],
            "representative_vital_facts": rows03[rep_id],
            "evidence_cards": selected_cards,
            "active_feature_names": feature_names,
            "allowed_evidence_ids": sorted(selected_ids | set(rows03[rep_id].get("supporting_evidence_ids", []))),
            "resource_oof": resource,
            "supervision_quality": sorted(quality_by_case.get(case_id, set())),
        }
    return traces, prompt_payloads


def prompt_for(payload: dict[str, Any]) -> tuple[str, str]:
    system = """You are the audit-only clinical error-attribution component of TRIBOT.
You do not predict a new ESI label and you do not invent missing clinical facts.
Trace the supplied raw dialogue through evidence cards, qualified trajectories, vital facts, and the OOF resource prediction.
In resource_oof, resource_bucket is the ESI-derived audit target; raw_prediction and threshold_prediction are model outputs.
Prediction disagreement alone never proves that the proxy target is wrong. Judge proxy_label_support from the raw observable evidence.
Decide the earliest material error owner: wrong card versus raw=01; wrong cross-turn state/policy composition=02;
wrong measurement consumption=03; correct inputs but wrong weighting/boundary=04; observable evidence conflicts with proxy=supervision.
Use exact strings from allowed_evidence_ids and active_feature_names, or return an empty list. Never abbreviate a feature name.
Do not treat a nurse question as a patient finding. Quarantined measurements are not consumed model features.
Return one JSON object with exactly these fields:
case_id, primary_error_type, owner_module, proxy_label_support, recommended_action,
cited_evidence_ids, cited_feature_names, rationale, reusable_error_pattern.
primary_error_type must be one of: """ + ", ".join(sorted(ERROR_TYPES)) + ".\nowner_module: " + ", ".join(sorted(OWNERS)) + ".\nrecommended_action: " + ", ".join(sorted(ACTIONS)) + ".\nproxy_label_support: supported, contradicted, or insufficient."
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return system, user


def validate(payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors = []
    if result.get("case_id") != payload["case_id"]:
        errors.append("case_id_mismatch")
    if result.get("primary_error_type") not in ERROR_TYPES:
        errors.append("invalid_error_type")
    if result.get("owner_module") not in OWNERS:
        errors.append("invalid_owner")
    if result.get("recommended_action") not in ACTIONS:
        errors.append("invalid_action")
    if result.get("proxy_label_support") not in {"supported", "contradicted", "insufficient"}:
        errors.append("invalid_proxy_support")
    allowed_ids = set(payload["allowed_evidence_ids"])
    cited_ids = result.get("cited_evidence_ids", [])
    if not isinstance(cited_ids, list) or set(cited_ids) - allowed_ids:
        errors.append("hallucinated_evidence_id")
    allowed_features = set(payload["active_feature_names"])
    cited_features = result.get("cited_feature_names", [])
    if not isinstance(cited_features, list) or set(cited_features) - allowed_features:
        errors.append("hallucinated_feature_name")
    if not isinstance(result.get("rationale"), str) or not result["rationale"].strip():
        errors.append("missing_rationale")
    return errors


def normalize_citations(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    parsed = dict(result.get("parsed_response") or {})
    allowed_ids = set(payload["allowed_evidence_ids"])
    allowed_features = set(payload["active_feature_names"])
    original_ids = parsed.get("cited_evidence_ids", [])
    original_features = parsed.get("cited_feature_names", [])
    parsed["cited_evidence_ids"] = [item for item in original_ids if item in allowed_ids]
    parsed["cited_feature_names"] = [item for item in original_features if item in allowed_features]
    normalized["parsed_response"] = parsed
    normalized["normalization_changes"] = {
        "removed_evidence_ids": len(original_ids) - len(parsed["cited_evidence_ids"]),
        "removed_feature_names": len(original_features) - len(parsed["cited_feature_names"]),
    }
    normalized["validator_errors"] = validate(payload, parsed) if normalized.get("api_success") else ["api_or_parse_failure"]
    return normalized


def call_deepseek(client: httpx.Client, url: str, key: str, model: str, max_tokens: int, retries: int, payload: dict[str, Any]) -> dict[str, Any]:
    system, user = prompt_for(payload)
    prompt_hash = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = parse_object(raw)
            return {
                "case_id": payload["case_id"],
                "model": model,
                "prompt_sha256": prompt_hash,
                "attempt_count": attempt,
                "api_success": True,
                "parsed_response": parsed,
                "validator_errors": validate(payload, parsed),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt)
    return {
        "case_id": payload["case_id"],
        "model": model,
        "prompt_sha256": prompt_hash,
        "attempt_count": retries,
        "api_success": False,
        "error": last_error,
        "parsed_response": None,
        "validator_errors": ["api_or_parse_failure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--stage01-cards", type=Path, required=True)
    parser.add_argument("--stage02", type=Path, required=True)
    parser.add_argument("--stage03", type=Path, required=True)
    parser.add_argument("--stage04-features", type=Path, required=True)
    parser.add_argument("--stage04-supervision", type=Path, required=True)
    parser.add_argument("--stage04-oof", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-margin", type=float, default=0.12)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces, payloads = compile_traces(args)
    trace_path = args.output_dir / "05_case_error_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    load_env(args.env_file)
    key = os.environ["DEEPSEEK_API_KEY"]
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    max_tokens = int(os.environ.get("MAX_TOKENS", "4096"))
    retries = int(os.environ.get("MAX_RETRY", "2"))
    selected_ids = sorted(payloads)
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    cache_path = args.output_dir / "05_deepseek_cache.jsonl"
    cached = {row["case_id"]: row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    results = []
    with httpx.Client(timeout=180.0) as client, cache_path.open("a", encoding="utf-8") as cache:
        for index, case_id in enumerate(selected_ids, 1):
            system, user = prompt_for(payloads[case_id])
            prompt_hash = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
            prior = cached.get(case_id)
            if prior and prior.get("model") == model and prior.get("prompt_sha256") == prompt_hash:
                result = prior
                source = "cache"
            else:
                result = call_deepseek(client, url, key, model, max_tokens, retries, payloads[case_id])
                cache.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                cache.flush()
                source = "api"
            result = normalize_citations(payloads[case_id], result)
            results.append(result)
            print(f"DeepSeek harness ({index}/{len(selected_ids)}) case={case_id} source={source} valid={not result['validator_errors']}")

    result_path = args.output_dir / "05_error_attributions.jsonl"
    with result_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    valid = [row["parsed_response"] for row in results if row["api_success"] and not row["validator_errors"]]
    clusters = {
        "primary_error_type_counts": Counter(row["primary_error_type"] for row in valid),
        "owner_module_counts": Counter(row["owner_module"] for row in valid),
        "recommended_action_counts": Counter(row["recommended_action"] for row in valid),
        "proxy_label_support_counts": Counter(row["proxy_label_support"] for row in valid),
        "reusable_error_pattern_counts": Counter(row.get("reusable_error_pattern", "") for row in valid),
    }
    (args.output_dir / "05_error_clusters.json").write_text(json.dumps(clusters, indent=2, default=dict), encoding="utf-8")

    input_paths = {
        "stage01_cards": args.stage01_cards,
        "stage02": args.stage02,
        "stage03": args.stage03,
        "stage04_features": args.stage04_features,
        "stage04_supervision": args.stage04_supervision,
        "stage04_oof": args.stage04_oof,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "audit_only": True,
        "case_count": len(traces),
        "selected_case_count": len(payloads),
        "processed_deepseek_case_count": len(results),
        "valid_attribution_count": len(valid),
        "api_failure_count": sum(not row["api_success"] for row in results),
        "validator_failure_count": sum(bool(row["validator_errors"]) for row in results),
        "citation_normalization_count": sum(
            row.get("normalization_changes", {}).get("removed_evidence_ids", 0)
            + row.get("normalization_changes", {}).get("removed_feature_names", 0)
            for row in results
        ),
        "observed_issue_counts": Counter(issue for row in traces for issue in row["observed_issues"]),
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "hard_gates": {
            "all_cases_traced": len(traces) == 541,
            "all_selected_cases_processed": len(results) == min(len(payloads), args.limit or len(payloads)),
            "api_failures_zero": all(row["api_success"] for row in results),
            "validator_failures_zero": all(not row["validator_errors"] for row in results),
            "attributions_audit_only": True,
        },
    }
    audit["release_gate_passed"] = all(audit["hard_gates"].values())
    (args.output_dir / "05_build_audit.json").write_text(json.dumps(audit, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=dict))


if __name__ == "__main__":
    main()
