#!/usr/bin/env python3
"""Complete stage 02 with evidence-locked DeepSeek policy qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


SCHEMA_VERSION = "02_evidence_locked_policy_trajectories_v1.1_deepseek"
QUALIFIER_VERSION = "deepseek_policy_qualifier_v1.1"
ALLOWED_DECISIONS = {
    "confirmed_policy_trigger",
    "confirmed_symptom_only",
    "uncertain_policy_review",
    "reject_context_only",
}
ALLOWED_REASONS = {
    "threshold_met",
    "threshold_not_met",
    "insufficient_evidence",
    "ambiguous_scope_or_state",
    "context_not_current_patient",
}
FIELD_GUIDANCE = {
    "critical_airway_or_gas_exchange": "Step A only for current severe airway or gas-exchange failure requiring immediate life-saving intervention; isolated wording or mild dyspnea is insufficient.",
    "active_seizure_or_unresponsiveness": "Step A only for a currently observed active seizure or unresponsiveness requiring immediate life-saving intervention.",
    "recent_seizure_high_risk": "Step B for a recent/current seizure creating a high-risk situation; historical seizure disorder alone is insufficient.",
    "critical_perfusion_or_hemorrhage": "Step A only for current hemorrhage with evidence of critical perfusion compromise requiring immediate life-saving intervention.",
    "respiratory_high_risk": "Step B for current respiratory distress, rest dyspnea, marked deterioration, or other directly supported high-risk respiratory features; mild exertional symptoms alone are insufficient.",
    "acute_coronary_risk": "Step B only when the current chest symptom pattern creates a plausible high-risk coronary/cardiopulmonary situation; clearly benign, local, reproducible, or weakly supported discomfort is symptom-only or uncertain.",
    "focal_neurologic_risk": "Step B for a current focal neurologic deficit or directly supported stroke-like presentation.",
    "acute_altered_mental_status": "Step B for new current confusion, lethargy, disorientation, or comparable acute mental-status change; family history, disputed reports, or baseline state are insufficient.",
    "active_self_or_other_harm": "Step B for current suicidal/homicidal ideation, plan, intent, or directly supported active danger; ordinary medication use and historical thoughts alone are insufficient.",
    "toxic_ingestion_high_risk": "Step B for actual or credible possible toxic ingestion requiring urgent evaluation; ordinary therapeutic medication use is insufficient.",
    "pregnancy_high_risk": "Step B only when current pregnancy/postpartum context is combined with a directly supported high-risk symptom such as concerning pain, bleeding, syncope, dyspnea, chest symptoms, or severe headache.",
    "immunocompromised_infection": "Step B for immunocompromised/transplant context with a current infectious presentation that constitutes high risk.",
    "active_bleeding_high_risk": "Step B for current clinically significant or uncontrolled bleeding with directly supported severity or risk context; minor, stopped, historical, or uncertain bleeding is insufficient.",
    "severe_pain_or_distress": "Step B only for current severe pain/distress supported by a valid pain score, severe language, functional limitation, treatment failure, or similarly strong evidence; ordinary pain mentions are insufficient.",
    "high_risk_headache": "Step B for thunderclap headache or an explicitly current headache with a directly supported red flag; a historical headache, medication indication, or unclear current headache is insufficient.",
    "organ_or_limb_threat": "Step B only when current eye/testicular/limb evidence supports a plausible organ- or limb-threatening condition, not merely the symptom name.",
    "high_risk_trauma": "Step B only for an actual current traumatic event with high-risk mechanism, neurologic deficit, functional loss, syncope, or altered mental status. Near-fall, dizziness-related unsteadiness, or a minor wound alone is insufficient.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_env(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def prompt_for(candidates: list[dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are a bounded ESI v5 Step A/B policy qualifier. Use only the supplied evidence bundle. "
        "Do not diagnose, invent facts, infer missing vital signs, change the policy field or policy step, "
        "or treat a nurse question as patient-positive evidence. A symptom being present does not by itself "
        "prove a policy trigger. If deterministic_route is confirmed_symptom_or_context_only, required policy "
        "conditions are missing and confirmed_policy_trigger is forbidden. When scope, current state, severity, or threshold is not directly supported, "
        "choose uncertain_policy_review. Return one JSON object only."
    )
    items = [{
        "candidate_id": candidate["policy_candidate_id"],
        "policy_field": candidate["policy_field"],
        "fixed_policy_step": candidate["policy_step"],
        "policy_rule_id": candidate["policy_rule_id"],
        "field_guidance": FIELD_GUIDANCE[candidate["policy_field"]],
        "anchor_atom_ids": candidate["anchor_atom_ids"],
        "modifier_atom_ids": candidate["modifier_atom_ids"],
        "semantic_modifiers": candidate["semantic_modifiers"],
        "deterministic_route": candidate["deterministic_route"],
        "missing_requirements": candidate["missing_requirements"],
        "evidence_bundle": candidate["evidence_bundle"],
    } for candidate in candidates]
    payload = {
        "candidates": items,
        "allowed_decisions": sorted(ALLOWED_DECISIONS),
        "allowed_reason_codes": sorted(ALLOWED_REASONS),
        "response_contract": {
            "decisions": [{
                "candidate_id": "must exactly match one input",
                "policy_field": "must exactly match that input",
                "decision": "one allowed decision",
                "reason_code": "one allowed reason code",
                "supporting_evidence_ids": "only evidence_id values supplied for that candidate",
                "rationale": "one concise evidence-bound sentence",
            }],
        },
    }
    return system, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def candidate_hash(candidate: dict[str, Any]) -> str:
    payload = {"qualifier_version": QUALIFIER_VERSION, "candidate": candidate}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def parse_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def validate(candidate: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    errors = []
    if parsed.get("candidate_id") != candidate["policy_candidate_id"]:
        errors.append("candidate_id_mismatch")
    if parsed.get("policy_field") != candidate["policy_field"]:
        errors.append("policy_field_mismatch")
    if parsed.get("decision") not in ALLOWED_DECISIONS:
        errors.append("invalid_decision")
    if parsed.get("reason_code") not in ALLOWED_REASONS:
        errors.append("invalid_reason_code")
    used = parsed.get("supporting_evidence_ids")
    if not isinstance(used, list) or any(not isinstance(item, str) for item in used):
        errors.append("invalid_evidence_id_list")
        used = []
    supplied = set(candidate["evidence_ids"])
    if set(used) - supplied:
        errors.append("hallucinated_evidence_id")
    if parsed.get("decision") == "confirmed_policy_trigger":
        if candidate["deterministic_route"] != "needs_llm_policy_qualification":
            errors.append("deterministic_policy_gate_not_met")
        anchor_ids = {
            item["evidence_id"]
            for item in candidate["evidence_bundle"]
            if item.get("clinical_atom_id") in set(candidate["anchor_atom_ids"])
        }
        if not set(used) & anchor_ids:
            errors.append("confirmed_trigger_missing_anchor_evidence")
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("missing_rationale")
    return errors


def call_model(
    client: httpx.Client,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    retries: int,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system, user = prompt_for(candidates)
    prompt_hash = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = parse_object(raw)
            items = parsed.get("decisions")
            if not isinstance(items, list):
                raise ValueError("missing decisions list")
            by_id = {item.get("candidate_id"): item for item in items if isinstance(item, dict)}
            if set(by_id) != {item["policy_candidate_id"] for item in candidates}:
                raise ValueError("batch candidate coverage mismatch")
            return [{
                "policy_candidate_id": candidate["policy_candidate_id"],
                "candidate_sha256": candidate_hash(candidate),
                "model": model,
                "prompt_sha256": prompt_hash,
                "attempt_count": attempt,
                "api_success": True,
                "raw_response": raw,
                "parsed_response": by_id[candidate["policy_candidate_id"]],
                "validator_errors": validate(candidate, by_id[candidate["policy_candidate_id"]]),
            } for candidate in candidates]
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return [{
        "policy_candidate_id": candidate["policy_candidate_id"],
        "candidate_sha256": candidate_hash(candidate),
        "model": model,
        "prompt_sha256": prompt_hash,
        "attempt_count": retries,
        "api_success": False,
        "error": last_error,
        "raw_response": None,
        "parsed_response": None,
        "validator_errors": ["api_or_parse_failure"],
    } for candidate in candidates]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--expected-instance-count", type=int, default=1010)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.environ["DEEPSEEK_API_KEY"]
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    max_tokens = int(os.environ.get("MAX_TOKENS", "500"))
    retries = int(os.environ.get("MAX_RETRY", "3"))
    url = base_url + "/chat/completions"

    dry_audit = json.loads((args.dry_output_dir / "02_build_audit.json").read_text(encoding="utf-8-sig"))
    if not dry_audit.get("release_gate_passed") or not dry_audit.get("dry_run"):
        raise SystemExit("Dry-run artifact has not passed its release gate")
    candidates = read_jsonl(args.dry_output_dir / "02_policy_candidates.jsonl")
    trajectory_rows = read_jsonl(args.dry_output_dir / "02_qualified_trajectories.jsonl")
    model_candidates = candidates
    if args.limit is not None:
        model_candidates = model_candidates[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "02_llm_response_cache.jsonl"
    cached = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cached[row["policy_candidate_id"]] = row

    decisions = []
    with httpx.Client(timeout=120.0) as client, cache_path.open("a", encoding="utf-8") as cache_stream:
        for start in range(0, len(model_candidates), args.batch_size):
            batch = model_candidates[start : start + args.batch_size]
            batch_results = {}
            pending = []
            for candidate in batch:
                prior = cached.get(candidate["policy_candidate_id"])
                if prior and prior.get("model") == model and prior.get("candidate_sha256") == candidate_hash(candidate):
                    reused = dict(prior)
                    reused["result_source"] = "cache"
                    batch_results[candidate["policy_candidate_id"]] = reused
                else:
                    pending.append(candidate)
            for result in (call_model(client, url, api_key, model, max_tokens, retries, pending) if pending else []):
                cache_stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                cache_stream.flush()
                result["result_source"] = "api"
                batch_results[result["policy_candidate_id"]] = result
            for offset, candidate in enumerate(batch, 1):
                result = batch_results[candidate["policy_candidate_id"]]
                decisions.append(result)
                index = start + offset
                print(f"DeepSeek 02 ({index}/{len(model_candidates)}) {candidate['policy_candidate_id']} {result['result_source']} success={result['api_success']}")

    decision_by_id = {row["policy_candidate_id"]: row for row in decisions}
    final_candidates = []
    preference_pairs = []
    validation_errors = []
    for candidate in candidates:
        final = dict(candidate)
        final["dry_run"] = False
        decision = decision_by_id.get(candidate["policy_candidate_id"])
        if decision is None:
            final["validated_status"] = "uncertain_policy_review"
            final["fallback_reason"] = "missing_llm_decision"
            validation_errors.append({"policy_candidate_id": candidate["policy_candidate_id"], "type": "missing_llm_decision"})
            final_candidates.append(final)
            continue
        parsed = decision.get("parsed_response") or {}
        raw_decision = parsed.get("decision")
        errors = decision["validator_errors"]
        if not decision["api_success"] or errors:
            validated = "uncertain_policy_review"
        elif raw_decision == "confirmed_policy_trigger":
            validated = "confirmed_step_a_trigger" if candidate["policy_step"] == "A" else "confirmed_step_b_policy_trigger"
        else:
            validated = raw_decision
        final.update({
            "model_raw_decision": raw_decision,
            "validated_status": validated,
            "llm_reason_code": parsed.get("reason_code"),
            "llm_supporting_evidence_ids": parsed.get("supporting_evidence_ids") or [],
            "llm_rationale": parsed.get("rationale"),
            "validator_errors": errors,
            "llm_model": decision["model"],
        })
        if errors:
            preference_pairs.append({
                "policy_candidate_id": candidate["policy_candidate_id"],
                "rejected_model_decision": raw_decision,
                "chosen_validated_status": validated,
                "validator_errors": errors,
                "audit_only": True,
            })
        final_candidates.append(final)

    final_by_id = {row["policy_candidate_id"]: row for row in final_candidates}
    final_outputs = []
    for row in trajectory_rows:
        merged = [final_by_id[item["policy_candidate_id"]] for item in row["policy_candidates"]]
        current = dict(row)
        current["schema_version"] = SCHEMA_VERSION
        current["policy_candidates"] = merged
        current["confirmed_step_a_signals"] = [item for item in merged if item.get("validated_status") == "confirmed_step_a_trigger"]
        current["confirmed_step_b_signals"] = [item for item in merged if item.get("validated_status") == "confirmed_step_b_policy_trigger"]
        current["dry_run"] = False
        final_outputs.append(current)

    status_counts = Counter(row.get("validated_status") for row in final_candidates)
    positive_by_field = Counter(
        row["policy_field"]
        for row in final_candidates
        if row.get("validated_status") in {"confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"}
    )
    hallucination_count = sum("hallucinated_evidence_id" in row["validator_errors"] for row in decisions)
    fallback_count = sum(not row["api_success"] for row in decisions)
    downgrade_count = sum(bool(row["validator_errors"]) for row in decisions)
    missing_count = sum(row.get("fallback_reason") == "missing_llm_decision" for row in final_candidates)
    approved_with_error = sum(
        bool(row.get("validator_errors"))
        and row.get("validated_status") in {"confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"}
        for row in final_candidates
    )
    hard_gates = {
        "dry_preflight_release_gate": bool(dry_audit.get("release_gate_passed")),
        "expected_instance_count": len(final_outputs) == args.expected_instance_count,
        "all_policy_candidates_decided": len(decisions) == len(model_candidates) and missing_count == 0,
        "api_failure_fallback_zero": fallback_count == 0,
        "hallucinated_evidence_approved_zero": hallucination_count == 0,
        "approved_claim_with_validator_error_zero": approved_with_error == 0,
        "invalid_measurement_consumption_zero": dry_audit.get("invalid_measurement_consumed_count") == 0,
        "upstream_conflict_zero": dry_audit.get("conflict_count") == 0,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "qualifier_version": QUALIFIER_VERSION,
        "model": model,
        "dry_output_dir": str(args.dry_output_dir.resolve()),
        "dry_candidate_sha256": sha256(args.dry_output_dir / "02_policy_candidates.jsonl"),
        "processed_instance_count": len(final_outputs),
        "policy_candidate_count": len(final_candidates),
        "llm_candidate_count": len(model_candidates),
        "deterministic_llm_eligible_candidate_count": sum(row["deterministic_route"] == "needs_llm_policy_qualification" for row in candidates),
        "llm_success_count": sum(row["api_success"] for row in decisions),
        "llm_failure_fallback_count": fallback_count,
        "validator_downgrade_count": downgrade_count,
        "evidence_id_hallucination_count": hallucination_count,
        "validated_status_counts": dict(status_counts),
        "positive_policy_field_counts": dict(positive_by_field),
        "confirmed_step_a_signal_count": status_counts["confirmed_step_a_trigger"],
        "confirmed_step_b_signal_count": status_counts["confirmed_step_b_policy_trigger"],
        "preference_pair_count": len(preference_pairs),
        "validation_error_count": len(validation_errors),
        "hard_gates": hard_gates,
        "development_gate_passed": all(hard_gates.values()),
        "release_gate_passed": all(hard_gates.values()),
        "dry_run": False,
    }

    write_jsonl(args.output_dir / "02_qualified_trajectories.jsonl", final_outputs)
    write_jsonl(args.output_dir / "02_policy_candidates.jsonl", final_candidates)
    write_jsonl(args.output_dir / "02_llm_decisions.jsonl", decisions)
    write_jsonl(cache_path, decisions)
    write_jsonl(args.output_dir / "02_preference_pairs.jsonl", preference_pairs)
    write_jsonl(args.output_dir / "validation_errors.jsonl", validation_errors)
    (args.output_dir / "02_build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    artifact_names = (
        "02_qualified_trajectories.jsonl", "02_policy_candidates.jsonl", "02_llm_decisions.jsonl",
        "02_preference_pairs.jsonl", "validation_errors.jsonl", "02_build_audit.json",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "dry_audit": {"path": str((args.dry_output_dir / "02_build_audit.json").resolve()), "sha256": sha256(args.dry_output_dir / "02_build_audit.json")},
            "dry_candidates": {"path": str((args.dry_output_dir / "02_policy_candidates.jsonl").resolve()), "sha256": sha256(args.dry_output_dir / "02_policy_candidates.jsonl")},
        },
        "source_sha256": {Path(__file__).name: sha256(Path(__file__))},
        "artifacts": {name: sha256(args.output_dir / name) for name in artifact_names},
    }
    (args.output_dir / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
