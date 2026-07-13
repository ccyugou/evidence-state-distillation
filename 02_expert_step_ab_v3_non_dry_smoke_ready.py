#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence-locked Claim Qualification Engine, 02 v3.

This module consumes only the prediction-safe 01 evidence timeline.  It does
not read labels, audit metadata, triage traces, personas, or raw transcripts.
Its contract ends at claim state and bounded Step A/B policy qualification;
final ESI, current-state reconciliation, and resource prediction belong to
downstream modules.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LLM_IMPORT_ERROR: Optional[str] = None
try:
    from llm_client import call_gpt
except Exception as exc:
    call_gpt = None
    LLM_IMPORT_ERROR = str(exc)


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "01_timelines_v4_5_4_narrow_semantic_patch_final" / "evidence_timeline.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "02_claim_qualification_v3_non_dry_smoke50_v1"
SCHEMA_VERSION = "02_claim_qualification_v3.0_evidence_locked_group_relative"
EXPECTED_TIMELINE_SCHEMA = "01_prediction_safe_timeline_v4.5.4_frozen_repaired"
EXPECTED_INSTANCE_COUNT = 688
EXPECTED_CASE_COUNT = 541
EXPECTED_DUPLICATE_CASE_COUNT = 147
LLM_MAX_RETRIES = 2
LLM_TIMEOUT_SECONDS = 120

FORBIDDEN_KEYS = {
    "ground_truth", "acuity", "triage", "recorded_triage", "turn_triage",
    "triage_state", "persona", "patient_persona", "nurse_persona", "model",
    "label", "gold_label", "cross_realization_metadata",
}
STRUCTURED_MEASUREMENT_TYPES = {"structured_vital", "structured_pain_score", "vital"}
QUESTION_LAYERS = {"nurse_question", "nurse_instruction_or_process", "nurse_mixed_utterance"}
PROCESS_LAYERS = {"nurse_instruction_or_process"}
ALLOWED_STEP_A_FIELDS = {
    "cardiac_arrest", "apnea", "unresponsive", "airway_failure",
    "shock_or_hypoperfusion", "profound_hypoglycemia",
    "immediate_lifesaving_intervention_required",
    "major_hemorrhage_requiring_immediate_intervention",
}
ALLOWED_STEP_B_FIELDS = {
    "chest_pain_or_acs_concern", "dyspnea", "respiratory_distress",
    "stroke_signs", "altered_mental_status", "syncope", "dizziness",
    "active_bleeding", "gi_bleeding_melena", "severe_pain", "severe_distress",
    "suicidal_ideation", "homicidal_ideation", "psychosis_or_violent_behavior",
    "pregnancy_postpartum_high_risk", "ectopic_pregnancy_suspected",
    "immunocompromised_fever", "transplant_fever_or_infection", "seizure_postictal",
    "anaphylaxis_or_airway_threat", "toxic_ingestion_or_overdose",
    "sexual_assault_or_domestic_violence", "high_risk_trauma",
    "testicular_or_ovarian_torsion", "severe_flank_pain_or_renal_colic",
    "pediatric_fever",
}
ALLOWED_FIELDS = ALLOWED_STEP_A_FIELDS | ALLOWED_STEP_B_FIELDS
# Conflict hard-gating is limited to formal Step A/B claims plus pregnancy
# status. Broad symptom families can contain distinct targets (for example,
# fever versus chills or pain in different locations) and must not be treated
# as same-atom contradictions merely because they share a broad field name.
CLAIM_CONFLICT_AUDIT_FIELDS = ALLOWED_FIELDS | {"pregnancy_status_or_context"}

CUE_FIELD_REGISTRY = {
    "chest_pain_or_pressure_mention": "chest_pain_or_acs_concern",
    "dyspnea_or_respiratory_symptom_mention": "dyspnea",
    "respiratory_distress_observable": "respiratory_distress",
    "stroke_or_focal_neuro_deficit": "stroke_signs",
    "new_ams_confusion_lethargy_disorientation": "altered_mental_status",
    "syncope_or_near_syncope": "syncope",
    "dizziness_or_lightheadedness": "dizziness",
    "gi_bleeding_melena_or_hemodynamic_concern": "gi_bleeding_melena",
    "active_bleeding": "active_bleeding",
    "fever_or_infection": "fever_or_infection",
    "abdominal_pain_mention": "abdominal_pain",
    "severe_pain_language": "severe_pain",
    "active_suicidal_or_homicidal_risk": "suicidal_ideation",
    "psychosis_or_violent_behavior": "psychosis_or_violent_behavior",
    "pregnancy_postpartum_high_risk": "pregnancy_status_or_context",
    "immunocompromised_or_transplant_with_fever_infection": "immunocompromised_fever",
    "high_risk_trauma_mechanism_or_penetrating_trauma": "high_risk_trauma",
    "seizure_postictal": "seizure_postictal",
    "anaphylaxis_or_airway_threat": "anaphylaxis_or_airway_threat",
    "actual_toxic_ingestion_or_overdose": "toxic_ingestion_or_overdose",
    "overdose_method_ideation_or_plan": "suicidal_ideation",
    "sexual_assault_or_domestic_violence_distress": "sexual_assault_or_domestic_violence",
    "testicular_or_ovarian_torsion_concern": "testicular_or_ovarian_torsion",
    "severe_flank_pain_or_renal_colic": "severe_flank_pain_or_renal_colic",
}

LLM_ALLOWED_STATUSES = {
    "confirmed_step_a_trigger",
    "confirmed_step_b_policy_trigger",
    "confirmed_symptom_only",
    "explicit_negative",
    "uncertain_policy_review",
    "reject_insufficient_context",
}
STRONG_SEVERE_PAIN_RE = re.compile(
    r"\b(unbearable|excruciating|pain\s+is\s+so\s+bad|hurts?\s+too\s+much|"
    r"burning\s+like\s+fire|very\s+bad\s+burning|can'?t\s+(?:stand|walk|move)|"
    r"can\s+barely\s+stand)\b",
    re.IGNORECASE,
)

STEP_A_CUE_TO_FIELD = {
    "cardiac_arrest": "cardiac_arrest", "apnea": "apnea", "unresponsive": "unresponsive",
    "airway_failure": "airway_failure", "shock_or_hypoperfusion": "shock_or_hypoperfusion",
    "profound_hypoglycemia": "profound_hypoglycemia",
    "immediate_lifesaving_intervention_required": "immediate_lifesaving_intervention_required",
    "major_hemorrhage_requiring_immediate_intervention": "major_hemorrhage_requiring_immediate_intervention",
}


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows is not None and len(rows) >= max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def contains_forbidden_key(value: Any, path: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            if key_lower in FORBIDDEN_KEYS or any(
                token in key_lower for token in ("recorded_triage", "turn_triage", "triage_state")
            ):
                found.append(f"{path}.{key}" if path else str(key))
            found.extend(contains_forbidden_key(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(contains_forbidden_key(child, f"{path}[{idx}]"))
    return found


def policy_of(cue: Dict[str, Any]) -> Dict[str, Any]:
    return cue.get("local_evidence_use_policy") or cue.get("evidence_use_policy") or {}


def measurement_usable(event: Dict[str, Any]) -> bool:
    return bool(
        event.get("usable_for_clinical_reasoning") is True
        and (event.get("evidence_use_policy") or {}).get("allowed") is True
        and not event.get("quarantine_reason")
    )


def preflight_timeline(timeline: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    errors: List[str] = []
    quarantined: List[Dict[str, Any]] = []
    if timeline.get("timeline_schema_version") != EXPECTED_TIMELINE_SCHEMA:
        errors.append("unexpected_timeline_schema")
    for key in ("instance_id", "case_id", "run_uuid", "events", "prompt_safe_lines"):
        if key not in timeline:
            errors.append(f"missing_top_level:{key}")
    errors.extend(f"forbidden_key:{x}" for x in contains_forbidden_key(timeline))
    events = timeline.get("events") if isinstance(timeline.get("events"), list) else []
    prompts = timeline.get("prompt_safe_lines") if isinstance(timeline.get("prompt_safe_lines"), list) else []
    if len(events) != len(prompts):
        errors.append("event_prompt_count_mismatch")
    ids = [ev.get("evidence_id") for ev in events if isinstance(ev, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_evidence_id")
    event_ids = set(ids)
    required_cue_fields = {
        "local_polarity", "local_temporality", "clinical_status", "scope_resolution",
        "local_source_layer", "local_evidence_use_policy", "clause_span_id",
        "inherits_from_evidence_id",
    }
    for ev in events:
        if not isinstance(ev, dict):
            errors.append("non_object_event")
            continue
        eid = ev.get("evidence_id")
        if ev.get("event_type") in STRUCTURED_MEASUREMENT_TYPES:
            if "usable_for_clinical_reasoning" not in ev or "quarantine_reason" not in ev:
                errors.append(f"measurement_contract_missing:{eid}")
            if not measurement_usable(ev):
                quarantined.append({
                    "evidence_id": eid,
                    "event_type": ev.get("event_type"),
                    "quarantine_reason": ev.get("quarantine_reason"),
                    "raw_value": ev.get("raw_value"),
                })
        for cue in ev.get("clinical_cue_spans") or []:
            missing = sorted(required_cue_fields - set(cue))
            if missing:
                errors.append(f"cue_contract_missing:{eid}:{','.join(missing)}")
            parent = cue.get("inherits_from_evidence_id")
            if parent and parent not in event_ids:
                errors.append(f"cue_parent_missing:{eid}:{parent}")
            span = cue.get("clause_span_id")
            if span and not str(span).startswith(f"{eid}:"):
                errors.append(f"cue_clause_span_mismatch:{eid}:{span}")
    return sorted(set(errors)), quarantined


def field_for_cue(cue: Dict[str, Any]) -> Tuple[Optional[str], str]:
    cue_type = str(cue.get("cue_type") or "")
    if cue_type == "numeric_severity_scale_mention":
        target = str(cue.get("scale_target_hint") or "unknown")
        if target == "pain":
            return "severe_pain", "pain_scale_support"
        if target == "non_pain":
            return "severity_support", "non_pain_scale_support"
        return "unresolved_scale_support", "unresolved_scale_support"
    if cue_type == "pain_mention":
        return "pain", "pain_anchor"
    if cue_type == "radiation_or_referred_pain_mention":
        return "chest_pain_or_acs_concern", "modifier_only"
    if cue_type in STEP_A_CUE_TO_FIELD:
        return STEP_A_CUE_TO_FIELD[cue_type], "step_a_candidate"
    return CUE_FIELD_REGISTRY.get(cue_type), "clinical_cue"


def llm_candidate_allowed(atom: Dict[str, Any]) -> Tuple[bool, str]:
    """Strict bounded-LLM entry contract; LLM cannot repair upstream state."""
    if atom.get("claim_state") != "present":
        return False, "claim_not_present"
    if atom.get("clinical_assertion_allowed") is not True:
        return False, "clinical_assertion_not_allowed"
    if atom.get("positive_trigger_allowed") is not True:
        return False, "positive_trigger_not_allowed"
    if atom.get("context_only") is True:
        return False, "context_only"
    if atom.get("local_source_layer") in QUESTION_LAYERS | PROCESS_LAYERS:
        return False, "question_or_process_source"
    if atom.get("local_temporality") in {"historical", "past", "resolved", "mixed_current_historical", "uncertain"}:
        return False, "non_current_temporality"
    if atom.get("scale_support_only") or atom.get("atom_role", "").endswith("scale_support"):
        return False, "support_only"
    if atom.get("field_hint") not in ALLOWED_STEP_B_FIELDS:
        return False, "field_not_step_b_policy_field"
    eligibility = atom.get("step_b_candidate_eligible")
    if eligibility is False:
        return False, "step_b_candidate_explicitly_disallowed"
    if atom.get("field_hint") == "pregnancy_status_or_context":
        return False, "pregnancy_status_requires_composition"
    return True, "threshold_ambiguity_policy_review"


def build_llm_prompt(candidate: Dict[str, Any], support_bundle: List[Dict[str, Any]]) -> Tuple[str, str]:
    field = candidate.get("field")
    supports = [
        {"evidence_id": s.get("support_evidence_id"), "support_type": s.get("support_type"),
         "support_role": s.get("support_role"), "span_text": s.get("span_text")}
        for s in support_bundle
        if s.get("support_evidence_id") == candidate.get("evidence_id") or s.get("field_hint") == field
    ][:12]
    system = (
        "You are a bounded ESI Step A/B claim qualifier. Return one JSON object only. "
        "Do not assign final ESI. Do not change polarity, temporality, source layer, or evidence IDs. "
        "A symptom alone is confirmed_symptom_only, not a Step B policy trigger. "
        "Use uncertain_policy_review when a legal current patient claim is high-risk but threshold is unclear."
    )
    user = json.dumps({
        "field": field,
        "candidate_claim": {
            "span_text": candidate.get("span_text"),
            "clause_text": candidate.get("clause_text"),
            "claim_state": candidate.get("claim_state"),
            "local_polarity": candidate.get("local_polarity"),
            "local_temporality": candidate.get("local_temporality"),
            "source_layer": candidate.get("local_source_layer"),
            "scope_resolution": candidate.get("scope_resolution"),
        },
        "supporting_evidence": supports,
        "required_output": {
            "status": sorted(LLM_ALLOWED_STATUSES),
            "policy_strength": ["none", "weak", "moderate", "strong"],
            "supporting_evidence_ids": "array of supplied IDs only",
            "rationale_code": "short enum-like string, no chain of thought",
        },
    }, ensure_ascii=False)
    return system, user


def parse_llm_json(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("llm_response_not_text")
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("malformed_json_boundaries")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("llm_response_not_object")
    return value


def normalize_llm_result(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    model_raw_status = raw.get("status")
    status = str(model_raw_status or "uncertain_policy_review")
    if status not in LLM_ALLOWED_STATUSES:
        status = "uncertain_policy_review"
    allowed_ids = {candidate.get("evidence_id")} | set(candidate.get("_allowed_supporting_evidence_ids") or [])
    supplied = raw.get("supporting_evidence_ids") or []
    supporting = [str(x) for x in supplied if str(x) in allowed_ids]
    if not supporting and candidate.get("evidence_id") and status in {
        "confirmed_step_a_trigger", "confirmed_step_b_policy_trigger", "confirmed_symptom_only"
    }:
        supporting = [candidate.get("evidence_id")]
    # LLM status cannot override upstream negative/question/history contracts.
    if candidate.get("claim_state") != "present" or candidate.get("positive_trigger_allowed") is not True:
        status = "uncertain_policy_review" if candidate.get("claim_state") == "present" else "reject_insufficient_context"
    if candidate.get("field") == "pregnancy_status_or_context":
        status = "confirmed_symptom_only" if candidate.get("claim_state") == "present" else "uncertain_policy_review"
    return {
        "status": status,
        "route": "bounded_llm",
        "route_reason": raw.get("rationale_code") or "bounded_policy_threshold_review",
        "adjudicator": "bounded_llm",
        "policy_strength": str(raw.get("policy_strength") or "none"),
        "supporting_evidence_ids": supporting,
        "raw_llm_result": raw,
        "raw_status": status,
        "model_raw_status": model_raw_status,
        "normalized_raw_status": status,
        "llm_fallback": False,
    }


def call_bounded_llm(candidate: Dict[str, Any], supports: List[Dict[str, Any]]) -> Dict[str, Any]:
    if call_gpt is None:
        return {
            "status": "uncertain_policy_review", "route": "bounded_llm_fallback",
            "route_reason": "llm_unavailable", "adjudicator": "program_fallback",
            "supporting_evidence_ids": [], "raw_status": "llm_unavailable",
            "model_raw_status": None, "normalized_raw_status": "uncertain_policy_review",
            "llm_fallback": True, "llm_failure_reason": "llm_unavailable",
        }
    system, user = build_llm_prompt(candidate, supports)
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(call_gpt, system, user)
        try:
            raw = future.result(timeout=LLM_TIMEOUT_SECONDS)
            executor.shutdown(wait=False, cancel_futures=True)
            candidate_for_validation = dict(candidate)
            candidate_for_validation["_allowed_supporting_evidence_ids"] = {
                s.get("support_evidence_id") for s in supports if s.get("support_evidence_id")
            }
            return normalize_llm_result(parse_llm_json(raw), candidate_for_validation)
        except Exception as exc:
            last_error = str(exc)
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
    return {
        "status": "uncertain_policy_review", "route": "bounded_llm_fallback",
        "route_reason": "llm_failed_safe_abstention", "adjudicator": "program_fallback",
        "supporting_evidence_ids": [], "raw_status": "llm_failed",
        "model_raw_status": None, "normalized_raw_status": "uncertain_policy_review",
        "llm_fallback": True, "llm_failure_reason": last_error or "unknown_llm_failure",
    }


def claim_state(cue: Dict[str, Any], source_layer: str, policy: Dict[str, Any]) -> str:
    status = str(cue.get("clinical_status") or "").lower()
    polarity = str(cue.get("local_polarity") or cue.get("polarity_hint") or "").lower()
    temporality = str(cue.get("local_temporality") or cue.get("temporality_hint") or "").lower()
    if source_layer in QUESTION_LAYERS or source_layer in PROCESS_LAYERS or policy.get("context_only") is True:
        return "question_context"
    if policy.get("clinical_assertion_allowed") is False:
        return "context_only"
    if status in {"question_context", "question"} or polarity == "question":
        return "question_context"
    if status in {"absent", "negated"} or polarity in {"negated", "negative", "absent"}:
        return "absent"
    if temporality in {"historical", "past", "resolved"}:
        return "historical_present"
    if status in {"uncertain", "possible", "suspected", "mixed_scope_possible"} or polarity in {"uncertain", "ambiguous", "mixed"}:
        return "uncertain"
    if polarity == "positive" or status in {"present", "asserted", "current"}:
        return "present"
    return "unsupported"


def make_atom(instance_id: str, event: Dict[str, Any], cue: Dict[str, Any]) -> Dict[str, Any]:
    field, role = field_for_cue(cue)
    policy = policy_of(cue)
    source = cue.get("local_source_layer") or cue.get("source_layer") or event.get("source_layer")
    atom_id = f"{instance_id}:{cue.get('clause_span_id') or event.get('evidence_id')}:{cue.get('cue_type')}:{cue.get('start_char')}"
    eligibility = cue.get("step_b_candidate_eligible") if "step_b_candidate_eligible" in cue else None
    return {
        "evidence_atom_id": atom_id,
        "parent_evidence_id": cue.get("inherits_from_evidence_id") or event.get("evidence_id"),
        "clause_span_id": cue.get("clause_span_id"),
        "cue_type": cue.get("cue_type"),
        "span_text": cue.get("span_text"),
        "start_char": cue.get("start_char"),
        "end_char": cue.get("end_char"),
        "clause_text": cue.get("clause_text") or (cue.get("derived_clause_span") or {}).get("span_text") or event.get("text"),
        "field_hint": field,
        "atom_role": role,
        "local_source_layer": source,
        "local_polarity": cue.get("local_polarity", cue.get("polarity_hint")),
        "local_temporality": cue.get("local_temporality", cue.get("temporality_hint")),
        "clinical_status": cue.get("clinical_status"),
        "scope_resolution": cue.get("scope_resolution"),
        "claim_state": claim_state(cue, source, policy),
        "clinical_assertion_allowed": policy.get("clinical_assertion_allowed") is True,
        "positive_trigger_allowed": policy.get("positive_trigger_allowed") is True,
        "negative_evidence_allowed": policy.get("negative_evidence_allowed") is True,
        "independent_evidence": policy.get("independent_evidence") is True,
        "context_only": policy.get("context_only") is True,
        "candidate_policy": cue.get("candidate_policy") or policy.get("candidate_policy"),
        "step_b_candidate_eligible": eligibility,
        "scale_target_hint": cue.get("scale_target_hint"),
        "scale_support_only": cue.get("scale_support_only") is True or role.endswith("scale_support"),
        "event_seq_idx": event.get("event_seq_idx"),
        "turn": event.get("turn"),
    }


def build_atoms(timeline: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    instance_id = str(timeline.get("instance_id"))
    atoms: List[Dict[str, Any]] = []
    supports: List[Dict[str, Any]] = []
    counters = Counter()
    for event in timeline.get("events") or []:
        etype = event.get("event_type")
        if etype in STRUCTURED_MEASUREMENT_TYPES:
            if not measurement_usable(event):
                counters["invalid_measurement_seen"] += 1
                continue
            if etype == "structured_pain_score":
                support = {
                    "support_evidence_id": event.get("evidence_id"),
                    "support_type": "structured_pain_score",
                    "support_role": "severity_support_not_standalone_trigger",
                    "field_hint": "severe_pain",
                    "value": event.get("normalized_value", event.get("value_numeric", event.get("value"))),
                    "usable_for_clinical_reasoning": True,
                }
                supports.append(support)
                counters["valid_pain_support"] += 1
            else:
                counters["valid_vital_measurement_seen"] += 1
            continue
        if etype == "chief_complaint":
            # Chief complaint is retained as a symptom anchor, never a direct
            # Step B trigger. Field-specific cues remain cue-local below.
            if event.get("value"):
                atoms.append({
                    "evidence_atom_id": f"{instance_id}:{event.get('evidence_id')}:chief_complaint",
                    "parent_evidence_id": event.get("evidence_id"),
                    "clause_span_id": None,
                    "cue_type": "chief_complaint",
                    "span_text": str(event.get("value")),
                    "clause_text": str(event.get("value")),
                    "field_hint": "chief_complaint",
                    "atom_role": "structured_symptom_anchor",
                    "local_source_layer": "structured_clinical_input",
                    "local_polarity": "positive",
                    "local_temporality": "current",
                    "clinical_status": "present",
                    "scope_resolution": "structured_presenting_complaint",
                    "claim_state": "present",
                    "clinical_assertion_allowed": True,
                    "positive_trigger_allowed": False,
                    "negative_evidence_allowed": True,
                    "independent_evidence": True,
                    "context_only": False,
                    "candidate_policy": "anchor_only",
                    "step_b_candidate_eligible": False,
                    "scale_target_hint": None,
                    "scale_support_only": False,
                    "event_seq_idx": event.get("event_seq_idx"),
                    "turn": event.get("turn"),
                })
            continue
        for cue in event.get("clinical_cue_spans") or []:
            atom = make_atom(instance_id, event, cue)
            if atom["local_source_layer"] in QUESTION_LAYERS or atom["claim_state"] in {"question_context", "context_only"}:
                counters["question_or_context_atoms"] += 1
            if atom["atom_role"].endswith("scale_support") or atom["atom_role"] == "pain_anchor":
                supports.append({
                    "support_evidence_id": atom["parent_evidence_id"],
                    "support_type": atom["cue_type"],
                    "support_role": atom["atom_role"],
                    "field_hint": atom["field_hint"],
                    "span_text": atom["span_text"],
                    "clause_span_id": atom["clause_span_id"],
                    "usable_for_clinical_reasoning": atom["clinical_assertion_allowed"],
                })
            atoms.append(atom)
    return atoms, supports, dict(counters)


def qualification_for_atom(atom: Dict[str, Any], dry_run: bool) -> Tuple[Dict[str, Any], str]:
    state = atom["claim_state"]
    field = atom.get("field_hint")
    reason = ""
    status = "reject_insufficient_context"
    route = "deterministic"
    if state == "question_context":
        status, reason = "question_context", "question_or_process_context_only"
    elif state == "context_only":
        status, reason = "reject_not_patient_endorsed", "source_policy_context_only"
    elif state == "absent":
        status, reason = "explicit_negative", "cue_local_negation"
    elif state == "historical_present":
        status, reason = "historical_claim", "historical_temporality_no_current_trigger"
    elif state == "uncertain":
        status, reason = "uncertain_policy_review", "cue_local_uncertainty"
    elif state == "present":
        if field == "chief_complaint":
            status, reason = "confirmed_symptom_only", "structured_chief_complaint_anchor_only"
        elif field == "pain":
            status, reason = "confirmed_symptom_only", "ordinary_pain_anchor_not_severe_pain_trigger"
        elif atom.get("scale_support_only") or atom.get("atom_role", "").endswith("scale_support"):
            status, reason = "support_only", "numeric_scale_support_not_standalone_trigger"
        elif atom.get("positive_trigger_allowed") is not True:
            status, reason = "confirmed_symptom_only", "cue_policy_disallows_standalone_step_b_trigger"
        elif field in ALLOWED_STEP_A_FIELDS:
            status, reason = "confirmed_step_a_trigger", "step_a_cue_local_positive_policy"
        elif field in ALLOWED_STEP_B_FIELDS:
            eligible, eligibility_reason = llm_candidate_allowed(atom)
            if not eligible:
                status, reason = "confirmed_symptom_only", eligibility_reason
            else:
                status = "uncertain_policy_review" if dry_run else "needs_llm_policy_qualification"
                reason = "bounded_policy_threshold_review"
                route = "bounded_llm" if not dry_run else "dry_run_llm_not_called"
        elif field is None:
            status, reason = "reject_insufficient_context", "cue_not_in_claim_registry"
        else:
            status, reason = "confirmed_symptom_only", "unregistered_field"
    else:
        status, reason = "reject_insufficient_context", "unsupported_claim_state"
    return {
        "status": status,
        "route": route,
        "route_reason": reason,
        "adjudicator": "rule_router" if route == "deterministic" else "dry_run_router",
        "confirmed_symptom": status in {"confirmed_symptom_only", "confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"},
        "step_a_trigger": status == "confirmed_step_a_trigger",
        "step_b_policy_trigger": status == "confirmed_step_b_policy_trigger",
        "supporting_evidence_ids": [atom["parent_evidence_id"]] if status in {"confirmed_step_a_trigger", "confirmed_step_b_policy_trigger", "confirmed_symptom_only", "explicit_negative", "historical_claim", "support_only"} else [],
        "raw_status": status,
        "model_raw_status": None,
        "normalized_raw_status": status,
        "llm_fallback": False,
    }, route


def build_candidate(atom: Dict[str, Any], idx: int, qualification: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": f"{atom.get('parent_evidence_id')}::claim::{idx}",
        "evidence_atom_id": atom.get("evidence_atom_id"),
        "evidence_id": atom.get("parent_evidence_id"),
        "clause_span_id": atom.get("clause_span_id"),
        "field": atom.get("field_hint"),
        "cue_type": atom.get("cue_type"),
        "span_text": atom.get("span_text"),
        "start_char": atom.get("start_char"),
        "end_char": atom.get("end_char"),
        "clause_text": atom.get("clause_text"),
        "claim_state": atom.get("claim_state"),
        "clinical_status": atom.get("clinical_status"),
        "local_polarity": atom.get("local_polarity"),
        "local_temporality": atom.get("local_temporality"),
        "local_source_layer": atom.get("local_source_layer"),
        "scope_resolution": atom.get("scope_resolution"),
        "clinical_assertion_allowed": atom.get("clinical_assertion_allowed"),
        "positive_trigger_allowed": atom.get("positive_trigger_allowed"),
        "negative_evidence_allowed": atom.get("negative_evidence_allowed"),
        "independent_evidence": atom.get("independent_evidence"),
        "context_only": atom.get("context_only"),
        "candidate_policy": atom.get("candidate_policy"),
        "step_b_candidate_eligible": atom.get("step_b_candidate_eligible"),
        "scale_target_hint": atom.get("scale_target_hint"),
        "scale_support_only": atom.get("scale_support_only", False),
        "qualification": qualification,
    }


def validate_qualification(candidate: Dict[str, Any], qualification: Dict[str, Any]) -> Dict[str, Any]:
    status = qualification.get("status")
    errors: List[str] = []
    llm_conflicts: List[str] = []
    if qualification.get("route", "").startswith("bounded_llm"):
        if status == "confirmed_step_a_trigger" and candidate.get("field") not in ALLOWED_STEP_A_FIELDS:
            llm_conflicts.append("llm_step_a_status_on_non_step_a_field")
        if status == "confirmed_step_b_policy_trigger" and candidate.get("field") not in ALLOWED_STEP_B_FIELDS:
            llm_conflicts.append("llm_step_b_status_on_non_step_b_field")
        if status == "explicit_negative" and candidate.get("claim_state") == "present":
            llm_conflicts.append("llm_negative_status_conflicts_with_present_claim")
        if candidate.get("claim_state") != "present":
            llm_conflicts.append("llm_target_claim_state_conflict")
    errors.extend(llm_conflicts)
    if status in {"confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"}:
        if candidate.get("claim_state") != "present":
            errors.append("trigger_requires_present_claim")
        if candidate.get("local_source_layer") in QUESTION_LAYERS | PROCESS_LAYERS or candidate.get("context_only"):
            errors.append("trigger_from_question_or_process_source")
        if candidate.get("local_temporality") in {"historical", "past", "resolved", "uncertain"}:
            errors.append("trigger_temporality_not_current")
        if candidate.get("positive_trigger_allowed") is not True:
            errors.append("trigger_policy_not_allowed")
        if candidate.get("field") not in ALLOWED_FIELDS:
            errors.append("trigger_field_not_registered")
    if status == "confirmed_step_b_policy_trigger" and candidate.get("field") == "toxic_ingestion_or_overdose":
        if candidate.get("local_temporality") != "current" or candidate.get("local_polarity") != "positive":
            errors.append("toxic_ingestion_not_current_positive")
    if candidate.get("scale_support_only") and status == "confirmed_step_b_policy_trigger":
        errors.append("numeric_scale_standalone_trigger")
    if candidate.get("field") == "pain" and status in {"confirmed_step_a_trigger", "confirmed_step_b_policy_trigger"}:
        errors.append("ordinary_pain_standalone_trigger")
    if errors:
        qualification = dict(qualification)
        qualification.update({
            "status": "uncertain_policy_review",
            "route": "validator_safe_downgrade",
            "route_reason": ";".join(errors),
            "adjudicator": "program_validator",
            "confirmed_symptom": candidate.get("claim_state") == "present",
            "step_a_trigger": False,
            "step_b_policy_trigger": False,
            "validator_errors": errors,
            "safe_downgrade": True,
            "llm_target_contract_conflicts": llm_conflicts,
            "safe_downgrade_to_uncertain": True,
        })
    else:
        qualification = dict(qualification)
        qualification.setdefault("validator_errors", [])
        qualification.setdefault("safe_downgrade", False)
        qualification.setdefault("llm_target_contract_conflicts", [])
        qualification.setdefault("safe_downgrade_to_uncertain", False)
    return qualification


def build_claim_row(candidate: Dict[str, Any], qualification: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "evidence_atom_id": candidate["evidence_atom_id"],
        "evidence_id": candidate["evidence_id"],
        "clause_span_id": candidate["clause_span_id"],
        "field": candidate["field"],
        "cue_type": candidate["cue_type"],
        "span_text": candidate.get("span_text"),
        "start_char": candidate.get("start_char"),
        "end_char": candidate.get("end_char"),
        "clause_text": candidate.get("clause_text"),
        "claim_state": candidate["claim_state"],
        "status": qualification.get("status"),
        "route": qualification.get("route"),
        "route_reason": qualification.get("route_reason"),
        "local_source_layer": candidate["local_source_layer"],
        "local_polarity": candidate["local_polarity"],
        "local_temporality": candidate["local_temporality"],
        "clinical_status": candidate.get("clinical_status"),
        "scope_resolution": candidate["scope_resolution"],
        "scale_target_hint": candidate.get("scale_target_hint"),
        "scale_support_only": candidate.get("scale_support_only", False),
        "candidate_policy": candidate.get("candidate_policy"),
        "positive_trigger_allowed": candidate.get("positive_trigger_allowed"),
        "clinical_assertion_allowed": candidate.get("clinical_assertion_allowed"),
        "step_b_candidate_eligible": candidate.get("step_b_candidate_eligible"),
        "context_only": candidate.get("context_only"),
        "supporting_evidence_ids": qualification.get("supporting_evidence_ids", []),
        "confirmed_symptom": qualification.get("confirmed_symptom", False),
        "step_a_trigger": qualification.get("step_a_trigger", False),
        "step_b_policy_trigger": qualification.get("step_b_policy_trigger", False),
        "validator_errors": qualification.get("validator_errors", []),
        "safe_downgrade": qualification.get("safe_downgrade", False),
        "raw_status": qualification.get("raw_status"),
        "model_raw_status": qualification.get("model_raw_status"),
        "normalized_raw_status": qualification.get("normalized_raw_status", qualification.get("raw_status")),
        "validated_status": qualification.get("validated_status", qualification.get("status")),
        "llm_fallback": qualification.get("llm_fallback", False),
        "llm_failure_reason": qualification.get("llm_failure_reason"),
        "llm_target_contract_conflicts": qualification.get("llm_target_contract_conflicts", []),
    }


def conflict_audit(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Preserve the conflict for Hermes: this classifies scope geometry but
    # never resolves the clinical disagreement automatically.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        field = claim.get("field")
        if field in CLAIM_CONFLICT_AUDIT_FIELDS and claim.get("claim_state") in {"present", "absent"}:
            groups[(claim.get("evidence_id"), claim.get("clause_span_id"), claim.get("field"))].append(claim)
    rows = []
    for (evidence_id, clause_span_id, field), values in groups.items():
        positives = [x for x in values if x.get("claim_state") == "present"]
        negatives = [x for x in values if x.get("claim_state") == "absent"]
        if not positives or not negatives:
            continue
        category = "unresolved_same_clause_same_field_conflict"
        pair_rows = []
        for positive in positives:
            for negative in negatives:
                same_atom = positive.get("evidence_atom_id") == negative.get("evidence_atom_id")
                p_start, p_end = positive.get("start_char"), positive.get("end_char")
                n_start, n_end = negative.get("start_char"), negative.get("end_char")
                overlap = (
                    isinstance(p_start, int) and isinstance(p_end, int)
                    and isinstance(n_start, int) and isinstance(n_end, int)
                    and p_start < n_end and n_start < p_end
                )
                pair_rows.append({
                    "positive_candidate_id": positive.get("candidate_id"),
                    "negative_candidate_id": negative.get("candidate_id"),
                    "same_evidence_atom": same_atom,
                    "overlapping_span": overlap,
                })
                if same_atom:
                    category = "same_atom_exact_conflict"
                elif overlap and category != "same_atom_exact_conflict":
                    category = "same_target_overlapping_span_conflict"
        rows.append({
            "evidence_id": evidence_id,
            "clause_span_id": clause_span_id,
            "field": field,
            "cue_type": sorted({x.get("cue_type") for x in values}),
            "claim_states": ["absent", "present"],
            "candidate_ids": [x.get("candidate_id") for x in values],
            "claims": [{
                "candidate_id": x.get("candidate_id"),
                "evidence_atom_id": x.get("evidence_atom_id"),
                "span_text": x.get("span_text"),
                "start_char": x.get("start_char"),
                "end_char": x.get("end_char"),
                "local_polarity": x.get("local_polarity"),
                "local_temporality": x.get("local_temporality"),
                "clinical_status": x.get("clinical_status"),
                "scope_resolution": x.get("scope_resolution"),
                "positive_trigger_allowed": x.get("positive_trigger_allowed"),
            } for x in values],
            "conflict_category": category,
            "pair_geometry": pair_rows,
            "requires_review": True,
            "resolution_policy": "no_automatic_conflict_resolution",
            "reason": category,
        })
    return rows


def recall_gap_audit(
    timeline: Dict[str, Any], atoms: List[Dict[str, Any]], supports: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Record upstream gaps without reading raw text or creating a claim."""
    patient_pain = [
        atom for atom in atoms
        if atom.get("field_hint") == "pain"
        and atom.get("claim_state") == "present"
        and atom.get("local_source_layer") == "patient_reported"
    ]
    valid_pain_support = [
        support for support in supports
        if support.get("support_type") == "structured_pain_score"
        and support.get("usable_for_clinical_reasoning") is True
        and isinstance(support.get("value"), (int, float))
        and support.get("value") >= 7
    ]
    severe_present = any(
        atom.get("field_hint") == "severe_pain" and atom.get("claim_state") == "present"
        for atom in atoms
    )
    if patient_pain and valid_pain_support and not severe_present:
        local_text = " ".join(str(a.get("clause_text") or a.get("span_text") or "") for a in patient_pain)
        if STRONG_SEVERE_PAIN_RE.search(local_text):
            gap_type = "probable_upstream_severe_pain_miss"
        elif any(
            a.get("clinical_status") in {"present_with_treatment_failure", "present_persistent_symptom"}
            or "cannot" in str(a.get("clause_text") or "").lower()
            for a in patient_pain
        ):
            gap_type = "policy_composition_needed"
        else:
            gap_type = "expected_support_only"
        return [{
            "instance_id": timeline.get("instance_id"),
            "case_id": timeline.get("case_id"),
            "gap_type": gap_type,
            "reason": "valid_structured_pain_support_plus_patient_pain_anchor_without_severe_pain_cue",
            "patient_pain_evidence_ids": sorted({a.get("parent_evidence_id") for a in patient_pain}),
            "support_evidence_ids": sorted({s.get("support_evidence_id") for s in valid_pain_support}),
            "audit_only": True,
            "automatic_positive_claim": False,
            "strong_expression_detected": bool(STRONG_SEVERE_PAIN_RE.search(local_text)),
        }]
    return []


def preference_rows(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for c in claims:
        raw_status = c.get("raw_status")
        validated_status = c.get("validated_status", c.get("status"))
        corrected = raw_status and raw_status != validated_status
        if not corrected and not c.get("llm_fallback"):
            continue
        rows.append({
            "candidate_id": c["candidate_id"],
            "chosen_status": validated_status,
            "rejected_status": raw_status,
            "preference_label": "validator_correction" if corrected else "safe_abstention_after_llm_failure",
            "feedback_dimensions": {
                "evidence_lock": not bool(c.get("validator_errors")),
                "scope_consistency": not bool(c.get("llm_target_contract_conflicts")),
                "temporality_consistency": c.get("local_temporality") not in {"historical", "mixed_current_historical"},
                "source_policy": c.get("clinical_assertion_allowed") is True and not c.get("context_only"),
                "field_binding": bool(c.get("field")),
                "policy_threshold": validated_status != "confirmed_step_b_policy_trigger" or not c.get("validator_errors"),
                "safe_abstention": validated_status in {"uncertain_policy_review", "reject_insufficient_context"},
            },
            "audit_only": True,
        })
    return rows


def process_timeline(timeline: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    errors, quarantined = preflight_timeline(timeline)
    instance_id = timeline.get("instance_id")
    empty_llm_runtime_counts = {
        "eligible_candidate_count": 0,
        "call_success_count": 0,
        "failure_fallback_count": 0,
        "validator_downgrade_count": 0,
        "normalization_change_count": 0,
    }
    if errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_id": instance_id,
            "case_id": timeline.get("case_id"),
            "run_uuid": timeline.get("run_uuid"),
            "preflight_failed": True,
            "preflight_errors": errors,
            "claims": [],
            "claim_candidates": [],
            "support_bundle": [],
            "invalid_measurements": quarantined,
            "preference_pairs": [],
            "field_candidate_group_diagnostics": [],
            "same_evidence_conflicts": [],
            "recall_gaps": [],
            "llm_runtime_counts": empty_llm_runtime_counts,
        }
    atoms, supports, atom_counts = build_atoms(timeline)
    recall_gaps = recall_gap_audit(timeline, atoms, supports)
    candidates: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    route_counts = Counter()
    llm_runtime_counts = Counter()
    for idx, atom in enumerate(atoms):
        qualification, route = qualification_for_atom(atom, dry_run)
        candidate = build_candidate(atom, idx, qualification)
        candidate["scale_support_only"] = atom.get("scale_support_only")

        if route in {"bounded_llm", "dry_run_llm_not_called"}:
            llm_runtime_counts["eligible_candidate_count"] += 1

        if route == "bounded_llm" and not dry_run:
            qualification = call_bounded_llm(candidate, supports)
            if qualification.get("llm_fallback"):
                llm_runtime_counts["failure_fallback_count"] += 1
            else:
                llm_runtime_counts["call_success_count"] += 1
            if qualification.get("model_raw_status") != qualification.get("normalized_raw_status"):
                llm_runtime_counts["normalization_change_count"] += 1

        qualification.setdefault("raw_status", qualification.get("status"))
        qualification.setdefault("raw_qualification", dict(qualification))
        qualification = validate_qualification(candidate, qualification)
        qualification["validated_status"] = qualification.get("status")
        if qualification.get("safe_downgrade"):
            llm_runtime_counts["validator_downgrade_count"] += 1
        candidate["qualification"] = qualification
        claim = build_claim_row(candidate, qualification)
        candidates.append(candidate)
        claims.append(claim)
        route_counts[qualification.get("status", "missing")] += 1
    conflicts = conflict_audit(claims)
    diagnostics = []
    by_field = defaultdict(list)
    for c in claims:
        if c.get("field"):
            by_field[c["field"]].append(c)
    for field, values in by_field.items():
        diagnostics.append({
            "instance_id": instance_id,
            "field": field,
            "candidate_count": len(values),
            "present_count": sum(v.get("claim_state") == "present" for v in values),
            "absent_count": sum(v.get("claim_state") == "absent" for v in values),
            "uncertain_count": sum(v.get("claim_state") == "uncertain" for v in values),
            "step_a_count": sum(v.get("step_a_trigger") for v in values),
            "step_b_count": sum(v.get("step_b_policy_trigger") for v in values),
            "safe_abstention_count": sum(v.get("status") in {"uncertain_policy_review", "reject_insufficient_context"} for v in values),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "case_id": timeline.get("case_id"),
        "run_uuid": timeline.get("run_uuid"),
        "preflight_failed": False,
        "preflight_errors": [],
        "claims": claims,
        "claim_candidates": candidates,
        "support_bundle": supports,
        "invalid_measurements": quarantined,
        "preference_pairs": preference_rows(claims),
        "field_candidate_group_diagnostics": diagnostics,
        "same_evidence_conflicts": conflicts,
        "recall_gaps": recall_gaps,
        "atom_counts": atom_counts,
        "route_counts": dict(route_counts),
        "llm_runtime_counts": {
            key: int(llm_runtime_counts.get(key, 0))
            for key in empty_llm_runtime_counts
        },
    }


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "main": output_dir / "02_qualified_claims.jsonl",
        "contract": output_dir / "02_to_02d_claim_contract.jsonl",
        "candidate_catalog": output_dir / "claim_candidate_catalog.jsonl",
        "support_catalog": output_dir / "support_bundle_catalog.jsonl",
        "recall_audit": output_dir / "upstream_recall_gap_audit.jsonl",
        "measurement_audit": output_dir / "invalid_measurement_consumption_audit.jsonl",
        "conflict_audit": output_dir / "same_evidence_conflict_audit.jsonl",
        "llm_conflicts": output_dir / "llm_contract_conflicts.jsonl",
        "preference": output_dir / "preference_pairs.jsonl",
        "diagnostics": output_dir / "field_candidate_group_diagnostics.jsonl",
        "audit": output_dir / "02_build_audit.json",
    }


def resume_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "checkpoint": output_dir / "resume_checkpoint.jsonl",
        "manifest": output_dir / "resume_manifest.json",
    }


def write_resume_manifest(path: Path, input_path: Path, timelines: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "input_file": str(input_path.resolve()),
        "dry_run": bool(args.dry_run),
        "limit_cases": args.limit_cases,
        "instance_ids": [timeline.get("instance_id") for timeline in timelines],
    })


def load_resume_checkpoint(
    paths: Dict[str, Path], input_path: Path, timelines: List[Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Dict[str, Any]]:
    if not paths["checkpoint"].exists() or not paths["manifest"].exists():
        raise RuntimeError("--resume requires resume_checkpoint.jsonl and resume_manifest.json in the output directory.")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_ids = [timeline.get("instance_id") for timeline in timelines]
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("input_file") != str(input_path.resolve())
        or bool(manifest.get("dry_run")) != bool(args.dry_run)
        or manifest.get("limit_cases") != args.limit_cases
        or manifest.get("instance_ids") != expected_ids
    ):
        raise RuntimeError("Resume manifest does not match the requested input, mode, or case limit.")
    rows = read_jsonl(paths["checkpoint"])
    completed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id or instance_id in completed:
            raise RuntimeError("Resume checkpoint contains a missing or duplicate instance_id.")
        completed[instance_id] = row
    return completed


def append_resume_checkpoint(path: Path, result: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()


def write_outputs(path_map: Dict[str, Path], results: List[Dict[str, Any]]) -> None:
    claims = []
    contracts = []
    candidates = []
    supports = []
    measurements = []
    conflicts = []
    recall_gaps = []
    preferences = []
    diagnostics = []
    llm_conflicts = []
    for result in results:
        claims.append({
            "schema_version": result["schema_version"],
            "instance_id": result["instance_id"],
            "case_id": result["case_id"],
            "run_uuid": result["run_uuid"],
            "preflight_failed": result["preflight_failed"],
            "preflight_errors": result["preflight_errors"],
            "claims": result["claims"],
        })
        contracts.append({
            "instance_id": result["instance_id"],
            "case_id": result["case_id"],
            "run_uuid": result["run_uuid"],
            "claims": result["claims"],
            "preflight_failed": result["preflight_failed"],
        })
        candidates.extend({"instance_id": result["instance_id"], **x} for x in result["claim_candidates"])
        supports.extend({"instance_id": result["instance_id"], **x} for x in result["support_bundle"])
        measurements.extend({"instance_id": result["instance_id"], **x} for x in result["invalid_measurements"])
        conflicts.extend({"instance_id": result["instance_id"], **x} for x in result["same_evidence_conflicts"])
        # This file is intentionally audit-only: it cannot feed qualification.
        recall_gaps.extend(result.get("recall_gaps", []))
        preferences.extend({"instance_id": result["instance_id"], **x} for x in result["preference_pairs"])
        diagnostics.extend(result["field_candidate_group_diagnostics"])
        for claim in result["claims"]:
            for conflict in claim.get("llm_target_contract_conflicts", []):
                llm_conflicts.append({
                    "instance_id": result["instance_id"],
                    "candidate_id": claim.get("candidate_id"),
                    "conflict": conflict,
                })
    # The recall floor is deliberately an audit-only placeholder in v3; no
    # raw-text fallback is allowed to manufacture a clinical candidate.
    write_jsonl(path_map["main"], claims)
    write_jsonl(path_map["contract"], contracts)
    write_jsonl(path_map["candidate_catalog"], candidates)
    write_jsonl(path_map["support_catalog"], supports)
    write_jsonl(path_map["recall_audit"], recall_gaps)
    write_jsonl(path_map["measurement_audit"], measurements)
    write_jsonl(path_map["conflict_audit"], conflicts)
    write_jsonl(path_map["llm_conflicts"], llm_conflicts)
    write_jsonl(path_map["preference"], preferences)
    write_jsonl(path_map["diagnostics"], diagnostics)


def summarize(results: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    counters = Counter()
    statuses = Counter()
    raw_statuses = Counter()
    validated_statuses = Counter()
    fields = Counter()
    for result in results:
        counters["processed"] += 1
        counters["preflight_failed"] += int(result["preflight_failed"])
        counters["candidate_count"] += len(result["claim_candidates"])
        counters["support_bundle_count"] += len(result["support_bundle"])
        counters["invalid_measurement_seen"] += len(result["invalid_measurements"])
        counters["same_evidence_conflict_count"] += len(result["same_evidence_conflicts"])
        counters["same_atom_exact_conflict_count"] += sum(
            x.get("conflict_category") == "same_atom_exact_conflict"
            for x in result["same_evidence_conflicts"]
        )
        counters["overlapping_span_conflict_count"] += sum(
            x.get("conflict_category") == "same_target_overlapping_span_conflict"
            for x in result["same_evidence_conflicts"]
        )
        counters["unresolved_same_clause_conflict_count"] += sum(
            x.get("conflict_category") == "unresolved_same_clause_same_field_conflict"
            for x in result["same_evidence_conflicts"]
        )
        counters["recall_gap_count"] += len(result.get("recall_gaps", []))
        for key, value in (result.get("llm_runtime_counts") or {}).items():
            counters[f"llm_{key}"] += int(value or 0)
        for claim in result["claims"]:
            statuses[claim.get("status", "missing")] += 1
            raw_statuses[claim.get("raw_status", "missing")] += 1
            validated_statuses[claim.get("validated_status", claim.get("status", "missing"))] += 1
            if claim.get("field"):
                fields[claim["field"]] += 1
    input_ids = [r.get("instance_id") for r in results]
    case_ids = [r.get("case_id") for r in results]
    count_check = {
        "timeline_count": len(results),
        "unique_instance_id_count": len(set(input_ids)),
        "unique_case_id_count": len(set(case_ids)),
        "duplicate_case_id_count": len(case_ids) - len(set(case_ids)),
        "limit_cases": args.limit_cases,
        "expected_instance_count": args.expected_instance_count,
        "expected_case_count": args.expected_case_count,
        "expected_duplicate_case_count": args.expected_duplicate_case_count,
    }
    expected_ok = True
    if args.limit_cases is None:
        if args.expected_instance_count is not None and len(results) != args.expected_instance_count:
            expected_ok = False
        if args.expected_case_count is not None and len(set(case_ids)) != args.expected_case_count:
            expected_ok = False
        if args.expected_duplicate_case_count is not None and count_check["duplicate_case_id_count"] != args.expected_duplicate_case_count:
            expected_ok = False
    else:
        expected_ok = len(results) == args.limit_cases
    count_check["passed"] = expected_ok
    claims_all = [c for r in results for c in r["claims"]]
    question_process_positive_count = sum(
        1 for c in claims_all
        if c.get("step_a_trigger") or c.get("step_b_policy_trigger")
        if c.get("local_source_layer") in QUESTION_LAYERS | PROCESS_LAYERS or c.get("context_only")
    )
    quarantined_consumed_count = sum(
        1 for r in results for c in r["claims"]
        if c.get("evidence_id") in {m.get("evidence_id") for m in r["invalid_measurements"]}
    )
    numeric_standalone_count = sum(
        1 for c in claims_all
        if c.get("step_b_policy_trigger") and c.get("cue_type") == "numeric_severity_scale_mention"
    )
    ordinary_pain_trigger_count = sum(
        1 for c in claims_all
        if c.get("step_b_policy_trigger") and c.get("field") == "pain"
    )
    nonpain_scale_severe_count = sum(
        1 for c in claims_all
        if c.get("cue_type") == "numeric_severity_scale_mention"
        and c.get("field") == "severe_pain"
        and c.get("scale_target_hint") != "pain"
    )
    overdose_ideation_actual_count = sum(
        1 for c in claims_all
        if c.get("cue_type") == "overdose_method_ideation_or_plan"
        and (c.get("field") == "toxic_ingestion_or_overdose" or c.get("step_b_policy_trigger"))
    )
    pregnancy_status_step_b_count = sum(
        1 for c in claims_all
        if c.get("field") == "pregnancy_status_or_context" and c.get("step_b_policy_trigger")
    )
    llm_eligible_count = counters["llm_eligible_candidate_count"]
    llm_success_count = counters["llm_call_success_count"]
    llm_fallback_count = counters["llm_failure_fallback_count"]
    llm_validator_downgrade_count = counters["llm_validator_downgrade_count"]
    llm_normalization_change_count = counters["llm_normalization_change_count"]
    llm_conflict_count = sum(len(c.get("llm_target_contract_conflicts") or []) for c in claims_all)

    deterministic_gates = {
        "preflight_failed_zero": counters["preflight_failed"] == 0,
        "question_process_positive_zero": question_process_positive_count == 0,
        "quarantined_measurement_consumed_zero": quarantined_consumed_count == 0,
        "historical_or_uncertain_step_b_zero": sum(
            1 for c in claims_all
            if c.get("step_b_policy_trigger") and c.get("claim_state") in {"historical_present", "uncertain"}
        ) == 0,
        "numeric_standalone_trigger_zero": numeric_standalone_count == 0,
        "ordinary_pain_standalone_trigger_zero": ordinary_pain_trigger_count == 0,
        "nonpain_scale_to_severe_pain_zero": nonpain_scale_severe_count == 0,
        "overdose_ideation_to_actual_zero": overdose_ideation_actual_count == 0,
        "pregnancy_status_not_step_b": pregnancy_status_step_b_count == 0,
        "same_evidence_conflict_zero": counters["same_evidence_conflict_count"] == 0,
        "count_gate": count_check["passed"],
    }
    llm_runtime_gates = {
        "llm_eligible_candidate_positive": llm_eligible_count > 0,
        "llm_call_success_positive": llm_success_count > 0,
        "llm_call_accounting_complete": llm_success_count + llm_fallback_count == llm_eligible_count,
        "llm_not_all_fallback": llm_fallback_count < llm_eligible_count,
    }
    hard_gates = dict(deterministic_gates)
    if not args.dry_run:
        hard_gates.update(llm_runtime_gates)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "input_file": str(Path(args.input_file).resolve()),
        "input_source_policy": "single_prediction_safe_evidence_timeline_only",
        "dry_run": bool(args.dry_run),
        "processed": counters["processed"],
        "candidate_count": counters["candidate_count"],
        "support_bundle_count": counters["support_bundle_count"],
        "status_counts": dict(statuses),
        "raw_qualification_status_counts": dict(raw_statuses),
        "validated_status_counts": dict(validated_statuses),
        "field_counts": dict(fields),
        "invalid_measurement_seen_count": counters["invalid_measurement_seen"],
        "invalid_measurement_consumed_support_count": quarantined_consumed_count,
        "invalid_measurement_candidate_count": 0,
        "invalid_measurement_llm_prompt_count": 0,
        "question_process_positive_count": question_process_positive_count,
        "numeric_standalone_trigger_count": numeric_standalone_count,
        "ordinary_pain_standalone_trigger_count": ordinary_pain_trigger_count,
        "nonpain_scale_to_severe_pain_count": nonpain_scale_severe_count,
        "overdose_ideation_to_actual_count": overdose_ideation_actual_count,
        "same_evidence_conflict_count": counters["same_evidence_conflict_count"],
        "same_atom_exact_conflict_count": counters["same_atom_exact_conflict_count"],
        "same_target_overlapping_span_conflict_count": counters["overlapping_span_conflict_count"],
        "unresolved_same_clause_same_field_conflict_count": counters["unresolved_same_clause_conflict_count"],
        "llm_target_contract_conflict_count": llm_conflict_count,
        "llm_eligible_candidate_count": llm_eligible_count,
        "llm_call_success_count": llm_success_count,
        "llm_failure_fallback_count": llm_fallback_count,
        "llm_validator_downgrade_count": llm_validator_downgrade_count,
        "llm_normalization_change_count": llm_normalization_change_count,
        "pregnancy_status_step_b_count": pregnancy_status_step_b_count,
        "upstream_recall_gap_count": counters["recall_gap_count"],
        "preflight_failed_count": counters["preflight_failed"],
        "input_count_check": count_check,
        "deterministic_hard_gates": deterministic_gates,
        "llm_runtime_gates": llm_runtime_gates,
        "hard_gates": hard_gates,
        "development_gate_passed": all(deterministic_gates.values()),
        "release_gate_passed": (not args.dry_run) and all(hard_gates.values()),
        "dpo_style_preference_pairs": sum(len(r["preference_pairs"]) for r in results),
        "grpo_style_field_group_diagnostics": sum(len(r["field_candidate_group_diagnostics"]) for r in results),
        "no_raw_text_recall_fallback": True,
        "no_current_state_override": True,
        "no_label_or_audit_input": True,
    }
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 02 v3 evidence-locked claim qualification.")
    parser.add_argument("--input-file", default=str(INPUT_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--expected-instance-count", type=int, default=None)
    parser.add_argument("--expected-case-count", type=int, default=None)
    parser.add_argument("--expected-duplicate-case-count", type=int, default=None)
    parser.add_argument("--use-default-count-expectations", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; bounded candidates stay uncertain.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from per-instance checkpoint in output-dir.")
    args = parser.parse_args()
    if args.use_default_count_expectations:
        args.expected_instance_count = args.expected_instance_count or EXPECTED_INSTANCE_COUNT
        args.expected_case_count = args.expected_case_count or EXPECTED_CASE_COUNT
        args.expected_duplicate_case_count = args.expected_duplicate_case_count or EXPECTED_DUPLICATE_CASE_COUNT
    return args


def enforce_input_contract(input_file: str) -> Path:
    path = Path(input_file).resolve()
    expected = INPUT_FILE.resolve()
    if os.path.normcase(str(path)) != os.path.normcase(str(expected)):
        raise RuntimeError(
            "02 v3 accepts only the frozen prediction-safe timeline: "
            f"{expected}"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    if not args.dry_run and call_gpt is None:
        detail = f" Import error: {LLM_IMPORT_ERROR}" if LLM_IMPORT_ERROR else ""
        raise RuntimeError(
            "Non-dry run requires a working llm_client.call_gpt; the client could not be imported."
            + detail
        )
    input_path = enforce_input_contract(args.input_file)
    timelines = read_jsonl(input_path, max_rows=args.limit_cases)
    if not timelines:
        raise RuntimeError("No timelines loaded")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    resume = resume_paths(output_dir)
    if args.resume and args.overwrite:
        raise RuntimeError("--resume and --overwrite cannot be used together.")
    existing_outputs = [path for path in paths.values() if path.exists()]
    if args.resume:
        if existing_outputs:
            raise RuntimeError("--resume expects an interrupted run without completed output files; use a new output-dir.")
        completed = load_resume_checkpoint(resume, input_path, timelines, args)
    else:
        if (resume["checkpoint"].exists() or resume["manifest"].exists()) and not args.overwrite:
            raise FileExistsError("Resume checkpoint exists; pass --resume to continue or --overwrite to restart.")
        if args.overwrite:
            for path in resume.values():
                if path.exists():
                    path.unlink()
        if existing_outputs and not args.overwrite:
            raise FileExistsError(
                "Output files already exist; pass --overwrite to replace them: "
                + ", ".join(str(path) for path in existing_outputs[:5])
            )
        completed = {}
        write_resume_manifest(resume["manifest"], input_path, timelines, args)
    if len(completed) > len(timelines):
        raise RuntimeError("Resume checkpoint contains more rows than the requested input.")
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_outputs[:5])
        )
    results: List[Dict[str, Any]] = []
    print(f"Loaded target timelines: {len(timelines)}")
    print(f"Input: {input_path}")
    print("Source policy: evidence_timeline.jsonl only")
    print(f"Dry run: {args.dry_run}")
    for index, timeline in enumerate(timelines, start=1):
        instance_id = timeline.get("instance_id")
        if instance_id in completed:
            print(f"Resuming completed ({index}/{len(timelines)}) instance_id={instance_id}")
            results.append(completed[instance_id])
            continue
        print(f"Running Claim Qualification v3 ({index}/{len(timelines)}) instance_id={timeline.get('instance_id')}")
        result = process_timeline(timeline, dry_run=args.dry_run)
        results.append(result)
        append_resume_checkpoint(resume["checkpoint"], result)
    write_outputs(paths, results)
    audit = summarize(results, args)
    write_json(paths["audit"], audit)
    for path in resume.values():
        if path.exists():
            path.unlink()
    print("Done")
    print(f"Processed total: {audit['processed']}")
    print(f"Claim candidates total: {audit['candidate_count']}")
    print(f"Support bundles total: {audit['support_bundle_count']}")
    print(f"Preflight failures total: {audit['preflight_failed_count']}")
    print(f"Invalid measurements seen: {audit['invalid_measurement_seen_count']}")
    print(f"Same-evidence conflicts total: {audit['same_evidence_conflict_count']}")
    print(f"LLM eligible candidates: {audit['llm_eligible_candidate_count']}")
    print(f"LLM successful calls: {audit['llm_call_success_count']}")
    print(f"LLM failure fallbacks: {audit['llm_failure_fallback_count']}")
    print(f"LLM validator downgrades: {audit['llm_validator_downgrade_count']}")
    print(f"LLM normalization changes: {audit['llm_normalization_change_count']}")
    print(f"Development gate passed: {audit['development_gate_passed']}")
    gate_name = "development_gate_passed" if args.dry_run else "release_gate_passed"
    gate_passed = bool(audit.get(gate_name))
    print(f"{gate_name}: {gate_passed}")
    print(f"Output saved to: {paths['main']}")
    print(f"Build audit saved to: {paths['audit']}")
    if not gate_passed:
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
