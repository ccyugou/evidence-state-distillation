#!/usr/bin/env python3
"""Closed ESI-v5 policy manifest consumed by stage 02.

This file maps stage-01 clinical atoms to policy roles. It contains no
dataset labels, learned weights, or free-text extraction rules.
"""

from __future__ import annotations


MANIFEST_VERSION = "esi_v5_step_ab_policy_manifest_v1.1"


POLICIES = {
    "critical_airway_or_gas_exchange": {
        "step": "A",
        "mode": "critical_airway",
        "anchors": {"gasping", "cyanosis", "stridor", "tripod_position", "short_sentence_speech"},
        "modifiers": {"dyspnea_at_rest", "dyspnea", "altered_mental_status"},
        "minimum_anchor_count": 1,
        "rule_id": "ESI5_DP_A_AIRWAY_GAS_EXCHANGE",
    },
    "active_seizure_or_unresponsiveness": {
        "step": "A",
        "mode": "direct_observation",
        "anchors": {"seizure"},
        "modifiers": set(),
        "rule_id": "ESI5_DP_A_SEIZURE_UNRESPONSIVE",
    },
    "recent_seizure_high_risk": {
        "step": "B",
        "mode": "current_anchor",
        "anchors": {"seizure"},
        "modifiers": {"altered_mental_status"},
        "rule_id": "ESI5_DP_B_RECENT_SEIZURE",
    },
    "critical_perfusion_or_hemorrhage": {
        "step": "A",
        "mode": "anchor_plus_modifier",
        "anchors": {"active_bleeding", "vaginal_bleeding", "hematemesis", "rectal_bleeding"},
        "modifiers": {"syncope", "near_syncope", "pallor", "diaphoresis", "altered_mental_status"},
        "rule_id": "ESI5_DP_A_PERFUSION_HEMORRHAGE",
    },
    "respiratory_high_risk": {
        "step": "B",
        "mode": "respiratory",
        "anchors": {"dyspnea", "dyspnea_at_rest", "dyspnea_exertional"},
        "modifiers": {"gasping", "tripod_position", "short_sentence_speech", "stridor", "wheezing", "cyanosis"},
        "rule_id": "ESI5_DP_B_RESPIRATORY_DISTRESS",
    },
    "acute_coronary_risk": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"chest_pain", "chest_pressure"},
        "modifiers": {"dyspnea", "dyspnea_at_rest", "diaphoresis", "nausea", "vomiting", "dizziness", "lightheadedness", "syncope", "pallor"},
        "rule_id": "ESI5_DP_B_ACS_CONCERN",
    },
    "focal_neurologic_risk": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"focal_weakness", "focal_numbness", "speech_deficit", "facial_droop"},
        "modifiers": {"headache", "altered_mental_status", "dizziness", "lightheadedness", "syncope"},
        "rule_id": "ESI5_DP_B_STROKE_SIGNS",
    },
    "acute_altered_mental_status": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"altered_mental_status", "lethargy"},
        "modifiers": {"agitation_or_violent_behavior", "hallucinations"},
        "rule_id": "ESI5_DP_B_NEW_AMS",
    },
    "active_self_or_other_harm": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"suicidal_ideation", "homicidal_ideation", "overdose_method_ideation"},
        "modifiers": {"agitation_or_violent_behavior", "hallucinations"},
        "rule_id": "ESI5_DP_B_SELF_OTHER_HARM",
    },
    "toxic_ingestion_high_risk": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"actual_toxic_ingestion", "possible_toxic_ingestion"},
        "modifiers": {"altered_mental_status", "dyspnea", "palpitations", "syncope", "seizure"},
        "rule_id": "ESI5_DP_B_TOXIC_INGESTION",
    },
    "pregnancy_high_risk": {
        "step": "B",
        "mode": "context_plus_modifier",
        "anchors": {"pregnancy_status", "postpartum_status"},
        "modifiers": {"abdominal_pain", "pelvic_pain", "vaginal_bleeding", "active_bleeding", "syncope", "near_syncope", "chest_pain", "chest_pressure", "dyspnea", "headache"},
        "rule_id": "ESI5_DP_B_PREGNANCY_POSTPARTUM_RISK",
    },
    "immunocompromised_infection": {
        "step": "B",
        "mode": "context_plus_modifier",
        "anchors": {"chemotherapy_or_immunosuppression", "transplant_recipient"},
        "modifiers": {"fever", "chills_or_rigors", "pneumonia", "cellulitis", "urinary_tract_infection"},
        "rule_id": "ESI5_DP_B_IMMUNOCOMPROMISED_INFECTION",
    },
    "active_bleeding_high_risk": {
        "step": "B",
        "mode": "bleeding",
        "anchors": {"active_bleeding", "vaginal_bleeding", "hematemesis", "melena", "rectal_bleeding", "hemoptysis", "epistaxis"},
        "modifiers": {"anticoagulant_use", "pallor", "diaphoresis", "syncope", "near_syncope", "dyspnea", "pregnancy_status", "postpartum_status"},
        "rule_id": "ESI5_DP_B_HIGH_RISK_BLEEDING",
    },
    "severe_pain_or_distress": {
        "step": "B",
        "mode": "pain_context",
        "anchors": {"pain_present", "abdominal_pain", "flank_pain", "back_pain", "neck_pain", "pelvic_pain", "testicular_or_scrotal_pain", "limb_or_joint_pain", "eye_pain_or_visual_change", "headache"},
        "modifiers": {"functional_limitation", "sickle_cell_disease"},
        "rule_id": "ESI5_DP_B_SEVERE_PAIN_DISTRESS",
    },
    "high_risk_headache": {
        "step": "B",
        "mode": "headache",
        "anchors": {"thunderclap_headache", "headache"},
        "modifiers": {"neck_pain", "fever", "vomiting", "altered_mental_status", "focal_weakness", "focal_numbness", "speech_deficit", "facial_droop"},
        "rule_id": "ESI5_DP_B_HIGH_RISK_HEADACHE",
    },
    "organ_or_limb_threat": {
        "step": "B",
        "mode": "anchor_any",
        "anchors": {"testicular_or_scrotal_pain", "eye_pain_or_visual_change"},
        "modifiers": {"severe_language", "functional_limitation"},
        "rule_id": "ESI5_DP_B_ORGAN_LIMB_THREAT",
    },
    "high_risk_trauma": {
        "step": "B",
        "mode": "anchor_plus_modifier",
        "anchors": {"fall_or_trauma", "laceration_or_wound", "fracture"},
        "modifiers": {"focal_weakness", "focal_numbness", "functional_limitation", "syncope", "altered_mental_status"},
        "rule_id": "ESI5_DP_B_HIGH_RISK_TRAUMA",
    },
}


POLICY_ATOMS = frozenset(
    atom_id
    for policy in POLICIES.values()
    for atom_id in policy["anchors"] | policy["modifiers"]
    if atom_id not in {"severe_language"}
)
