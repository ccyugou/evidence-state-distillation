# step5_prompts.py
import json

CLASSIFICATION_SYSTEM_PROMPT = """You are an emergency triage classification agent.

Your task is to assign an ESI level (1–5) to a patient based on structured clinical
evidence. You operate under an EVIDENCE-LOCKED constraint: you may only use the
features provided to you. You do not have access to the original patient text.

This constraint exists by design. The features were extracted and verified in previous
steps. If information is absent, treat it as not reported — do not infer from clinical
knowledge what the patient "probably" has.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION SEQUENCE — MANDATORY ORDER
Source: ESI Handbook v5 (ENA, 2023)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always follow Steps A → B → C → D in order.
Stop at the first step that triggers.

STEP A — ESI 1 (Immediate lifesaving intervention required)
Assign ESI 1 if patient currently requires any of:
  - Airway management or intubation
  - Apneic
  - Pulseless / cardiac arrest
  - Unresponsive or AVPU P or U
  - Severe respiratory distress (labored breathing, accessory muscle use,
    inability to speak, cyanosis, SpO2 below age threshold)
  - Profound hypotension or shock
  - Profound hypoglycemia
  - Emergency medications, fluid resuscitation, or blood products immediately
  - Penetrating trauma to head, neck, chest, or abdomen needing immediate
    lifesaving intervention
→ If any Step A criterion is met: predicted_esi = 1. Stop.

STEP B — ESI 2 (High-risk / altered mental status / severe pain or distress)
Assign ESI 2 if at least ONE of the following is present:

  High-risk situation:
    - Active chest pain suspicious for ACS
    - Signs of stroke: facial droop, arm weakness, slurred speech, sudden onset
    - Suspected ectopic pregnancy (hemodynamically stable)
    - Suicidal ideation with plan or intent, homicidal, or actively violent
    - Immunocompromised or chemotherapy patient with fever
    - Threatened airway or respiratory compromise not yet at Level 1
    - Active major bleeding

  Altered mental status (must be NEW, not chronic baseline):
    - New confusion, lethargy, or disorientation
    - Infant or child with abnormal level of consciousness

  Severe pain or distress:
    - Pain ≥ 7/10 by patient report, or described as severe/excruciating/worst ever
    - Severe physiological or psychological distress

  Pediatric fever (requires both age and temperature to be present):
    - 1–28 days:  T > 38.0°C → ESI 2
    - 1–3 months: T > 38.0°C → consider ESI 2
    - ≥ 3 months: T > 39.0°C or T < 36.0°C, with incomplete immunisations
                  or no obvious source → consider ESI 2/3
→ If any Step B criterion is met: predicted_esi = 2. Stop.

STEP C — Resource estimation (ESI 3, 4, 5)
Count different types of resources, not individual tests.

  Count as resources:
    Labs (blood, urine) | ECG, X-ray | CT, MRI, ultrasound, angiography
    IV fluids / hydration | IV, IM, or nebulised medications
    Specialty consultation | Simple procedure (×1) | Complex procedure (×2)

  Do NOT count as resources:
    History and physical exam | Point-of-care testing | Oral medications
    Prescription refill | Saline or heparin lock | Simple wound dressing or recheck
    Crutches, splints, slings

  Assignment:
    0 resource types → ESI 5
    1 resource type  → ESI 4
    2+ resource types → candidate ESI 3 (proceed to Step D)

STEP D — High-risk vital sign reassessment (ESI 3 candidates only)
Apply age-specific thresholds. Only trigger if the vital value is present.
If age is unknown, note this in reasoning and do not apply a threshold.

  Age          HR (bpm)  RR (/min)  SpO2
  < 1 month    > 190     > 60       < 92%
  1–12 months  > 180     > 55       < 92%
  1–3 years    > 140     > 40       < 92%
  3–5 years    > 120     > 35       < 92%
  5–12 years   > 120     > 30       < 92%
  12–18 years  > 100     > 20       < 92%
  > 18 years   > 100     > 20       < 92%

  If high-risk vital signs are present → consider upgrade to ESI 2.
  If upgrading: predicted_esi = 2 with the vital sign as primary evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFICATION AGENT FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The verification agent may have identified missed ESI 1 or 2 signals.
Treat high-confidence missed signals as additional evidence — they were found
in the original text but not captured in Step 3 features.
Low-confidence signals are advisory only; do not let them solely determine ESI 1 or 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Pure JSON only. No markdown, no code fences, no explanation.
- predicted_esi must be an integer 1–5.
- Every item in primary_evidence must come from the provided features or
  verification findings — not from general clinical knowledge.
- reasoning must cite specific features; one or two sentences maximum.
- If information is insufficient to determine a step, note it in reasoning
  and err toward higher acuity.
"""


def _format_features(feat):
    lines = []

    # demographics
    demo = feat.get("demographics", {})
    age  = demo.get("age_group")
    sex  = demo.get("sex")
    if isinstance(age, dict) and age.get("value"):
        lines.append(f"age_group: {age['value']}  [{age.get('evidence_text', '')}]")
    if isinstance(sex, dict) and sex.get("value"):
        lines.append(f"sex: {sex['value']}")

    # positive symptoms
    sym_lines = []
    for field, val in feat.get("symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == 1:
            sym_lines.append(
                f"  {field}: YES  [{val.get('evidence_text', '')}]"
                f"  (source: {val.get('source_layer', '')})"
            )
        elif isinstance(val, dict) and val.get("value") == 0:
            sym_lines.append(f"  {field}: explicitly denied")
    if sym_lines:
        lines.append("symptoms:")
        lines.extend(sym_lines)

    # severity
    sev_lines = []
    for field, val in feat.get("severity", {}).items():
        if isinstance(val, dict) and val.get("value") not in (None, "unknown"):
            sev_lines.append(
                f"  {field}: {val['value']}  [{val.get('evidence_text', '')}]"
            )
    if sev_lines:
        lines.append("severity:")
        lines.extend(sev_lines)

    # risk factors
    risk_lines = []
    for field, val in feat.get("risk_factors", {}).items():
        if isinstance(val, dict) and val.get("value") == 1:
            risk_lines.append(
                f"  {field}: YES  [{val.get('evidence_text', '')}]"
            )
    if risk_lines:
        lines.append("risk_factors:")
        lines.extend(risk_lines)

    # temporal pattern
    temp_lines = []
    for field, val in feat.get("temporal_pattern", {}).items():
        if isinstance(val, dict) and val.get("value") not in (None, "unknown"):
            temp_lines.append(f"  {field}: {val['value']}")
    if temp_lines:
        lines.append("temporal_pattern:")
        lines.extend(temp_lines)

    # resource clues
    res = feat.get("resource_clues", {})
    res_lines = []
    for field in ["needs_labs", "needs_imaging", "needs_iv_access",
                  "needs_specialist", "needs_procedure"]:
        val = res.get(field)
        if isinstance(val, dict) and val.get("value") == 1:
            res_lines.append(f"  {field}: YES  [{val.get('evidence_text', '')}]")
    er = res.get("estimated_resources")
    if isinstance(er, dict) and er.get("value") not in (None, "unknown"):
        res_lines.append(
            f"  estimated_resources: {er['value']}  [{er.get('evidence_text', '')}]"
        )
    if res_lines:
        lines.append("resource_clues:")
        lines.extend(res_lines)

    return "\n".join(lines) if lines else "(no features extracted)"


def _format_verification(verif):
    lines = []

    esi1 = verif.get("missed_esi1_signals", [])
    esi2 = verif.get("missed_esi2_signals", [])
    rec  = verif.get("upgrade_recommendation", {})

    if esi1:
        lines.append("missed ESI 1 signals (from verification agent):")
        for s in esi1:
            lines.append(
                f"  [{s.get('confidence','?')}] {s.get('criterion','')}:"
                f" \"{s.get('evidence_text','')}\""
            )

    if esi2:
        lines.append("missed ESI 2 signals (from verification agent):")
        for s in esi2:
            lines.append(
                f"  [{s.get('confidence','?')}] {s.get('criterion','')}:"
                f" \"{s.get('evidence_text','')}\""
            )

    if rec.get("recommend_upgrade"):
        lines.append(
            f"upgrade recommendation: to ESI {rec.get('to_level')} — "
            f"{rec.get('primary_reason','')}"
        )

    return "\n".join(lines) if lines else "(no missed signals found)"


def _format_resources(resource_record):
    if not resource_record:
        return "(resource estimation not available)"

    count     = resource_record.get("resource_count", "unknown")
    reasoning = resource_record.get("resource_reasoning", "")
    resources = resource_record.get("likely_resources", [])

    lines = [f"estimated resource count: {count}"]

    high_med = [r for r in resources if r.get("confidence") in ("high", "medium")]
    if high_med:
        lines.append("likely resource types:")
        for r in high_med:
            lines.append(
                f"  [{r['confidence']}] {r['resource_type']}"
                f" — {r.get('clinical_indication', '')}"
            )
    else:
        lines.append("  no high/medium confidence resources identified")

    if reasoning:
        lines.append(f"reasoning: {reasoning}")

    lines.append("")
    lines.append("NOTE: use this estimate for Step C only.")
    lines.append("If Step A or B is already triggered, skip Step C entirely.")

    return "\n".join(lines)


def build_classification_prompt(feat_record, verif_record, resource_record=None):
    feat   = feat_record["extracted_features"]
    vitals = feat_record["objective_vitals"]

    present_vitals = {k: v for k, v in vitals.items() if v is not None}
    vitals_str = json.dumps(present_vitals, indent=2) \
                 if present_vitals else "none available"

    features_str     = _format_features(feat)
    verification_str = _format_verification(verif_record)
    resource_str     = _format_resources(resource_record)

    return f"""Classify this patient using Steps A → B → C → D in order.
You may only use the evidence below. Do not access the original patient text.

─── EXTRACTED FEATURES (Step 3) ───────────────────────────────────
{features_str}

─── OBJECTIVE VITAL SIGNS ──────────────────────────────────────────
{vitals_str}

─── VERIFICATION AGENT FINDINGS (Step 4) ───────────────────────────
{verification_str}

─── RESOURCE ESTIMATION (Step 4b) ──────────────────────────────────
{resource_str}

─── OUTPUT FORMAT ───────────────────────────────────────────────────
{{
  "case_id": "{feat_record['case_id']}",
  "predicted_esi": <integer 1-5>,
  "confidence": "<high | medium | low>",
  "decision_path": {{
    "step_a_triggered": <true | false>,
    "step_b_triggered": <true | false>,
    "step_b_criterion": "<criterion name or null>",
    "step_c_resource_count": <integer or null>,
    "step_d_triggered": <true | false>
  }},
  "primary_evidence": [
    {{
      "feature": "<feature name>",
      "value": "<feature value>",
      "evidence_text": "<quote from extracted features>",
      "decision_point": "<A | B | C | D>"
    }}
  ],
  "reasoning": "<one or two sentences citing specific features>"
}}
"""