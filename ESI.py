# step4_prompts.py
import json

VERIFICATION_SYSTEM_PROMPT = """You are an emergency triage safety reviewer.

Your task is to identify ESI Level 1 or Level 2 signals that were missed in a previous
feature extraction step. You will be given the original patient text, already-extracted
features, objective vital signs, and missing information.

Your approach is ADVERSARIAL: assume the previous extraction may have missed high-risk
signals. Work through the ESI decision criteria systematically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE RULES — READ BEFORE CHECKING ANY CRITERION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Evidence must come directly from the text. Do not perform multi-step clinical inference.
   WRONG: "patient is postpartum + has fever → could have mastitis → ESI 2"
   RIGHT: text explicitly states the patient is immunocompromised, on chemotherapy,
          or receiving radiation therapy

2. Signals must reflect the CURRENT presentation, not history or resolved symptoms.
   WRONG: "started wheezing a few days ago" → severe respiratory distress
   RIGHT: "currently struggling to breathe" / "using accessory muscles" /
          "unable to speak in full sentences" / "cyanotic"

3. "Severe respiratory distress" requires at least one of the following currently:
   labored or accessory-muscle breathing, inability to complete sentences,
   central cyanosis, severe air hunger, or SpO2 below age-specific threshold.
   Mild shortness of breath, chronic wheeze, or past respiratory symptoms do NOT qualify.

4. Pediatric fever signals require BOTH age AND temperature to be directly evidenced
   in the text. Do not assume age from context; it must be stated.

5. confidence calibration:
   "high"   = criterion is explicitly and unambiguously stated in the text
   "medium" = current presentation strongly and directly implies the criterion,
              no inference chain needed
   "low"    = indirect or uncertain — do NOT trigger upgrade_recommendation for
              low-confidence findings alone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP A — ESI Level 1 (Immediate lifesaving intervention required)
Source: ESI Handbook v5, Chapter 3 (ENA, 2023)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check if the patient currently requires any of the following:

Airway / breathing:
  - Intubated or needs immediate intubation
  - Apneic
  - Severe respiratory distress (see definition above)

Circulation / consciousness:
  - Pulseless / cardiac arrest
  - Unresponsive, acutely nonverbal and not following commands,
    or requires noxious stimulus (AVPU P or U)
  - Profound hypotension or shock
  - Profound hypoglycemia

Intervention:
  - Needs emergency medications immediately
  - Needs fluid resuscitation or blood products immediately
  - Penetrating trauma to head, neck, chest, or abdomen requiring
    immediate lifesaving intervention

If any Level 1 trigger is present → flag as missed_esi1_signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP B — ESI Level 2 (High-risk / altered mental status / severe pain or distress)
Source: ESI Handbook v5, Chapter 4 (ENA, 2023)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check if the patient currently has at least ONE of the following:

High-risk situation:
  - Concerning chest pain (active, suspicious for ACS)
  - Signs of stroke: facial droop, arm weakness, slurred speech, sudden onset
  - Suspected ectopic pregnancy (hemodynamically stable but high risk)
  - Suicidal ideation with plan or intent, homicidal, or actively violent
  - High-risk pediatric presentation (see pediatric fever below)
  - Threatened airway or respiratory compromise not yet meeting Level 1
  - Active major bleeding

Altered mental status (must be NEW, not chronic/baseline):
  - New confusion, lethargy, or disorientation
  - Infant or child with depressed or abnormal level of consciousness

Severe pain or distress:
  - Pain ≥ 7/10 by patient report, or described as severe/excruciating/worst ever
  - Severe physiological or psychological distress currently observed

Pediatric fever (age-specific, must have text evidence for both age and temperature):
  - 1–28 days:  T > 38.0°C  → ESI 2
  - 1–3 months: T > 38.0°C  → consider ESI 2
  - ≥ 3 months: T > 39.0°C or T < 36.0°C, incomplete immunisations,
                or no obvious fever source → consider ESI 2 or 3

If any Level 2 trigger is present → flag as missed_esi2_signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP D — High-risk vital signs (Decision Point D, applies to ESI 3/4/5 candidates)
Source: ESI Handbook v5, Chapter 6 (ENA, 2023)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use age-specific thresholds. Only flag if the vital sign value is present
in objective_vitals AND exceeds the threshold below.

  Age          HR (bpm)   RR (breaths/min)   SpO2
  < 1 month    > 190      > 60               < 92%
  1–12 months  > 180      > 55               < 92%
  1–3 years    > 140      > 40               < 92%
  3–5 years    > 120      > 35               < 92%
  5–12 years   > 120      > 30               < 92%
  12–18 years  > 100      > 20               < 92%
  > 18 years   > 100      > 20               < 92%

If a high-risk vital sign is present → flag as missed_esi2_signal with
criterion "Decision Point D: high-risk vital sign" and specify which threshold
was exceeded and at what value.

Note: Do not apply vital sign thresholds without knowing age.
If age is not reported, note it in primary_reason but do not flag a vital sign
threshold violation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Check Steps A, B, and D in order.
2. For each criterion, verify whether Step 3 already captured it.
3. Only report signals NOT already captured in the Step 3 features.
4. Apply the evidence rules strictly — no multi-step inference, current presentation only.
5. If nothing was missed, output empty lists.
6. Do not re-assign an ESI level; only flag missed signals and recommend upgrade.

OUTPUT RULES:
- Pure JSON only. No markdown, no code fences, no explanation.
- upgrade_recommendation.to_level must be null if recommend_upgrade is false.
- Do not recommend upgrade based on low-confidence findings alone.
"""


def build_verification_prompt(record, state):
    vitals  = record["objective_vitals"]
    feat    = record["extracted_features"]
    text    = state["clean_text"]
    missing = state["evidence_state"]["missing_information"]

    already_found = []

    for field, val in feat.get("symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == 1:
            already_found.append(
                f"symptom '{field}': \"{val.get('evidence_text', '')}\""
            )

    for field, val in feat.get("severity", {}).items():
        v = val.get("value") if isinstance(val, dict) else None
        if v not in (None, "unknown", "none", "calm", "normal"):
            already_found.append(
                f"severity '{field}' = {v}: \"{val.get('evidence_text', '')}\""
            )

    for field, val in feat.get("risk_factors", {}).items():
        if isinstance(val, dict) and val.get("value") == 1:
            already_found.append(
                f"risk_factor '{field}': \"{val.get('evidence_text', '')}\""
            )

    age = feat.get("demographics", {}).get("age_group")
    if isinstance(age, dict) and age.get("value"):
        already_found.append(
            f"age_group = {age.get('value')}: \"{age.get('evidence_text', '')}\""
        )

    already_str = "\n".join(f"  - {f}" for f in already_found) \
                  if already_found else "  (no positive signals captured)"

    present_vitals = {k: v for k, v in vitals.items() if v is not None}
    vitals_str  = json.dumps(present_vitals, indent=2) if present_vitals \
                  else "none available"
    missing_str = ", ".join(missing) if missing else "none"

    return f"""Patient presentation text:
{text}

Objective vital signs (extracted by rule-based system, do not modify):
{vitals_str}

Missing vital fields: {missing_str}

Positive signals already captured by Step 3 (do NOT re-report):
{already_str}

Check for missed ESI Level 1 or Level 2 signals following Steps A, B, and D.
Apply the evidence rules strictly before reporting any finding.

Output in this exact JSON format:

{{
  "case_id": "{record['case_id']}",
  "missed_esi1_signals": [
    {{
      "criterion": "<which Step A criterion>",
      "evidence_text": "<exact or near-exact quote>",
      "source_layer": "<patient_reported | nurse_observed | objective_vitals>",
      "confidence": "<high | medium | low>"
    }}
  ],
  "missed_esi2_signals": [
    {{
      "criterion": "<which Step B or D criterion>",
      "evidence_text": "<exact or near-exact quote>",
      "source_layer": "<patient_reported | nurse_observed | objective_vitals>",
      "confidence": "<high | medium | low>"
    }}
  ],
  "upgrade_recommendation": {{
    "recommend_upgrade": <true | false>,
    "to_level": <1 | 2 | null>,
    "primary_reason": "<cite the specific criterion and evidence; null if no upgrade>"
  }}
}}
"""