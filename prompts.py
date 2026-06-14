import json
# system prompt and user prompt builder 

SYSTEM_PROMPT = """You are a clinical information extraction assistant for emergency triage.

Extract structured features from a patient presentation. Follow the schema exactly.
Do NOT add, rename, or omit any fields.

VITALS RULES:
1. Objective vital signs are already provided in objective_vitals. Do not modify or re-extract them.
2. Missing vitals = missing information only. Do not treat absent vitals as normal or abnormal.
3. Do not apply adult thresholds to pediatric patients. Only flag vitals abnormal if the text says so.

OUTPUT RULES:
- Pure JSON only. No markdown, no code fences, no explanation.
- Binary fields (symptoms, risk_factors, resource binary):
    1    = explicitly present in text
    0    = explicitly denied ("denies X", "no X")
    null = not mentioned or unclear
- Categorical fields: one of the listed values, or null.
- Non-null fields must include evidence_text and source_layer.
- Null fields: output null directly, not an empty object.
- estimated_resources: base on explicitly mentioned tests/procedures only.
  Do NOT reverse-engineer from your guess of the ESI level.

source_layer values: patient_reported, nurse_observed, objective_vitals,
                     question_intent, revision, mixed
"""


SCHEMA_TEMPLATE = {
    "demographics": {
        "age_group": None,   # neonate|infant_under_3mo|child|adolescent|adult|older_adult|unknown
        "sex": None          # male|female|other|unknown
    },
    "symptoms": {
        "chest_pain": None,
        "dyspnea": None,
        "altered_mental_status": None,
        "active_bleeding": None,
        "seizure": None,
        "syncope": None,
        "cyanosis": None,
        "vomiting": None,
        "fever_reported": None,
        "abdominal_pain": None,
        "headache": None,
        "rash": None,
        "palpitations": None,
        "weakness": None,
        "trauma_mechanism": None,
        "respiratory_distress": None,
        "suicidal_ideation": None
    },
    "severity": {
        "pain_severity": None,       # none|mild|moderate|severe|unknown
        "respiratory_effort": None,  # normal|mild|moderate|severe|unknown
        "overall_distress": None     # calm|mild|moderate|severe|unknown
    },
    "risk_factors": {
        "pregnancy_related": None,
        "pediatric_high_risk": None,
        "elderly_risk": None,
        "immunocompromised": None,
        "cardiac_history": None,
        "anticoagulant_use": None,
        "oncologic": None,
        "mental_health_concern": None
    },
    "temporal_pattern": {
        "onset": None,       # sudden|gradual|unknown
        "duration": None,    # hours|days|weeks|chronic|unknown
        "trajectory": None   # worsening|stable|improving|unknown
    },
    "resource_clues": {
        "needs_labs": None,
        "needs_imaging": None,
        "needs_iv_access": None,
        "needs_specialist": None,
        "needs_procedure": None,
        "estimated_resources": None  # 0|1|2+|unknown
    }
}


def build_user_prompt(state):
    vitals  = state["evidence_state"]["nurse_observed"]["objective_vitals"]
    missing = state["evidence_state"]["missing_information"]

    missing_str = ", ".join(missing) if missing else "none"
    vitals_str  = json.dumps(vitals, indent=2)
    schema_str  = json.dumps(SCHEMA_TEMPLATE, indent=2)

    return f"""Patient presentation:
{state["clean_text"]}

Already extracted objective vitals (do not modify):
{vitals_str}

Missing vital fields: {missing_str}

Extract clinical features using this exact schema.
When a field has supporting evidence, replace null with:
  {{"value": <value>, "evidence_text": "<quote from text>", "source_layer": "<layer>"}}

{schema_str}
"""