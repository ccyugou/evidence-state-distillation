import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from pipeline_identity import make_instance_id
except Exception:
    def make_instance_id(case_id, run_uuid):
        return f"{case_id}__{run_uuid}"

from clinical_data_contract import (
    PAIN_CANONICAL_UNIT,
    PAIN_MAX,
    PAIN_MIN,
    VITAL_CANONICAL_NAMES,
    VITAL_CONTRACTS,
    VITAL_NAME_ALIASES,
    VITAL_PLAUSIBILITY_RANGES,
    classify_pain_contract,
    classify_vital_contract,
    normalize_contract_value,
)

try:
    from clinical_data_contract import is_scale_question_text as _shared_is_scale_question_text
except ImportError:
    _shared_is_scale_question_text = None


def is_scale_question_text(text):
    """Strict scale-question detector; excludes vital-rate questions."""
    value = (text or "").strip().lower()
    if not value:
        return False
    if re.search(r"\b(?:heart|pulse|respiratory|breathing)\s+rate\b", value):
        return False
    if _shared_is_scale_question_text is not None:
        return bool(_shared_is_scale_question_text(value))
    patterns = [
        r"\bon\s+a\s+scale\b",
        r"\bout\s+of\s+(?:ten|10)\b",
        r"\bfrom\s+(?:zero|0)\s+to\s+(?:ten|10)\b",
        r"\bhow\s+(?:bad|severe)\b",
        r"\bhow\s+would\s+you\s+rate\b",
        r"\brate\s+(?:your|the|this|it)\b",
        r"\bwhat\s+(?:number|score|rating)\b",
        r"\b(?:pain|fatigue|distress|anxiety|nausea|dizziness|breathlessness|symptom)\s+(?:score|rating|severity)\b",
    ]
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


RAW_DATA_DIR = "transcripts/mimic"
OUTPUT_DIR = "outputs/01_timelines"

PREDICTION_TIMELINE_FILE = os.path.join(OUTPUT_DIR, "evidence_timeline.jsonl")
AUDIT_METADATA_FILE = os.path.join(OUTPUT_DIR, "timeline_audit_metadata.jsonl")
BUILD_AUDIT_FILE = os.path.join(OUTPUT_DIR, "timeline_build_audit.json")
CROSS_REALIZATION_FILE = os.path.join(OUTPUT_DIR, "cross_realization_metadata.json")
TRIAGE_TRACE_AUDIT_ONLY_FILE = os.path.join(OUTPUT_DIR, "triage_state_trace_audit_only.jsonl")
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")

TIMELINE_SCHEMA_VERSION = "01_prediction_safe_timeline_v4.5.4_frozen_repaired"

# These keys must never appear anywhere inside prediction-facing timeline rows.
PREDICTION_FORBIDDEN_KEYS = {
    "triage",
    "acuity",
    "ground_truth",
    "patient_persona",
    "nurse_persona",
    "specialisation",
    "specialization",
    "risk_tolerance",
    "guideline_adherence",
    "instruction",
    "persona",
    "seed",
    "pairing",
    "model",
}

# These markers must not appear in prompt-facing strings after sanitization.
PROMPT_FORBIDDEN_TOKEN_RE = re.compile(
    r"\b("
    r"triage|acuity|ground_truth|patient_persona|nurse_persona|"
    r"specialisation|specialization|risk_tolerance|guideline_adherence|"
    r"instruction|persona|seed|pairing|model"
    r")\b",
    re.IGNORECASE,
)

ALLOWED_TIMELINE_TOP_LEVEL_FIELDS = {
    "timeline_schema_version",
    "instance_id",
    "case_id",
    "group_id",
    "run_uuid",
    "realization_id",
    "dataset",
    "safe_vignette_context",
    "events",
    "prompt_safe_lines",
    "quality_flags",
}

VITAL_KEYS = ["temperature", "heartrate", "resprate", "o2sat", "sbp"]
STRUCTURED_VIGNETTE_KEYS = ["gender", "arrival_transport", "chiefcomplaint"]
STRUCTURED_CLINICAL_FIELDS = STRUCTURED_VIGNETTE_KEYS + VITAL_KEYS + ["pain"]
PATIENT_CONTEXT_FIELDS = ["age_group", "cognitive_state", "recall_accuracy", "language_proficiency"]
VITAL_UNIT_ASSUMPTIONS = {
    key: contract["canonical_unit"]
    for key, contract in VITAL_CONTRACTS.items()
}

CANONICAL_VITAL_NAMES = VITAL_CANONICAL_NAMES

PAIN_VALID_RANGE = (PAIN_MIN, PAIN_MAX)

VITAL_DISPLAY_NAMES = {
    "temperature": "temperature",
    "heartrate": "heart_rate",
    "resprate": "respiratory_rate",
    "o2sat": "oxygen_saturation",
    "sbp": "systolic_blood_pressure",
}

VITAL_DISPLAY_UNITS = {
    "temperature": "fahrenheit",
    "heartrate": "beats_per_minute",
    "resprate": "breaths_per_minute",
    "o2sat": "percent",
    "sbp": "mmHg",
}


QUESTION_START_RE = re.compile(
    r"^\s*(any|are|is|was|were|do|does|did|have|has|had|can|could|would|will|"
    r"how|what|when|where|why|which|who|tell me|describe|on a scale|scale)\b",
    re.IGNORECASE,
)

NURSE_OBSERVATION_RE = re.compile(
    "|".join([
        r"\byou (look|seem|appear|sound)\b",
        r"\byou(?:'re| are) (pale|sweating|diaphoretic|short of breath|struggling|confused|drowsy|lethargic)\b",
        r"\bi (can )?(see|notice|observe)\b",
        r"\byour (skin|breathing|speech|pulse|color|colour|lips|face)\b.*\b(is|are|looks|seems|appears)\b",
        r"\bthat (looks|seems|appears)\b",
        r"\byour (oxygen|o2|blood pressure|heart rate|respiratory rate|temperature)\b"
        r"[^.?!;\n]{0,60}\b(is|are|looks|seems|appears|low|high|elevated|abnormal)\b",
    ]),
    re.IGNORECASE,
)

NURSE_PROCESS_RE = re.compile(
    "|".join([
        r"\bi'?m (going to|gonna)\b",
        r"\bwe (need to|will|are going to)\b",
        r"\bi'?ll\b",
        r"\bplease (sit|wait|hold|come|take|follow|stay)\b",
        r"\blet'?s\b",
        r"\bi need to\b",
    ]),
    re.IGNORECASE,
)

NURSE_QUESTION_START_RE = re.compile(
    r"\b(any|are|is|was|were|do|does|did|have|has|had|can|could|would|will|"
    r"how|what|when|where|why|which|tell me|describe|on a scale|scale)\b",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\b(no|not|never|without|no longer)\b|\bdenies?\b|\bhaven[’']t\b|\bhasn[’']t\b|\bdidn[’']t\b",
    re.IGNORECASE,
)
SCOPE_BREAKER_RE = re.compile(
    r"\b(but|however|though|although|except|just|only|rather|instead|apart from|other than|besides|still|yet)\b",
    re.IGNORECASE,
)
POSITIVE_REASSERTION_RE = re.compile(
    r"\b(just|only|except|but)\s+(this|the|my|that)?\s*"
    r"(chest pain|pain|shortness of breath|trouble breathing|bleeding|abdominal pain|belly pain|rib pain|ribs?)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(maybe|perhaps|possibly|possible|suspected|suspect|potential|not sure|not certain|kind of|sort of|seems like)\b|"
    r"\bi don'?t know(?: whether| if)?\b|\b(?:can'?t|cannot) tell if\b",
    re.IGNORECASE,
)
TARGET_LOCAL_NEGATION_RE = re.compile(
    r"\b(?:don[’']t|dont|do not)\s+(?:have|feel)\b|"
    r"\b(?:doesn[’']t|doesnt|does not)\s+hurt(?:s)?\b",
    re.IGNORECASE,
)
TARGET_LOCAL_UNCERTAINTY_RE = re.compile(
    r"\b(?:don[’']t|dont|do not)\s+(?:think|recall)\b|"
    r"\b(?:can[’']t|cant|cannot)\s+remember\s+if\b",
    re.IGNORECASE,
)
TARGET_UNCERTAINTY_RE = re.compile(
    r"\b(?:not sure|not certain|don'?t know whether|don'?t know if|cannot tell if|can'?t tell if)\b",
    re.IGNORECASE,
)
PATIENT_DIAGNOSTIC_QUESTION_RE = re.compile(
    r"\b(?:how(?: exactly)?\s+do\s+you\s+rule\s+out|could\s+this\s+be|"
    r"is\s+it\s+possible|am\s+i|do\s+i\s+have|what\s+if|"
    r"is\s+this\s+an?)\b|\brule\s+out\b",
    re.IGNORECASE,
)
CONTRAST_SCOPE_BREAKER_RE = re.compile(
    r"\b(but|however|though|although|except|just|only|rather|instead|"
    r"apart from|other than|besides|yet)\b",
    re.IGNORECASE,
)
CONTRACTED_BLEEDING_NEGATION_RE = re.compile(
    r"\b(?:isn[’']t|aren[’']t|wasn[’']t|weren[’']t)\s+(?:really\s+)?(?:actively\s+)?bleeding\b|"
    r"\bnot\s+actively\s+bleeding\b",
    re.IGNORECASE,
)
HISTORICAL_RE = re.compile(
    r"\b(history|historical|hx of|used to|in the past|previously|previous|yesterday|last night|last week|days ago|years ago|months ago|last year)\b|\bold\b.*\binjury\b",
    re.IGNORECASE,
)
CURRENT_RE = re.compile(
    r"\b(now|today|currently|right now|this morning|tonight|since|started|began|getting worse|worsening)\b|"
    r"\b(?:an?\s+|a few\s+)?\d+\s*(?:minutes?|hours?)\s+ago\b|\ban hour ago\b|\b(?:earlier today|just now|recently)\b|"
    r"\bfor \d+\s*(hours?|days?|weeks?)\b|"
    r"\b(when|while)\s+i\s+\w+\b|\b(on exertion|with minimal activity|when walking|when climbing)\b",
    re.IGNORECASE,
)
TREATMENT_FAILURE_RE = re.compile(
    r"\b(didn'?t|did not|hasn'?t|has not|not)\s+(help|work|improve|relieve|resolve|stop)\b|"
    r"\b(no response|failed to improve|failed to relieve|not responding to)\b",
    re.IGNORECASE,
)
SYMPTOM_PERSISTENCE_RE = re.compile(
    r"\b(still|continues?|persistent|persisting|ongoing)\b",
    re.IGNORECASE,
)
RESOLVED_SYMPTOM_RE = re.compile(
    r"\b(gone|resolved|stopped|no longer|has stopped|has resolved)\b",
    re.IGNORECASE,
)
SYMPTOM_RESOLUTION_PATTERNS = {
    "active_bleeding": [r"\bbleeding\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b", r"\bstopped\s+(?:the\s+)?bleeding\b"],
    "gi_bleeding_melena_or_hemodynamic_concern": [r"\b(?:melena|black tarry stool|rectal bleeding)\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b"],
    "chest_pain_or_pressure_mention": [r"\b(?:chest pain|chest pressure|chest tightness|chest discomfort)\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b"],
    "dyspnea_or_respiratory_symptom_mention": [r"\b(?:shortness of breath|trouble breathing|difficulty breathing|breathlessness)\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b", r"\bno longer\s+(?:short of breath|breathless)\b"],
    "pain_mention": [r"\b(?:pain|ache|aching|soreness?)\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b", r"\bstopped\s+(?:the\s+)?(?:pain|aching)\b"],
    "abdominal_pain_mention": [r"\b(?:abdominal|stomach|belly) pain\b\s+(?:is|has|was|were)?\s*(?:gone|resolved|stopped)\b"],
}
RADIATION_NEGATION_RE = re.compile(
    r"\b(doesn'?t|does not|do not|doesn't|not|no)\b[^.?!;\n]{0,40}\b"
    r"(radiat(e|es|ed|ing)|go|spread|travel|move)\b|"
    r"\b(no radiation|stays? in (the )?chest)\b",
    re.IGNORECASE,
)

CLINICAL_CUE_PATTERNS = {
    "chest_pain_or_pressure_mention": [
        r"\bchest (pain|pressure|tightness|discomfort)\b",
        r"\bpain in (my|the) chest\b",
        r"\b(pressure|tightness|heaviness)\s+(?:in|on|across)\s+(?:my|the) chest\b",
        r"\bchest\s+(?:feels?\s+)?(heavy|tight|pressured?)\b",
        r"\b(?:just|only)\s+(pressure|tightness|heaviness)\b",
        r"\bchest\b[^.?!;\n]{0,100}\b(sweat|sweating|short(ness)? of breath|nausea|vomit|dizzy|pressure)\b",
    ],
    "dyspnea_or_respiratory_symptom_mention": [
        r"\bshort(ness)? of breath\b",
        r"\btrouble breathing\b",
        r"\btrouble(?:\s|[.,…-]){1,6}breathing\b",
        r"\bdifficulty breathing\b",
        r"\bdifficulty(?:\s|[.,…-]){1,6}breathing\b",
        r"\bcan'?t breathe\b",
        r"\bbreathless\b",
    ],
    "respiratory_distress_observable": [
        r"\btripod(ding)?\b",
        r"\bstridor\b",
        r"\bwheez(e|ing)\b",
        r"\bspeaking in (short|two|2|three|3)[-\s]?(word)? sentences\b",
        r"\bworking hard to breathe\b",
        r"\busing accessory muscles\b",
    ],
    "stroke_or_focal_neuro_deficit": [
        r"\bslurred speech\b",
        r"\bfacial droop\b",
        r"\bone[-\s]?sided (weakness|numbness)\b",
        r"\bweakness on (one|the) side\b",
        r"\baphasia\b",
        r"\bdysarthria\b",
        r"\barm drift\b",
        r"\bstroke\b",
    ],
    "new_ams_confusion_lethargy_disorientation": [
        r"\bconfus(ed|ion)\b",
        r"\bdisorient(ed|ation)\b",
        r"\bletharg(ic|y)\b",
        r"\bdrowsy\b",
        r"\bnot making sense\b",
        r"\bhard to wake\b",
        r"\bunresponsive\b",
        r"\bnot following commands\b",
    ],
    "syncope_or_near_syncope": [
        r"\bfaint(ed|ing)?\b",
        r"\bpassed out\b",
        r"\bblack(ed)? out\b",
        r"\bnear[-\s]?syncope\b",
        r"\bsyncope\b",
    ],
    "dizziness_or_lightheadedness": [
        r"\blightheaded\b",
        r"\bdizz(y|iness)\b",
    ],
    "gi_bleeding_melena_or_hemodynamic_concern": [
        r"\bblood in (my|the) (stool|vomit)\b",
        r"\bvomiting blood\b",
        r"\bblack tarry stool\b",
        r"\bmelena\b",
        r"\brectal bleeding\b",
    ],
    "active_bleeding": [
        r"\bbleeding\b",
        r"\bblood in (my|the) urine\b",
        r"\bheavy vaginal bleeding\b",
        r"\bsoaking (through )?a pad\b",
        r"\bclots?\b",
    ],
    "abdominal_pain_mention": [
        r"\babdominal pain\b",
        r"\bstomach pain\b",
        r"\bbelly pain\b",
        r"\blower quadrant pain\b",
        r"\bupper (abdominal|stomach) pain\b",
    ],
    "pain_mention": [
        r"\b(pain|ache|aching|hurt|hurts|sore|soreness|burning|cramping)\b",
    ],
    "radiation_or_referred_pain_mention": [
        r"\bradiat(e|es|ed|ing)\b",
        r"\b(radiation|referred pain)\b",
        r"\bpain\b[^.?!;\n]{0,60}\b(to|into)\b[^.?!;\n]{0,30}\b(arm|jaw|back|shoulder|neck)\b",
    ],
    "severe_pain_language": [
        r"\b(severe|terrible|excruciating|unbearable|worst)\b[^,.?!;\n]{0,80}\bpain\b",
        r"\bpain\b[^,.?!;\n]{0,80}\b(severe|terrible|excruciating|unbearable|worst)\b",
        r"\bcan barely stand\b",
        r"\bcan'?t (stand|walk|move) because of pain\b",
    ],
    "numeric_severity_scale_mention": [
        r"\b(7|8|9|10)\s*/\s*10\b",
        r"\b(seven|eight|nine|ten)\s+(out of|over)\s+ten\b",
        r"\babout (a|an)?\s*(seven|eight|nine|ten)\b",
        r"\baround (a|an)?\s*(seven|eight|nine|ten)\b",
        r"\b(seven|eight|nine|ten)\s+or\s+(seven|eight|nine|ten)\b",
        r"\b(7|8|9|10)\s+or\s+(7|8|9|10)\b",
    ],
    "active_suicidal_or_homicidal_risk": [
        r"\bsuicid(al|e)\b",
        r"\bkill myself\b",
        r"\bhurt myself\b",
        r"\bend my life\b",
        r"\b(not here anymore|better if i (wasn'?t|were not) here|end it all)\b",
        r"\bhomicid(al|e)\b",
        r"\bhurt someone\b",
        r"\bkill someone\b",
    ],
    "psychosis_or_violent_behavior": [
        r"\bpsychotic\b",
        r"\bhearing voices\b",
        r"\bvoices telling\b",
        r"\bhallucinat(ion|ing)\b",
        r"\bviolent\b",
        r"\bcombative\b",
        r"\baggressive\b",
    ],
    "pregnancy_postpartum_high_risk": [
        r"\bpregnan(t|cy)\b",
        r"\bpostpartum\b",
        r"\brecently gave birth\b",
        r"\blmp\b",
        r"\blast menstrual period\b",
    ],
    "immunocompromised_or_transplant_with_fever_infection": [
        r"\bchemotherapy\b",
        r"\bchemo\b",
        r"\btransplant\b",
        r"\bimmunocompromised\b",
        r"\bimmunosuppressed\b",
        r"\bsteroids?\b",
    ],
    "fever_or_infection": [
        r"\bfever\b",
        r"\bfebrile\b",
        r"\bchills\b",
        r"\brigors\b",
        r"\binfection\b",
        r"\bsepsis\b",
    ],
    "high_risk_trauma_mechanism_or_penetrating_trauma": [
        r"\bpenetrating\b",
        r"\bstab\b",
        r"\bgunshot\b",
        r"\beject(ed|ion)\b",
        r"\bfall\b.*\b(20|twenty)\s*(feet|ft)\b",
        r"\bextrication\b",
        r"\bamputation\b",
        r"\bneurovascular\b",
        r"\bcompartment syndrome\b",
    ],
    "seizure_postictal": [
        r"\bseizure\b",
        r"\bpost[-\s]?ictal\b",
    ],
    "anaphylaxis_or_airway_threat": [
        r"\banaphylaxis\b",
        r"\ballergic reaction\b",
        r"\bfacial swelling\b",
        r"\blip swelling\b",
        r"\btongue swelling\b",
        r"\bthroat closing\b",
    ],
    "actual_toxic_ingestion_or_overdose": [
        r"\boverdosed\b",
        r"\b(?:possible|suspected|potential)\s+(?:actual\s+)?ingestion\b",
        r"\b(?:possible|suspected|potential)\s+(?:drug\s+)?overdose\b",
        r"\b(?:overdose|overdosed)\s+history\b",
        r"\bhistory\s+of\s+(?:an?\s+)?overdose\b",
        r"\b(?:had|has had|have had)\s+an?\s+overdose\b(?!\s+(?:plan|thought|idea|history|risk|concern))",
        r"\bingested\b[^.?!;\n]{0,35}\b(?:pills?|tablets?|medications?|medicine|bleach)\b",
        r"\bdrank\b[^.?!;\n]{0,20}\bbleach\b",
        r"\b(?:i|we)\s+(?:take|am\s+taking|are\s+taking)\s+(?:too\s+many|more\s+than\s+prescribed|a\s+handful\s+of|a\s+bottle\s+of)\s+(?:pills?|tablets?|medications?|medicine)\b",
        r"\b(?:took|swallowed)\s+(?:\d{2,}|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one\s+hundred|too\s+many|more\s+than\s+prescribed|a\s+handful|a\s+bottle)\b[^.?!;\n]{0,20}\b(?:pills?|tablets?|medications?|medicine)\b",
        r"\bswallowed\s+(?:a\s+)?(?:bottle|handful)\b",
        r"\b(?:was|were|has been|have been)\s+poisoned\b",
    ],
    "overdose_method_ideation_or_plan": [
        r"\b(?:thinking|thought|considering|planning|plan(?:ning)?)\b(?:(?!\band then\b|\bthen\b|\bafter that\b|\bbut eventually\b)[^.?!;\n]){0,50}\b(?:pills?|tablets?|medications?|medicine)\b",
        r"\b(?:thinking|thought|considering|planning|plan(?:ning)?)\b(?:(?!\band then\b|\bthen\b|\bafter that\b|\bbut eventually\b)[^.?!;\n]){0,40}\boverdos(?:e|ing)\b",
        r"\b(?:overdose|overdosing)\s+(?:plan|thought|idea)\b",
        r"\bplan\s+to\s+overdose\b",
        r"\bwondered about\s+(?:overdos(?:e|ing)|taking too many)\b",
        r"\b(?:take|taking|swallow|swallowing)\s+(?:too many|more than prescribed|a handful of|a bottle of)\s+(?:pills?|tablets?|medications?|medicine)\b",
        r"\b(?:take|taking|swallow|swallowing)\b[^.?!;\n]{0,30}\b(?:pills?|tablets?|medications?|medicine)\b[^.?!;\n]{0,30}\b(?:to|so i can|because i want to)\b[^.?!;\n]{0,20}\b(?:hurt|harm|kill)\s+myself\b",
        r"\boverdose\b[^.?!;\n]{0,30}\b(?:to|so i can|because i want to)\b[^.?!;\n]{0,20}\b(?:hurt|harm|kill|not wake up)\b",
    ],
    "sexual_assault_or_domestic_violence_distress": [
        r"\bsexual assault\b",
        r"\brape\b",
        r"\bdomestic violence\b",
        r"\bassaulted\b",
    ],
    "testicular_or_ovarian_torsion_concern": [
        r"\btesticular pain\b",
        r"\bscrotal pain\b",
        r"\bovarian torsion\b",
        r"\btesticular torsion\b",
        r"\bunilateral lower quadrant pain\b",
    ],
    "severe_flank_pain_or_renal_colic": [
        r"\bflank pain\b",
        r"\brenal colic\b",
        r"\bkidney stone\b",
    ],
}

NUMBER_WORDS = {
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
SCALE_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
SHORT_NUMERIC_SCALE_RE = re.compile(
    r"(?:(?:it(?:'s| is)|i(?:'d| would)\s+say|probably|maybe)\s+)?"
    r"(?:an?\s+)?(?P<value>0|1|2|3|4|5|6|7|8|9|10|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s*(?:out of|/)\s*10)?",
    re.IGNORECASE,
)
SCALE_PAIN_HINT_RE = re.compile(
    r"\b(pain|ache|aching|hurt|hurts|sore|soreness|discomfort|burning|cramping)\b",
    re.IGNORECASE,
)
SCALE_NON_PAIN_HINT_RE = re.compile(
    r"\b(fatigue|distress|tired|drained|weakness|weak|breathlessness|shortness of breath|"
    r"breathing difficulty|anxiety|nausea|dizziness|dizzy|heartbeat|heart rate|palpitations|"
    r"pounding heart|racing heart|symptom severity)\b",
    re.IGNORECASE,
)
CLAUSE_BREAK_RE = re.compile(r"[,;!?\n]+|(?<!\.)\.(?!\.)")
SCOPE_CONNECTOR_RE = re.compile(
    r"\b(but|however|though|although|except|just|only|rather|instead|"
    r"apart from|other than|besides|still|yet)\b",
    re.IGNORECASE,
)
HISTORICAL_ONLY_RE = re.compile(
    r"\b(history of|hx of|used to|in the past|previously|previous|years ago|months ago|last year|"
    r"when i was|as a child)\b",
    re.IGNORECASE,
)

COMPILED_CUE_PATTERNS = {
    cue_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for cue_type, patterns in CLINICAL_CUE_PATTERNS.items()
}


def ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)


def list_json_files(base_dir):
    files = []
    for root, _, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    files.sort()
    return files


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, str(exc)


def save_jsonl_row(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            save_jsonl_row(f, row)


def make_evidence_id(instance_id, idx):
    return f"{instance_id}_ev_{idx:04d}"


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return value


def classify_numeric_plausibility(name, value):
    result = classify_vital_contract(name, value)
    result["plausibility_reason"] = result.get("quarantine_reason")
    result["unit_assumption"] = result.get("raw_unit") or VITAL_UNIT_ASSUMPTIONS.get(name)
    return result


def classify_pain_validity(value):
    result = classify_pain_contract(value)
    result["validity_reason"] = result.get("quarantine_reason")
    return result


def is_question_text(text):
    stripped = (text or "").strip()
    return bool("?" in stripped or QUESTION_START_RE.search(stripped))


def classify_nurse_utterance(text):
    original = text or ""
    lower = original.lower()
    if is_question_text(original):
        return "nurse_question"
    if NURSE_OBSERVATION_RE.search(lower):
        return "nurse_observed_statement"
    if NURSE_PROCESS_RE.search(lower):
        return "nurse_instruction_or_process"
    return "nurse_other"


def build_assertion_scope_flags(text):
    """Surface-only scope warning; target-level assertion is handled downstream."""
    lower = (text or "").lower()
    flags = []
    negation_present = bool(NEGATION_RE.search(lower))
    scope_breaker_present = bool(SCOPE_BREAKER_RE.search(lower))
    positive_reassertion_present = bool(POSITIVE_REASSERTION_RE.search(lower))
    clinical_cue_starts = []
    for patterns in COMPILED_CUE_PATTERNS.values():
        for pattern in patterns:
            clinical_cue_starts.extend(match.start() for match in pattern.finditer(lower))
    negation_starts = [match.start() for match in NEGATION_RE.finditer(lower)]
    breaker_starts = [match.start() for match in SCOPE_BREAKER_RE.finditer(lower)]
    negation_before_cue = any(
        neg < cue
        for neg in negation_starts
        for cue in clinical_cue_starts
    )
    clinical_cue_before_negation = any(
        cue < neg
        for cue in clinical_cue_starts
        for neg in negation_starts
    )
    scope_breaker_between_negation_and_cue = any(
        neg < breaker < cue
        for neg in negation_starts
        for breaker in breaker_starts
        for cue in clinical_cue_starts
    )
    if negation_present:
        flags.append("negation_cue_present")
    if scope_breaker_present:
        flags.append("scope_breaker_present")
    if positive_reassertion_present:
        flags.append("positive_reassertion_possible")
    if TARGET_LOCAL_NEGATION_RE.search(lower):
        flags.append("target_local_negation_present")
    if TARGET_LOCAL_UNCERTAINTY_RE.search(lower):
        flags.append("target_local_uncertainty_present")
    if negation_present and clinical_cue_before_negation and not negation_before_cue:
        flags.append("post_clinical_negation_cue_present")
    if negation_present and negation_before_cue and (
        positive_reassertion_present or scope_breaker_between_negation_and_cue or clinical_cue_before_negation
    ):
        flags.append("scope_requires_target_level_resolution")
    elif negation_present and (negation_before_cue or not clinical_cue_starts):
        flags.append("local_negation_likely")
    if UNCERTAINTY_RE.search(lower):
        flags.append("uncertainty_cue_present")
    return flags


def infer_event_surface_polarity_hint(text, source_layer):
    if source_layer == "nurse_question":
        return "question"
    flags = build_assertion_scope_flags(text)
    if "scope_requires_target_level_resolution" in flags:
        return "mixed_scope_possible"
    if "local_negation_likely" in flags:
        return "negation_cue_present"
    if "post_clinical_negation_cue_present" in flags:
        return "positive_surface"
    if "uncertainty_cue_present" in flags:
        return "uncertainty_cue_present"
    return "positive_surface"


def infer_polarity_hint(text, source_layer):
    lower = (text or "").lower()
    if source_layer == "nurse_question":
        return "question"
    flags = build_assertion_scope_flags(lower)
    if "scope_requires_target_level_resolution" in flags:
        return "mixed_scope_possible"
    if "local_negation_likely" in flags:
        return "negated"
    if UNCERTAINTY_RE.search(lower):
        return "uncertain"
    return "positive"


def infer_temporality_hint(text):
    lower = (text or "").lower()
    historical = bool(HISTORICAL_RE.search(lower))
    current = bool(CURRENT_RE.search(lower))
    if current and not historical:
        return "current"
    if historical and not current:
        return "historical"
    if current and historical:
        return "mixed_current_historical"
    return "unknown"


def sanitize_prediction_text(text):
    text = text or ""
    text = re.sub(r"\btriage\s+nurses?\b", "nurse", text, flags=re.IGNORECASE)
    text = re.sub(r"\btriage\s+area\b", "care area", text, flags=re.IGNORECASE)
    text = re.sub(r"\bat\s+triage\b", "during initial assessment", text, flags=re.IGNORECASE)
    text = re.sub(r"\btriage\b", "initial assessment", text, flags=re.IGNORECASE)
    return text



def clean_transcript_text(text):
    """Remove transcript/prosody artifacts before clinical cue extraction."""
    text = text or ""
    text = text.replace("#", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_text_for_event(item):
    original = clean_transcript_text(sanitize_prediction_text(item.get("original", "")))
    utterance = clean_transcript_text(sanitize_prediction_text(item.get("utterance", "")))
    prediction_text = original or utterance or ""
    return prediction_text, prediction_text

def split_clause_ranges(text):
    """Return deterministic local clause ranges without changing parent text."""
    if not text:
        return []
    boundaries = {0, len(text)}
    for match in CLAUSE_BREAK_RE.finditer(text):
        # Preserve comma-separated denial lists until a contrast connector:
        # "no bleeding, fainting, or shoulder pain, just abdominal pain".
        if match.group(0).strip() == ",":
            prior = text[:match.start()]
            negations = list(NEGATION_RE.finditer(prior))
            if negations:
                last_negation = negations[-1]
                between = text[last_negation.end():match.start()]
                after = text[match.end():]
                starts_with_contrast = bool(re.match(
                    r"\s*(?:but|however|except|just|only|though|although|yet)\b",
                    after,
                    re.IGNORECASE,
                ))
                if not CONTRAST_SCOPE_BREAKER_RE.search(between) and not starts_with_contrast:
                    continue
        boundaries.add(match.start())
        boundaries.add(match.end())
    for match in SCOPE_CONNECTOR_RE.finditer(text):
        boundaries.add(match.start())
    ordered = sorted(boundaries)
    ranges = []
    for start, end in zip(ordered, ordered[1:]):
        left = start
        right = end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right and text[left:right].strip(" ,;.!?\n"):
            ranges.append((left, right))
    return ranges or [(0, len(text))]


def build_nurse_utterance_segments(text):
    """Split mixed nurse text into prediction-safe observation/question/process spans."""
    if not text:
        return []
    boundaries = {0, len(text)}
    split_re = re.compile(
        r"(?<!\.)[.!?](?!\.)\s+|,\s+(?=(?:any|are|is|was|were|do|does|did|have|has|had|can|could|"
        r"would|will|how|what|when|where|why|which|tell me|describe|on a scale|scale)\b)",
        re.IGNORECASE,
    )
    for match in split_re.finditer(text):
        boundaries.add(match.end())
    ordered = sorted(boundaries)
    segments = []
    for start, end in zip(ordered, ordered[1:]):
        left = start
        right = end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        segment_text = text[left:right]
        if not segment_text:
            continue
        segments.append({
            "span_text": segment_text,
            "start_char": left,
            "end_char": right,
            "segment_source_layer": classify_nurse_utterance(segment_text),
            "is_question": is_question_text(segment_text),
        })
    return segments or [{
        "span_text": text,
        "start_char": 0,
        "end_char": len(text),
        "segment_source_layer": classify_nurse_utterance(text),
        "is_question": is_question_text(text),
    }]


def nurse_parent_source_layer(segments, fallback):
    layers = {seg.get("segment_source_layer") for seg in segments if seg.get("segment_source_layer")}
    return next(iter(layers)) if len(layers) == 1 else ("nurse_mixed_utterance" if layers else fallback)


def segment_for_span(segments, start, end):
    for segment in segments or []:
        if segment.get("start_char", 0) <= start and end <= segment.get("end_char", 0):
            return segment
    return None


def numeric_values_from_text(text):
    values = []
    for match in re.finditer(r"\b(7|8|9|10|seven|eight|nine|ten)\b", text or "", re.IGNORECASE):
        token = match.group(1).lower()
        values.append(int(token) if token.isdigit() else NUMBER_WORDS[token])
    return values


def scale_values_from_text(text):
    values = []
    for match in re.finditer(
        r"\b(0|1|2|3|4|5|6|7|8|9|10|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        text or "",
        re.IGNORECASE,
    ):
        token = match.group(1).lower()
        values.append(int(token) if token.isdigit() else SCALE_NUMBER_WORDS[token])
    return values


def classify_scale_target(clause_text):
    has_pain = bool(SCALE_PAIN_HINT_RE.search(clause_text or ""))
    has_non_pain = bool(SCALE_NON_PAIN_HINT_RE.search(clause_text or ""))
    if has_pain and has_non_pain:
        return "multi_target", "low"
    if has_pain:
        return "pain", "medium"
    if has_non_pain:
        return "non_pain", "medium"
    return "unknown", "low"


def classify_nearest_scale_target(text, cue_start, cue_end, clause_text):
    target, confidence = classify_scale_target(clause_text)
    if target != "unknown":
        return target, confidence, "local_clause"
    window_start = max(0, cue_start - 90)
    window_end = min(len(text), cue_end + 90)
    window = text[window_start:window_end]
    mentions = []
    for match in SCALE_PAIN_HINT_RE.finditer(window):
        mentions.append((abs(window_start + match.start() - cue_start), "pain"))
    for match in SCALE_NON_PAIN_HINT_RE.finditer(window):
        mentions.append((abs(window_start + match.start() - cue_start), "non_pain"))
    if not mentions:
        return "unknown", "low", "unresolved"
    mentions.sort(key=lambda item: item[0])
    nearest_distance = mentions[0][0]
    nearest = {target for distance, target in mentions if distance == nearest_distance}
    if len(nearest) > 1:
        return "multi_target", "low", "same_utterance_nearest_target"
    return next(iter(nearest)), "medium", "same_utterance_nearest_target"


def build_local_evidence_policy(source_layer, cue_type):
    clinical_positive = source_layer in {"patient_reported", "nurse_observed_statement"}
    independent = clinical_positive
    context_only = not clinical_positive
    if source_layer == "nurse_observed_statement":
        allowed_uses = ["symptom_reasoning", "safety_reasoning", "resource_reasoning"]
        disallowed_uses = []
    elif source_layer == "patient_reported":
        allowed_uses = ["symptom_reasoning", "safety_reasoning", "resource_reasoning"]
        disallowed_uses = []
    else:
        allowed_uses = ["conversation_context", "answer_interpretation_context"]
        disallowed_uses = ["standalone_positive_clinical_evidence", "direct_resource_vote", "standalone_high_risk_trigger"]
    policy = {
        "allowed": True,
        "clinical_positive_allowed": clinical_positive,
        "clinical_assertion_allowed": clinical_positive,
        "positive_trigger_allowed": clinical_positive,
        "negative_evidence_allowed": clinical_positive,
        "independent_evidence": independent,
        "context_only": context_only,
        "allowed_uses": allowed_uses,
        "disallowed_uses": disallowed_uses,
    }
    if cue_type == "pain_mention":
        policy["candidate_policy"] = "anchor_only"
        policy["step_b_candidate_eligible"] = False
        policy["positive_trigger_allowed"] = False
        policy["allowed_uses"] = ["scale_target_binding", "body_site_coreference", "treatment_failure_resolution"]
    elif cue_type == "dizziness_or_lightheadedness":
        policy["candidate_policy"] = "symptom_only_or_contextual_upgrade"
        policy["step_b_candidate_eligible"] = False
        policy["positive_trigger_allowed"] = False
    elif not clinical_positive:
        policy["candidate_policy"] = "context_only"
        policy["step_b_candidate_eligible"] = False
        policy["positive_trigger_allowed"] = False
    return policy


def symptom_resolution_applies_to_target(clause_text, cue_type):
    patterns = SYMPTOM_RESOLUTION_PATTERNS.get(cue_type, [])
    return any(re.search(pattern, clause_text or "", re.IGNORECASE) for pattern in patterns)


def patient_diagnostic_question_applies(clause_text, cue_type):
    """Questions/hypotheses are context, not patient-endorsed positives."""
    lower = (clause_text or "").lower()
    if not ("?" in lower or PATIENT_DIAGNOSTIC_QUESTION_RE.search(lower)):
        return False
    if cue_type in {"pregnancy_postpartum_high_risk", "active_bleeding", "syncope_or_near_syncope",
                    "chest_pain_or_pressure_mention", "dyspnea_or_respiratory_symptom_mention"}:
        return True
    return bool(PATIENT_DIAGNOSTIC_QUESTION_RE.search(lower))


def target_uncertainty_applies(lower, cue_start):
    before = lower[:max(0, cue_start)]
    window = before[-120:]
    match = TARGET_UNCERTAINTY_RE.search(window)
    if not match:
        return False
    tail = window[match.end():]
    if re.search(r"[?;\n]|--|—|\b(?:but|however|though|although)\b", tail):
        return False
    return True


def target_local_denial_resolution(lower, cue_start, cue_end=None, cue_type=None):
    """Resolve only explicit target-local don't-have/feel/think forms.

    The active denial may span a coordinated list (``chest pain or dyspnea``),
    but it stops at a contrast or a new subject proposition.  This deliberately
    does not treat generic ``can't`` or ``don't`` as negation.
    """
    before = lower[:max(0, cue_start)]
    breakers = list(CONTRAST_SCOPE_BREAKER_RE.finditer(before))
    active = before[breakers[-1].end():] if breakers else before

    denial = TARGET_LOCAL_NEGATION_RE.search(active)
    if denial:
        tail = active[denial.end():]
        if not re.search(r"\b(?:and|or)\s+(?:i|i'm|i am|it|he|she|they|we)\b", tail):
            return "absent", "target_local_denial_applies_to_target"

    cue_text = lower[cue_start:cue_end] if isinstance(cue_end, int) else ""
    if cue_type in {"pain_mention", "abdominal_pain_mention", "chest_pain_or_pressure_mention"}:
        if re.search(r"\b(?:doesn[’']t|doesnt|does not)\s+hurt(?:s)?\b", active + cue_text):
            return "absent", "target_local_does_not_hurt"

    uncertainty = TARGET_LOCAL_UNCERTAINTY_RE.search(active)
    if uncertainty:
        tail = active[uncertainty.end():]
        if not re.search(r"\b(?:and|or)\s+(?:i|i'm|i am|it|he|she|they|we)\b", tail):
            return "uncertain", "target_local_uncertainty_applies_to_target"
    return None, None


def persistence_applies_to_target(lower, cue_start):
    """Keep persistence local; do not cross a new question or clause."""
    for match in SYMPTOM_PERSISTENCE_RE.finditer(lower):
        if match.start() >= cue_start:
            continue
        between = lower[match.end():cue_start]
        if len(between) > 70 or re.search(r"[?;\n]|--|—", between):
            continue
        if re.search(r"\b(have|has|had|feel|feels|hurt|hurts|bleed|bleeding|continue|"
                     r"experience|experiencing|remain|remains|ongoing)\b", between):
            return True
    return False


def coordinated_negation_applies(lower, cue_start):
    """Apply a leading denial only before a contrast/scope breaker."""
    negations = list(NEGATION_RE.finditer(lower[:cue_start]))
    if not negations:
        return False
    last_negation = negations[-1]
    between = lower[last_negation.end():cue_start]
    return not CONTRAST_SCOPE_BREAKER_RE.search(between)


def resolve_clause_local_assertion(clause_text, cue_start, cue_end, source_layer, cue_type=None):
    lower = (clause_text or "").lower()
    before = lower[:max(0, cue_start)]
    cue_text = lower[max(0, cue_start):max(0, cue_end)]
    target_local_polarity, target_local_scope = target_local_denial_resolution(
        lower,
        cue_start,
        cue_end=cue_end,
        cue_type=cue_type,
    )
    negation = bool(NEGATION_RE.search(before)) or bool(
        re.search(r"^\s*(no|none|nothing|not really|not at all)\b", lower)
    )
    contracted_negation = cue_type in {"active_bleeding", "gi_bleeding_melena_or_hemodynamic_concern"} and bool(
        CONTRACTED_BLEEDING_NEGATION_RE.search(lower)
    )
    uncertainty = target_local_polarity == "uncertain" or target_uncertainty_applies(lower, cue_start) or bool(
        re.search(r"\b(?:maybe|perhaps|possibly|suspected|potential)\b", cue_text)
    )
    historical = bool(HISTORICAL_RE.search(lower))
    current = bool(CURRENT_RE.search(lower))
    treatment_failure = bool(TREATMENT_FAILURE_RE.search(lower))
    symptom_persistence = persistence_applies_to_target(lower, cue_start)
    resolved_symptom = bool(RESOLVED_SYMPTOM_RE.search(lower)) and symptom_resolution_applies_to_target(
        lower,
        cue_type,
    )
    treatment_negation = bool(re.search(
        r"\b(didn'?t|did not|hasn'?t|has not|not)\s+(help|work|improve|relieve|resolve|stop)\b",
        lower,
        re.IGNORECASE,
    ))
    direct_target_negation = (
        target_local_polarity == "absent"
        or ((negation or contracted_negation) and not treatment_negation and (
        contracted_negation or coordinated_negation_applies(lower, cue_start)
        ))
    )
    radiation_denial = cue_type == "radiation_or_referred_pain_mention" and bool(
        RADIATION_NEGATION_RE.search(lower)
    )
    positive_after_negation = bool(
        negation and re.search(r"\b(but|however|though|although|except|just|only|instead|still|yet)\b", before)
    )
    diagnostic_question = (
        source_layer == "patient_reported"
        and patient_diagnostic_question_applies(lower, cue_type)
        and not target_uncertainty_applies(lower, cue_start)
    )
    if source_layer == "nurse_question" or diagnostic_question:
        polarity = "question"
        certainty = "interrogative"
        clinical_status = "question_context"
        scope_resolution = "patient_diagnostic_question_context" if diagnostic_question else "question_context_only"
    elif uncertainty:
        polarity = "uncertain"
        certainty = "uncertain"
        clinical_status = "present_but_uncertain"
        scope_resolution = target_local_scope or "uncertainty_applies_to_target"
    elif radiation_denial:
        polarity = "negated"
        certainty = "asserted"
        clinical_status = "associated_feature_absent"
        scope_resolution = "radiation_feature_denied"
    elif resolved_symptom or direct_target_negation:
        polarity = "negated"
        certainty = "asserted"
        clinical_status = "absent"
        scope_resolution = target_local_scope or (
            "local_negation_applies_to_target" if direct_target_negation else "resolved_target_absent"
        )
    elif treatment_failure and re.search(r"\b(pain|symptom|breath|breathing|nausea|dizziness)\b", lower):
        polarity = "positive"
        certainty = "asserted"
        clinical_status = "present_with_treatment_failure"
        scope_resolution = "treatment_failure_does_not_negate_target"
    elif symptom_persistence:
        polarity = "positive"
        certainty = "asserted"
        clinical_status = "present_persistent_symptom"
        scope_resolution = "symptom_persistence_does_not_negate_target"
    elif positive_after_negation:
        polarity = "positive"
        certainty = "asserted"
        clinical_status = "present_after_local_denial_or_scope_breaker"
        scope_resolution = "positive_reassertion_after_local_negation"
    elif negation:
        polarity = "negated"
        certainty = "asserted"
        clinical_status = "absent"
        scope_resolution = "local_negation_applies_to_target"
    else:
        polarity = "positive"
        certainty = "asserted"
        clinical_status = "present"
        scope_resolution = "local_positive_assertion"
    if current and historical:
        temporality = "mixed_current_historical"
    elif current:
        temporality = "current"
    elif historical:
        temporality = "historical"
    else:
        temporality = "unknown"
    if historical and current:
        temporal_resolution = "current_historical_mixed_requires_resolution"
    elif historical:
        temporal_resolution = "historical_only_or_historical_context"
    elif current:
        temporal_resolution = "current_marker_present"
    else:
        temporal_resolution = "no_explicit_temporal_marker"
    if cue_type == "actual_toxic_ingestion_or_overdose":
        if historical and not current:
            clinical_status = "historical_actual_overdose"
            scope_resolution = "historical_context_only"
        elif polarity == "uncertain":
            clinical_status = "possible_actual_ingestion"
            scope_resolution = "uncertainty_applies_to_target"
    if positive_after_negation or (negation and ("but" in lower or "however" in lower)):
        scope_confidence = "medium"
    elif negation or source_layer == "nurse_question":
        scope_confidence = "high"
    else:
        scope_confidence = "medium"
    return {
        "local_polarity": polarity,
        "local_temporality": temporality,
        "clinical_status": clinical_status,
        "certainty": certainty,
        "scope_resolution": scope_resolution,
        "scope_confidence": scope_confidence,
        "temporal_resolution": temporal_resolution,
        "cue_text": cue_text,
        "diagnostic_question": diagnostic_question,
        "contracted_local_negation": contracted_negation,
        "persistence_applies_to_target": symptom_persistence,
    }


def extract_clinical_cue_spans(text, source_layer, utterance_segments=None):
    if not text:
        return []
    cues = []
    event_scope_flags = build_assertion_scope_flags(text)
    event_scope_requires_resolution = "scope_requires_target_level_resolution" in event_scope_flags
    clause_ranges = split_clause_ranges(text)
    for clause_start, clause_end in clause_ranges:
        clause_text = text[clause_start:clause_end]
        for cue_type, patterns in COMPILED_CUE_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(clause_text):
                    absolute_start = clause_start + match.start()
                    absolute_end = clause_start + match.end()
                    if absolute_end > clause_end:
                        continue
                    if cue_type == "chest_pain_or_pressure_mention" and re.fullmatch(
                        r"(?:just|only)\s+(?:pressure|tightness|heaviness)",
                        match.group(0).strip(),
                        re.IGNORECASE,
                    ):
                        preceding = text[:absolute_start]
                        if not re.search(r"\bchest\b|\bpressure\s+in\s+(?:my|the)\s+chest\b", preceding, re.IGNORECASE):
                            continue
                    segment = segment_for_span(utterance_segments, absolute_start, absolute_end)
                    local_source_layer = segment.get("segment_source_layer") if segment else source_layer
                    local = resolve_clause_local_assertion(
                        clause_text,
                        match.start(),
                        match.end(),
                        local_source_layer,
                        cue_type=cue_type,
                    )
                    local_policy = build_local_evidence_policy(local_source_layer, cue_type)
                    if local.get("clinical_status") == "question_context" or local.get("local_polarity") == "question":
                        local_policy.update({
                            "allowed": True,
                            "clinical_assertion_allowed": False,
                            "positive_trigger_allowed": False,
                            "negative_evidence_allowed": False,
                            "independent_evidence": False,
                            "context_only": True,
                            "candidate_policy": "anchor_only" if cue_type == "pain_mention" else "question_context",
                            "step_b_candidate_eligible": False,
                        })
                        if cue_type == "pain_mention":
                            local_policy["allowed_uses"] = [
                                "scale_target_binding", "body_site_coreference", "treatment_failure_resolution"
                            ]
                    elif local.get("local_polarity") == "uncertain":
                        local_policy["positive_trigger_allowed"] = False
                        if cue_type == "pain_mention":
                            local_policy["candidate_policy"] = "anchor_only"
                            local_policy["step_b_candidate_eligible"] = False
                        else:
                            local_policy["candidate_policy"] = "uncertain_requires_review"
                            local_policy["step_b_candidate_eligible"] = None
                    local_policy["positive_trigger_allowed"] = bool(
                        local_policy.get("positive_trigger_allowed") is True
                        and local["local_polarity"] == "positive"
                        and not (
                            cue_type == "actual_toxic_ingestion_or_overdose"
                            and local.get("local_temporality") in {"historical", "mixed_current_historical"}
                        )
                        and not (
                            cue_type == "actual_toxic_ingestion_or_overdose"
                            and local.get("local_polarity") != "positive"
                        )
                    )
                    cue = {
                        "cue_type": cue_type,
                        "span_text": text[absolute_start:absolute_end],
                        "start_char": absolute_start,
                        "end_char": absolute_end,
                        "polarity_hint": local["local_polarity"],
                        "temporality_hint": local["local_temporality"],
                        "source_layer": local_source_layer,
                        "local_source_layer": local_source_layer,
                        "rule": pattern.pattern,
                        "clause_text": clause_text,
                        "clause_start_char": clause_start,
                        "clause_end_char": clause_end,
                        "local_polarity": local["local_polarity"],
                        "local_temporality": local["local_temporality"],
                        "clinical_status": local["clinical_status"],
                        "certainty": local["certainty"],
                        "scope_resolution": local["scope_resolution"],
                        "scope_confidence": local["scope_confidence"],
                        "temporal_resolution": local["temporal_resolution"],
                        "local_independent_evidence": local_policy.get("independent_evidence"),
                        "local_context_only": local_policy.get("context_only"),
                        "local_evidence_use_policy": local_policy,
                        "candidate_policy": local_policy.get("candidate_policy"),
                        "step_b_candidate_eligible": local_policy.get("step_b_candidate_eligible"),
                        "event_scope_flags": event_scope_flags,
                        "event_scope_requires_target_level_resolution": event_scope_requires_resolution,
                        "derived_clause_span": {
                            "span_text": clause_text,
                            "start_char": clause_start,
                            "end_char": clause_end,
                            "local_polarity": local["local_polarity"],
                            "local_temporality": local["local_temporality"],
                            "scope_resolution": local["scope_resolution"],
                            "scope_confidence": local["scope_confidence"],
                            "temporal_resolution": local["temporal_resolution"],
                            "source_layer": local_source_layer,
                            "local_source_layer": local_source_layer,
                            "local_independent_evidence": local_policy.get("independent_evidence"),
                            "local_context_only": local_policy.get("context_only"),
                            "local_evidence_use_policy": local_policy,
                            "event_scope_flags": event_scope_flags,
                            "event_scope_requires_target_level_resolution": event_scope_requires_resolution,
                        },
                    }
                    if cue_type == "numeric_severity_scale_mention":
                        scale_target, target_confidence, target_source = classify_nearest_scale_target(
                            text,
                            absolute_start,
                            absolute_end,
                            clause_text,
                        )
                        cue["numeric_values"] = numeric_values_from_text(cue["span_text"])
                        cue["scale_target_hint"] = scale_target
                        cue["scale_target_confidence"] = target_confidence
                        cue["scale_target_source"] = target_source
                        cue["scale_target_resolution"] = (
                            "requires_dialogue_or_field_binding"
                            if scale_target in {"unknown", "multi_target"}
                            else "target_hint_from_local_or_nearest_context"
                        )
                        cue["scale_policy"] = "generic_numeric_severity_not_pain_by_default"
                        scale_is_patient_clinical = local_source_layer in {
                            "patient_reported",
                            "nurse_observed_statement",
                        }
                        if scale_target == "pain" and scale_is_patient_clinical:
                            cue["candidate_policy"] = "pain_scale_support_only"
                            cue["scale_support_only"] = True
                        else:
                            cue["candidate_policy"] = "non_pain_or_unresolved_scale"
                            cue["scale_support_only"] = False
                        cue["step_b_candidate_eligible"] = False
                        local_policy["candidate_policy"] = cue["candidate_policy"]
                        local_policy["step_b_candidate_eligible"] = False
                        local_policy["scale_support_only"] = cue["scale_support_only"]
                        local_policy["positive_trigger_allowed"] = False
                        cue["local_evidence_use_policy"] = local_policy
                        cue["derived_clause_span"]["local_evidence_use_policy"] = local_policy
                    cues.append(cue)
    seen = set()
    deduped = []
    for cue in cues:
        key = (cue["cue_type"], cue["start_char"], cue["end_char"], cue["span_text"].lower())
        if key in seen:
            continue
        seen.add(key)
        overlaps = []
        for index, existing in enumerate(deduped):
            same_clause = (
                cue.get("cue_type") == existing.get("cue_type")
                and cue.get("clause_start_char") == existing.get("clause_start_char")
                and cue.get("clause_end_char") == existing.get("clause_end_char")
            )
            ranges_overlap = cue["start_char"] < existing["end_char"] and existing["start_char"] < cue["end_char"]
            if same_clause and ranges_overlap:
                overlaps.append(index)
        if not overlaps:
            deduped.append(cue)
            continue
        longest_index = max(
            overlaps + [len(deduped)],
            key=lambda index: (
                len(cue["span_text"]) if index == len(deduped) else len(deduped[index]["span_text"]),
                -(cue["start_char"] if index == len(deduped) else deduped[index]["start_char"]),
            ),
        )
        if longest_index == len(deduped):
            for index in reversed(overlaps):
                deduped.pop(index)
            deduped.append(cue)
    actual_ranges = [
        (cue["start_char"], cue["end_char"])
        for cue in deduped
        if cue.get("cue_type") == "actual_toxic_ingestion_or_overdose"
    ]
    deduped = [
        cue for cue in deduped
        if not (
            cue.get("cue_type") == "overdose_method_ideation_or_plan"
            and any(
                cue["start_char"] < end and start < cue["end_char"]
                for start, end in actual_ranges
            )
        )
    ]
    return deduped


def make_event_assertion(event, text_for_hints=""):
    """Build a conservative event-level assertion object for downstream inheritance."""
    source_layer = event.get("source_layer")
    polarity = event.get("polarity_hint") or infer_polarity_hint(text_for_hints, source_layer)
    temporality = event.get("temporality_hint") or infer_temporality_hint(text_for_hints)
    certainty = "asserted"
    if polarity in {"uncertain", "question"}:
        certainty = "uncertain" if polarity == "uncertain" else "interrogative"
    if polarity == "mixed_scope_possible":
        certainty = "requires_target_resolution"
    if source_layer == "nurse_question":
        certainty = "interrogative"
    return {
        "polarity": polarity,
        "temporality": temporality,
        "experiencer": "patient",
        "certainty": certainty,
        "assertion_source": "event_text_inferred",
    }


def inherit_assertion_for_cues(event):
    """Attach parent provenance while retaining clause-local cue semantics."""
    assertion = event.get("assertion") or {}
    parent_id = event.get("evidence_id")
    scope_flags = event.get("assertion_scope_flags", []) or []
    clause_spans = {}
    clause_id_by_key = {}
    fixed = []
    for cue in event.get("clinical_cue_spans", []) or []:
        cue = dict(cue)
        cue["inherits_from_evidence_id"] = parent_id
        cue["assertion_source"] = "clause_local_resolver"
        cue["polarity"] = cue.get("local_polarity", assertion.get("polarity", cue.get("polarity_hint")))
        cue["temporality"] = cue.get("local_temporality", assertion.get("temporality", cue.get("temporality_hint")))
        cue["experiencer"] = assertion.get("experiencer", "patient")
        cue["assertion"] = {
            "polarity": cue["polarity"],
            "temporality": cue["temporality"],
            "experiencer": cue["experiencer"],
            "certainty": cue.get("certainty", "asserted"),
            "assertion_source": "clause_local_resolver",
        }
        if scope_flags:
            cue["assertion_scope_flags"] = list(scope_flags)
        clause_key = (cue.get("clause_start_char"), cue.get("clause_end_char"))
        if clause_key not in clause_spans:
            clause_spans[clause_key] = cue.get("derived_clause_span", {})
        if clause_key not in clause_id_by_key:
            clause_id_by_key[clause_key] = f"{parent_id}:clause_{len(clause_id_by_key)}"
        cue["clause_span_id"] = clause_id_by_key[clause_key]
        cue["polarity_hint"] = cue["polarity"]
        cue["temporality_hint"] = cue["temporality"]
        fixed.append(cue)
    event["clinical_cue_spans"] = fixed
    derived = []
    for clause_key, span in clause_spans.items():
        span = dict(span)
        span["clause_span_id"] = clause_id_by_key[clause_key]
        span["inherits_from_evidence_id"] = parent_id
        derived.append(span)
    event["derived_clause_spans"] = derived
    return event




def attach_default_event_policy_and_provenance(event):
    """Attach a complete evidence policy/provenance schema to free-text events.

    Structured clinical inputs and optional patient_context events define their own policy.
    This helper mainly normalizes nurse/patient utterance events so downstream modules
    never need to infer whether an event is independent evidence or context-only.
    """
    source_layer = event.get("source_layer")
    actor = event.get("actor")

    if source_layer in {"patient_reported", "nurse_observed_statement"}:
        independent = True
        context_only = False
        allowed_uses = ["symptom_reasoning", "resource_reasoning", "safety_reasoning"]
        disallowed_uses = []
    elif source_layer == "nurse_question":
        # A nurse question may contain symptom words and cue spans, but is not a
        # positive clinical fact. Keep it as a cue-bearing non-independent event
        # so the cue registry can preserve polarity=question without violating
        # the rule that context_only events must not carry clinical cues.
        independent = False
        context_only = False
        allowed_uses = ["conversation_context", "answer_interpretation_context"]
        disallowed_uses = [
            "standalone_positive_clinical_evidence",
            "direct_resource_vote",
            "standalone_high_risk_trigger",
        ]
    elif source_layer in {"nurse_instruction_or_process", "nurse_other", "nurse_mixed_utterance"}:
        independent = False
        context_only = False
        allowed_uses = ["conversation_context", "workflow_context", "clause_local_resolution"]
        disallowed_uses = [
            "standalone_positive_clinical_evidence",
            "direct_resource_vote",
            "standalone_high_risk_trigger",
        ]
    else:
        independent = True
        context_only = False
        allowed_uses = ["clinical_reasoning"]
        disallowed_uses = []

    event.setdefault("independent_evidence", independent)
    event.setdefault("context_only", context_only)
    event.setdefault("evidence_use_policy", {
        "allowed": True,
        "independent_evidence": independent,
        "context_only": context_only,
        "allowed_uses": allowed_uses,
        "disallowed_uses": disallowed_uses,
    })
    event.setdefault("provenance", {
        "source_field": event.get("source_field") or f"history.{actor}.original_or_utterance",
        "derived_from_evidence_ids": [],
        "transformation": "cleaned_original_or_utterance_copy",
    })
    return event


def add_event_common_fields(event, text_for_hints=None):
    text = text_for_hints if text_for_hints is not None else (event.get("original") or event.get("text") or "")
    source_layer = event.get("source_layer")
    event["event_surface_polarity_hint"] = infer_event_surface_polarity_hint(text, source_layer)
    event["assertion_scope_flags"] = build_assertion_scope_flags(text)
    event["polarity_hint"] = infer_polarity_hint(text, source_layer)
    event["temporality_hint"] = infer_temporality_hint(text)
    event["assertion"] = make_event_assertion(event, text)
    event["clinical_cue_spans"] = extract_clinical_cue_spans(
        text,
        source_layer,
        utterance_segments=event.get("utterance_segments"),
    )
    inherit_assertion_for_cues(event)
    attach_default_event_policy_and_provenance(event)
    return event


def get_vignette_vitals(vignette):
    return {key: vignette.get(key) for key in VITAL_KEYS if vignette.get(key) is not None}


def get_ground_truth_vitals(ground_truth):
    vitals = ground_truth.get("vitals", {}) if isinstance(ground_truth, dict) else {}
    if not isinstance(vitals, dict):
        return {}
    return {key: vitals.get(key) for key in VITAL_KEYS if vitals.get(key) is not None}


def values_match(left, right):
    left_norm = normalize_value(left)
    right_norm = normalize_value(right)
    if isinstance(left_norm, (int, float)) and isinstance(right_norm, (int, float)):
        return abs(float(left_norm) - float(right_norm)) < 1e-9
    return left_norm == right_norm



def choose_structured_value(field, vignette, ground_truth, allow_nonlabel_ground_truth_fallback=False):
    vignette_value = vignette.get(field)
    if field in VITAL_KEYS:
        gt_value = (ground_truth.get("vitals", {}) or {}).get(field)
    else:
        gt_value = ground_truth.get(field)
    if vignette_value is not None:
        return vignette_value, "primary_vignette_structured_input"
    if gt_value is not None and allow_nonlabel_ground_truth_fallback:
        return gt_value, "non_label_ground_truth_fallback"
    if gt_value is not None and not allow_nonlabel_ground_truth_fallback:
        return None, "missing_vignette_ground_truth_fallback_disabled"
    return None, "missing"


def build_structured_consistency_audit(vignette, ground_truth, allow_nonlabel_ground_truth_fallback=False):
    audit = {
        "canonical_policy": "vignette_first_non_label_ground_truth_fallback_only_if_enabled",
        "allow_nonlabel_ground_truth_fallback": bool(allow_nonlabel_ground_truth_fallback),
        "label_fields_withheld_from_prediction": ["vignette.acuity", "ground_truth.acuity"],
        "forbidden_structured_fields_withheld_from_prediction": ["vignette.specialisation", "vignette.specialization"],
        "field_audits": {},
        "vignette_missing_but_non_label_ground_truth_present": [],
        "non_label_ground_truth_missing_but_vignette_present": [],
        "conflict_fields": [],
        "conflict_count": 0,
        "fallback_fields": [],
        "fallback_field_count": 0,
        "ground_truth_nonlabel_available_but_fallback_disabled": [],
        "acuity_consistency_audit_only": {
            "vignette_acuity": vignette.get("acuity"),
            "ground_truth_acuity": ground_truth.get("acuity"),
            "matches": values_match(vignette.get("acuity"), ground_truth.get("acuity")),
            "excluded_from_prediction_timeline": True,
        },
    }
    for field in STRUCTURED_CLINICAL_FIELDS:
        vignette_value = vignette.get(field)
        if field in VITAL_KEYS:
            gt_value = (ground_truth.get("vitals", {}) or {}).get(field)
        else:
            gt_value = ground_truth.get(field)
        if isinstance(vignette_value, str):
            vignette_value = sanitize_prediction_text(vignette_value)
        if isinstance(gt_value, str):
            gt_value = sanitize_prediction_text(gt_value)
        chosen_value, chosen_origin = choose_structured_value(
            field,
            vignette,
            ground_truth,
            allow_nonlabel_ground_truth_fallback=allow_nonlabel_ground_truth_fallback,
        )
        match = values_match(vignette_value, gt_value) if vignette_value is not None and gt_value is not None else None
        if vignette_value is None and gt_value is not None:
            audit["vignette_missing_but_non_label_ground_truth_present"].append(field)
            if allow_nonlabel_ground_truth_fallback:
                audit["fallback_fields"].append(field)
            else:
                audit["ground_truth_nonlabel_available_but_fallback_disabled"].append(field)
        if gt_value is None and vignette_value is not None:
            audit["non_label_ground_truth_missing_but_vignette_present"].append(field)
        if match is False:
            audit["conflict_fields"].append(field)
        audit["field_audits"][field] = {
            "vignette_present": vignette_value is not None,
            "non_label_ground_truth_present": gt_value is not None,
            "values_match_when_both_present": match,
            "prediction_value_origin": chosen_origin,
            "prediction_value_present": chosen_value is not None,
            "non_label_ground_truth_used_for_prediction": chosen_origin == "non_label_ground_truth_fallback",
        }
    audit["conflict_count"] = len(audit["conflict_fields"])
    audit["fallback_field_count"] = len(audit["fallback_fields"])
    audit["vignette_vs_ground_truth_nonlabel_all_match_when_both_present"] = audit["conflict_count"] == 0
    return audit


def merge_structured_clinical_input(vignette, ground_truth, allow_nonlabel_ground_truth_fallback=False):
    structured_input = {}
    field_origins = {}
    for field in STRUCTURED_CLINICAL_FIELDS:
        value, origin = choose_structured_value(
            field,
            vignette,
            ground_truth,
            allow_nonlabel_ground_truth_fallback=allow_nonlabel_ground_truth_fallback,
        )
        if isinstance(value, str):
            value = sanitize_prediction_text(value)
        if value is None:
            continue
        structured_input[field] = value
        field_origins[field] = origin
    audit = build_structured_consistency_audit(
        vignette,
        ground_truth,
        allow_nonlabel_ground_truth_fallback=allow_nonlabel_ground_truth_fallback,
    )
    audit["prediction_field_origins"] = field_origins
    audit["prediction_field_count"] = len(structured_input)
    audit["prediction_fields"] = sorted(structured_input.keys())
    audit["non_label_ground_truth_fallback_used"] = any(
        origin == "non_label_ground_truth_fallback" for origin in field_origins.values()
    )
    return structured_input, audit


def classify_structured_pain_hint(value):
    normalized = normalize_contract_value(value)
    if normalized is None or not isinstance(normalized, (int, float)):
        return "pain_measurement_unknown"
    if normalized < PAIN_MIN or normalized > PAIN_MAX:
        return "pain_measurement_unusable"
    if normalized == 0:
        return "no_pain"
    if 1 <= normalized <= 3:
        return "mild_pain_score"
    if 4 <= normalized <= 6:
        return "moderate_pain_score"
    if normalized >= 7:
        return "severe_pain_score"
    return "pain_measurement_unknown"


def build_structured_text(field, value):
    if field == "chiefcomplaint":
        return f"chief complaint: {value}"
    if field == "gender":
        return f"gender: {value}"
    if field == "arrival_transport":
        return f"arrival transport: {value}"
    if field in VITAL_KEYS:
        display_name = VITAL_DISPLAY_NAMES.get(field, field)
        unit = VITAL_DISPLAY_UNITS.get(field)
        unit_part = f" {unit}" if unit else ""
        return f"{display_name}={value}{unit_part}"
    if field == "pain":
        return f"pain_score={value}/10"
    return f"{field}={value}"


def apply_structured_event_hints(event, field, value):
    """Apply deterministic semantics to structured fields instead of text-style inference."""
    if field in VITAL_KEYS or field == "pain":
        usable = event.get("usable_for_clinical_reasoning") is True
        if not usable:
            event["event_surface_polarity_hint"] = "structured_input_quarantined"
            event["assertion_scope_flags"] = ["measurement_quarantined"]
            event["polarity_hint"] = "measurement_unusable"
            event["temporality_hint"] = "not_usable_for_clinical_reasoning"
            event["assertion"] = {
                "polarity": "measurement_unusable",
                "temporality": "unknown",
                "experiencer": "patient",
                "certainty": "invalid_or_unusable_measurement",
                "assertion_source": "measurement_contract_quarantine",
            }
            event["independent_evidence"] = False
            event["context_only"] = False
            event["evidence_use_policy"] = {
                "allowed": False,
                "independent_evidence": False,
                "context_only": False,
                "support_only": False,
                "allowed_uses": [],
                "disallowed_uses": ["clinical_reasoning", "support_evidence", "positive_trigger"],
            }
            event["clinical_cue_spans"] = []
            return event
    event["event_surface_polarity_hint"] = "structured_input"
    event["assertion_scope_flags"] = []
    if field == "chiefcomplaint":
        event["polarity_hint"] = "presenting_complaint"
        event["temporality_hint"] = "initial_assessment_current"
        event["assertion"] = {
            "polarity": "positive",
            "temporality": "current",
            "experiencer": "patient",
            "certainty": "asserted",
            "assertion_source": "structured_chief_complaint_default",
        }
        event["evidence_use_policy"] = {
            "allowed": True,
            "independent_evidence": True,
            "context_only": False,
            "standalone_step_b_trigger": False,
            "allowed_uses": ["symptom_reasoning", "resource_reasoning", "safety_reasoning"],
            "disallowed_uses": [],
        }
        event["clinical_cue_spans"] = extract_clinical_cue_spans(event.get("original", ""), event.get("source_layer"))
        inherit_assertion_for_cues(event)
    elif field in VITAL_KEYS:
        event["polarity_hint"] = "measured_value"
        event["temporality_hint"] = "initial_assessment_current"
        event["assertion"] = {
            "polarity": "measured_value",
            "temporality": "current",
            "experiencer": "patient",
            "certainty": "measured",
            "assertion_source": "structured_measurement_default",
        }
        event["clinical_cue_spans"] = []
        event["evidence_use_policy"] = {
            "allowed": True,
            "independent_evidence": True,
            "context_only": False,
            "standalone_step_b_trigger": False,
            "allowed_uses": ["vital_sign_reasoning", "safety_reasoning", "resource_reasoning"],
            "disallowed_uses": [],
        }
    elif field == "pain":
        event["polarity_hint"] = classify_structured_pain_hint(value)
        event["temporality_hint"] = "initial_assessment_current"
        event["assertion"] = {
            "polarity": event["polarity_hint"],
            "temporality": "current",
            "experiencer": "patient",
            "certainty": "measured",
            "assertion_source": "structured_pain_score_default",
        }
        event["clinical_cue_spans"] = []
        event["evidence_use_policy"] = {
            "allowed": True,
            "independent_evidence": True,
            "context_only": False,
            "support_only": True,
            "standalone_step_b_trigger": False,
            "allowed_uses": ["pain_reasoning", "resource_reasoning", "safety_reasoning"],
            "disallowed_uses": [],
        }
    elif field in {"gender", "arrival_transport"}:
        event["independent_evidence"] = False
        event["context_only"] = True
        event["polarity_hint"] = "context"
        event["temporality_hint"] = "context"
        event["assertion"] = {
            "polarity": "context",
            "temporality": "context",
            "experiencer": "patient",
            "certainty": "asserted",
            "assertion_source": "structured_context_default",
        }
        event["clinical_cue_spans"] = []
        event["evidence_use_policy"] = {
            "allowed": True,
            "independent_evidence": False,
            "context_only": True,
            "standalone_step_b_trigger": False,
            "allowed_uses": ["demographic_or_arrival_context", "resource_reasoning", "safety_reasoning"],
            "disallowed_uses": ["standalone_high_risk_trigger"],
        }
    else:
        add_event_common_fields(event, event.get("original", ""))
    return event


def build_structured_clinical_input_events(instance_id, structured_input, start_idx):
    events = []
    ev_idx = start_idx
    ordered_fields = ["gender", "arrival_transport", "chiefcomplaint"] + VITAL_KEYS + ["pain"]
    for field in ordered_fields:
        if field not in structured_input:
            continue
        value = structured_input.get(field)
        if field == "chiefcomplaint":
            event_type = "chief_complaint"
            event_name = field
        elif field == "gender":
            event_type = "structured_demographic"
            event_name = field
        elif field == "arrival_transport":
            event_type = "structured_arrival_context"
            event_name = field
        elif field in VITAL_KEYS:
            event_type = "structured_vital"
            event_name = CANONICAL_VITAL_NAMES.get(field, field)
        elif field == "pain":
            event_type = "structured_pain_score"
            event_name = "pain_score"
        else:
            event_type = "structured_clinical_field"
            event_name = field
        text = build_structured_text(field, value)
        event = {
            "evidence_id": make_evidence_id(instance_id, ev_idx),
            "turn": -1,
            "actor": "structured_clinical_input",
            "event_type": event_type,
            "name": event_name,
            "value": value,
            "text": text,
            "original": text,
            "source_layer": "structured_clinical_input",
            "source_field": f"structured_clinical_input.{event_name}",
            "prediction_visibility": "allowed",
            "independent_evidence": True,
            "context_only": False,
            "provenance": {
                "source_field": f"structured_clinical_input.{event_name}",
                "derived_from_evidence_ids": [],
                "transformation": "deterministic_whitelist_merge",
            },
        }
        if field in VITAL_KEYS:
            event["raw_name"] = field
            event["canonical_vital_name"] = CANONICAL_VITAL_NAMES.get(field, field)
            event["canonical_vital_value"] = normalize_value(value)
            event["measurement_source"] = "structured_clinical_input"
            event["unit_assumption"] = VITAL_UNIT_ASSUMPTIONS.get(field)
            event["measurement_identity"] = "canonical_case_measurement"
            event["independent_measurement"] = True
            event["canonical_measurement_evidence_id"] = event["evidence_id"]
            event.update(classify_numeric_plausibility(field, value))
        if field == "pain":
            event["raw_name"] = "pain"
            event["is_structured_pain_score"] = True
            event["measurement_source"] = "structured_clinical_input"
            event["measurement_identity"] = "canonical_case_measurement"
            event["independent_measurement"] = True
            event["canonical_measurement_evidence_id"] = event["evidence_id"]
            event.update(classify_pain_validity(value))
        apply_structured_event_hints(event, field, value)
        events.append(event)
        ev_idx += 1
    return events, ev_idx


def build_patient_context_events(instance_id, patient_persona, start_idx, include_patient_context=False):
    """Optional persona-derived context. Disabled by default; never positive clinical evidence."""
    events = []
    ev_idx = start_idx
    if not include_patient_context:
        return events, ev_idx
    if not isinstance(patient_persona, dict):
        return events, ev_idx
    for field in PATIENT_CONTEXT_FIELDS:
        value = patient_persona.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            value = clean_transcript_text(sanitize_prediction_text(value))
        text = f"patient context {field}: {value}"
        event = {
            "evidence_id": make_evidence_id(instance_id, ev_idx),
            "turn": -1,
            "actor": "patient_context",
            "event_type": "structured_patient_context",
            "name": field,
            "value": value,
            "text": text,
            "original": text,
            "source_layer": "structured_patient_context",
            "source_field": f"patient_context.{field}",
            "prediction_visibility": "allowed",
            "independent_evidence": False,
            "context_only": True,
            "polarity_hint": "context_only",
            "temporality_hint": "context",
            "assertion": {
                "polarity": "context_only",
                "temporality": "context",
                "experiencer": "patient",
                "certainty": "context",
                "assertion_source": "patient_context_whitelist",
            },
            "clinical_cue_spans": [],
            "evidence_use_policy": {
                "allowed": True,
                "independent_evidence": False,
                "context_only": True,
                "allowed_uses": ["communication_barrier", "reliability_adjustment", "review_flag"],
                "disallowed_uses": ["direct_acuity_vote", "direct_resource_vote", "standalone_high_risk_trigger"],
            },
            "context_use_limit": "context_or_evidence_reliability_only_not_step_trigger",
            "provenance": {
                "source_field": f"patient_context.{field}",
                "derived_from_evidence_ids": [],
                "transformation": "whitelist_map_from_patient_persona",
            },
        }
        events.append(event)
        ev_idx += 1
    return events, ev_idx


def get_source_layer_for_history_item(actor, event_type, text):
    if actor == "patient":
        return "patient_reported"
    if actor == "nurse":
        return classify_nurse_utterance(text)
    if actor == "system":
        return "system_vital" if event_type == "vital" else "system_event"
    return "unknown"


def build_history_events(instance_id, history, start_idx, structured_vital_lookup=None):
    structured_vital_lookup = structured_vital_lookup or {}
    events = []
    ev_idx = start_idx
    audit = {
        "history_triage_entry_count": 0,
        "history_triage_by_turn": [],
        "system_events_excluded_from_prediction": [],
        "unknown_actor_events_excluded_raw_content": 0,
        "unknown_actor_events_excluded_from_prediction": [],
    }
    for history_index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        actor = item.get("actor", "unknown")
        turn = item.get("turn")
        if "triage" in item:
            audit["history_triage_entry_count"] += 1
            audit["history_triage_by_turn"].append({
                "history_index": history_index,
                "turn": turn,
                "actor": actor,
                "triage": item.get("triage"),
            })
        if actor in ["nurse", "patient"]:
            text, original = get_text_for_event(item)
            source_layer = get_source_layer_for_history_item(actor, "utterance", original)
            utterance_segments = build_nurse_utterance_segments(original) if actor == "nurse" else []
            if utterance_segments:
                source_layer = nurse_parent_source_layer(utterance_segments, source_layer)
            event = {
                "evidence_id": make_evidence_id(instance_id, ev_idx),
                "history_index": history_index,
                "turn": turn,
                "actor": actor,
                "event_type": "utterance",
                "text": text,
                "original": original,
                "source_layer": source_layer,
                "prediction_visibility": "allowed",
            }
            if utterance_segments:
                event["utterance_segments"] = utterance_segments
            add_event_common_fields(event, original)
            events.append(event)
            ev_idx += 1
        elif actor == "system":
            event_type = item.get("event", "system_event")
            if event_type != "vital":
                audit["system_events_excluded_from_prediction"].append({
                    "turn": turn,
                    "event": event_type,
                    "reason": "non_clinical_system_process_marker",
                })
                continue
            name = item.get("name")
            value = item.get("value")
            raw_name = name
            canonical_name = CANONICAL_VITAL_NAMES.get(raw_name, raw_name)
            structured_counterpart = structured_vital_lookup.get(raw_name) or structured_vital_lookup.get(canonical_name)
            contract_key = VITAL_NAME_ALIASES.get(str(raw_name).strip().lower(), raw_name)
            unit = VITAL_DISPLAY_UNITS.get(contract_key)
            display_name = VITAL_DISPLAY_NAMES.get(raw_name, canonical_name)
            unit_part = f" {unit}" if unit else ""
            vital_text = f"{display_name}={value}{unit_part}"
            measurement_contract = classify_numeric_plausibility(contract_key, value)
            event = {
                "evidence_id": make_evidence_id(instance_id, ev_idx),
                "history_index": history_index,
                "turn": turn,
                "actor": "system",
                "event_type": "vital",
                "event_subtype": "vital_reveal",
                "source_layer": "system_vital",
                "prediction_visibility": "allowed",
                "name": canonical_name,
                "raw_name": raw_name,
                "value": value,
                "text": vital_text,
                "original": vital_text,
                "source_field": f"history.system.{canonical_name}",
                "canonical_vital_name": canonical_name,
                "canonical_vital_value": normalize_value(value),
                "measurement_source": "history_system_vital",
                "unit_assumption": measurement_contract.get("raw_unit") or VITAL_UNIT_ASSUMPTIONS.get(contract_key),
                "measurement_identity": "dialogue_reveal",
                "independent_measurement": False,
                "canonical_measurement_evidence_id": structured_counterpart.get("evidence_id") if structured_counterpart else None,
                "structured_counterpart_present": structured_counterpart is not None,
                "structured_counterpart_evidence_id": structured_counterpart.get("evidence_id") if structured_counterpart else None,
                "value_matches_structured": values_match(value, structured_counterpart.get("value")) if structured_counterpart else None,
                "event_surface_polarity_hint": "measured_value",
                "assertion_scope_flags": [],
                "polarity_hint": "measured_value",
                "temporality_hint": "initial_assessment_current",
                "clinical_cue_spans": [],
                "independent_evidence": False,
                "context_only": True,
                "evidence_use_policy": {
                    "allowed": True,
                    "independent_evidence": False,
                    "context_only": True,
                    "support_only": True,
                    "allowed_uses": ["vital_reveal_context"],
                    "disallowed_uses": [],
                },
                "assertion": {
                    "polarity": "measured_value",
                    "temporality": "current",
                    "experiencer": "patient",
                    "certainty": "measured",
                    "assertion_source": "system_vital_measurement_default",
                },
                "provenance": {
                    "source_field": f"history.system.{canonical_name}",
                    "derived_from_evidence_ids": [],
                    "transformation": "raw_copy_system_vital_event",
                },
            }
            event.update(measurement_contract)
            if event.get("usable_for_clinical_reasoning") is False:
                event["event_surface_polarity_hint"] = "vital_reveal_quarantined"
                event["polarity_hint"] = "measurement_unusable"
                event["temporality_hint"] = "not_usable_for_clinical_reasoning"
                event["assertion"] = {
                    "polarity": "measurement_unusable",
                    "temporality": "unknown",
                    "experiencer": "patient",
                    "certainty": "invalid_or_unusable_measurement",
                    "assertion_source": "measurement_contract_quarantine",
                }
                event["evidence_use_policy"] = {
                    "allowed": False,
                    "independent_evidence": False,
                    "context_only": False,
                    "support_only": False,
                    "allowed_uses": [],
                    "disallowed_uses": ["clinical_reasoning", "support_evidence", "positive_trigger"],
                }
            events.append(event)
            ev_idx += 1
        else:
            audit["unknown_actor_events_excluded_raw_content"] += 1
            audit["unknown_actor_events_excluded_from_prediction"].append({
                "turn": turn,
                "actor": actor,
                "reason": "unknown_actor_raw_content_not_copied_to_prediction_timeline",
            })
    return events, ev_idx, audit


def build_dialogue_derived_numeric_cue(event, start_char, end_char, target, confidence):
    text = event.get("original") or event.get("text") or ""
    clause_start, clause_end = next(
        (
            start,
            end,
        )
        for start, end in split_clause_ranges(text)
        if start <= start_char and end_char <= end
    )
    clause_text = text[clause_start:clause_end]
    local = resolve_clause_local_assertion(
        clause_text,
        start_char - clause_start,
        end_char - clause_start,
        "patient_reported",
        cue_type="numeric_severity_scale_mention",
    )
    local_policy = build_local_evidence_policy("patient_reported", "numeric_severity_scale_mention")
    local_policy["candidate_policy"] = (
        "pain_scale_support_only" if target == "pain" else "non_pain_or_unresolved_scale"
    )
    local_policy["scale_support_only"] = target == "pain"
    local_policy["step_b_candidate_eligible"] = False
    local_policy["positive_trigger_allowed"] = False
    event_scope_flags = build_assertion_scope_flags(text)
    event_scope_requires_resolution = "scope_requires_target_level_resolution" in event_scope_flags
    return {
        "cue_type": "numeric_severity_scale_mention",
        "span_text": text[start_char:end_char],
        "start_char": start_char,
        "end_char": end_char,
        "polarity_hint": local["local_polarity"],
        "temporality_hint": local["local_temporality"],
        "source_layer": "patient_reported",
        "local_source_layer": "patient_reported",
        "rule": "dialogue_derived_short_numeric_scale_answer",
        "clause_text": clause_text,
        "clause_start_char": clause_start,
        "clause_end_char": clause_end,
        "local_polarity": local["local_polarity"],
        "local_temporality": local["local_temporality"],
        "clinical_status": local["clinical_status"],
        "certainty": local["certainty"],
        "scope_resolution": local["scope_resolution"],
        "scope_confidence": local["scope_confidence"],
        "temporal_resolution": local["temporal_resolution"],
        "local_independent_evidence": True,
        "local_context_only": False,
        "local_evidence_use_policy": local_policy,
        "candidate_policy": local_policy["candidate_policy"],
        "step_b_candidate_eligible": False,
        "event_scope_flags": event_scope_flags,
        "event_scope_requires_target_level_resolution": event_scope_requires_resolution,
        "numeric_values": scale_values_from_text(text[start_char:end_char]),
        "scale_target_hint": target,
        "scale_target_confidence": confidence,
        "scale_target_source": "previous_question",
        "scale_target_resolution": (
            "resolved_from_previous_scale_question"
            if target in {"pain", "non_pain"}
            else "multi_target_previous_question_requires_downstream_resolution"
        ),
        "scale_policy": "generic_numeric_severity_not_pain_by_default",
        "scale_support_only": target == "pain",
        "derived_clause_span": {
            "span_text": clause_text,
            "start_char": clause_start,
            "end_char": clause_end,
            "local_polarity": local["local_polarity"],
            "local_temporality": local["local_temporality"],
            "scope_resolution": local["scope_resolution"],
            "scope_confidence": local["scope_confidence"],
            "temporal_resolution": local["temporal_resolution"],
            "source_layer": "patient_reported",
            "local_source_layer": "patient_reported",
            "local_independent_evidence": True,
            "local_context_only": False,
            "local_evidence_use_policy": local_policy,
            "event_scope_flags": event_scope_flags,
            "event_scope_requires_target_level_resolution": event_scope_requires_resolution,
        },
    }


def attach_dialogue_scaffold(events):
    """Attach prediction-safe dialogue structure links without adding label data."""
    last_question_id = None
    last_question_event = None
    last_patient_id = None
    for idx, event in enumerate(events):
        event["event_seq_idx"] = idx
        event["prev_evidence_id"] = events[idx - 1].get("evidence_id") if idx > 0 else None
        event["next_evidence_id"] = events[idx + 1].get("evidence_id") if idx + 1 < len(events) else None
        source_layer = event.get("source_layer")
        actor = event.get("actor")
        link = {
            "previous_question_evidence_id": last_question_id,
            "answer_to_question_evidence_id": last_question_id if actor == "patient" else None,
            "previous_patient_evidence_id": last_patient_id,
            "next_patient_evidence_id": None,
        }
        event["dialogue_link"] = link
        if source_layer == "nurse_question":
            last_question_id = event.get("evidence_id")
            last_question_event = event
        elif source_layer == "nurse_mixed_utterance":
            question_segments = [
                seg.get("span_text", "")
                for seg in event.get("utterance_segments", []) or []
                if seg.get("segment_source_layer") == "nurse_question"
            ]
            if question_segments:
                last_question_id = event.get("evidence_id")
                last_question_event = dict(event)
                last_question_event["question_context_text"] = " ".join(question_segments)
        if actor == "patient":
            question_text = ""
            if last_question_event:
                question_text = last_question_event.get("question_context_text") or last_question_event.get("original") or ""
            if question_text and is_scale_question_text(question_text):
                question_target, question_confidence = classify_scale_target(question_text)
                for cue in event.get("clinical_cue_spans", []) or []:
                    if (
                        cue.get("cue_type") == "numeric_severity_scale_mention"
                        and cue.get("scale_target_hint") in {"unknown", "multi_target"}
                        and question_target not in {"unknown", "multi_target"}
                    ):
                        cue["scale_target_hint"] = question_target
                        cue["scale_target_confidence"] = question_confidence
                        cue["scale_target_source"] = "previous_question"
                        cue["scale_target_resolution"] = "resolved_from_previous_scale_question"
                        question_is_pain = question_target == "pain"
                        cue["candidate_policy"] = (
                            "pain_scale_support_only"
                            if question_is_pain
                            else "non_pain_or_unresolved_scale"
                        )
                        cue["scale_support_only"] = question_is_pain
                        cue["step_b_candidate_eligible"] = False
                        local_policy = cue.get("local_evidence_use_policy", {}) or {}
                        local_policy["candidate_policy"] = cue["candidate_policy"]
                        local_policy["scale_support_only"] = question_is_pain
                        local_policy["step_b_candidate_eligible"] = False
                        cue["local_evidence_use_policy"] = local_policy
                        for span in event.get("derived_clause_spans", []) or []:
                            if span.get("clause_span_id") == cue.get("clause_span_id"):
                                span["local_evidence_use_policy"] = local_policy
                                break
                if question_target != "unknown":
                    existing_spans = {
                        (cue.get("start_char"), cue.get("end_char"))
                        for cue in event.get("clinical_cue_spans", []) or []
                        if cue.get("cue_type") == "numeric_severity_scale_mention"
                    }
                    for clause_start, clause_end in split_clause_ranges(event.get("original") or ""):
                        clause_text = (event.get("original") or "")[clause_start:clause_end].strip()
                        if not clause_text:
                            continue
                        match = SHORT_NUMERIC_SCALE_RE.fullmatch(clause_text)
                        if not match:
                            continue
                        answer_start = clause_start + ((event.get("original") or "")[clause_start:clause_end].find(clause_text))
                        answer_end = answer_start + len(clause_text)
                        if (answer_start, answer_end) in existing_spans:
                            continue
                        event.setdefault("clinical_cue_spans", []).append(
                            build_dialogue_derived_numeric_cue(
                                event,
                                answer_start,
                                answer_end,
                                question_target,
                                question_confidence,
                            )
                        )
                        existing_spans.add((answer_start, answer_end))
                    if event.get("clinical_cue_spans"):
                        inherit_assertion_for_cues(event)
            last_patient_id = event.get("evidence_id")

    next_patient_id = None
    for event in reversed(events):
        event.setdefault("dialogue_link", {})["next_patient_evidence_id"] = next_patient_id
        if event.get("actor") == "patient":
            next_patient_id = event.get("evidence_id")
    return events


def make_prompt_safe_line(ev):
    eid = ev.get("evidence_id")
    turn = ev.get("turn")
    actor = ev.get("actor")
    layer = ev.get("source_layer")
    event_type = ev.get("event_type")
    text = sanitize_prediction_text(ev.get("original") or ev.get("text") or "")
    name = ev.get("name")
    value = ev.get("value")
    if ev.get("usable_for_clinical_reasoning") is False and ev.get("quarantine_reason"):
        name_value_part = f" {name}=QUARANTINED" if name is not None else ""
        text = "[quarantined measurement omitted from clinical reasoning]"
    else:
        name_value_part = f" {name}={value}" if name is not None and value is not None else ""
    polarity = ev.get("polarity_hint")
    temporality = ev.get("temporality_hint")
    return (
        f"[{eid}] turn={turn} actor={actor} layer={layer} type={event_type}"
        f"{name_value_part} polarity={polarity} temporality={temporality}: {text}"
    )



def build_safe_vignette_context(structured_events):
    """Evidence-locked derived view over structured events; not an independent evidence source."""
    context = {
        "summary_type": "safe_vignette_context",
        "evidence_locked_summary": True,
        "independent_evidence": False,
        "context_only": True,
        "not_an_independent_evidence_source": True,
        "derivation_method": "deterministic_whitelist_merge_from_structured_events",
        "downstream_evidence_use_allowed": True,
        "downstream_must_cite_attached_evidence_id": True,
        "may_not_increment_evidence_count": True,
        "may_not_override_source_assertion": True,
        "derived_from_evidence_ids": [],
        "quarantined_measurements": [],
        "fields": {},
        "vitals": {},
        "evidence_use_policy": {
            "visible_to_downstream": True,
            "may_be_cited": True,
            "must_resolve_to_source_evidence": True,
            "may_not_increment_evidence_count": True,
            "may_not_override_source_assertion": True,
        },
    }
    for ev in structured_events:
        if ev.get("source_layer") != "structured_clinical_input":
            continue
        event_type = ev.get("event_type")
        name = ev.get("name")
        raw_name = ev.get("raw_name")
        eid = ev.get("evidence_id")
        if not eid:
            continue
        if ev.get("usable_for_clinical_reasoning") is False:
            context["quarantined_measurements"].append({
                "source_evidence_id": eid,
                "name": name,
                "canonical_unit": ev.get("canonical_unit"),
                "value_plausibility": ev.get("value_plausibility"),
                "quarantine_reason": ev.get("quarantine_reason"),
                "raw_value_quarantined": True,
                "allowed_for_clinical_reasoning": False,
            })
            continue
        context["derived_from_evidence_ids"].append(eid)
        value_entry = {
            "value": ev.get("value"),
            "source_evidence_id": eid,
            "source_layer": ev.get("source_layer"),
            "event_type": event_type,
            "assertion": ev.get("assertion"),
            "independent_evidence": False,
            "not_an_independent_evidence_source": True,
        }
        if event_type == "structured_vital":
            vital_key = raw_name or name
            value_entry.update({
                "name": name,
                "raw_name": raw_name,
                "canonical_vital_name": ev.get("canonical_vital_name"),
                "unit_assumption": ev.get("unit_assumption"),
            "value_plausibility": ev.get("value_plausibility"),
            "usable_for_clinical_reasoning": ev.get("usable_for_clinical_reasoning"),
            })
            context["vitals"][vital_key] = value_entry
        else:
            field_key = raw_name or name
            value_entry["name"] = name
            if raw_name:
                value_entry["raw_name"] = raw_name
            if event_type == "structured_pain_score":
                value_entry["pain_quality"] = classify_pain_validity(ev.get("value"))
            context["fields"][field_key] = value_entry
    context["derived_from_evidence_ids"] = sorted(set(context["derived_from_evidence_ids"]))
    return context


def collect_quality_flags(events, history_audit):
    flags = {
        "stripped_history_triage_entries": history_audit.get("history_triage_entry_count", 0),
        "excluded_system_process_events": len(history_audit.get("system_events_excluded_from_prediction", [])),
        "implausible_vital_events": [],
        "invalid_pain_events": [],
        "quarantined_measurements": [],
        "unknown_source_events": [],
        "forbidden_event_field_count": 0,
        "forbidden_event_fields": [],
        "structured_clinical_input_event_count": 0,
        "structured_patient_context_event_count": 0,
        "structured_vital_event_count": 0,
        "derived_clause_span_count": 0,
        "nurse_segment_source_layer_counts": {},
        "observation_segment_count": 0,
        "question_segment_count": 0,
        "process_segment_count": 0,
        "discourse_only_segment_count": 0,
        "mixed_scope_context_cue_count": 0,
        "silent_denial_positive_count": 0,
        "silent_denial_positive_examples": [],
    }
    for ev in events:
        eid = ev.get("evidence_id")
        if ev.get("value_plausibility") == "implausible":
            flags["implausible_vital_events"].append({
                "evidence_id": eid,
                "name": ev.get("name"),
                "value": ev.get("value"),
                "reason": ev.get("plausibility_reason"),
            })
        if ev.get("event_type") == "structured_pain_score" and ev.get("value_validity") == "out_of_range":
            flags["invalid_pain_events"].append({
                "evidence_id": eid,
                "value": ev.get("value"),
                "reason": ev.get("validity_reason"),
            })
        if ev.get("quarantine_reason"):
            flags["quarantined_measurements"].append({
                "evidence_id": eid,
                "event_type": ev.get("event_type"),
                "name": ev.get("name"),
                "raw_value": ev.get("raw_value", ev.get("value")),
                "quarantine_reason": ev.get("quarantine_reason"),
                "usable_for_clinical_reasoning": ev.get("usable_for_clinical_reasoning"),
            })
        if ev.get("source_layer") == "structured_clinical_input":
            flags["structured_clinical_input_event_count"] += 1
        if ev.get("source_layer") == "structured_patient_context":
            flags["structured_patient_context_event_count"] += 1
        if ev.get("event_type") == "structured_vital":
            flags["structured_vital_event_count"] += 1
        flags["derived_clause_span_count"] += len(ev.get("derived_clause_spans", []) or [])
        for segment in ev.get("utterance_segments", []) or []:
            segment_layer = segment.get("segment_source_layer") or "unknown"
            counts = flags["nurse_segment_source_layer_counts"]
            counts[segment_layer] = counts.get(segment_layer, 0) + 1
            if segment_layer == "nurse_observed_statement":
                flags["observation_segment_count"] += 1
            elif segment_layer == "nurse_question":
                flags["question_segment_count"] += 1
            elif segment_layer == "nurse_instruction_or_process":
                flags["process_segment_count"] += 1
            elif segment_layer == "nurse_other":
                flags["discourse_only_segment_count"] += 1
        flags["mixed_scope_context_cue_count"] += sum(
            1 for cue in ev.get("clinical_cue_spans", []) or []
            if cue.get("event_scope_requires_target_level_resolution") is True
        )
        for cue in ev.get("clinical_cue_spans", []) or []:
            if cue.get("local_polarity") != "positive":
                continue
            relative_start = cue.get("start_char", 0) - cue.get("clause_start_char", 0)
            target_polarity, target_scope = target_local_denial_resolution(
                cue.get("clause_text") or "",
                relative_start,
                cue_end=relative_start + len(cue.get("span_text") or ""),
                cue_type=cue.get("cue_type"),
            )
            if target_polarity in {"absent", "uncertain"}:
                flags["silent_denial_positive_count"] += 1
                if len(flags["silent_denial_positive_examples"]) < 20:
                    flags["silent_denial_positive_examples"].append({
                        "evidence_id": eid,
                        "cue_type": cue.get("cue_type"),
                        "span_text": cue.get("span_text"),
                        "local_polarity": cue.get("local_polarity"),
                        "target_local_polarity": target_polarity,
                        "target_local_scope": target_scope,
                    })
        if ev.get("source_layer") == "unknown":
            flags["unknown_source_events"].append(eid)
        forbidden_here = sorted(set(ev.keys()) & PREDICTION_FORBIDDEN_KEYS)
        if forbidden_here:
            flags["forbidden_event_field_count"] += len(forbidden_here)
            flags["forbidden_event_fields"].append({"evidence_id": eid, "fields": forbidden_here})
    return flags


def scan_forbidden_keys(obj, path="$"):
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            if key in PREDICTION_FORBIDDEN_KEYS:
                hits.append({"path": child_path, "key": key})
            hits.extend(scan_forbidden_keys(value, child_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(scan_forbidden_keys(value, f"{path}[{idx}]"))
    return hits


def scan_prediction_string_values_forbidden_tokens(obj, path="$"):
    hits = []
    if isinstance(obj, dict):
        evidence_id = obj.get("evidence_id")
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            if isinstance(value, str):
                matches = sorted(set(m.group(0).lower() for m in PROMPT_FORBIDDEN_TOKEN_RE.finditer(value)))
                if matches:
                    hit = {"path": child_path, "markers": matches, "text": value[:300]}
                    if evidence_id:
                        hit["evidence_id"] = evidence_id
                    hits.append(hit)
            else:
                hits.extend(scan_prediction_string_values_forbidden_tokens(value, child_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(scan_prediction_string_values_forbidden_tokens(value, f"{path}[{idx}]"))
    return hits


def scan_prompt_forbidden_tokens(timeline):
    return scan_prediction_string_values_forbidden_tokens(timeline)


def validate_prediction_timeline(timeline, include_patient_context=False):
    errors = []
    top_extra = sorted(set(timeline.keys()) - ALLOWED_TIMELINE_TOP_LEVEL_FIELDS)
    if top_extra:
        errors.append({"type": "unexpected_top_level_fields", "fields": top_extra})
    key_hits = scan_forbidden_keys(timeline)
    if key_hits:
        errors.append({"type": "forbidden_keys_recursive", "hits": key_hits[:100], "hit_count": len(key_hits)})
    token_hits = scan_prompt_forbidden_tokens(timeline)
    if token_hits:
        errors.append({"type": "forbidden_prompt_tokens", "hits": token_hits[:100], "hit_count": len(token_hits)})
    if len(timeline.get("events", [])) != len(timeline.get("prompt_safe_lines", [])):
        errors.append({
            "type": "event_prompt_safe_line_count_mismatch",
            "event_count": len(timeline.get("events", [])),
            "prompt_safe_line_count": len(timeline.get("prompt_safe_lines", [])),
        })
    if (timeline.get("quality_flags", {}) or {}).get("silent_denial_positive_count", 0):
        errors.append({
            "type": "silent_denial_positive",
            "count": (timeline.get("quality_flags", {}) or {}).get("silent_denial_positive_count"),
            "examples": (timeline.get("quality_flags", {}) or {}).get("silent_denial_positive_examples", [])[:20],
        })

    events = timeline.get("events", []) or []
    event_ids = {ev.get("evidence_id") for ev in events if ev.get("evidence_id")}

    # Patient context is disabled by default and must never be independent evidence.
    patient_context_events = [ev for ev in events if ev.get("source_layer") == "structured_patient_context"]
    if patient_context_events and not include_patient_context:
        errors.append({
            "type": "patient_context_present_while_disabled",
            "count": len(patient_context_events),
            "evidence_ids": [ev.get("evidence_id") for ev in patient_context_events[:50]],
        })
    for ev in patient_context_events:
        policy = ev.get("evidence_use_policy", {}) or {}
        if ev.get("independent_evidence") is not False or ev.get("context_only") is not True:
            errors.append({"type": "patient_context_not_context_only", "evidence_id": ev.get("evidence_id")})
        if ev.get("clinical_cue_spans"):
            errors.append({"type": "patient_context_has_clinical_cue_spans", "evidence_id": ev.get("evidence_id")})
        if policy.get("independent_evidence") is not False or policy.get("context_only") is not True:
            errors.append({"type": "patient_context_policy_not_context_only", "evidence_id": ev.get("evidence_id")})

    # Structured chief complaint must carry deterministic assertion, and cue spans must inherit from parent.
    for ev in events:
        eid = ev.get("evidence_id")
        event_type = ev.get("event_type")
        derived_clause_spans = ev.get("derived_clause_spans", []) or []
        derived_ids = {span.get("clause_span_id") for span in derived_clause_spans}
        event_text = ev.get("original") or ev.get("text") or ""
        span_by_id = {span.get("clause_span_id"): span for span in derived_clause_spans}
        actual_ingestion_cues = [
            cue for cue in ev.get("clinical_cue_spans", []) or []
            if cue.get("cue_type") == "actual_toxic_ingestion_or_overdose"
        ]
        ideation_cues = [
            cue for cue in ev.get("clinical_cue_spans", []) or []
            if cue.get("cue_type") == "overdose_method_ideation_or_plan"
        ]
        for actual in actual_ingestion_cues:
            for ideation in ideation_cues:
                if actual.get("start_char", 0) < ideation.get("end_char", 0) and ideation.get("start_char", 0) < actual.get("end_char", 0):
                    errors.append({
                        "type": "toxic_ingestion_ideation_span_overlap",
                        "evidence_id": eid,
                        "actual": actual,
                        "ideation": ideation,
                    })
        if ev.get("clinical_cue_spans") and not isinstance(derived_clause_spans, list):
            errors.append({"type": "clinical_cue_derived_clause_spans_missing", "evidence_id": eid})
        for span in derived_clause_spans:
            if span.get("inherits_from_evidence_id") != eid:
                errors.append({"type": "derived_clause_parent_mismatch", "evidence_id": eid, "span": span})
            if span.get("local_polarity") is None or span.get("local_temporality") is None:
                errors.append({"type": "derived_clause_local_assertion_missing", "evidence_id": eid, "span": span})
            if span.get("local_source_layer") is None or not isinstance(span.get("local_evidence_use_policy"), dict):
                errors.append({"type": "derived_clause_local_policy_missing", "evidence_id": eid, "span": span})
            start = span.get("start_char")
            end = span.get("end_char")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(event_text):
                errors.append({"type": "derived_clause_offset_invalid", "evidence_id": eid, "span": span})
            elif span.get("span_text") != event_text[start:end]:
                errors.append({"type": "derived_clause_text_offset_mismatch", "evidence_id": eid, "span": span})
        for cue in ev.get("clinical_cue_spans", []) or []:
            clause_id = cue.get("clause_span_id")
            referenced_span = span_by_id.get(clause_id)
            if clause_id not in derived_ids:
                errors.append({"type": "cue_clause_span_link_missing", "evidence_id": eid, "cue": cue})
            elif (
                cue.get("clause_start_char") != referenced_span.get("start_char")
                or cue.get("clause_end_char") != referenced_span.get("end_char")
            ):
                errors.append({"type": "cue_clause_offset_reference_mismatch", "evidence_id": eid, "cue": cue})
            for field in ["local_polarity", "local_temporality", "scope_confidence", "scope_resolution"]:
                if cue.get(field) is None:
                    errors.append({"type": "cue_local_semantic_field_missing", "evidence_id": eid, "field": field, "cue": cue})
            local_policy = cue.get("local_evidence_use_policy", {}) or {}
            if cue.get("local_source_layer") != cue.get("source_layer"):
                errors.append({"type": "cue_local_source_layer_mismatch", "evidence_id": eid, "cue": cue})
            if local_policy.get("independent_evidence") is not cue.get("local_independent_evidence"):
                errors.append({"type": "cue_local_policy_independence_mismatch", "evidence_id": eid, "cue": cue})
            if local_policy.get("context_only") is not cue.get("local_context_only"):
                errors.append({"type": "cue_local_policy_context_mismatch", "evidence_id": eid, "cue": cue})
            expected_assertion_allowed = cue.get("local_source_layer") in {
                "patient_reported",
                "nurse_observed_statement",
            }
            if cue.get("clinical_status") == "question_context" or cue.get("local_polarity") == "question":
                expected_assertion_allowed = False
            if local_policy.get("clinical_assertion_allowed") is not expected_assertion_allowed:
                errors.append({"type": "cue_clinical_assertion_policy_mismatch", "evidence_id": eid, "cue": cue})
            if local_policy.get("negative_evidence_allowed") is not expected_assertion_allowed:
                errors.append({"type": "cue_negative_evidence_policy_mismatch", "evidence_id": eid, "cue": cue})
            if cue.get("local_polarity") != "positive" and local_policy.get("positive_trigger_allowed") is not False:
                errors.append({"type": "nonpositive_cue_trigger_policy_not_blocked", "evidence_id": eid, "cue": cue})
            if not isinstance(cue.get("event_scope_flags"), list) or not isinstance(
                cue.get("event_scope_requires_target_level_resolution"), bool
            ):
                errors.append({"type": "cue_event_scope_context_missing", "evidence_id": eid, "cue": cue})
            if cue.get("source_layer") == "nurse_question" and (
                cue.get("local_independent_evidence") is not False
                or local_policy.get("clinical_positive_allowed") is not False
            ):
                errors.append({"type": "nurse_question_span_marked_positive_evidence", "evidence_id": eid, "cue": cue})
            if cue.get("source_layer") in {"nurse_instruction_or_process", "nurse_other"} and (
                cue.get("local_independent_evidence") is not False
                or local_policy.get("clinical_positive_allowed") is not False
            ):
                errors.append({"type": "nurse_non_observation_span_marked_positive_evidence", "evidence_id": eid, "cue": cue})
            if cue.get("cue_type") == "pain_mention" and (
                local_policy.get("candidate_policy") != "anchor_only"
                or local_policy.get("step_b_candidate_eligible") is not False
            ):
                errors.append({"type": "ordinary_pain_cue_not_anchor_only", "evidence_id": eid, "cue": cue})
            if cue.get("cue_type") == "actual_toxic_ingestion_or_overdose":
                if cue.get("local_polarity") == "uncertain" and cue.get("clinical_status") != "possible_actual_ingestion":
                    errors.append({"type": "uncertain_actual_ingestion_status_missing", "evidence_id": eid, "cue": cue})
                if cue.get("local_temporality") == "historical" and cue.get("clinical_status") != "historical_actual_overdose":
                    errors.append({"type": "historical_actual_overdose_status_missing", "evidence_id": eid, "cue": cue})
                if (
                    cue.get("local_polarity") != "positive"
                    or cue.get("local_temporality") == "historical"
                ) and local_policy.get("positive_trigger_allowed") is not False:
                    errors.append({
                        "type": "noncurrent_or_nonpositive_actual_ingestion_trigger_not_blocked",
                        "evidence_id": eid,
                        "cue": cue,
                    })
            if cue.get("cue_type") == "numeric_severity_scale_mention":
                if (
                    cue.get("scale_target_hint") is None
                    or cue.get("numeric_values") is None
                    or cue.get("scale_target_resolution") is None
                ):
                    errors.append({"type": "numeric_scale_contract_missing", "evidence_id": eid, "cue": cue})
                if cue.get("step_b_candidate_eligible") is not False:
                    errors.append({"type": "numeric_scale_step_b_eligibility_not_blocked", "evidence_id": eid, "cue": cue})
                target = cue.get("scale_target_hint")
                expected_policy = (
                    "pain_scale_support_only"
                    if target == "pain" and cue.get("local_source_layer") in {"patient_reported", "nurse_observed_statement"}
                    else "non_pain_or_unresolved_scale"
                )
                if cue.get("candidate_policy") != expected_policy:
                    errors.append({
                        "type": "numeric_scale_candidate_policy_mismatch",
                        "evidence_id": eid,
                        "expected": expected_policy,
                        "actual": cue.get("candidate_policy"),
                        "cue": cue,
                    })
        if ev.get("source_layer") == "nurse_mixed_utterance":
            segments = ev.get("utterance_segments", []) or []
            if len(segments) < 2 or len({seg.get("segment_source_layer") for seg in segments}) < 2:
                errors.append({"type": "nurse_mixed_utterance_not_segmented", "evidence_id": eid})
        policy = ev.get("evidence_use_policy", {}) or {}
        is_measurement = event_type in {"structured_vital", "structured_pain_score", "vital"}
        if is_measurement:
            required_contract_fields = [
                "raw_value",
                "raw_unit",
                "normalized_value",
                "canonical_unit",
                "value_plausibility",
                "quarantine_reason",
                "usable_for_clinical_reasoning",
                "evidence_use_policy",
            ]
            missing_contract_fields = [field for field in required_contract_fields if field not in ev]
            if missing_contract_fields:
                errors.append({
                    "type": "measurement_contract_fields_missing",
                    "evidence_id": eid,
                    "fields": missing_contract_fields,
                })
            policy = ev.get("evidence_use_policy", {}) or {}
            usable = ev.get("usable_for_clinical_reasoning") is True
            if not usable:
                if ev.get("quarantine_reason") in {None, ""}:
                    errors.append({"type": "quarantined_measurement_missing_reason", "evidence_id": eid})
                if policy.get("allowed") is not False or policy.get("support_only") is not False:
                    errors.append({"type": "quarantined_measurement_policy_not_blocked", "evidence_id": eid})
                if ev.get("independent_evidence") is not False:
                    errors.append({"type": "quarantined_measurement_marked_independent", "evidence_id": eid})
            else:
                if ev.get("quarantine_reason") is not None:
                    errors.append({"type": "usable_measurement_has_quarantine_reason", "evidence_id": eid})
                if policy.get("allowed") is not True:
                    errors.append({"type": "usable_measurement_policy_not_allowed", "evidence_id": eid})
            if event_type == "vital" and ev.get("measurement_identity") != "dialogue_reveal":
                errors.append({"type": "history_vital_missing_dialogue_reveal_identity", "evidence_id": eid})
            if event_type == "vital" and ev.get("independent_measurement") is not False:
                errors.append({"type": "history_vital_marked_independent_measurement", "evidence_id": eid})
            if event_type == "vital" and ev.get("context_only") is not True:
                errors.append({"type": "history_vital_reveal_not_context_only", "evidence_id": eid})
        if event_type == "structured_pain_score" and ev.get("usable_for_clinical_reasoning") is True:
            if policy.get("support_only") is not True or policy.get("standalone_step_b_trigger") is not False:
                errors.append({"type": "structured_pain_policy_not_support_only", "evidence_id": eid})
        if event_type in {"structured_demographic", "structured_arrival_context"}:
            if ev.get("independent_evidence") is not False or ev.get("context_only") is not True:
                errors.append({"type": "structured_context_marked_independent", "evidence_id": eid})
        if event_type == "chief_complaint" and ev.get("source_layer") == "structured_clinical_input":
            if policy.get("standalone_step_b_trigger") is not False:
                errors.append({"type": "structured_chief_complaint_standalone_trigger_not_blocked", "evidence_id": eid})
        if ev.get("event_type") == "chief_complaint" and ev.get("source_layer") == "structured_clinical_input":
            assertion = ev.get("assertion") or {}
            if assertion.get("polarity") != "positive" or assertion.get("temporality") != "current" or assertion.get("experiencer") != "patient":
                errors.append({"type": "structured_chief_complaint_bad_assertion", "evidence_id": eid, "assertion": assertion})
        for cue in ev.get("clinical_cue_spans", []) or []:
            if cue.get("inherits_from_evidence_id") != eid:
                errors.append({"type": "clinical_cue_span_missing_or_bad_parent", "evidence_id": eid, "cue": cue})
            if not cue.get("assertion_source"):
                errors.append({"type": "clinical_cue_span_missing_assertion_source", "evidence_id": eid, "cue": cue})

    # Evidence-locked summary must be a derived non-independent view with valid source IDs.
    summary = timeline.get("safe_vignette_context", {}) or {}
    if summary.get("evidence_locked_summary") is not True:
        errors.append({"type": "summary_not_marked_evidence_locked"})
    if summary.get("independent_evidence") is not False or summary.get("not_an_independent_evidence_source") is not True:
        errors.append({"type": "summary_independence_flags_invalid"})
    if summary.get("downstream_must_cite_attached_evidence_id") is not True:
        errors.append({"type": "summary_missing_downstream_citation_requirement"})
    for section in ["fields", "vitals"]:
        values = summary.get(section, {}) or {}
        for key, entry in values.items():
            sid = entry.get("source_evidence_id") if isinstance(entry, dict) else None
            if not sid:
                errors.append({"type": "summary_field_missing_source_evidence_id", "section": section, "field": key})
            elif sid not in event_ids:
                errors.append({"type": "summary_field_source_evidence_id_not_in_events", "section": section, "field": key, "source_evidence_id": sid})
            if isinstance(entry, dict) and entry.get("independent_evidence") is not False:
                errors.append({"type": "summary_field_marked_independent", "section": section, "field": key})
    for sid in summary.get("derived_from_evidence_ids", []) or []:
        if sid not in event_ids:
            errors.append({"type": "summary_derived_from_missing_event", "source_evidence_id": sid})

    return errors


def extract_raw_history_triage_audit(history):
    values = []
    by_turn = []
    for history_index, item in enumerate(history):
        if isinstance(item, dict) and "triage" in item:
            value = item.get("triage")
            values.append(value)
            by_turn.append({
                "history_index": history_index,
                "turn": item.get("turn"),
                "actor": item.get("actor"),
                "triage": value,
            })
    counts = Counter(str(v) for v in values)
    return {
        "history_triage_entry_count": len(values),
        "history_triage_values": sorted(set(str(v) for v in values)),
        "history_triage_value_counts": dict(sorted(counts.items())),
        "history_triage_by_turn": by_turn,
        "history_triage_stripped_from_prediction_timeline": True,
    }



def require_identity(case, file_path):
    case_id = case.get("case_id")
    run_uuid = case.get("run_uuid")
    if case_id is None or str(case_id).strip() == "":
        raise ValueError(f"missing required case_id in {file_path}")
    if run_uuid is None or str(run_uuid).strip() == "":
        raise ValueError(f"missing required run_uuid in {file_path}")
    return str(case_id), str(run_uuid)


def build_one_timeline(case, file_path, allow_nonlabel_ground_truth_fallback=False, include_patient_context=False):
    if include_patient_context:
        raise ValueError("patient_persona is audit-only and cannot enter the prediction timeline")
    dataset = case.get("dataset")
    case_id, run_uuid = require_identity(case, file_path)
    instance_id = make_instance_id(case_id, run_uuid)
    group_id = case_id
    realization_id = run_uuid
    vignette = case.get("vignette", {}) or {}
    ground_truth = case.get("ground_truth", {}) or {}
    history = case.get("history", []) or []

    events = []
    ev_idx = 0
    structured_input, structured_consistency_audit = merge_structured_clinical_input(
        vignette,
        ground_truth,
        allow_nonlabel_ground_truth_fallback=allow_nonlabel_ground_truth_fallback,
    )
    structured_events, ev_idx = build_structured_clinical_input_events(instance_id, structured_input, ev_idx)
    events.extend(structured_events)
    patient_context_events, ev_idx = build_patient_context_events(
        instance_id,
        case.get("patient_persona", {}),
        ev_idx,
        include_patient_context=include_patient_context,
    )
    events.extend(patient_context_events)
    structured_vital_lookup = {}
    for ev in structured_events:
        if ev.get("event_type") != "structured_vital":
            continue
        if ev.get("raw_name"):
            structured_vital_lookup[ev.get("raw_name")] = ev
        if ev.get("name"):
            structured_vital_lookup[ev.get("name")] = ev
        if ev.get("canonical_vital_name"):
            structured_vital_lookup[ev.get("canonical_vital_name")] = ev
    history_events, ev_idx, history_audit = build_history_events(instance_id, history, ev_idx, structured_vital_lookup)
    events.extend(history_events)
    attach_dialogue_scaffold(events)

    prompt_safe_lines = [make_prompt_safe_line(ev) for ev in events]
    quality_flags = collect_quality_flags(events, history_audit)
    timeline = {
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "instance_id": instance_id,
        "case_id": case_id,
        "group_id": group_id,
        "run_uuid": run_uuid,
        "realization_id": realization_id,
        "dataset": dataset,
        "safe_vignette_context": build_safe_vignette_context(structured_events),
        "events": events,
        "prompt_safe_lines": prompt_safe_lines,
        "quality_flags": quality_flags,
    }
    validation_errors = validate_prediction_timeline(timeline, include_patient_context=include_patient_context)
    audit_metadata = {
        "instance_id": instance_id,
        "case_id": case_id,
        "group_id": group_id,
        "run_uuid": run_uuid,
        "realization_id": realization_id,
        "dataset": dataset,
        "raw_file_path": file_path,
        "raw_file_name": os.path.basename(file_path),
        "file_name_expected_to_match_run_uuid": os.path.splitext(os.path.basename(file_path))[0] == run_uuid,
        "seed": case.get("seed"),
        "model": case.get("model"),
        "pairing": case.get("pairing", {}),
        "ground_truth": {
            "acuity": ground_truth.get("acuity"),
            "pain": ground_truth.get("pain"),
            "chiefcomplaint": ground_truth.get("chiefcomplaint"),
            "vitals": ground_truth.get("vitals", {}),
        },
        "vignette_audit_only": {
            "stay_id": vignette.get("stay_id"),
            "acuity": vignette.get("acuity"),
            "specialisation": vignette.get("specialisation"),
            "gender": vignette.get("gender"),
            "arrival_transport": vignette.get("arrival_transport"),
            "chiefcomplaint": vignette.get("chiefcomplaint"),
            "vitals": get_vignette_vitals(vignette),
            "pain": vignette.get("pain"),
        },
        "structured_clinical_input_audit": {
            "main_prediction_source": "vignette_non_label_structured_fields",
            "non_label_ground_truth_role": "consistency_audit_or_fallback_only",
            "allow_nonlabel_ground_truth_fallback": bool(allow_nonlabel_ground_truth_fallback),
            "prediction_fields": structured_consistency_audit.get("prediction_fields", []),
            "prediction_field_count": structured_consistency_audit.get("prediction_field_count", 0),
            "prediction_field_origins": structured_consistency_audit.get("prediction_field_origins", {}),
            "source_layer": "structured_clinical_input",
            "non_label_fallback_used": structured_consistency_audit.get("non_label_ground_truth_fallback_used", False),
            "fallback_fields": structured_consistency_audit.get("fallback_fields", []),
            "fallback_field_count": structured_consistency_audit.get("fallback_field_count", 0),
        },
        "structured_input_consistency_audit": structured_consistency_audit,
        "persona_audit_only": {
            "patient_persona": case.get("patient_persona", {}),
            "nurse_persona": case.get("nurse_persona", {}),
            "patient_context_prediction_policy": "audit_only_never_prediction",
            "nurse_persona_prediction_policy": "audit_only_never_prediction",
        },
        "raw_history_triage_audit": extract_raw_history_triage_audit(history),
        "history_processing_audit": history_audit,
        "timeline_quality_flags": quality_flags,
        "prediction_timeline_validation_errors": validation_errors,
    }
    return timeline, audit_metadata, validation_errors

def summarize_cross_realization(metadata_rows):
    by_group = defaultdict(list)
    for row in metadata_rows:
        by_group[row["group_id"]].append(row)
    duplicate_groups = {}
    group_size_counts = Counter()
    for group_id, rows in by_group.items():
        group_size_counts[len(rows)] += 1
        if len(rows) <= 1:
            continue
        acuities = Counter(str(row.get("ground_truth", {}).get("acuity")) for row in rows)
        ground_truth_chief_complaints = Counter(
            str(row.get("ground_truth", {}).get("chiefcomplaint"))
            for row in rows
        )
        vignette_chief_complaints = Counter(
            str(row.get("vignette_audit_only", {}).get("chiefcomplaint"))
            for row in rows
        )
        duplicate_groups[group_id] = {
            "group_id": group_id,
            "instance_count": len(rows),
            "instance_ids": [row["instance_id"] for row in rows],
            "run_uuids": [row["run_uuid"] for row in rows],
            "ground_truth_acuity_counts": dict(sorted(acuities.items())),
            "all_same_ground_truth_acuity": len(acuities) == 1,
            "ground_truth_chiefcomplaint_counts": dict(sorted(ground_truth_chief_complaints.items())),
            "all_same_ground_truth_chiefcomplaint": len(ground_truth_chief_complaints) == 1,
            "vignette_chiefcomplaint_counts": dict(sorted(vignette_chief_complaints.items())),
            "all_same_vignette_chiefcomplaint": len(vignette_chief_complaints) == 1,
            "pairings": [row.get("pairing", {}) for row in rows],
            "patient_persona_summary": [
                {
                    "instance_id": row["instance_id"],
                    "age_group": row.get("persona_audit_only", {}).get("patient_persona", {}).get("age_group"),
                    "pain_expression": row.get("persona_audit_only", {}).get("patient_persona", {}).get("pain_expression"),
                    "recall_accuracy": row.get("persona_audit_only", {}).get("patient_persona", {}).get("recall_accuracy"),
                    "cognitive_state": row.get("persona_audit_only", {}).get("patient_persona", {}).get("cognitive_state"),
                    "topic_drift": row.get("persona_audit_only", {}).get("patient_persona", {}).get("topic_drift"),
                    "response_length": row.get("persona_audit_only", {}).get("patient_persona", {}).get("response_length"),
                }
                for row in rows
            ],
            "nurse_persona_summary": [
                {
                    "instance_id": row["instance_id"],
                    "experience_level": row.get("persona_audit_only", {}).get("nurse_persona", {}).get("experience_level"),
                    "risk_tolerance": row.get("persona_audit_only", {}).get("nurse_persona", {}).get("risk_tolerance"),
                    "guideline_adherence": row.get("persona_audit_only", {}).get("nurse_persona", {}).get("guideline_adherence"),
                    "communication_style": row.get("persona_audit_only", {}).get("nurse_persona", {}).get("communication_style"),
                    "verbosity": row.get("persona_audit_only", {}).get("nurse_persona", {}).get("verbosity"),
                }
                for row in rows
            ],
        }
    return {
        "group_key": "case_id / group_id",
        "instance_key": "instance_id = case_id__run_uuid",
        "total_groups": len(by_group),
        "group_size_counts": dict(sorted((str(k), v) for k, v in group_size_counts.items())),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }


def events_between_history_positions(events, start_history_index, end_history_index):
    selected = []
    if start_history_index is None or end_history_index is None:
        return selected
    for event in events:
        history_index = event.get("history_index")
        if history_index is None:
            continue
        if start_history_index < history_index <= end_history_index:
            selected.append(event)
    return selected


def build_triage_state_trace_audit_rows(timeline_rows, metadata_rows):
    timeline_by_instance = {row.get("instance_id"): row for row in timeline_rows}
    rows = []
    for meta in metadata_rows:
        instance_id = meta.get("instance_id")
        timeline = timeline_by_instance.get(instance_id, {}) or {}
        events = timeline.get("events", []) or []
        entries = ((meta.get("raw_history_triage_audit") or {}).get("history_triage_by_turn") or [])
        turn_entries = []
        for entry in entries:
            turn_entries.append({
                "history_index": entry.get("history_index"),
                "turn": entry.get("turn"),
                "actor": entry.get("actor"),
                "recorded_triage": entry.get("triage"),
            })
        change_events = []
        for prev_entry, curr_entry in zip(turn_entries, turn_entries[1:]):
            if str(prev_entry.get("recorded_triage")) == str(curr_entry.get("recorded_triage")):
                continue
            between = events_between_history_positions(
                events,
                prev_entry.get("history_index"),
                curr_entry.get("history_index"),
            )
            change_events.append({
                "from_recorded_triage": prev_entry.get("recorded_triage"),
                "to_recorded_triage": curr_entry.get("recorded_triage"),
                "from_history_index": prev_entry.get("history_index"),
                "to_history_index": curr_entry.get("history_index"),
                "from_turn": prev_entry.get("turn"),
                "to_turn": curr_entry.get("turn"),
                "new_evidence_ids_between": [ev.get("evidence_id") for ev in between if ev.get("evidence_id")],
                "new_patient_evidence_ids_between": [
                    ev.get("evidence_id") for ev in between
                    if ev.get("actor") == "patient" and ev.get("evidence_id")
                ],
                "new_vital_evidence_ids_between": [
                    ev.get("evidence_id") for ev in between
                    if ev.get("event_type") == "vital" and ev.get("evidence_id")
                ],
                "new_question_context_evidence_ids_between": [
                    ev.get("evidence_id") for ev in between
                    if ev.get("source_layer") == "nurse_question" and ev.get("evidence_id")
                ],
            })
        rows.append({
            "instance_id": instance_id,
            "case_id": meta.get("case_id"),
            "group_id": meta.get("group_id"),
            "run_uuid": meta.get("run_uuid"),
            "audit_only": True,
            "not_prediction_input": True,
            "strict_prediction_modules_must_not_read": True,
            "turn_triage_entries": turn_entries,
            "triage_change_events": change_events,
            "turn_triage_entry_count": len(turn_entries),
            "triage_change_event_count": len(change_events),
        })
    return rows


def build_audit_summary(
    timeline_rows,
    metadata_rows,
    load_errors,
    validation_error_rows,
    raw_data_dir,
    output_dir,
    expected_instance_count=None,
    expected_case_count=None,
    primary_experiment_unit="case",
    case_aggregation="majority",
    duplicate_weighting="equal_case",
    include_patient_context=False,
):
    source_layer_counts = Counter()
    event_type_counts = Counter()
    cue_type_counts = Counter()
    polarity_counts = Counter()
    event_surface_polarity_counts = Counter()
    assertion_scope_flag_counts = Counter()
    temporality_counts = Counter()
    structured_input_field_counts = Counter()
    patient_context_field_counts = Counter()
    fallback_case_count = 0
    fallback_field_counts = Counter()
    conflict_case_count = 0
    file_name_run_uuid_mismatch_count = 0
    summary_orphan_field_count = 0
    summary_field_count = 0
    cue_inheritance_mismatch_count = 0
    dialogue_scaffold_missing_count = 0
    forbidden_stripped_total = 0
    excluded_system_process_total = 0
    implausible_vital_event_count = 0
    invalid_pain_event_count = 0
    derived_clause_span_count = 0
    nurse_segment_source_layer_counts = Counter()
    observation_segment_count = 0
    question_segment_count = 0
    process_segment_count = 0
    discourse_only_segment_count = 0
    mixed_scope_context_cue_count = 0
    silent_denial_positive_count = 0
    example_buckets = {
        "validation_errors": [],
        "implausible_vitals": [],
        "invalid_pain": [],
        "nurse_observed_statement": [],
        "clinical_cue_spans": [],
    }
    for timeline in timeline_rows:
        event_ids = {ev.get("evidence_id") for ev in timeline.get("events", []) if ev.get("evidence_id")}
        summary = timeline.get("safe_vignette_context", {}) or {}
        for section in ["fields", "vitals"]:
            for _, entry in (summary.get(section, {}) or {}).items():
                summary_field_count += 1
                sid = entry.get("source_evidence_id") if isinstance(entry, dict) else None
                if not sid or sid not in event_ids:
                    summary_orphan_field_count += 1
        qf = timeline.get("quality_flags", {})
        forbidden_stripped_total += qf.get("stripped_history_triage_entries", 0)
        excluded_system_process_total += qf.get("excluded_system_process_events", 0)
        implausible_vital_event_count += len(qf.get("implausible_vital_events", []))
        invalid_pain_event_count += len(qf.get("invalid_pain_events", []))
        derived_clause_span_count += qf.get("derived_clause_span_count", 0)
        nurse_segment_source_layer_counts.update(qf.get("nurse_segment_source_layer_counts", {}))
        observation_segment_count += qf.get("observation_segment_count", 0)
        question_segment_count += qf.get("question_segment_count", 0)
        process_segment_count += qf.get("process_segment_count", 0)
        discourse_only_segment_count += qf.get("discourse_only_segment_count", 0)
        mixed_scope_context_cue_count += qf.get("mixed_scope_context_cue_count", 0)
        silent_denial_positive_count += qf.get("silent_denial_positive_count", 0)
        for item in qf.get("implausible_vital_events", [])[:2]:
            if len(example_buckets["implausible_vitals"]) < 50:
                example_buckets["implausible_vitals"].append({"instance_id": timeline.get("instance_id"), **item})
        for item in qf.get("invalid_pain_events", [])[:2]:
            if len(example_buckets["invalid_pain"]) < 50:
                example_buckets["invalid_pain"].append({"instance_id": timeline.get("instance_id"), **item})
        for ev in timeline.get("events", []):
            source_layer_counts[ev.get("source_layer")] += 1
            event_type_counts[ev.get("event_type")] += 1
            polarity_counts[ev.get("polarity_hint")] += 1
            event_surface_polarity_counts[ev.get("event_surface_polarity_hint")] += 1
            temporality_counts[ev.get("temporality_hint")] += 1
            for flag in ev.get("assertion_scope_flags", []) or []:
                assertion_scope_flag_counts[flag] += 1
            if "event_seq_idx" not in ev or "dialogue_link" not in ev:
                dialogue_scaffold_missing_count += 1
            if ev.get("source_layer") == "structured_clinical_input":
                structured_input_field_counts[ev.get("name")] += 1
            if ev.get("source_layer") == "structured_patient_context":
                patient_context_field_counts[ev.get("name")] += 1
            if ev.get("source_layer") == "nurse_observed_statement" and len(example_buckets["nurse_observed_statement"]) < 50:
                example_buckets["nurse_observed_statement"].append({
                    "instance_id": timeline.get("instance_id"),
                    "evidence_id": ev.get("evidence_id"),
                    "turn": ev.get("turn"),
                    "original": ev.get("original"),
                })
            for cue in ev.get("clinical_cue_spans", []):
                cue_type_counts[cue.get("cue_type")] += 1
                if cue.get("inherits_from_evidence_id") != ev.get("evidence_id"):
                    cue_inheritance_mismatch_count += 1
                if len(example_buckets["clinical_cue_spans"]) < 100:
                    example_buckets["clinical_cue_spans"].append({
                        "instance_id": timeline.get("instance_id"),
                        "evidence_id": ev.get("evidence_id"),
                        "turn": ev.get("turn"),
                        "source_layer": ev.get("source_layer"),
                        "cue": cue,
                        "original": ev.get("original"),
                    })
    for row in validation_error_rows:
        if len(example_buckets["validation_errors"]) < 50:
            example_buckets["validation_errors"].append(row)
    for row in metadata_rows:
        if not row.get("file_name_expected_to_match_run_uuid", True):
            file_name_run_uuid_mismatch_count += 1
        scia = row.get("structured_clinical_input_audit", {})
        if scia.get("non_label_fallback_used"):
            fallback_case_count += 1
            for field in scia.get("fallback_fields", []):
                fallback_field_counts[field] += 1
        if row.get("structured_input_consistency_audit", {}).get("conflict_count", 0) > 0:
            conflict_case_count += 1
    label_counts = Counter(str(row.get("ground_truth", {}).get("acuity")) for row in metadata_rows)
    case_id_counts = Counter(row.get("case_id") for row in metadata_rows)
    instance_id_counts = Counter(row.get("instance_id") for row in metadata_rows)
    duplicate_case_ids = {k: v for k, v in case_id_counts.items() if v > 1}
    duplicate_instance_ids = {k: v for k, v in instance_id_counts.items() if v > 1}
    instance_count_matches = None if expected_instance_count is None else len(timeline_rows) == expected_instance_count
    case_count_matches = None if expected_case_count is None else len(case_id_counts) == expected_case_count
    audit_summary = {
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "raw_data_dir": raw_data_dir,
        "output_dir": output_dir,
        "prediction_timeline_file": os.path.join(output_dir, "evidence_timeline.jsonl"),
        "audit_metadata_file": os.path.join(output_dir, "timeline_audit_metadata.jsonl"),
        "total_timelines": len(timeline_rows),
        "load_error_count": len(load_errors),
        "load_errors": load_errors[:50],
        "validation_error_row_count": len(validation_error_rows),
        "validation_error_count": sum(len(row.get("errors", [])) for row in validation_error_rows),
        "id_audit": {
            "case_id_unique": len(case_id_counts),
            "case_id_duplicate_groups": len(duplicate_case_ids),
            "case_id_duplicate_total_extra_rows": sum(v - 1 for v in duplicate_case_ids.values()),
            "instance_id_unique": len(instance_id_counts),
            "instance_id_duplicate_groups": len(duplicate_instance_ids),
            "instance_id_duplicate_total_extra_rows": sum(v - 1 for v in duplicate_instance_ids.values()),
            "run_uuid_unique": len(set(row.get("run_uuid") for row in metadata_rows)),
            "instance_id_definition": "case_id__run_uuid",
            "file_name_expected_to_match_run_uuid_mismatch_count": file_name_run_uuid_mismatch_count,
        },
        "dataset_unit_audit": {
            "raw_file_count": len(timeline_rows) + len(load_errors),
            "timeline_count": len(timeline_rows),
            "unique_case_id_count": len(case_id_counts),
            "unique_instance_id_count": len(instance_id_counts),
            "duplicate_case_id_count": len(duplicate_case_ids),
            "duplicate_instance_id_count": len(duplicate_instance_ids),
            "expected_instance_count": expected_instance_count,
            "expected_case_count": expected_case_count,
            "instance_count_matches": instance_count_matches,
            "case_count_matches": case_count_matches,
            "primary_experiment_unit": primary_experiment_unit,
            "case_aggregation": case_aggregation,
            "duplicate_weighting": duplicate_weighting,
            "primary_metrics_policy": "case_level_541_expected_when_configured",
            "instance_metrics_policy": "secondary_robustness_688_expected_when_configured",
        },
        "label_distribution_audit_only": dict(sorted(label_counts.items())),
        "source_layer_counts": dict(sorted((str(k), v) for k, v in source_layer_counts.items())),
        "event_type_counts": dict(sorted((str(k), v) for k, v in event_type_counts.items())),
        "structured_input_field_counts": dict(sorted((str(k), v) for k, v in structured_input_field_counts.items())),
        "patient_context_field_counts": dict(sorted((str(k), v) for k, v in patient_context_field_counts.items())),
        "structured_fallback_audit": {
            "non_label_ground_truth_fallback_case_count": fallback_case_count,
            "non_label_ground_truth_fallback_field_counts": dict(sorted((str(k), v) for k, v in fallback_field_counts.items())),
            "structured_conflict_case_count": conflict_case_count,
        },
        "provenance_audit": {
            "summary_field_count": summary_field_count,
            "summary_orphan_field_count": summary_orphan_field_count,
            "summary_orphan_field_rate": (summary_orphan_field_count / summary_field_count) if summary_field_count else 0.0,
            "cue_inheritance_mismatch_count": cue_inheritance_mismatch_count,
        },
        "patient_context_policy": {
            "include_patient_context": include_patient_context,
            "default_in_prediction_timeline": False,
            "if_enabled_independent_evidence": False,
            "if_enabled_context_only": True,
        },
        "polarity_hint_counts": dict(sorted((str(k), v) for k, v in polarity_counts.items())),
        "event_surface_polarity_hint_counts": dict(sorted((str(k), v) for k, v in event_surface_polarity_counts.items())),
        "assertion_scope_flag_counts": dict(sorted((str(k), v) for k, v in assertion_scope_flag_counts.items())),
        "temporality_hint_counts": dict(sorted((str(k), v) for k, v in temporality_counts.items())),
        "clinical_cue_type_counts": dict(sorted((str(k), v) for k, v in cue_type_counts.items())),
        "quality_audit": {
            "history_triage_entries_stripped_total": forbidden_stripped_total,
            "excluded_system_process_event_total": excluded_system_process_total,
            "implausible_vital_event_count": implausible_vital_event_count,
            "invalid_pain_event_count": invalid_pain_event_count,
            "derived_clause_span_count": derived_clause_span_count,
            "nurse_segment_source_layer_counts": dict(sorted(nurse_segment_source_layer_counts.items())),
            "observation_segment_count": observation_segment_count,
            "question_segment_count": question_segment_count,
            "process_segment_count": process_segment_count,
            "discourse_only_segment_count": discourse_only_segment_count,
            "mixed_scope_context_cue_count": mixed_scope_context_cue_count,
            "silent_denial_positive_count": silent_denial_positive_count,
            "dialogue_scaffold_missing_event_count": dialogue_scaffold_missing_count,
            "acuity_leakage_check": "enforced_by_prediction_timeline_validation",
        },
        "triage_state_trace_audit_only_policy": {
            "file": os.path.join(output_dir, "triage_state_trace_audit_only.jsonl"),
            "audit_only": True,
            "not_prediction_input": True,
            "strict_prediction_modules_must_not_read": True,
            "not_written_inside_prediction_timeline": True,
        },
        "forbidden_field_policy": {
            "prediction_forbidden_keys": sorted(PREDICTION_FORBIDDEN_KEYS),
            "prediction_timeline_top_level_fields_allowed": sorted(ALLOWED_TIMELINE_TOP_LEVEL_FIELDS),
            "history_triage_copied_to_prediction_timeline": False,
            "ground_truth_copied_to_prediction_timeline": False,
            "specialisation_copied_to_prediction_timeline": False,
            "persona_copied_to_prediction_timeline": False,
            "pairing_or_model_copied_to_prediction_timeline": False,
            "audit_metadata_contains_labels_and_persona_for_audit_only": True,
            "downstream_prediction_modules_must_not_read_audit_metadata": True,
            "prediction_modules_allowed_input_file": "evidence_timeline.jsonl",
            "validation_mode": "fail_fast_before_prediction_file_is_committed",
        },
        "example_files": {
            "validation_errors": os.path.join(output_dir, "examples", "validation_errors.jsonl"),
            "implausible_vitals": os.path.join(output_dir, "examples", "implausible_vitals.jsonl"),
            "invalid_pain": os.path.join(output_dir, "examples", "invalid_pain.jsonl"),
            "nurse_observed_statement": os.path.join(output_dir, "examples", "nurse_observed_statement_examples.jsonl"),
            "clinical_cue_spans": os.path.join(output_dir, "examples", "clinical_cue_span_examples.jsonl"),
            "triage_state_trace_audit_only": os.path.join(output_dir, "triage_state_trace_audit_only.jsonl"),
        },
    }
    return audit_summary, example_buckets


def output_paths(output_dir):
    examples_dir = os.path.join(output_dir, "examples")
    return {
        "prediction": os.path.join(output_dir, "evidence_timeline.jsonl"),
        "metadata": os.path.join(output_dir, "timeline_audit_metadata.jsonl"),
        "build_audit": os.path.join(output_dir, "timeline_build_audit.json"),
        "cross_realization": os.path.join(output_dir, "cross_realization_metadata.json"),
        "triage_trace_audit_only": os.path.join(output_dir, "triage_state_trace_audit_only.jsonl"),
        "examples_dir": examples_dir,
    }



def parse_args():
    parser = argparse.ArgumentParser(description="Build prediction-safe evidence timelines.")
    parser.add_argument("--raw-data-dir", default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--allow-validation-errors", action="store_true",
                        help="Write outputs even if prediction safety validation fails.")
    parser.add_argument("--allow-nonlabel-ground-truth-fallback", action="store_true",
                        help="Allow non-label ground_truth chiefcomplaint/vitals/pain only when vignette fields are missing. Default is off for stricter auditability.")
    parser.add_argument("--expected-instance-count", type=int, default=None,
                        help="Expected number of instance-level timelines; mismatch is a hard validation error when provided.")
    parser.add_argument("--expected-case-count", type=int, default=None,
                        help="Expected number of unique case_id groups; mismatch is a hard validation error when provided.")
    parser.add_argument("--primary-experiment-unit", choices=["case", "instance"], default="case",
                        help="Declared primary experiment unit. For this project, use case as the primary unit.")
    parser.add_argument("--case-aggregation", choices=["canonical_run", "majority", "safety_first"], default="majority",
                        help="Case-level aggregation policy to be consumed by evaluation scripts.")
    parser.add_argument("--duplicate-weighting", choices=["equal_case", "equal_instance"], default="equal_case",
                        help="Declared weighting policy for downstream evaluation.")
    parser.add_argument("--hard-fail-on-count-mismatch", action="store_true",
                        help="Make configured expected count mismatches hard failures. Count mismatches are hard failures whenever expected values are provided.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = output_paths(args.output_dir)
    ensure_dirs(args.output_dir)
    json_files = list_json_files(args.raw_data_dir)
    print(f"Found {len(json_files)} JSON files")

    timeline_rows = []
    metadata_rows = []
    validation_error_rows = []
    load_errors = []

    for i, file_path in enumerate(json_files, start=1):
        case, err = load_json(file_path)
        if case is None:
            load_errors.append({"file_path": file_path, "error": err})
            continue
        try:
            timeline, metadata, validation_errors = build_one_timeline(
                case,
                file_path,
                allow_nonlabel_ground_truth_fallback=args.allow_nonlabel_ground_truth_fallback,
            )
        except Exception as exc:
            validation_error_rows.append({
                "instance_id": None,
                "case_id": case.get("case_id") if isinstance(case, dict) else None,
                "file_path": file_path,
                "errors": [{"type": "build_one_timeline_exception", "message": str(exc)}],
            })
            continue
        if validation_errors:
            validation_error_rows.append({
                "instance_id": timeline.get("instance_id"),
                "case_id": timeline.get("case_id"),
                "file_path": file_path,
                "errors": validation_errors,
            })
        timeline_rows.append(timeline)
        metadata_rows.append(metadata)
        if i % 50 == 0:
            print(f"Processed {i}/{len(json_files)} files")

    case_id_counts = Counter(row.get("case_id") for row in metadata_rows)
    instance_counts = Counter(row.get("instance_id") for row in metadata_rows)
    duplicate_instance_ids = {k: v for k, v in instance_counts.items() if v > 1}
    if duplicate_instance_ids:
        validation_error_rows.append({
            "instance_id": None,
            "case_id": None,
            "file_path": None,
            "errors": [{
                "type": "duplicate_instance_id_hard_fail",
                "duplicate_instance_ids": duplicate_instance_ids,
            }],
        })
    if args.expected_instance_count is not None and len(timeline_rows) != args.expected_instance_count:
        validation_error_rows.append({
            "instance_id": None,
            "case_id": None,
            "file_path": None,
            "errors": [{
                "type": "expected_instance_count_mismatch",
                "expected_instance_count": args.expected_instance_count,
                "actual_instance_count": len(timeline_rows),
            }],
        })
    if args.expected_case_count is not None and len(case_id_counts) != args.expected_case_count:
        validation_error_rows.append({
            "instance_id": None,
            "case_id": None,
            "file_path": None,
            "errors": [{
                "type": "expected_case_count_mismatch",
                "expected_case_count": args.expected_case_count,
                "actual_unique_case_count": len(case_id_counts),
            }],
        })
    if args.primary_experiment_unit == "case" and len(case_id_counts) < len(timeline_rows) and not args.case_aggregation:
        validation_error_rows.append({
            "instance_id": None,
            "case_id": None,
            "file_path": None,
            "errors": [{"type": "duplicate_cases_without_case_aggregation_policy"}],
        })

    cross_realization = summarize_cross_realization(metadata_rows)
    triage_trace_audit_rows = build_triage_state_trace_audit_rows(timeline_rows, metadata_rows)
    audit_summary, example_buckets = build_audit_summary(
        timeline_rows,
        metadata_rows,
        load_errors,
        validation_error_rows,
        args.raw_data_dir,
        args.output_dir,
        expected_instance_count=args.expected_instance_count,
        expected_case_count=args.expected_case_count,
        primary_experiment_unit=args.primary_experiment_unit,
        case_aggregation=args.case_aggregation,
        duplicate_weighting=args.duplicate_weighting,
        include_patient_context=False,
    )

    if validation_error_rows and not args.allow_validation_errors:
        write_json(paths["build_audit"], audit_summary)
        write_jsonl(os.path.join(paths["examples_dir"], "validation_errors.jsonl"), example_buckets["validation_errors"])
        print("Prediction timeline validation failed; evidence_timeline.jsonl was not written.")
        print(f"Validation error rows: {len(validation_error_rows)}")
        print(f"Build audit saved to: {paths['build_audit']}")
        sys.exit(2)

    write_jsonl(paths["prediction"], timeline_rows)
    write_jsonl(paths["metadata"], metadata_rows)
    write_jsonl(paths["triage_trace_audit_only"], triage_trace_audit_rows)
    write_json(paths["cross_realization"], cross_realization)
    write_json(paths["build_audit"], audit_summary)
    write_jsonl(os.path.join(paths["examples_dir"], "validation_errors.jsonl"), example_buckets["validation_errors"])
    write_jsonl(os.path.join(paths["examples_dir"], "implausible_vitals.jsonl"), example_buckets["implausible_vitals"])
    write_jsonl(os.path.join(paths["examples_dir"], "invalid_pain.jsonl"), example_buckets["invalid_pain"])
    write_jsonl(os.path.join(paths["examples_dir"], "nurse_observed_statement_examples.jsonl"), example_buckets["nurse_observed_statement"])
    write_jsonl(os.path.join(paths["examples_dir"], "clinical_cue_span_examples.jsonl"), example_buckets["clinical_cue_spans"])

    print("Done")
    print(f"Written prediction-safe timelines: {len(timeline_rows)}")
    print(f"Unique case_id count: {len(case_id_counts)}")
    print(f"Load errors: {len(load_errors)}")
    print(f"Validation error rows: {len(validation_error_rows)}")
    print(f"Output saved to: {paths['prediction']}")
    print(f"Audit metadata saved to: {paths['metadata']}")
    print(f"Triage trace audit-only saved to: {paths['triage_trace_audit_only']}")
    print(f"Build audit saved to: {paths['build_audit']}")
    print(f"Cross-realization metadata saved to: {paths['cross_realization']}")


if __name__ == "__main__":
    main()
