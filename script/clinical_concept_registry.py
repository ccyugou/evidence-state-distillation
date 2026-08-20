#!/usr/bin/env python3
"""Closed clinical mention registry for the TRIBOT stage-01 compiler.

The registry normalizes surface language to symptom/finding concepts. It does
not contain ESI labels, resource labels, diagnoses, or policy decisions.
"""

from __future__ import annotations

import re
from typing import Any


REGISTRY_VERSION = "tribot_clinical_atom_registry_v1.0"


def atom(
    label: str,
    parent: str,
    mention_type: str,
    patterns: tuple[str, ...],
    *,
    observable: bool = False,
    priority: int = 50,
) -> dict[str, Any]:
    return {
        "label": label,
        "concept_parent": parent,
        "mention_type": mention_type,
        "patterns": patterns,
        "observable": observable,
        "priority": priority,
    }


# Specific atoms intentionally precede broad atoms. The parent keeps related
# expressions together without discarding clinically meaningful detail.
CLINICAL_ATOMS: dict[str, dict[str, Any]] = {
    "dyspnea_at_rest": atom(
        "Dyspnea at rest",
        "dyspnea",
        "symptom",
        (
            r"\bshort(?:ness)? of breath\b[^.!?;]{0,45}\b(?:at rest|resting|sitting|lying down)\b",
            r"\b(?:breathless|trouble breathing|difficulty breathing)\b[^.!?;]{0,45}\b(?:at rest|sitting|lying down)\b",
        ),
        priority=100,
    ),
    "dyspnea_exertional": atom(
        "Exertional dyspnea",
        "dyspnea",
        "symptom",
        (
            r"\bshort(?:ness)? of breath\b[^.!?;]{0,45}\b(?:walking|walk|stairs|exertion|activity)\b",
            r"\b(?:walking|walk|stairs|exertion|activity)\b[^.!?;]{0,45}\b(?:short of breath|breathless|winded)\b",
            r"\bdyspnea on exertion\b",
        ),
        priority=100,
    ),
    "dyspnea": atom(
        "Dyspnea",
        "dyspnea",
        "symptom",
        (
            r"\bshort(?:ness)? of breath\b",
            r"\bbreathless\b|\bwinded\b",
            r"\b(?:trouble|difficulty|problem) breathing\b",
            r"\b(?:hard|unable) to breathe\b|\bcan(?:not|'t) breathe\b",
            r"\bcan(?:not|'t) catch (?:my )?(?:breath|enough air)\b",
            r"\bnot getting (?:enough )?air\b",
        ),
        priority=90,
    ),
    "gasping": atom(
        "Gasping",
        "respiratory_distress",
        "observable_finding",
        (r"\bgasp(?:ing|s)?\b",),
        observable=True,
        priority=100,
    ),
    "tripod_position": atom(
        "Tripod positioning",
        "respiratory_distress",
        "observable_finding",
        (r"\btripod(?:ding)?\b", r"\bleaning forward\b[^.!?;]{0,35}\bto breathe\b"),
        observable=True,
        priority=100,
    ),
    "short_sentence_speech": atom(
        "Speech limited by breathlessness",
        "respiratory_distress",
        "observable_finding",
        (
            r"\b(?:cannot|can't|unable to) (?:finish|speak in full)\b[^.!?;]{0,20}\bsentences?\b",
            r"\b(?:can only|only able to) speak\b[^.!?;]{0,15}\b(?:two|three|2|3) words?\b",
            r"\b(?:two|three|2|3)[- ]word sentences?\b",
        ),
        observable=True,
        priority=100,
    ),
    "stridor": atom("Stridor", "respiratory_distress", "finding", (r"\bstridor\b",), observable=True, priority=100),
    "wheezing": atom("Wheezing", "wheezing", "symptom_or_finding", (r"\bwheez(?:e|es|ing)\b",), observable=True),
    "cough": atom("Cough", "cough", "symptom", (r"\bcough(?:ing|s|ed)?\b",)),
    "hemoptysis": atom(
        "Hemoptysis",
        "respiratory_bleeding",
        "symptom",
        (r"\bhemoptysis\b", r"\bcough(?:ing|ed)?\b[^.!?;]{0,35}\bblood\b", r"\bblood\b[^.!?;]{0,35}\bcough"),
        priority=100,
    ),
    "chest_pressure": atom(
        "Chest pressure or tightness",
        "chest_pain_or_discomfort",
        "symptom",
        (
            r"\bchest\b[^.!?;]{0,35}\b(?:pressure|tight(?:ness)?|heav(?:y|iness)|squeez(?:e|ing))\b",
            r"\b(?:pressure|tight(?:ness)?|heav(?:y|iness)|squeez(?:e|ing))\b"
            r"(?:(?!\b(?:but|just|except)\b)[^.!?;]){0,35}\bchest\b",
        ),
        priority=100,
    ),
    "chest_pain": atom(
        "Chest pain",
        "chest_pain_or_discomfort",
        "symptom",
        (
            r"\bchest\b[^.!?;]{0,35}\b(?:pain|ache|hurt|discomfort)\b",
            r"\b(?:pain|ache|hurt|discomfort)\b[^.!?;]{0,35}\bchest\b",
        ),
        priority=95,
    ),
    "palpitations": atom(
        "Palpitations",
        "palpitations",
        "symptom",
        (
            r"\bpalpitations?\b",
            r"\bheart\b[^.!?;]{0,35}\b(?:pounding|racing|fluttering|skipping|beating fast)\b",
            r"\b(?:pounding|racing|fluttering) heartbeat\b",
        ),
        priority=90,
    ),
    "syncope": atom(
        "Syncope",
        "syncope",
        "event",
        (r"\b(?:fainted|fainting|passed out|lost consciousness|blacked out)\b",),
        priority=100,
    ),
    "near_syncope": atom(
        "Near-syncope",
        "syncope",
        "symptom",
        (r"\b(?:almost|nearly|about to) faint\w*\b", r"\bnear[- ]syncope\b", r"\bclose to passing out\b", r"\bfeel(?:ing)? faint\b"),
        priority=100,
    ),
    "lightheadedness": atom(
        "Lightheadedness",
        "dizziness",
        "symptom",
        (r"\blight[- ]?headed(?:ness)?\b",),
        priority=85,
    ),
    "dizziness": atom(
        "Dizziness",
        "dizziness",
        "symptom",
        (r"\bdizz(?:y|iness)\b", r"\bunsteady\b"),
        priority=80,
    ),
    "vertigo": atom(
        "Vertigo",
        "dizziness",
        "symptom",
        (r"\bvertigo\b", r"\b(?:room|world|everything)\b[^.!?;]{0,35}\bspinn(?:ing|s)?\b"),
        priority=95,
    ),
    "altered_mental_status": atom(
        "Altered mental status",
        "altered_mental_status",
        "symptom_or_finding",
        (
            r"\bconfus(?:ed|ion)\b|\bdisorient(?:ed|ation)\b",
            r"\bwhere am i\b",
            r"\b(?:cannot|can't) think straight\b",
            r"\b(?:do not|don't) know where i am\b",
            r"\b(?:do not|don't) remember how i got here\b",
        ),
        observable=True,
        priority=100,
    ),
    "lethargy": atom("Lethargy", "altered_mental_status", "symptom_or_finding", (r"\bletharg(?:y|ic)\b", r"\bobtund(?:ed|ation)\b"), observable=True, priority=95),
    "hallucinations": atom("Hallucinations", "psychosis", "symptom", (r"\bhallucinat\w*\b", r"\bhearing voices\b", r"\bseeing things\b[^.!?;]{0,35}\bnot there\b"), priority=95),
    "agitation_or_violent_behavior": atom(
        "Agitation or violent behavior",
        "behavioral_safety_risk",
        "symptom_or_finding",
        (r"\bagitat(?:ed|ion)\b", r"\bcombat(?:ive|iveness)\b", r"\bviolent\b|\baggressive\b"),
        observable=True,
        priority=90,
    ),
    "suicidal_ideation": atom(
        "Suicidal ideation",
        "self_harm_risk",
        "thought_or_intent",
        (
            r"\bsuicid(?:al|e)\b",
            r"\b(?:kill|hurt|harm) myself\b",
            r"\bend my life\b",
            r"\b(?:do not|don't) want to live\b",
            r"\bwish i (?:were|was) dead\b",
        ),
        priority=100,
    ),
    "homicidal_ideation": atom(
        "Homicidal ideation",
        "other_harm_risk",
        "thought_or_intent",
        (
            r"\bhomicid(?:al|e)\b",
            r"\b(?:kill|hurt|harm)\s+(?:him|her|them|someone|people)\b",
            r"\b(?:want|plan|intend|thinking|thought|urge|going)\b[^.!?;]{0,25}\b(?:kill|hurt|harm)\b[^.!?;]{0,18}\b(?:him|her|them|someone|people)\b",
        ),
        priority=100,
    ),
    "overdose_method_ideation": atom(
        "Overdose method ideation or plan",
        "self_harm_risk",
        "thought_or_intent",
        (
            r"\b(?:thinking|thought|considering|planning|plan|wondered)\b[^.!?;]{0,65}\b(?:overdose|pills?|tablets?|medication)\b",
            r"\b(?:take|swallow)\b[^.!?;]{0,40}\b(?:pills?|tablets?)\b[^.!?;]{0,35}\b(?:hurt|harm|kill) myself\b",
        ),
        priority=110,
    ),
    "actual_toxic_ingestion": atom(
        "Actual toxic ingestion or overdose",
        "toxic_ingestion",
        "event",
        (
            r"\boverdosed\b",
            r"\b(?:took|swallowed|ingested|drank)\b[^.!?;]{0,55}\b(?:too many|a handful|overdose|poison|bleach|chemical)\b",
            r"\b(?:took|swallowed|ingested)\b[^.!?;]{0,30}\b(?:eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|[89]|[1-9][0-9])\b[^.!?;]{0,18}\b(?:pills?|tablets?|medications?)\b",
        ),
        priority=105,
    ),
    "possible_toxic_ingestion": atom(
        "Possible toxic ingestion",
        "toxic_ingestion",
        "event_concern",
        (r"\b(?:possible|suspected) (?:ingestion|overdose)\b",),
        priority=100,
    ),
    "seizure": atom("Seizure", "seizure", "event", (r"\bseizure\b|\bconvulsion\b|\bpost[- ]?ictal\b",), priority=95),
    "focal_weakness": atom(
        "Focal weakness",
        "focal_neurologic_deficit",
        "symptom",
        (
            r"\b(?:left|right|one)[- ]?(?:sided?|side)\b[^.!?;]{0,55}\bweak(?:ness)?\b",
            r"\bweak(?:ness)?\b[^.!?;]{0,55}\b(?:left|right|one)[- ]?(?:sided?|side)\b",
            r"\b(?:left|right) (?:arm|leg) weakness\b",
        ),
        priority=100,
    ),
    "focal_numbness": atom(
        "Focal numbness",
        "focal_neurologic_deficit",
        "symptom",
        (
            r"\b(?:left|right|one)[- ]?(?:sided?|side)\b[^.!?;]{0,55}\bnumb(?:ness)?\b",
            r"\bnumb(?:ness)?\b[^.!?;]{0,55}\b(?:left|right|one)[- ]?(?:sided?|side)\b",
            r"\b(?:left|right) (?:arm|leg|face|foot|hand) numb(?:ness)?\b",
        ),
        priority=100,
    ),
    "speech_deficit": atom("Speech deficit", "focal_neurologic_deficit", "symptom_or_finding", (r"\bslurred speech\b", r"\bexpressive aphasia\b", r"\b(?:cannot|can't) get (?:my )?words out\b"), observable=True, priority=100),
    "facial_droop": atom("Facial droop", "focal_neurologic_deficit", "symptom_or_finding", (r"\b(?:face|facial) droop\w*\b",), observable=True, priority=100),
    "headache": atom("Headache", "headache", "symptom", (r"\bheadache\b", r"\bhead (?:pain|hurts|ache)\b")),
    "thunderclap_headache": atom("Thunderclap headache", "headache", "symptom", (r"\bthunderclap headache\b", r"\bworst headache\b[^.!?;]{0,35}\b(?:sudden|life)\b"), priority=105),
    "abdominal_pain": atom("Abdominal pain", "abdominal_pain", "symptom", (r"\b(?:abdominal|abdomen|stomach|belly)\b[^.!?;]{0,35}\b(?:pain|ache|hurt|cramp)\w*\b", r"\b(?:pain|ache|hurt|cramp)\w*\b[^.!?;]{0,35}\b(?:abdomen|abdominal|stomach|belly)\b"), priority=90),
    "flank_pain": atom("Flank pain", "flank_pain", "symptom", (r"\b(?:left|right)?\s*flank pain\b",), priority=95),
    "back_pain": atom("Back pain", "back_pain", "symptom", (r"\b(?:lower |upper |mid )?back\b[^.!?;]{0,25}\b(?:pain|ache|hurt)\w*\b", r"\bback pain\b")),
    "neck_pain": atom("Neck pain", "neck_pain", "symptom", (r"\bneck\b[^.!?;]{0,25}\b(?:pain|ache|hurt|stiff)\w*\b",)),
    "pelvic_pain": atom("Pelvic pain", "pelvic_pain", "symptom", (r"\bpelvic pain\b",)),
    "testicular_or_scrotal_pain": atom("Testicular or scrotal pain", "genitourinary_pain", "symptom", (r"\b(?:testicular|testicle|scrotal|scrotum)\b[^.!?;]{0,25}\b(?:pain|ache|hurt)\w*\b",), priority=95),
    "limb_or_joint_pain": atom("Limb or joint pain", "musculoskeletal_pain", "symptom", (r"\b(?:arm|leg|knee|ankle|foot|feet|toe|finger|hand|wrist|elbow|shoulder|hip|joint|clavicle)\b[^.!?;]{0,35}\b(?:pain|ache|hurt|sore|throbbing)\w*\b", r"\b(?:pain|ache|hurt|sore|throbbing)\w*\b[^.!?;]{0,35}\b(?:arm|leg|knee|ankle|foot|feet|toe|finger|hand|wrist|elbow|shoulder|hip|joint|clavicle)\b")),
    "pain_mention": atom("Pain mention", "pain", "symptom_anchor", (r"\bpain(?:ful)?\b", r"\bhurt(?:s|ing)?\b"), priority=10),
    "sore_throat": atom("Sore throat", "upper_airway_symptom", "symptom", (r"\bsore throat\b", r"\bthroat\b[^.!?;]{0,30}\b(?:pain|hurt|burn|fire)\w*\b")),
    "difficulty_swallowing": atom("Dysphagia", "upper_airway_symptom", "symptom", (r"\b(?:difficulty|trouble|unable to|can't|cannot) swallow\w*\b", r"\bdysphagia\b"), priority=95),
    "ear_pain": atom("Otalgia", "ear_symptom", "symptom", (r"\b(?:left |right )?ear\b[^.!?;]{0,25}\b(?:pain|ache|hurt|throbbing|screaming)\w*\b",)),
    "eye_pain_or_visual_change": atom("Eye pain or visual change", "ocular_symptom", "symptom", (r"\beye\b[^.!?;]{0,30}\b(?:pain|hurt|red|vision|blurr|double)\w*\b", r"\b(?:blurred vision|double vision|visual loss|vision loss|floaters|flashes)\b")),
    "nausea": atom("Nausea", "nausea", "symptom", (r"\bnausea\b|\bnauseous\b|\bfeel sick\b",)),
    "vomiting": atom("Vomiting", "vomiting", "symptom", (r"\bvomit(?:ing|ed|s)?\b", r"\bthrow(?:ing)? up\b|\bthrew up\b", r"\bnothing stays down\b")),
    "hematemesis": atom("Hematemesis", "gastrointestinal_bleeding", "symptom", (r"\bhematemesis\b", r"\b(?:vomit(?:ing|ed)?|threw up)\b[^.!?;]{0,35}\bblood\b", r"\bblood\b[^.!?;]{0,35}\b(?:vomit(?:ing|ed)?|throw(?:ing)? up)\b", r"\bcoffee[- ]ground (?:emesis|vomit)\b"), priority=105),
    "melena": atom("Melena", "gastrointestinal_bleeding", "symptom", (r"\bmelena\b", r"\bblack(?:,)? tarry stools?\b", r"\bblack stools?\b"), priority=105),
    "rectal_bleeding": atom("Rectal bleeding", "gastrointestinal_bleeding", "symptom", (r"\b(?:bright red blood per rectum|brbpr|rectal bleeding)\b", r"\bblood\b[^.!?;]{0,35}\b(?:stool|bowel movement)\b"), priority=100),
    "diarrhea": atom("Diarrhea", "diarrhea", "symptom", (r"\bdiarrh(?:ea|eal)\b", r"\bliquid stools?\b")),
    "constipation": atom("Constipation", "constipation", "symptom", (r"\bconstipat(?:ed|ion)\b", r"\bno bowel movement\b")),
    "active_bleeding": atom("Active bleeding", "active_bleeding", "symptom_or_finding", (r"\bbleed(?:ing|s)?\b", r"\booz(?:ing|es)?\b"), observable=True, priority=70),
    "vaginal_bleeding": atom("Vaginal bleeding", "active_bleeding", "symptom", (r"\bvaginal bleeding\b", r"\bbleeding\b[^.!?;]{0,30}\b(?:pad|vaginal|pregnan|postpartum)\b"), priority=105),
    "hematuria": atom("Hematuria", "urinary_bleeding", "symptom", (r"\bhematuria\b", r"\bblood\b[^.!?;]{0,30}\b(?:urine|pee|urinating)\b"), priority=100),
    "epistaxis": atom("Epistaxis", "nasal_bleeding", "symptom", (r"\bepistaxis\b|\bnosebleed\b|\bnose bleeding\b",), priority=100),
    "fever": atom("Fever", "fever", "symptom_or_measurement_context", (r"\bfever(?:ish|s)?\b", r"\btemperature\b[^.!?;]{0,20}\b(?:high|elevated)\b")),
    "chills_or_rigors": atom("Chills or rigors", "systemic_infection_symptom", "symptom", (r"\bchills?\b|\brigors?\b|\bshiver(?:ing|ed)?\b",)),
    "generalized_weakness": atom("Generalized weakness", "generalized_weakness", "symptom", (r"\bgeneral(?:ized|ly) weak(?:ness)?\b", r"\bfeel(?:ing)? weak\b|\bweak all over\b")),
    "fatigue": atom("Fatigue", "fatigue", "symptom", (r"\bfatigu(?:e|ed)\b|\bdrained\b|\bexhausted\b",)),
    "dysuria": atom(
        "Dysuria",
        "urinary_symptom",
        "symptom",
        (
            r"\bdysuria\b",
            r"\b(?:burn(?:s|ing|ed|y)?|pain|hurt\w*)\b[^.!?;]{0,30}\b(?:urination|urinating|urinate|pee|peeing|pass(?:ing)? urine|go(?:ing)? (?:to the )?toilet)\b",
            r"\b(?:urination|urinating|urinate|pee|peeing|pass(?:ing)? urine|go(?:ing)? (?:to the )?toilet)\b[^.!?;]{0,20}\b(?:burn(?:s|ing|ed|y)?|painful|hurt\w*|fire)\b",
            r"\bevery time I (?:urinate|pee|go (?:to the )?toilet)\b[^.!?;]{0,35}\b(?:burn|hurt|pain|fire)\w*\b",
        ),
        priority=95,
    ),
    "urinary_retention": atom("Urinary retention", "urinary_symptom", "symptom", (r"\burinary retention\b", r"\b(?:cannot|can't|unable to) (?:urinate|pee)\b"), priority=95),
    "poor_oral_intake": atom("Poor oral intake", "dehydration_or_poor_intake", "symptom", (r"\b(?:poor|decreased|reduced) (?:oral )?intake\b", r"\b(?:cannot|can't|unable to|barely) (?:eat|drink)\b")),
    "pregnancy_status": atom("Pregnancy status", "pregnancy_or_postpartum_context", "clinical_context", (r"\bpregnan(?:t|cy)\b", r"\b\d{1,2} weeks? along\b"), priority=90),
    "postpartum_status": atom("Postpartum status", "pregnancy_or_postpartum_context", "clinical_context", (r"\bpostpartum\b", r"\bgave birth\b[^.!?;]{0,45}\b(?:ago|last|recent)\b"), priority=90),
    "chemotherapy_or_immunosuppression": atom("Chemotherapy or immunosuppression", "immunocompromised_context", "clinical_context", (r"\bchemotherapy\b|\bon chemo\b", r"\bimmunocompromised\b|\bimmunosuppress(?:ed|ion|ive)\b"), priority=95),
    "transplant_recipient": atom("Transplant recipient status", "transplant_context", "clinical_context", (r"\b(?:organ |kidney |liver |heart |lung )?transplant(?: recipient| patient|ed)?\b",), priority=95),
    "anticoagulant_use": atom("Anticoagulant use", "anticoagulation_context", "clinical_context", (r"\b(?:warfarin|coumadin|apixaban|eliquis|rivaroxaban|xarelto|dabigatran|pradaxa|blood thinner|anticoagulan\w*)\b",), priority=95),
    "asthma_or_copd": atom("Asthma or chronic obstructive pulmonary disease", "chronic_respiratory_context", "clinical_context", (r"\basthma\b", r"\b(?:copd|chronic obstructive pulmonary disease)\b")),
    "sickle_cell_disease": atom("Sickle cell disease", "sickle_cell_context", "clinical_context", (r"\bsickle cell(?: disease| crisis)?\b",), priority=95),
    "diabetes": atom("Diabetes mellitus", "diabetes_context", "clinical_context", (r"\bdiabet(?:es|ic)\b",)),
    "atrial_fibrillation": atom("Atrial fibrillation", "cardiac_rhythm_context", "clinical_context", (r"\batrial fibrillation\b|\ba[- ]?fib\b",), priority=95),
    "hypertension": atom("Hypertension", "hypertension_context", "clinical_context", (r"\bhypertension\b|\bhigh blood pressure\b",)),
    "anemia": atom("Anemia", "anemia_context", "clinical_context", (r"\ban[ae]mia\b",)),
    "urinary_tract_infection": atom("Urinary tract infection", "urinary_infection_context", "clinical_context", (r"\burinary tract infection\b|\buti\b",)),
    "cellulitis": atom("Cellulitis", "localized_infection", "clinical_context", (r"\bcellulitis\b",)),
    "fracture": atom("Fracture", "fracture_context", "clinical_context", (r"\bfracture(?:d|s)?\b|\bbroken (?:bone|arm|leg|hip|wrist|finger|toe)\b",)),
    "pneumonia": atom("Pneumonia", "lower_respiratory_infection_context", "clinical_context", (r"\bpneumonia\b",)),
    "fall_or_trauma": atom("Fall or trauma", "trauma", "event", (r"\b(?:fell|fall|fallen)\b", r"\b(?:motor vehicle|motorcycle|bike) (?:collision|crash|accident)\b", r"\b(?:stab|gunshot|penetrating) wound\b")),
    "laceration_or_wound": atom("Laceration or wound", "wound_or_injury", "finding", (r"\blaceration\b|\bwound\b|\bcut\b",), observable=True),
    "swelling": atom("Swelling", "swelling", "symptom_or_finding", (r"\bswell(?:ing|s)?\b|\bswollen\b|\bedema\b",), observable=True),
    "erythema": atom("Erythema", "inflammatory_skin_finding", "finding", (r"\berythema\b|\bredness\b|\bred and swollen\b",), observable=True),
    "pallor": atom("Pallor", "perfusion_appearance", "observable_finding", (r"\bpale\b|\bpallor\b",), observable=True, priority=95),
    "diaphoresis": atom("Diaphoresis", "perfusion_appearance", "observable_finding", (r"\bdiaphoretic\b|\bprofusely sweating\b|\bclammy\b",), observable=True, priority=95),
    "cyanosis": atom("Cyanosis", "respiratory_appearance", "observable_finding", (r"\bcyanotic\b|\bblue lips?\b",), observable=True, priority=100),
    "abscess": atom("Abscess", "localized_infection", "finding_or_diagnosis_context", (r"\babscess\b|\bboil\b",), observable=True),
    "functional_limitation": atom(
        "Functional limitation",
        "functional_limitation",
        "functional_status",
        (
            r"\b(?:cannot|can't|unable to|not able to|barely)\b[^.!?;]{0,40}\b(?:walk|stand|move|bear weight|function)\b",
            r"\b(?:walk|standing|moving)\b[^.!?;]{0,25}\b(?:very hard|impossible)\b",
        ),
        priority=90,
    ),
}


FALSE_FRIEND_PATTERNS: dict[str, tuple[str, ...]] = {
    "active_bleeding": (
        r"\bblood (?:pressure|test|draw|work|result)\b",
        r"\bbleeding disorders?\b",
        r"\b(?:not|no) bleeding disorder\b[^.!?;]{0,35}\bfamily\b",
    ),
    "fever": (r"\bfever pitch\b",),
    "suicidal_ideation": (r"\bsuicide screening\b",),
    "overdose_method_ideation": (
        r"\b(?:can|able to|tolerate)\b[^.!?;]{0,25}\b(?:take|taking) tablets?\b",
        r"\b(?:took|taking)\b[^.!?;]{0,35}\b(?:paracetamol|acetaminophen|ibuprofen|aspirin|antibiotic)\b",
        r"\bmaybe (?:one|two|three|\d+) tablets?\b",
    ),
}


ABBREVIATIONS: dict[str, tuple[str, str]] = {
    "sob": ("dyspnea", "Shortness of breath"),
    "doe": ("dyspnea_exertional", "Dyspnea on exertion"),
    "cp": ("chest_pain", "Chest pain"),
    "ams": ("altered_mental_status", "Altered mental status"),
    "si": ("suicidal_ideation", "Suicidal ideation"),
    "hi": ("homicidal_ideation", "Homicidal ideation"),
    "loc": ("syncope", "Loss of consciousness"),
    "brbpr": ("rectal_bleeding", "Bright red blood per rectum"),
}


COMPILED_ATOMS: dict[str, tuple[re.Pattern[str], ...]] = {
    atom_id: tuple(re.compile(pattern, re.IGNORECASE) for pattern in spec["patterns"])
    for atom_id, spec in CLINICAL_ATOMS.items()
}

COMPILED_FALSE_FRIENDS: dict[str, tuple[re.Pattern[str], ...]] = {
    atom_id: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for atom_id, patterns in FALSE_FRIEND_PATTERNS.items()
}


def normalized_match_text(text: str) -> str:
    """Normalize punctuation without changing character count."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
    )


def standard_term(atom_id: str) -> dict[str, Any]:
    spec = CLINICAL_ATOMS[atom_id]
    return {
        "registry_version": REGISTRY_VERSION,
        "clinical_atom_id": atom_id,
        "clinical_atom_label": spec["label"],
        "concept_parent_id": spec["concept_parent"],
        "mention_type": spec["mention_type"],
        "external_ontology_links": [],
    }
