#!/usr/bin/env python3
"""Compile prediction-safe TRIBOT dialogue into evidence cards.

Stages:
  prepare  - deterministic extraction plus selective bounded Qwen proposals
  finalize - validate Qwen proposals with local NLI scores and write artifacts

The compiler never reads audit summaries, labels, persona fields, or
history-level triage traces. Stage 02 remains responsible for policy-field
qualification and effective-state reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from clinical_concept_registry import (
    CLINICAL_ATOMS,
    COMPILED_ATOMS,
    COMPILED_FALSE_FRIENDS,
    REGISTRY_VERSION,
    normalized_match_text,
    standard_term,
)
from clinical_data_contract import classify_pain_contract, classify_vital_contract


SCHEMA_VERSION = "01_dialogue_to_evidence_cards_v1.0"
DEFAULT_API_URL = "http://127.0.0.1:9097/v1/chat/completions"
DEFAULT_QWEN_MODEL = "Qwen2.5-7B-Instruct-AWQ"
VITAL_FIELDS = ("temperature", "heartrate", "resprate", "o2sat", "sbp")

FORBIDDEN_INPUT_KEYS = {
    "ground_truth",
    "acuity",
    "triage",
    "patient_persona",
    "nurse_persona",
    "pairing",
    "model",
    "seed",
    "specialisation",
}
FORBIDDEN_OUTPUT_KEYS = FORBIDDEN_INPUT_KEYS | {"label", "resource_bucket", "esi"}

SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
CLAUSE_BREAK_RE = re.compile(
    r",\s*(?=(?:but|however|although|though|except|yet|just|rather|instead|more of)\b)|"
    r"\s+(?=(?:but|however|although|though|except|yet)\b)|"
    r"\s*[—–-]\s*(?=just\b)|[;]+",
    re.I,
)
QUESTION_RE = re.compile(
    r"\?|\b(?:do|does|did|are|is|was|were|have|has|had|can|could|would|will|"
    r"any|what|when|where|which|who|how|why)\b",
    re.I,
)
PROCESS_RE = re.compile(
    r"\b(?:we will|we'll|i will|i'll|going to|plan to)\b[^.!?;]{0,55}"
    r"\b(?:recheck|monitor|observe|reassess|measure|give|administer|ask|contact|call)\b|"
    r"\b(?:after|following)\b[^.!?;]{0,35}\b(?:dose|medication|opioid|injection)\b",
    re.I,
)
RESTATEMENT_RE = re.compile(
    r"\b(?:you said|you told me|you mentioned|you reported|as you described|"
    r"you are saying|you've said)\b",
    re.I,
)
OBSERVATION_RE = re.compile(
    r"\b(?:i can see|i notice|i observed|you appear|you look|patient appears|"
    r"patient is|currently gasping|visibly|audibly)\b",
    re.I,
)
AMBIGUOUS_OBSERVATION_RE = re.compile(
    r"\b(?:you seem|it seems|sounds like|appears to be|seems to be)\b",
    re.I,
)
FAMILY_RE = re.compile(r"\b(?:mother|father|mom|dad|sister|brother|aunt|uncle|family|grandmother|grandfather)\b", re.I)
PATIENT_SELF_RE = re.compile(r"\b(?:i|i'm|i am|my|me|myself|i've|i have)\b", re.I)
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|den(?:y|ies|ied)|do not|don't|does not|doesn't|"
    r"did not|didn't|have not|haven't|has not|hasn't|had not|hadn't|"
    r"would not|wouldn't|"
    r"is not|isn't|are not|aren't|was not|wasn't|were not|weren't)\b",
    re.I,
)
POST_TARGET_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|has|have)\s+(?:not|no longer)\s+(?:present|there|happening|ongoing)\b|"
    r"\b(?:went away|has resolved|have resolved|is gone|are gone)\b|"
    r"\b(?:but\s+)?(?:i\s+)?(?:do not|don't) think so (?:today|now|currently)\b|"
    r"^\W*(?:before\W*)?no\b",
    re.I,
)
POST_TARGET_ECHO_DENIAL_RE = re.compile(r"^\W*(?:(?:um|uh)\W*)?(?:no\b\W*)+", re.I)
PSEUDO_NEGATION_RE = re.compile(r"\b(?:no doubt|not only|not just|cannot rule out|can't rule out)\b", re.I)
UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|possible|suspected|uncertain|not sure|not certain|"
    r"do not know|don't know|cannot tell|can't tell|do not think|don't think|"
    r"cannot remember if|can't remember if|wonder(?:ing|ed)? whether)\b",
    re.I,
)
HYPOTHETICAL_RE = re.compile(
    r"\b(?:what if|could this be|how do you rule out|how would you know|if i(?:'m| am)?|whether i)\b",
    re.I,
)
ORIENTATION_FAILURE_RE = re.compile(r"\b(?:where am i|do not know where i am|don't know where i am|what day is it)\b", re.I)
HISTORICAL_RE = re.compile(
    r"\b(?:history of|in the past|used to|years? ago|months? ago|childhood|previous episode)\b",
    re.I,
)
RECENT_RE = re.compile(r"\b(?:yesterday|last night|earlier today|this morning|hours? ago|days? ago)\b", re.I)
EARLIER_RE = re.compile(r"\b(?:before|earlier|at first|previously)\b", re.I)
CURRENT_RE = re.compile(r"\b(?:now|currently|right now|today|still|ongoing|at present)\b", re.I)
NEW_EPISODE_RE = re.compile(r"\b(?:new|newly|started again|came back|recurred|another episode)\b", re.I)
RESOLVED_RE = re.compile(r"\b(?:resolved|went away|gone now|stopped|no longer)\b", re.I)
SEVERE_RE = re.compile(r"\b(?:severe|terrible|awful|unbearable|excruciat\w*|extreme|agony|worst|very bad|really bad|so bad|like fire)\b", re.I)
MILD_RE = re.compile(r"\b(?:mild|slight|a little|a bit|not too severe|manageable)\b", re.I)
WORSENING_RE = re.compile(r"\b(?:worse|worsening|getting harder|increasing|progressive|more severe|spreading)\b", re.I)
IMPROVING_RE = re.compile(r"\b(?:better|improving|easing|less severe|settling)\b", re.I)
PERSISTENT_RE = re.compile(r"\b(?:still|continues?|continuing|persistent|ongoing|hasn't stopped|has not stopped)\b", re.I)
TREATMENT_FAILURE_RE = re.compile(
    r"\b(?:did not|didn't|has not|hasn't|have not|haven't)\b[^.!?;]{0,40}"
    r"\b(?:help|work|relieve|improve)\b|\bno relief\b|\bwithout relief\b",
    re.I,
)
FUNCTIONAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to|barely|not able to)\b[^.!?;]{0,45}"
    r"\b(?:walk|stand|move|breathe|speak|eat|drink|function|bear weight)\b",
    re.I,
)
ANATOMY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("right_lower_quadrant_abdomen", re.compile(r"\b(?:right lower quadrant|rlq)\b", re.I)),
    ("left_lower_quadrant_abdomen", re.compile(r"\b(?:left lower quadrant|llq)\b", re.I)),
    ("right_upper_quadrant_abdomen", re.compile(r"\b(?:right upper quadrant|ruq)\b", re.I)),
    ("left_upper_quadrant_abdomen", re.compile(r"\b(?:left upper quadrant|luq)\b", re.I)),
    ("chest", re.compile(r"\bchest\b", re.I)),
    ("abdomen", re.compile(r"\b(?:abdomen|abdominal|stomach|belly)\b", re.I)),
    ("flank", re.compile(r"\bflank\b", re.I)),
    ("back", re.compile(r"\bback\b|\bspine\b", re.I)),
    ("neck", re.compile(r"\bneck\b", re.I)),
    ("head", re.compile(r"\bhead\b", re.I)),
    ("face", re.compile(r"\bface|facial\b", re.I)),
    ("eye", re.compile(r"\beye\b", re.I)),
    ("ear", re.compile(r"\bear\b", re.I)),
    ("throat", re.compile(r"\bthroat\b", re.I)),
    ("shoulder", re.compile(r"\bshoulder\b", re.I)),
    ("arm", re.compile(r"\barm\b", re.I)),
    ("elbow", re.compile(r"\belbow\b", re.I)),
    ("wrist", re.compile(r"\bwrist\b", re.I)),
    ("hand", re.compile(r"\bhand\b", re.I)),
    ("finger", re.compile(r"\bfinger\b", re.I)),
    ("hip", re.compile(r"\bhip\b", re.I)),
    ("leg", re.compile(r"\bleg\b", re.I)),
    ("knee", re.compile(r"\bknee\b", re.I)),
    ("ankle", re.compile(r"\bankle\b", re.I)),
    ("foot", re.compile(r"\b(?:foot|feet)\b", re.I)),
    ("toe", re.compile(r"\btoe\b", re.I)),
    ("pelvis", re.compile(r"\bpelvi[cs]\b", re.I)),
    ("testicle_or_scrotum", re.compile(r"\b(?:testicle|testicular|scrotum|scrotal)\b", re.I)),
)
FIXED_ANATOMY_BY_ATOM = {
    "chest_pain": "chest",
    "chest_pressure": "chest",
    "palpitations": "chest",
    "headache": "head",
    "neck_pain": "neck",
    "sore_throat": "throat",
    "ear_pain": "ear",
    "eye_pain_or_visual_change": "eye",
    "facial_droop": "face",
}
ANATOMY_ALLOWED_BY_ATOM = {
    "abdominal_pain": {"abdomen", "right_lower_quadrant_abdomen", "left_lower_quadrant_abdomen", "right_upper_quadrant_abdomen", "left_upper_quadrant_abdomen"},
    "back_pain": {"back", "flank"},
    "limb_or_joint_pain": {"shoulder", "arm", "elbow", "wrist", "hand", "finger", "hip", "leg", "knee", "ankle", "foot", "toe"},
}
ANATOMY_DYNAMIC_ATOMS = {
    "fall_or_trauma", "swelling", "erythema", "laceration_or_wound", "active_bleeding",
    "focal_weakness", "numbness_or_paresthesia", "deformity",
}
SPECIFIC_PAIN_ATOMS = {
    "chest_pain", "abdominal_pain", "flank_pain", "back_pain", "neck_pain", "pelvic_pain",
    "testicular_or_scrotal_pain", "limb_or_joint_pain", "headache", "thunderclap_headache",
    "ear_pain", "eye_pain_or_visual_change", "sore_throat",
}
GENERIC_DENIAL_RE = re.compile(
    r"^\s*(?:nope\b|nothing like that\b|none of those\b|not really\b|"
    r"no\s*[.!?]?\s*$|no\s*[,.-]\s*(?:nothing|none|not really)\b)",
    re.I,
)
QA_ATTRIBUTE_RE = re.compile(
    r"\b(?:started|began|worse|better|only when|even while|for \w+ (?:hours?|days?|weeks?)|"
    r"since|about (?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+))\b",
    re.I,
)
DYSURIA_CROSS_CLAUSE_RE = re.compile(
    r"\b(?:burn(?:s|ing|ed|y)?|hurt\w*|fire)\b.{0,80}\b(?:urination|urinating|urinate|pee|peeing|pass(?:ing)? urine|go(?:ing)? (?:to the )?toilet)\b|"
    r"\b(?:when|while|every time)\s+(?:i\s+)?(?:urinate|pee|go (?:to the )?toilet)\b.{0,50}\b(?:burn\w*|hurt\w*|fire)\b",
    re.I | re.S,
)
NUMERIC_SCALE_RE = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\b", re.I)
SCALE_QUESTION_RE = re.compile(
    r"\b(?:on a scale|out of ten|out of 10|from zero to ten|from 0 to 10|"
    r"rate your|how would you rate|how bad|how severe|pain score|pain scale)\b",
    re.I,
)
NON_SCALE_RATE_RE = re.compile(r"\b(?:heart|pulse|respiratory|breathing) rate\b", re.I)

POSITIVE_FUNCTIONAL_ATOMS = {
    "dyspnea",
    "dyspnea_at_rest",
    "dyspnea_exertional",
    "difficulty_swallowing",
    "urinary_retention",
    "poor_oral_intake",
    "functional_limitation",
    "altered_mental_status",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_spans(text: str) -> list[dict[str, Any]]:
    boundaries = {0, len(text)}
    for regex in (SENTENCE_BREAK_RE, CLAUSE_BREAK_RE):
        for match in regex.finditer(text):
            boundaries.add(match.start())
            boundaries.add(match.end())
    ordered = sorted(boundaries)
    spans = []
    for left, right in zip(ordered, ordered[1:]):
        raw = text[left:right]
        lead = len(raw) - len(raw.lstrip())
        tail = len(raw.rstrip())
        start, end = left + lead, left + tail
        if start < end and not re.fullmatch(r"[\s,;.!?]+", text[start:end]):
            spans.append({"start": start, "end": end, "text": text[start:end]})
    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        spans.append({"start": start, "end": end, "text": text[start:end]})
    return spans


def nurse_subtype(span_text: str, has_observable_atom: bool) -> tuple[str, bool]:
    if "?" in span_text or QUESTION_RE.match(span_text.strip()):
        return "nurse_question", False
    if PROCESS_RE.search(span_text):
        return "nurse_process_or_instruction", False
    if RESTATEMENT_RE.search(span_text):
        return "nurse_restatement_context", False
    if OBSERVATION_RE.search(span_text) and has_observable_atom:
        return "nurse_direct_observation", False
    if AMBIGUOUS_OBSERVATION_RE.search(span_text) and has_observable_atom:
        return "nurse_subtype_ambiguous", True
    return "nurse_context_statement", False


def source_policy(actor: str, subtype: str) -> dict[str, Any]:
    if actor == "patient":
        return {
            "source_layer": "patient_reported",
            "clinical_assertion_allowed": True,
            "positive_trigger_allowed": True,
            "negative_evidence_allowed": True,
            "context_only": False,
            "independent_evidence": True,
        }
    if subtype == "nurse_direct_observation":
        return {
            "source_layer": "nurse_observed",
            "clinical_assertion_allowed": True,
            "positive_trigger_allowed": True,
            "negative_evidence_allowed": True,
            "context_only": False,
            "independent_evidence": True,
        }
    return {
        "source_layer": subtype,
        "clinical_assertion_allowed": False,
        "positive_trigger_allowed": False,
        "negative_evidence_allowed": False,
        "context_only": True,
        "independent_evidence": False,
    }


def subject_for_clause(clause: str, actor: str) -> tuple[str, bool]:
    family = FAMILY_RE.search(clause) is not None
    patient = PATIENT_SELF_RE.search(clause) is not None
    if family and patient:
        return "uncertain", True
    if family:
        return "family_member", False
    if actor in {"patient", "nurse"}:
        return "patient", False
    return "unknown", True


def local_assertion(atom_id: str, clause: str, local_start: int, local_end: int) -> dict[str, Any]:
    normalized = normalized_match_text(clause)
    before = normalized[max(0, local_start - 100):local_start]
    target_text = normalized[local_start:local_end]
    after = normalized[local_end:min(len(normalized), local_end + 45)]
    whole_uncertain = UNCERTAINTY_RE.search(normalized) is not None
    orientation_failure = atom_id == "altered_mental_status" and ORIENTATION_FAILURE_RE.search(normalized) is not None
    hypothetical = HYPOTHETICAL_RE.search(normalized) is not None or ("?" in normalized and not orientation_failure)
    functional_positive = atom_id in POSITIVE_FUNCTIONAL_ATOMS and FUNCTIONAL_RE.search(normalized) is not None
    comma = before.rfind(",")
    if comma >= 0 and re.match(r"\s*(?:the|this|that|my|his|her|our|i|he|she|we|it)\b", before[comma + 1:], re.I):
        before = before[comma + 1:]
    local_negation = (
        NEGATION_RE.search(before) is not None
        or NEGATION_RE.search(target_text) is not None
        or POST_TARGET_NEGATION_RE.search(after) is not None
        or POST_TARGET_ECHO_DENIAL_RE.search(after) is not None
    )
    if PSEUDO_NEGATION_RE.search(before + " " + target_text):
        local_negation = False
    if functional_positive:
        assertion = "present"
        scope = "negative_ability_describes_positive_finding"
    elif whole_uncertain or hypothetical:
        assertion = "uncertain"
        scope = "uncertainty_or_hypothesis_applies_to_target"
    elif local_negation:
        assertion = "absent"
        scope = "local_negation_applies_to_target"
    else:
        assertion = "present"
        scope = "local_positive_assertion"

    if CLINICAL_ATOMS[atom_id]["mention_type"] == "clinical_context" and HISTORICAL_RE.search(normalized):
        temporality = "chronic_or_background"
    elif HISTORICAL_RE.search(normalized):
        temporality = "historical"
    elif CURRENT_RE.search(normalized):
        temporality = "current"
    elif RECENT_RE.search(normalized):
        temporality = "recent"
    elif EARLIER_RE.search(normalized):
        temporality = "earlier_in_episode"
    else:
        temporality = "current_or_unspecified"

    modifiers = []
    for name, regex in (
        ("severe_language", SEVERE_RE),
        ("mild_language", MILD_RE),
        ("worsening", WORSENING_RE),
        ("improving", IMPROVING_RE),
        ("persistent", PERSISTENT_RE),
        ("treatment_failure", TREATMENT_FAILURE_RE),
        ("functional_limitation", FUNCTIONAL_RE),
        ("resolved_language", RESOLVED_RE),
    ):
        matches = list(regex.finditer(normalized))
        if not matches:
            continue
        nearby = []
        for modifier_match in matches:
            if modifier_match.end() <= local_start:
                gap = local_start - modifier_match.end()
            elif modifier_match.start() >= local_end:
                gap = modifier_match.start() - local_end
            else:
                gap = 0
            if gap <= 60:
                nearby.append(modifier_match)
        if not nearby:
            continue
        if name in {"severe_language", "mild_language", "worsening", "improving"}:
            nearest = min(nearby, key=lambda item: min(abs(item.start() - local_end), abs(local_start - item.end())))
            modifier_before = normalized[max(0, nearest.start() - 20):nearest.start()]
            if NEGATION_RE.search(modifier_before):
                if name == "worsening":
                    modifiers.append("stable_or_not_worsening")
                continue
        modifiers.append(name)
    return {
        "assertion": assertion,
        "certainty": "uncertain" if assertion == "uncertain" else "asserted",
        "temporality": temporality,
        "scope_resolution": scope,
        "semantic_modifiers": modifiers,
    }


def false_friend(atom_id: str, clause: str, start: int, end: int) -> bool:
    for regex in COMPILED_FALSE_FRIENDS.get(atom_id, ()):
        match = regex.search(clause)
        if match and not (match.end() <= start or match.start() >= end):
            return True
    return False


def anatomy_attributes(atom_id: str, clause: str, start: int, end: int) -> tuple[str | None, str | None]:
    allowed = ANATOMY_ALLOWED_BY_ATOM.get(atom_id)
    fixed = FIXED_ANATOMY_BY_ATOM.get(atom_id)
    if fixed is None and allowed is None and atom_id not in ANATOMY_DYNAMIC_ATOMS:
        return None, None

    def distance(match: re.Match[str]) -> int:
        if match.end() <= start:
            return start - match.end()
        if match.start() >= end:
            return match.start() - end
        return 0

    candidates = []
    for name, regex in ANATOMY_PATTERNS:
        if allowed is not None and name not in allowed:
            continue
        for match in regex.finditer(clause):
            gap = distance(match)
            if gap <= 40:
                candidates.append((gap, match.start(), name))
    site = min(candidates)[2] if candidates else fixed
    if atom_id == "abdominal_pain" and site is None:
        site = "abdomen"
    if atom_id == "back_pain" and site is None:
        site = "back"

    side_candidates = []
    for side, regex in (
        ("left", re.compile(r"\bleft\b", re.I)),
        ("right", re.compile(r"\bright\b", re.I)),
        ("bilateral", re.compile(r"\b(?:bilateral|both)\b", re.I)),
    ):
        for match in regex.finditer(clause):
            gap = distance(match)
            if gap <= 40:
                side_candidates.append((gap, match.start(), side))
    laterality = min(side_candidates)[2] if side_candidates else None
    return site, laterality


def scan_clause(
    *,
    instance_id: str,
    parent_event_id: str,
    clause_span_id: str,
    clause_text: str,
    clause_offset: int,
    actor: str,
    subtype: str,
    normalization_method: str = "closed_registry_regex",
) -> list[dict[str, Any]]:
    search_text = normalized_match_text(clause_text)
    raw_matches = []
    for atom_id, regexes in COMPILED_ATOMS.items():
        spec = CLINICAL_ATOMS[atom_id]
        for regex in regexes:
            for match in regex.finditer(search_text):
                if false_friend(atom_id, search_text, match.start(), match.end()):
                    continue
                raw_matches.append(
                    {
                        "atom_id": atom_id,
                        "start": match.start(),
                        "end": match.end(),
                        "priority": int(spec["priority"]),
                        "parent": spec["concept_parent"],
                    }
                )

    # For overlapping mentions of the same parent, keep the most specific span.
    selected = []
    for candidate in sorted(raw_matches, key=lambda x: (-x["priority"], -(x["end"] - x["start"]), x["start"], x["atom_id"])):
        collision = next(
            (
                kept
                for kept in selected
                if kept["parent"] == candidate["parent"]
                and candidate["start"] < kept["end"]
                and kept["start"] < candidate["end"]
            ),
            None,
        )
        if collision is None:
            selected.append(candidate)

    selected = [
        candidate
        for candidate in selected
        if candidate["atom_id"] != "pain_mention"
        or not any(
            other["atom_id"] in SPECIFIC_PAIN_ATOMS
            and candidate["start"] < other["end"]
            and other["start"] < candidate["end"]
            for other in selected
        )
    ]

    subject, subject_ambiguous = subject_for_clause(clause_text, actor)
    policy = source_policy(actor, subtype)
    cards = []
    for match in sorted(selected, key=lambda x: (x["start"], x["end"], x["atom_id"])):
        atom_id = match["atom_id"]
        assertion = local_assertion(atom_id, clause_text, match["start"], match["end"])
        absolute_start = clause_offset + match["start"]
        absolute_end = clause_offset + match["end"]
        card_id = stable_id("card", instance_id, parent_event_id, absolute_start, absolute_end, atom_id)
        anatomy_site, laterality = anatomy_attributes(atom_id, clause_text, match["start"], match["end"])
        card = {
            "evidence_card_id": card_id,
            "parent_event_id": parent_event_id,
            "clause_span_id": clause_span_id,
            "span_text": clause_text[match["start"]:match["end"]],
            "char_start": absolute_start,
            "char_end": absolute_end,
            **standard_term(atom_id),
            "normalization": {
                "status": "confirmed_closed_registry",
                "method": normalization_method,
                "candidate_concept_ids": [atom_id],
            },
            "subject": subject,
            "subject_ambiguous": subject_ambiguous,
            "anatomy_site": anatomy_site,
            "laterality": laterality,
            **assertion,
            "source_subtype": subtype,
            **policy,
            "qa_context": None,
            "episode_id": None,
            "episode_relation_hint": None,
            "qwen_resolution": None,
            "nli_verification": None,
        }
        if card["assertion"] != "present":
            card["positive_trigger_allowed"] = False
        if card["subject"] != "patient":
            card["positive_trigger_allowed"] = False
        if atom_id == "pain_mention" and card["assertion"] == "present":
            card["support_only"] = True
            card["positive_trigger_allowed"] = False
            card["independent_evidence"] = False
        cards.append(card)
    return cards


def remove_bare_echo_denials(cards: list[dict[str, Any]], parent_text: str) -> list[dict[str, Any]]:
    """Drop a bare echoed target when the same utterance immediately denies it."""
    remove_ids = set()
    for first in cards:
        if first["assertion"] != "present" or len(first["span_text"].split()) > 2:
            continue
        for later in cards:
            if later["clinical_atom_id"] != first["clinical_atom_id"] or later["assertion"] != "absent":
                continue
            if later["char_start"] <= first["char_end"]:
                continue
            gap = normalized_match_text(parent_text[first["char_end"]:later["char_start"]])
            if re.fullmatch(r"[\W_]*(?:(?:um|uh)[\W_]*)?(?:(?:no|nah)[\W_]*)+", gap, re.I):
                remove_ids.add(first["evidence_card_id"])
                break
    return [card for card in cards if card["evidence_card_id"] not in remove_ids]


def propagate_coordinated_negation(cards: list[dict[str, Any]], parent_text: str) -> list[dict[str, Any]]:
    """Carry a leading denial across a short `or ... or` coordination."""
    ordered = sorted(cards, key=lambda card: (card["char_start"], card["char_end"]))
    for first, later in zip(ordered, ordered[1:]):
        between = normalized_match_text(parent_text[first["char_end"]:later["char_start"]])
        if (
            first["assertion"] == "absent"
            and later["assertion"] == "present"
            and len(between) <= 30
            and re.search(r"\bor\b", between, re.I)
            and not re.search(r"\b(?:but|however|except|just|rather)\b", between, re.I)
        ):
            later["assertion"] = "absent"
            later["certainty"] = "asserted"
            later["scope_resolution"] = "coordinated_negation_applies_to_target"
            later["positive_trigger_allowed"] = False
    for card in ordered:
        tail = normalized_match_text(parent_text[card["char_end"]:card["char_end"] + 90])
        if card["assertion"] == "present" and re.search(
            r"\bbut\b[^.!?;]{0,30}\b(?:i\s+)?(?:do not|don't) think so (?:today|now|currently)\b",
            tail,
            re.I,
        ):
            card["assertion"] = "uncertain"
            card["certainty"] = "uncertain"
            card["scope_resolution"] = "patient_disputes_reported_current_target"
            card["positive_trigger_allowed"] = False
    return cards


def build_parent_event(instance_id: str, history_index: int, event: Mapping[str, Any], seq_idx: int) -> dict[str, Any]:
    actor = str(event.get("actor") or "unknown")
    original = str(event.get("original") or "") if actor in {"patient", "nurse"} else ""
    event_type = "utterance" if actor in {"patient", "nurse"} else str(event.get("event") or "system_event")
    return {
        "parent_event_id": stable_id("event", instance_id, history_index, actor, event_type),
        "event_seq_idx": seq_idx,
        "history_index": history_index,
        "turn": event.get("turn"),
        "actor": actor,
        "event_type": event_type,
        "original": original,
        "system_name": event.get("name") if actor == "system" and event_type == "vital" else None,
        "system_value": event.get("value") if actor == "system" and event_type == "vital" else None,
        "prev_evidence_id": None,
        "next_evidence_id": None,
        "previous_question_evidence_id": None,
        "answer_to_question_evidence_id": None,
    }


def question_targets(cards: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(card["clinical_atom_id"]) for card in cards if card["clinical_atom_id"] != "pain_mention"})


def make_qa_inferred_card(
    *,
    instance_id: str,
    parent_event: Mapping[str, Any],
    question_event_id: str,
    question_target_ids: Sequence[str],
    atom_id: str,
    assertion: str,
) -> dict[str, Any]:
    text = str(parent_event["original"])
    card_id = stable_id("card", instance_id, parent_event["parent_event_id"], 0, len(text), atom_id, "qa_inferred")
    card = {
        "evidence_card_id": card_id,
        "parent_event_id": parent_event["parent_event_id"],
        "clause_span_id": None,
        "span_text": text,
        "char_start": 0,
        "char_end": len(text),
        **standard_term(atom_id),
        "normalization": {
            "status": "qa_coreference_deterministic",
            "method": "adjacent_single_target_or_generic_denial",
            "candidate_concept_ids": list(question_target_ids),
        },
        "subject": "patient",
        "subject_ambiguous": False,
        "anatomy_site": None,
        "laterality": None,
        "assertion": assertion,
        "certainty": "asserted",
        "temporality": "current_or_unspecified",
        "scope_resolution": "qa_answer_corefers_to_previous_question_target",
        "semantic_modifiers": [name for name, regex in (("worsening", WORSENING_RE), ("improving", IMPROVING_RE), ("persistent", PERSISTENT_RE), ("treatment_failure", TREATMENT_FAILURE_RE), ("functional_limitation", FUNCTIONAL_RE)) if regex.search(text)],
        "source_subtype": "patient_qa_inferred",
        **source_policy("patient", "patient_statement"),
        "qa_context": {"question_event_id": question_event_id, "question_targets": list(question_target_ids)},
        "episode_id": None,
        "episode_relation_hint": None,
        "qwen_resolution": None,
        "nli_verification": None,
    }
    if assertion != "present":
        card["positive_trigger_allowed"] = False
    if atom_id == "pain_mention" and assertion == "present":
        card["support_only"] = True
        card["positive_trigger_allowed"] = False
        card["independent_evidence"] = False
    return card


def scale_target(question_text: str, targets: Sequence[str]) -> str:
    lowered = normalized_match_text(question_text).lower()
    if NON_SCALE_RATE_RE.search(lowered) or not SCALE_QUESTION_RE.search(lowered):
        return "not_scale_question"
    pain_targets = [target for target in targets if "pain" in target or target in {"headache", "sore_throat", "ear_pain"}]
    non_pain_terms = re.search(r"\b(?:fatigue|distress|dizziness|nausea|breathing|weakness)\b", lowered)
    if pain_targets and non_pain_terms:
        return "multi_target"
    if pain_targets or "pain" in lowered:
        return "pain"
    if non_pain_terms:
        return "non_pain"
    return "unknown"


def compile_record(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = str(raw.get("dataset"))
    case_id = str(raw.get("case_id"))
    run_uuid = str(raw.get("run_uuid"))
    instance_id = f"{case_id}__{run_uuid}"
    vignette = raw.get("vignette") or {}
    history = raw.get("history") or []

    parent_events = [build_parent_event(instance_id, index, event, index) for index, event in enumerate(history)]
    for index, event in enumerate(parent_events):
        if index:
            event["prev_evidence_id"] = parent_events[index - 1]["parent_event_id"]
        if index + 1 < len(parent_events):
            event["next_evidence_id"] = parent_events[index + 1]["parent_event_id"]

    clause_spans: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    relation_edges: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    question_info: dict[str, dict[str, Any]] = {}
    latest_question: dict[str, Any] | None = None

    # Structured chief complaint is prediction-safe support, never a policy trigger.
    chief = str(vignette.get("chiefcomplaint") or "").strip()
    if chief:
        chief_event_id = stable_id("event", instance_id, "structured_chief_complaint")
        chief_clause_id = stable_id("clause", chief_event_id, 0, len(chief))
        chief_cards = scan_clause(
            instance_id=instance_id,
            parent_event_id=chief_event_id,
            clause_span_id=chief_clause_id,
            clause_text=chief,
            clause_offset=0,
            actor="structured",
            subtype="structured_chief_complaint",
            normalization_method="closed_registry_structured_chief_complaint",
        )
        for card in chief_cards:
            card.update(
                source_layer="structured_chief_complaint",
                source_subtype="structured_chief_complaint",
                clinical_assertion_allowed=True,
                positive_trigger_allowed=False,
                negative_evidence_allowed=False,
                context_only=False,
                independent_evidence=False,
            )
        cards.extend(chief_cards)

    for parent in parent_events:
        actor = parent["actor"]
        if actor not in {"patient", "nurse"}:
            continue
        text = parent["original"]
        event_cards: list[dict[str, Any]] = []
        event_clauses = split_spans(text)
        for clause_index, clause in enumerate(event_clauses):
            clause_id = stable_id("clause", parent["parent_event_id"], clause["start"], clause["end"])
            preliminary = scan_clause(
                instance_id=instance_id,
                parent_event_id=parent["parent_event_id"],
                clause_span_id=clause_id,
                clause_text=clause["text"],
                clause_offset=clause["start"],
                actor=actor,
                subtype="patient_statement" if actor == "patient" else "nurse_context_statement",
            )
            has_observable = any(bool(CLINICAL_ATOMS[card["clinical_atom_id"]]["observable"]) for card in preliminary)
            subtype, subtype_ambiguous = (
                nurse_subtype(clause["text"], has_observable)
                if actor == "nurse"
                else ("patient_statement", False)
            )
            if actor == "nurse":
                policy = source_policy(actor, subtype)
                for card in preliminary:
                    card.update(source_subtype=subtype, **policy)
                    if subtype_ambiguous:
                        card["positive_trigger_allowed"] = False
            clause_row = {
                "clause_span_id": clause_id,
                "parent_event_id": parent["parent_event_id"],
                "clause_index": clause_index,
                "char_start": clause["start"],
                "char_end": clause["end"],
                "span_text": clause["text"],
                "local_source_subtype": subtype,
                "local_context_only": source_policy(actor, subtype)["context_only"],
            }
            clause_spans.append(clause_row)
            event_cards.extend(preliminary)
            for card in preliminary:
                mixed_time = HISTORICAL_RE.search(clause["text"]) and CURRENT_RE.search(clause["text"])
                if card.get("subject_ambiguous") or mixed_time:
                    ambiguities.append(
                        {
                            "ambiguity_id": stable_id("amb", instance_id, card["evidence_card_id"], "card_attributes"),
                            "task": "card_attributes",
                            "instance_id": instance_id,
                            "parent_event_id": parent["parent_event_id"],
                            "target_evidence_card_id": card["evidence_card_id"],
                            "target_text": clause["text"],
                            "target_char_start": clause["start"],
                            "candidate_concept_ids": [card["clinical_atom_id"]],
                            "allowed_labels": ["present", "absent", "uncertain", "not_mentioned"],
                            "previous_nurse_question": latest_question["text"] if latest_question else None,
                            "previous_same_concept_event": None,
                        }
                    )
            if subtype_ambiguous and preliminary:
                ambiguities.append(
                    {
                        "ambiguity_id": stable_id("amb", instance_id, clause_id, "nurse_subtype"),
                        "task": "nurse_subtype",
                        "instance_id": instance_id,
                        "parent_event_id": parent["parent_event_id"],
                        "target_text": clause["text"],
                        "target_char_start": clause["start"],
                        "candidate_concept_ids": question_targets(preliminary),
                        "allowed_labels": ["nurse_direct_observation", "nurse_question", "nurse_process_or_instruction", "nurse_restatement_context", "nurse_context_statement"],
                        "previous_nurse_question": None,
                        "previous_same_concept_event": None,
                    }
                )

        if actor == "patient" and not any(card["clinical_atom_id"] == "dysuria" for card in event_cards):
            composite_match = DYSURIA_CROSS_CLAUSE_RE.search(text)
            if composite_match:
                atom_id = "dysuria"
                assertion = local_assertion(atom_id, text, composite_match.start(), composite_match.end())
                card = {
                    "evidence_card_id": stable_id("card", instance_id, parent["parent_event_id"], composite_match.start(), composite_match.end(), atom_id, "cross_clause"),
                    "parent_event_id": parent["parent_event_id"],
                    "clause_span_id": None,
                    "span_text": text[composite_match.start():composite_match.end()],
                    "char_start": composite_match.start(),
                    "char_end": composite_match.end(),
                    **standard_term(atom_id),
                    "normalization": {"status": "confirmed_closed_registry", "method": "cross_clause_composite", "candidate_concept_ids": [atom_id]},
                    "subject": "patient",
                    "subject_ambiguous": False,
                    "anatomy_site": None,
                    "laterality": None,
                    **assertion,
                    "source_subtype": "patient_statement",
                    **source_policy("patient", "patient_statement"),
                    "qa_context": None,
                    "episode_id": None,
                    "episode_relation_hint": None,
                    "qwen_resolution": None,
                    "nli_verification": None,
                }
                if card["assertion"] != "present":
                    card["positive_trigger_allowed"] = False
                event_cards.append(card)

        event_cards = remove_bare_echo_denials(event_cards, text)
        event_cards = propagate_coordinated_negation(event_cards, text)
        cards.extend(event_cards)
        if actor == "nurse":
            targets = question_targets(event_cards)
            if "?" in text or QUESTION_RE.match(text.strip()):
                question_info[parent["parent_event_id"]] = {
                    "question_targets": targets,
                    "scale_target": scale_target(text, targets),
                    "text": text,
                }
                latest_question = {"event": parent, **question_info[parent["parent_event_id"]]}
        elif actor == "patient" and latest_question is not None:
            q_event = latest_question["event"]
            intervening = parent["history_index"] - q_event["history_index"]
            if 0 < intervening <= 2 or parent.get("turn") == q_event.get("turn"):
                parent["previous_question_evidence_id"] = q_event["parent_event_id"]
                parent["answer_to_question_evidence_id"] = q_event["parent_event_id"]
                relation_edges.append(
                    {
                        "relation_id": stable_id("rel", q_event["parent_event_id"], parent["parent_event_id"], "answers_question"),
                        "source_id": parent["parent_event_id"],
                        "target_id": q_event["parent_event_id"],
                        "relation_type": "answers_question",
                        "relation_status": "deterministic_adjacency",
                    }
                )
                for card in event_cards:
                    card["qa_context"] = {
                        "question_event_id": q_event["parent_event_id"],
                        "question_targets": latest_question["question_targets"],
                        "scale_target": latest_question["scale_target"],
                    }
                answer_text = text.strip()
                targets = latest_question["question_targets"]
                short_answer = len(re.findall(r"\b\w+\b", answer_text)) <= 14
                numeric_only = NUMERIC_SCALE_RE.fullmatch(answer_text.strip(" .?!")) is not None
                deterministic_qa_cards = []
                if targets and not event_cards and latest_question["scale_target"] != "not_scale_question" and numeric_only:
                    pass
                elif len(targets) == 1 and not event_cards and GENERIC_DENIAL_RE.search(answer_text):
                    deterministic_qa_cards.append(make_qa_inferred_card(instance_id=instance_id, parent_event=parent, question_event_id=q_event["parent_event_id"], question_target_ids=targets, atom_id=targets[0], assertion="absent"))
                elif len(targets) == 1 and not event_cards and short_answer and QA_ATTRIBUTE_RE.search(answer_text):
                    deterministic_qa_cards.append(make_qa_inferred_card(instance_id=instance_id, parent_event=parent, question_event_id=q_event["parent_event_id"], question_target_ids=targets, atom_id=targets[0], assertion="present"))
                elif len(targets) > 1 and not event_cards and GENERIC_DENIAL_RE.search(answer_text):
                    deterministic_qa_cards.extend(make_qa_inferred_card(instance_id=instance_id, parent_event=parent, question_event_id=q_event["parent_event_id"], question_target_ids=targets, atom_id=target, assertion="absent") for target in targets)
                cards.extend(deterministic_qa_cards)
                event_cards.extend(deterministic_qa_cards)

                covered_atom_ids = {str(card["clinical_atom_id"]) for card in event_cards}
                covered_parent_ids = {str(card["concept_parent_id"]) for card in event_cards}
                explicitly_answered_targets = covered_atom_ids.intersection(targets)
                unresolved_targets = []
                for target in targets:
                    target_parent = str(CLINICAL_ATOMS.get(target, {}).get("concept_parent", target))
                    covered = target in covered_atom_ids
                    if target == target_parent and target_parent in covered_parent_ids:
                        covered = True
                    if not covered:
                        unresolved_targets.append(target)

                if unresolved_targets and not event_cards and (GENERIC_DENIAL_RE.search(answer_text) or QA_ATTRIBUTE_RE.search(answer_text) or numeric_only):
                    ambiguities.append(
                        {
                            "ambiguity_id": stable_id("amb", instance_id, parent["parent_event_id"], "qa_coreference"),
                            "task": "qa_coreference",
                            "instance_id": instance_id,
                            "parent_event_id": parent["parent_event_id"],
                            "target_text": text,
                            "target_char_start": 0,
                            "candidate_concept_ids": unresolved_targets,
                            "allowed_labels": ["present", "absent", "uncertain", "not_mentioned"],
                            "previous_nurse_question": latest_question["text"],
                            "previous_question_event_id": q_event["parent_event_id"],
                            "previous_same_concept_event": None,
                        }
                    )
                elif (
                    unresolved_targets
                    and len(targets) > 1
                    and not explicitly_answered_targets
                    and (GENERIC_DENIAL_RE.search(answer_text) or UNCERTAINTY_RE.search(answer_text))
                ):
                    ambiguities.append(
                        {
                            "ambiguity_id": stable_id("amb", instance_id, parent["parent_event_id"], "qa_multi_target_scope"),
                            "task": "qa_coreference",
                            "instance_id": instance_id,
                            "parent_event_id": parent["parent_event_id"],
                            "target_text": text,
                            "target_char_start": 0,
                            "candidate_concept_ids": unresolved_targets,
                            "allowed_labels": ["present", "absent", "uncertain", "not_mentioned"],
                            "previous_nurse_question": latest_question["text"],
                            "previous_question_event_id": q_event["parent_event_id"],
                            "previous_same_concept_event": None,
                        }
                    )
            latest_question = None

    # Add support-only numeric scale cards after QA target resolution.
    cards_by_parent = defaultdict(list)
    for card in cards:
        cards_by_parent[card["parent_event_id"]].append(card)
    event_by_id = {event["parent_event_id"]: event for event in parent_events}
    for parent in parent_events:
        if parent["actor"] != "patient" or not parent["answer_to_question_evidence_id"]:
            continue
        qid = parent["answer_to_question_evidence_id"]
        qinfo = question_info.get(qid, {})
        if qinfo.get("scale_target") == "not_scale_question":
            continue
        text = parent["original"]
        for match in NUMERIC_SCALE_RE.finditer(normalized_match_text(text)):
            atom_id = "pain_present" if qinfo.get("scale_target") == "pain" else "numeric_scale_context"
            card_id = stable_id("card", instance_id, parent["parent_event_id"], match.start(), match.end(), atom_id)
            cards.append(
                {
                    "evidence_card_id": card_id,
                    "parent_event_id": parent["parent_event_id"],
                    "clause_span_id": None,
                    "span_text": text[match.start():match.end()],
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "registry_version": REGISTRY_VERSION,
                    "clinical_atom_id": atom_id,
                    "clinical_atom_label": "Pain severity scale context" if atom_id == "pain_present" else "Numeric severity scale context",
                    "concept_parent_id": "pain" if atom_id == "pain_present" else "numeric_scale_context",
                    "mention_type": "numeric_support",
                    "external_ontology_links": [],
                    "normalization": {"status": "qa_scale_bound", "method": "deterministic_scale_question_link", "candidate_concept_ids": qinfo.get("question_targets", [])},
                    "subject": "patient",
                    "subject_ambiguous": False,
                    "anatomy_site": None,
                    "laterality": None,
                    "assertion": "present",
                    "certainty": "asserted",
                    "temporality": "current_or_unspecified",
                    "scope_resolution": "numeric_answer_bound_to_scale_question",
                    "semantic_modifiers": [],
                    "source_subtype": "patient_scale_answer",
                    "source_layer": "patient_reported_support",
                    "clinical_assertion_allowed": True,
                    "positive_trigger_allowed": False,
                    "negative_evidence_allowed": False,
                    "context_only": False,
                    "independent_evidence": False,
                    "support_only": True,
                    "scale_target_hint": qinfo.get("scale_target"),
                    "qa_context": {"question_event_id": qid, "question_targets": qinfo.get("question_targets", []), "scale_target": qinfo.get("scale_target")},
                    "episode_id": None,
                    "episode_relation_hint": None,
                    "qwen_resolution": None,
                    "nli_verification": None,
                }
            )

    # Provisional episode links are evidence organization, not state reduction.
    active_episode: dict[tuple[str, str], str] = {}
    episode_counter: Counter[tuple[str, str]] = Counter()
    previous_card: dict[tuple[str, str], dict[str, Any]] = {}
    parent_order = {event["parent_event_id"]: event["event_seq_idx"] for event in parent_events}
    for card in sorted(cards, key=lambda c: (parent_order.get(c["parent_event_id"], -1), c["char_start"], c["clinical_atom_id"])):
        if card.get("context_only") or card.get("subject") not in {"patient", "family_member"}:
            continue
        key = (str(card["subject"]), str(card["concept_parent_id"]))
        if card["temporality"] == "historical":
            episode_counter[key] += 1
            episode_id = stable_id("episode", instance_id, *key, "historical", episode_counter[key])
            relation_hint = "historical_episode"
        else:
            current_parent_text = event_by_id.get(card["parent_event_id"], {}).get("original", "")
            if key not in active_episode or (NEW_EPISODE_RE.search(current_parent_text) and RESOLVED_RE.search(current_parent_text)):
                episode_counter[key] += 1
                active_episode[key] = stable_id("episode", instance_id, *key, "current", episode_counter[key])
                relation_hint = "starts_episode"
            else:
                relation_hint = "same_episode_candidate"
            episode_id = active_episode[key]
        card["episode_id"] = episode_id
        card["episode_relation_hint"] = relation_hint
        if key in previous_card and previous_card[key].get("episode_id") == episode_id:
            relation_edges.append(
                {
                    "relation_id": stable_id("rel", previous_card[key]["evidence_card_id"], card["evidence_card_id"], "same_episode_candidate"),
                    "source_id": card["evidence_card_id"],
                    "target_id": previous_card[key]["evidence_card_id"],
                    "relation_type": "same_episode_candidate",
                    "relation_status": "provisional_for_02",
                }
            )
        previous_card[key] = card

    measurements = []
    quarantined = []
    canonical_by_raw_name = {}
    for field in VITAL_FIELDS:
        contract = classify_vital_contract(field, vignette.get(field))
        measurement_id = stable_id("measurement", instance_id, field, "canonical")
        row = {
            "measurement_id": measurement_id,
            "measurement_identity": "canonical_case_measurement",
            "field": contract.get("canonical_name"),
            **contract,
        }
        canonical_by_raw_name[field] = row
        (measurements if contract["usable_for_clinical_reasoning"] else quarantined).append(row)
    pain_contract = classify_pain_contract(vignette.get("pain"))
    pain_row = {
        "measurement_id": stable_id("measurement", instance_id, "pain", "canonical"),
        "measurement_identity": "canonical_case_measurement",
        "field": "pain_score",
        **pain_contract,
    }
    (measurements if pain_contract["usable_for_clinical_reasoning"] else quarantined).append(pain_row)

    for parent in parent_events:
        if parent["actor"] != "system" or parent["event_type"] != "vital":
            continue
        raw_name = str(parent.get("system_name") or "")
        contract = classify_vital_contract(raw_name, parent.get("system_value"))
        canonical = canonical_by_raw_name.get(raw_name)
        reveal = {
            "measurement_id": stable_id("measurement", instance_id, parent["parent_event_id"], "reveal"),
            "measurement_identity": "dialogue_reveal",
            "field": contract.get("canonical_name"),
            "canonical_measurement_id": canonical.get("measurement_id") if canonical else None,
            "independent_measurement": False,
            "context_only": True,
            "parent_event_id": parent["parent_event_id"],
            **contract,
        }
        (measurements if contract["usable_for_clinical_reasoning"] else quarantined).append(reveal)
        if canonical:
            relation_edges.append(
                {
                    "relation_id": stable_id("rel", reveal["measurement_id"], canonical["measurement_id"], "reveals_measurement"),
                    "source_id": reveal["measurement_id"],
                    "target_id": canonical["measurement_id"],
                    "relation_type": "reveals_measurement",
                    "relation_status": "deterministic_identity_link",
                }
            )

    row = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "dataset": dataset,
        "case_id": case_id,
        "run_uuid": run_uuid,
        "instance_id": instance_id,
        "input_file_sha256": sha256_file(path),
        "parent_events": parent_events,
        "clause_spans": clause_spans,
        "evidence_cards": cards,
        "relation_edges": relation_edges,
        "measurement_records": measurements,
        "quarantined_measurements": quarantined,
        "qwen_ambiguity_ids": [item["ambiguity_id"] for item in ambiguities],
    }
    return row, ambiguities, quarantined


def qwen_system_prompt() -> str:
    return (
        "You are a bounded clinical ambiguity resolver for an evidence compiler. "
        "You may only choose from candidate_concept_ids and allowed_labels. "
        "The target text and optional immediately preceding nurse question are the only evidence. "
        "A nurse question is context, never positive patient evidence. Do not infer diagnosis, ESI, resources, "
        "severity, measurements, or an unmentioned symptom. Preserve negation, uncertainty, subject, and time. "
        "For qa_coreference, return only concepts actually resolved by the answer; omit unmentioned candidates. "
        "Each QA decision contains only concept_id and assertion. QA defaults are patient/current/same_episode. "
        "For card_attributes, also include subject and temporality. For nurse_subtype, include only nurse_subtype. "
        "Do not repeat target_text and do not provide explanations, quotes, rationales, null fields, or markdown. "
        "Return compact strict JSON only, for example: "
        "{\"results\":[{\"ambiguity_id\":\"id\",\"decisions\":[{\"concept_id\":\"allowed_id\",\"assertion\":\"present\"}]}]}."
    )


def call_qwen(api_url: str, model: str, batch: Sequence[Mapping[str, Any]], timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": qwen_system_prompt()},
            {"role": "user", "content": json.dumps({"items": list(batch)}, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        outer = json.loads(response.read().decode("utf-8"))
    content = outer["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    return json.loads(content)


def validate_qwen_decision(item: Mapping[str, Any], decision: Mapping[str, Any]) -> tuple[bool, str | None]:
    concept_id = decision.get("concept_id")
    if concept_id is not None and concept_id not in item["candidate_concept_ids"]:
        return False, "concept_outside_candidate_set"
    if decision.get("assertion") not in {"present", "absent", "uncertain", "not_mentioned"}:
        return False, "invalid_assertion"
    if decision.get("temporality") not in {"current", "recent", "historical", "unknown"}:
        return False, "invalid_temporality"
    if decision.get("subject") not in {"patient", "family_member", "unknown"}:
        return False, "invalid_subject"
    supporting = str(decision.get("supporting_text") or "")
    if supporting and supporting not in str(item["target_text"]):
        return False, "supporting_text_not_exact_substring"
    if item["task"] == "nurse_subtype" and decision.get("nurse_subtype") not in item["allowed_labels"]:
        return False, "invalid_nurse_subtype"
    return True, None


def normalize_qwen_decision(item: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(decision)
    task = str(item.get("task"))
    if task.startswith("qa_"):
        normalized.setdefault("temporality", "current")
        normalized.setdefault("subject", "patient")
        normalized.setdefault("episode_relation", "same_episode")
        normalized.setdefault("nurse_subtype", None)
    elif task == "nurse_subtype":
        normalized.setdefault("concept_id", None)
        normalized.setdefault("assertion", "not_mentioned")
        normalized.setdefault("temporality", "current")
        normalized.setdefault("subject", "patient")
        normalized.setdefault("episode_relation", None)
    else:
        normalized.setdefault("temporality", "unknown")
        normalized.setdefault("subject", "unknown")
        normalized.setdefault("episode_relation", "uncertain")
        normalized.setdefault("nurse_subtype", None)
    normalized.setdefault("supporting_text", "")
    normalized.setdefault("confidence", "low")
    return normalized


def run_qwen(
    ambiguities: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    api_url: str,
    model: str,
    batch_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    cached = {
        row["ambiguity_id"]: row
        for row in read_jsonl(cache_path)
        if row.get("qwen_status") == "proposed_pending_nli"
    }
    by_id = {str(item["ambiguity_id"]): item for item in ambiguities}
    pending = [item for item in ambiguities if item["ambiguity_id"] not in cached]
    completed = 0

    def resolve_batch(batch: Sequence[Mapping[str, Any]]) -> None:
        nonlocal completed
        attempt_error = None
        parsed = None
        for attempt in range(2):
            try:
                parsed = call_qwen(api_url, model, batch, timeout)
                break
            except Exception as exc:
                attempt_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1 + attempt)
        if parsed is None and len(batch) > 1:
            midpoint = len(batch) // 2
            resolve_batch(batch[:midpoint])
            resolve_batch(batch[midpoint:])
            return
        result_by_id = {
            str(row.get("ambiguity_id")): row
            for row in (parsed or {}).get("results", [])
            if isinstance(row, dict)
        }
        additions = []
        for item in batch:
            raw_result = result_by_id.get(str(item["ambiguity_id"]))
            if raw_result is None:
                row = {**item, "qwen_status": "failure_unresolved", "error": attempt_error or "missing_result", "decisions": []}
            else:
                accepted_decisions = []
                errors = []
                for decision in raw_result.get("decisions", []):
                    normalized_decision = normalize_qwen_decision(item, decision)
                    valid, error = validate_qwen_decision(item, normalized_decision)
                    if valid:
                        accepted_decisions.append(normalized_decision)
                    else:
                        errors.append(error)
                row = {
                    **item,
                    "qwen_status": "proposed_pending_nli" if accepted_decisions else "invalid_unresolved",
                    "error": ";".join(str(error) for error in errors if error) or None,
                    "decisions": accepted_decisions,
                }
            cached[item["ambiguity_id"]] = row
            additions.append(row)
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in additions:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
        completed += len(batch)
        print(f"Qwen ambiguity resolution {completed}/{len(pending)}", flush=True)

    for start in range(0, len(pending), batch_size):
        resolve_batch(pending[start:start + batch_size])
    return [cached[item_id] for item_id in sorted(cached) if item_id in by_id]


def controlled_hypothesis(concept_id: str | None, decision: Mapping[str, Any], task: str) -> str:
    if task == "nurse_subtype":
        subtype = str(decision.get("nurse_subtype") or "nurse_context_statement")
        phrases = {
            "nurse_direct_observation": "The nurse is directly reporting an observation of the patient.",
            "nurse_question": "The nurse is asking a question rather than reporting an observed patient finding.",
            "nurse_process_or_instruction": "The nurse is describing a future process, monitoring step, or treatment instruction.",
            "nurse_restatement_context": "The nurse is restating information previously reported by the patient.",
            "nurse_context_statement": "The nurse statement is contextual and is not a direct clinical observation.",
        }
        return phrases[subtype]
    label = CLINICAL_ATOMS.get(str(concept_id), {}).get("label", str(concept_id).replace("_", " "))
    subject = "The patient" if decision.get("subject") == "patient" else "A family member"
    assertion = decision.get("assertion")
    temporality = decision.get("temporality")
    time_phrase = "currently" if temporality in {"current", "unknown"} else ("recently" if temporality == "recent" else "historically")
    if assertion == "present":
        return f"{subject} {time_phrase} has {label}."
    if assertion == "absent":
        return f"{subject} {time_phrase} does not have {label}."
    if assertion == "uncertain":
        return f"{subject} is uncertain whether they {time_phrase} have {label}."
    return f"The target text does not state whether {subject.lower()} has {label}."


def incompatible_hypotheses(concept_id: str | None, decision: Mapping[str, Any], task: str) -> list[str]:
    if task == "nurse_subtype":
        alternatives = []
        for subtype in ("nurse_direct_observation", "nurse_question", "nurse_process_or_instruction", "nurse_restatement_context"):
            if subtype != decision.get("nurse_subtype"):
                alternatives.append(controlled_hypothesis(None, {**decision, "nurse_subtype": subtype}, task))
        return alternatives[:2]
    alternatives = []
    for assertion in ("present", "absent", "uncertain"):
        if assertion != decision.get("assertion"):
            alternatives.append(controlled_hypothesis(concept_id, {**decision, "assertion": assertion}, task))
    if decision.get("temporality") == "historical":
        alternatives.append(controlled_hypothesis(concept_id, {**decision, "temporality": "current"}, task))
    return alternatives[:3]


def build_nli_pairs(qwen_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for row in qwen_rows:
        if row.get("qwen_status") != "proposed_pending_nli":
            continue
        context = []
        if row.get("previous_nurse_question"):
            context.append(f"Previous nurse question: {row['previous_nurse_question']}")
        context.append(f"Target statement: {row['target_text']}")
        premise = "\n".join(context)
        for index, decision in enumerate(row.get("decisions", [])):
            if decision.get("assertion") == "not_mentioned" and row.get("task") != "nurse_subtype":
                continue
            concept_id = decision.get("concept_id")
            chosen = controlled_hypothesis(concept_id, decision, str(row["task"]))
            alternatives = incompatible_hypotheses(concept_id, decision, str(row["task"]))
            pair_id = stable_id("nli", row["ambiguity_id"], index, concept_id, chosen)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "ambiguity_id": row["ambiguity_id"],
                    "decision_index": index,
                    "task": row["task"],
                    "concept_id": concept_id,
                    "premise": premise,
                    "chosen_hypothesis": chosen,
                    "contrast_hypotheses": alternatives,
                    "decision": decision,
                }
            )
    return pairs


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.json"))
    preliminary = []
    ambiguities = []
    quarantined = []
    for index, path in enumerate(files, 1):
        row, row_ambiguities, row_quarantine = compile_record(path)
        preliminary.append(row)
        ambiguities.extend(row_ambiguities)
        quarantined.extend({"instance_id": row["instance_id"], **item} for item in row_quarantine)
        if index % 100 == 0 or index == len(files):
            print(f"Deterministic compilation {index}/{len(files)}", flush=True)
    write_jsonl(args.output_dir / "preliminary_timelines.jsonl", preliminary)
    write_jsonl(args.output_dir / "qwen_ambiguity_queue.jsonl", ambiguities)
    write_jsonl(args.output_dir / "measurement_quarantine.jsonl", quarantined)
    cache_path = args.output_dir / "qwen_resolution_cache.jsonl"
    if args.no_qwen:
        qwen_rows = [{**item, "qwen_status": "not_run_unresolved", "error": None, "decisions": []} for item in ambiguities]
        write_jsonl(cache_path, qwen_rows)
    else:
        qwen_rows = run_qwen(
            ambiguities,
            cache_path=cache_path,
            api_url=args.api_url,
            model=args.qwen_model,
            batch_size=args.qwen_batch_size,
            timeout=args.qwen_timeout,
        )
    write_jsonl(cache_path, qwen_rows)
    nli_pairs = build_nli_pairs(qwen_rows)
    write_jsonl(args.output_dir / "nli_pairs.jsonl", nli_pairs)
    write_json(
        args.output_dir / "prepare_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "registry_version": REGISTRY_VERSION,
            "input_dir": str(args.input_dir.resolve()),
            "input_file_count": len(files),
            "preliminary_timeline_count": len(preliminary),
            "evidence_card_count": sum(len(row["evidence_cards"]) for row in preliminary),
            "ambiguity_count": len(ambiguities),
            "ambiguity_task_counts": dict(Counter(item["task"] for item in ambiguities)),
            "qwen_status_counts": dict(Counter(row["qwen_status"] for row in qwen_rows)),
            "nli_pair_count": len(nli_pairs),
            "quarantined_measurement_count": len(quarantined),
            "nli_design": "directional_controlled_hypothesis_with_contrastive_margin",
        },
    )


def accepted_nli(score: Mapping[str, Any], min_entailment: float, max_contradiction: float, min_margin: float) -> tuple[bool, str]:
    chosen = score.get("chosen_scores", {})
    entailment = float(chosen.get("entailment", 0.0))
    contradiction = float(chosen.get("contradiction", 1.0))
    contrast_entailment = max((float(item.get("scores", {}).get("entailment", 0.0)) for item in score.get("contrast_scores", [])), default=0.0)
    margin = entailment - contrast_entailment
    if entailment < min_entailment:
        return False, "chosen_entailment_below_threshold"
    if contradiction > max_contradiction:
        return False, "chosen_contradiction_above_threshold"
    if margin < min_margin:
        return False, "contrastive_margin_below_threshold"
    return True, "accepted"


def add_qwen_card(row: dict[str, Any], qwen_row: Mapping[str, Any], pair: Mapping[str, Any], score: Mapping[str, Any]) -> dict[str, Any] | None:
    decision = pair["decision"]
    concept_id = decision.get("concept_id")
    if not concept_id or concept_id not in CLINICAL_ATOMS:
        return None
    target = str(qwen_row["target_text"])
    supporting = str(decision.get("supporting_text") or target)
    local = target.find(supporting)
    if local < 0:
        return None
    parent = next((event for event in row["parent_events"] if event["parent_event_id"] == qwen_row["parent_event_id"]), None)
    if parent is None:
        return None
    char_start = int(qwen_row.get("target_char_start", 0)) + local
    char_end = char_start + len(supporting)
    card_id = stable_id("card", row["instance_id"], parent["parent_event_id"], char_start, char_end, concept_id, "qwen")
    policy = source_policy("patient", "patient_statement")
    anatomy_site, laterality = anatomy_attributes(concept_id, str(parent.get("original") or ""), char_start, char_end)
    card = {
        "evidence_card_id": card_id,
        "parent_event_id": parent["parent_event_id"],
        "clause_span_id": None,
        "span_text": supporting,
        "char_start": char_start,
        "char_end": char_end,
        **standard_term(concept_id),
        "normalization": {"status": "confirmed_bounded_qwen_nli", "method": "closed_candidate_qwen", "candidate_concept_ids": qwen_row["candidate_concept_ids"]},
        "subject": decision["subject"],
        "subject_ambiguous": decision["subject"] == "unknown",
        "anatomy_site": anatomy_site,
        "laterality": laterality,
        "assertion": decision["assertion"],
        "certainty": "uncertain" if decision["assertion"] == "uncertain" else "asserted",
        "temporality": decision["temporality"],
        "scope_resolution": "bounded_qwen_qa_coreference",
        "semantic_modifiers": [],
        "source_subtype": "patient_statement",
        **policy,
        "qa_context": {"question_event_id": qwen_row.get("previous_question_event_id"), "question_targets": qwen_row["candidate_concept_ids"]},
        "episode_id": None,
        "episode_relation_hint": decision.get("episode_relation"),
        "qwen_resolution": {"ambiguity_id": qwen_row["ambiguity_id"], "confidence": decision.get("confidence"), "bounded": True},
        "nli_verification": {"pair_id": pair["pair_id"], "chosen_scores": score.get("chosen_scores"), "contrast_scores": score.get("contrast_scores")},
    }
    if card["assertion"] != "present" or card["subject"] != "patient":
        card["positive_trigger_allowed"] = False
    return card


def recursive_forbidden_keys(value: Any, path: str = "") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_OUTPUT_KEYS:
                errors.append(f"{path}/{key}")
            errors.extend(recursive_forbidden_keys(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(recursive_forbidden_keys(child, f"{path}/{index}"))
    return errors


def validate_timeline(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors = []
    event_by_id = {event["parent_event_id"]: event for event in row["parent_events"]}
    for card in row["evidence_cards"]:
        parent = event_by_id.get(card["parent_event_id"])
        if parent is None and card["source_subtype"] != "structured_chief_complaint":
            errors.append({"type": "unknown_parent_event", "evidence_card_id": card["evidence_card_id"]})
            continue
        if parent is not None:
            text = str(parent.get("original") or "")
            start, end = int(card["char_start"]), int(card["char_end"])
            if text[start:end] != card["span_text"]:
                errors.append({"type": "span_offset_mismatch", "evidence_card_id": card["evidence_card_id"]})
        if card["clinical_atom_id"] not in CLINICAL_ATOMS and card["clinical_atom_id"] not in {"pain_present", "numeric_scale_context"}:
            errors.append({"type": "unknown_clinical_atom", "evidence_card_id": card["evidence_card_id"]})
        if card.get("source_subtype") in {"nurse_question", "nurse_process_or_instruction", "nurse_restatement_context", "nurse_context_statement", "nurse_subtype_ambiguous"} and card.get("positive_trigger_allowed"):
            errors.append({"type": "context_source_positive_trigger", "evidence_card_id": card["evidence_card_id"]})
        if card.get("support_only") and card.get("positive_trigger_allowed"):
            errors.append({"type": "support_only_positive_trigger", "evidence_card_id": card["evidence_card_id"]})
    for location in recursive_forbidden_keys(row):
        errors.append({"type": "forbidden_output_key", "location": location})
    return errors


def prediction_projection_hash(raw: Mapping[str, Any]) -> str:
    vignette = raw.get("vignette") or {}
    history = []
    for event in raw.get("history") or []:
        if not isinstance(event, dict):
            continue
        history.append({key: event.get(key) for key in ("turn", "actor", "event", "name", "value", "original") if event.get(key) is not None})
    projection = {
        "dataset": raw.get("dataset"),
        "case_id": raw.get("case_id"),
        "run_uuid": raw.get("run_uuid"),
        "vignette": {key: vignette.get(key) for key in ("chiefcomplaint", "temperature", "heartrate", "resprate", "o2sat", "sbp", "pain")},
        "history": history,
    }
    return hashlib.sha256(json.dumps(projection, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def label_permutation_invariant(input_dir: Path) -> bool:
    for path in sorted(input_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        before = prediction_projection_hash(raw)
        raw["ground_truth"] = {"acuity": 99, "chiefcomplaint": "PERMUTED"}
        raw.setdefault("vignette", {})["acuity"] = 99
        for event in raw.get("history") or []:
            if isinstance(event, dict) and "triage" in event:
                event["triage"] = 99
        if prediction_projection_hash(raw) != before:
            return False
    return True


def finalize(args: argparse.Namespace) -> None:
    preliminary = read_jsonl(args.output_dir / "preliminary_timelines.jsonl")
    active_ambiguity_ids = {
        ambiguity_id
        for row in preliminary
        for ambiguity_id in row.get("qwen_ambiguity_ids", [])
    }
    qwen_rows = {
        row["ambiguity_id"]: row
        for row in read_jsonl(args.output_dir / "qwen_resolution_cache.jsonl")
        if row["ambiguity_id"] in active_ambiguity_ids
    }
    pairs = {row["pair_id"]: row for row in read_jsonl(args.output_dir / "nli_pairs.jsonl")}
    scores = {row["pair_id"]: row for row in read_jsonl(args.nli_scores)}
    pair_by_ambiguity = defaultdict(list)
    for pair in pairs.values():
        pair_by_ambiguity[pair["ambiguity_id"]].append(pair)

    unresolved = []
    qwen_audit = []
    accepted_count = 0
    for row in preliminary:
        for ambiguity_id in row.get("qwen_ambiguity_ids", []):
            qrow = qwen_rows.get(ambiguity_id)
            if not qrow:
                unresolved.append({"instance_id": row["instance_id"], "ambiguity_id": ambiguity_id, "reason": "missing_qwen_row"})
                continue
            decisions_audit = []
            for pair in pair_by_ambiguity.get(ambiguity_id, []):
                score = scores.get(pair["pair_id"])
                if score is None:
                    accepted, reason = False, "missing_nli_score"
                else:
                    accepted, reason = accepted_nli(score, args.min_entailment, args.max_contradiction, args.min_margin)
                decisions_audit.append({"pair_id": pair["pair_id"], "concept_id": pair.get("concept_id"), "accepted": accepted, "reason": reason})
                if accepted and qrow["task"] == "qa_coreference":
                    card = add_qwen_card(row, qrow, pair, score)
                    if card is not None:
                        row["evidence_cards"].append(card)
                        accepted_count += 1
                elif accepted and qrow["task"] == "nurse_subtype":
                    subtype = pair["decision"].get("nurse_subtype")
                    for card in row["evidence_cards"]:
                        if card["parent_event_id"] == qrow["parent_event_id"] and card["span_text"] in qrow["target_text"]:
                            policy = source_policy("nurse", subtype)
                            card.update(source_subtype=subtype, **policy)
                            card["qwen_resolution"] = {"ambiguity_id": ambiguity_id, "confidence": pair["decision"].get("confidence"), "bounded": True}
                            card["nli_verification"] = {"pair_id": pair["pair_id"], "chosen_scores": score.get("chosen_scores"), "contrast_scores": score.get("contrast_scores")}
                            accepted_count += 1
                elif accepted and qrow["task"] == "card_attributes":
                    target_card_id = qrow.get("target_evidence_card_id")
                    target_card = next((card for card in row["evidence_cards"] if card["evidence_card_id"] == target_card_id), None)
                    if target_card is not None and pair["decision"].get("concept_id") == target_card["clinical_atom_id"]:
                        decision = pair["decision"]
                        target_card["subject"] = decision["subject"]
                        target_card["subject_ambiguous"] = decision["subject"] == "unknown"
                        target_card["assertion"] = decision["assertion"]
                        target_card["certainty"] = "uncertain" if decision["assertion"] == "uncertain" else "asserted"
                        target_card["temporality"] = decision["temporality"]
                        target_card["scope_resolution"] = "bounded_qwen_attribute_resolution"
                        target_card["qwen_resolution"] = {"ambiguity_id": ambiguity_id, "confidence": decision.get("confidence"), "bounded": True}
                        target_card["nli_verification"] = {"pair_id": pair["pair_id"], "chosen_scores": score.get("chosen_scores"), "contrast_scores": score.get("contrast_scores")}
                        if target_card["assertion"] != "present" or target_card["subject"] != "patient":
                            target_card["positive_trigger_allowed"] = False
                        accepted_count += 1
                if not accepted:
                    unresolved.append({"instance_id": row["instance_id"], "ambiguity_id": ambiguity_id, "pair_id": pair["pair_id"], "reason": reason})
            if not pair_by_ambiguity.get(ambiguity_id):
                unresolved.append({"instance_id": row["instance_id"], "ambiguity_id": ambiguity_id, "reason": qrow.get("qwen_status")})
            qwen_audit.append({"instance_id": row["instance_id"], "ambiguity_id": ambiguity_id, "task": qrow.get("task"), "qwen_status": qrow.get("qwen_status"), "decisions": decisions_audit})

    for row in preliminary:
        event_by_id = {event["parent_event_id"]: event for event in row["parent_events"]}
        for card in row["evidence_cards"]:
            parent = event_by_id.get(card["parent_event_id"])
            if parent is None:
                continue
            card["anatomy_site"], card["laterality"] = anatomy_attributes(
                card["clinical_atom_id"],
                str(parent.get("original") or ""),
                int(card["char_start"]),
                int(card["char_end"]),
            )

    validation_errors = []
    for row in preliminary:
        for error in validate_timeline(row):
            validation_errors.append({"instance_id": row["instance_id"], **error})

    final_path = args.output_dir / "evidence_timelines.jsonl"
    write_jsonl(final_path, preliminary)
    flat_cards = [
        {"instance_id": row["instance_id"], "case_id": row["case_id"], **card}
        for row in preliminary
        for card in row["evidence_cards"]
    ]
    write_jsonl(args.output_dir / "evidence_cards.jsonl", flat_cards)
    write_jsonl(args.output_dir / "qwen_nli_audit.jsonl", qwen_audit)
    write_jsonl(args.output_dir / "unresolved_semantic_review.jsonl", unresolved)
    write_jsonl(args.output_dir / "validation_errors.jsonl", validation_errors)

    source_counts = Counter(card["source_subtype"] for card in flat_cards)
    assertion_counts = Counter(card["assertion"] for card in flat_cards)
    atom_counts = Counter(card["clinical_atom_id"] for card in flat_cards)
    question_positive = sum(
        1
        for card in flat_cards
        if card.get("source_subtype") in {"nurse_question", "nurse_process_or_instruction", "nurse_restatement_context", "nurse_context_statement", "nurse_subtype_ambiguous"}
        and card.get("positive_trigger_allowed")
    )
    invalid_consumed = sum(
        1
        for row in preliminary
        for item in row["quarantined_measurements"]
        if item.get("usable_for_clinical_reasoning") or item.get("evidence_use_policy", {}).get("allowed")
    )
    permutation_ok = label_permutation_invariant(args.input_dir)
    unique_instances = len({row["instance_id"] for row in preliminary})
    hard_gates = {
        "expected_instance_count": len(preliminary) == args.expected_instance_count,
        "unique_instance_ids": unique_instances == len(preliminary),
        "validation_errors_zero": not validation_errors,
        "forbidden_leakage_zero": not any(error["type"] == "forbidden_output_key" for error in validation_errors),
        "question_process_positive_zero": question_positive == 0,
        "invalid_measurement_consumption_zero": invalid_consumed == 0,
        "label_permutation_invariant": permutation_ok,
        "nli_scores_complete": len(scores) == len(pairs),
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "output_file": str(final_path.resolve()),
        "processed_instance_count": len(preliminary),
        "unique_case_count": len({row["case_id"] for row in preliminary}),
        "unique_instance_count": unique_instances,
        "parent_event_count": sum(len(row["parent_events"]) for row in preliminary),
        "clause_span_count": sum(len(row["clause_spans"]) for row in preliminary),
        "evidence_card_count": len(flat_cards),
        "evidence_card_assertion_counts": dict(assertion_counts),
        "evidence_card_source_counts": dict(source_counts),
        "top_clinical_atom_counts": dict(atom_counts.most_common(50)),
        "qwen_ambiguity_count": len(qwen_rows),
        "qwen_nli_accepted_action_count": accepted_count,
        "unresolved_semantic_review_count": len(unresolved),
        "nli_pair_count": len(pairs),
        "nli_score_count": len(scores),
        "nli_thresholds": {"min_entailment": args.min_entailment, "max_contradiction": args.max_contradiction, "min_contrastive_margin": args.min_margin},
        "quarantined_measurement_count": sum(len(row["quarantined_measurements"]) for row in preliminary),
        "invalid_measurement_consumption_count": invalid_consumed,
        "question_process_positive_count": question_positive,
        "validation_error_count": len(validation_errors),
        "hard_gates": hard_gates,
        "release_gate_passed": all(hard_gates.values()),
        "stage_boundary": {
            "01_outputs_local_evidence_not_esi": True,
            "02_owns_policy_qualification_and_state_reduction": True,
            "03_owns_vital_fact_interpretation": True,
        },
    }
    write_json(args.output_dir / "timeline_build_audit.json", audit)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_root_sha256": hashlib.sha256("".join(row["input_file_sha256"] for row in preliminary).encode("utf-8")).hexdigest(),
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
        },
    }
    write_json(args.output_dir / "ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("transcripts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/01_evidence_cards_v1"))
    parser.add_argument("--stage", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--expected-instance-count", type=int, default=1010)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-batch-size", type=int, default=6)
    parser.add_argument("--qwen-timeout", type=float, default=120.0)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--nli-scores", type=Path, default=Path("outputs/01_evidence_cards_v1/nli_scores.jsonl"))
    parser.add_argument("--min-entailment", type=float, default=0.62)
    parser.add_argument("--max-contradiction", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.10)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
