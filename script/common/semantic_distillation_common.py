from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get(
    "EVIDENCE_STATE_TEACHER_OUTPUT",
    ROOT / "outputs" / "teacher_distillation",
))

BINDINGS = ("DIRECT_TARGET", "TARGET_FACET", "OTHER_SUBJECT", "HISTORICAL_HYPOTHESIS", "UNSUPPORTED")
FACETS = (
    "existence", "severity", "frequency", "duration", "course", "trigger", "location",
    "radiation", "character", "associated_feature", "treatment_response", "temporality", "subject",
    "diagnostic_hypothesis", "unclear",
)
STANCES = ("affirm", "deny", "mixed", "uncertain", "revise", "no_answer")
SUBJECTS = ("patient", "caregiver", "family_other", "unknown")
TEMPORALITIES = ("current", "historical", "mixed", "unknown")
REQUIREMENTS = ("ANSWER_SUFFICIENT", "QUESTION_REQUIRED", "LONG_CONTEXT_REQUIRED", "INSUFFICIENT")
EXISTENCE_EFFECTS = (
    "SET_PRESENT", "SET_ABSENT", "KEEP_PRESENT", "MOVE_TO_HISTORY", "REOPEN",
    "NO_EXISTENCE_CHANGE", "HOLD",
)
FACET_EFFECTS = (
    "UPDATE_CURRENT_FACET", "UPDATE_HISTORICAL_FACET", "NO_FACET_UPDATE", "HOLD",
)

FIELD_ENUMS = {
    "target_binding": BINDINGS,
    "question_facet": FACETS,
    "answer_stance": STANCES,
    "subject": SUBJECTS,
    "temporality": TEMPORALITIES,
    "context_requirement": REQUIREMENTS,
    "existence_effect": EXISTENCE_EFFECTS,
    "facet_effect": FACET_EFFECTS,
}
CORE_FIELDS = tuple(FIELD_ENUMS)

EXISTENCE_GUIDE = {
    "SET_PRESENT": "回答使目标症状当前存在得到支持；即使同时给出时长或程度也可选择。",
    "SET_ABSENT": "回答明确目标症状当前不存在；不要求历史上曾存在。",
    "KEEP_PRESENT": "既往已支持当前存在，本轮只确认持续或更新侧面。",
    "MOVE_TO_HISTORY": "回答明确把此前目标症状整体限定到既往，当前存在不再成立。",
    "REOPEN": "目标症状此前已消失，回答明确当前复发。",
    "NO_EXISTENCE_CHANGE": "只处理历史/条件侧面、他人内容或无须改变当前存在层。",
    "HOLD": "当前存在层无法可靠确定。",
}
FACET_GUIDE = {
    "UPDATE_CURRENT_FACET": "更新当前症状的程度、频率、时长、性状、诱因等侧面。",
    "UPDATE_HISTORICAL_FACET": "只更新既往或历史条件中的侧面，不改变当前存在层。",
    "NO_FACET_UPDATE": "本轮没有合法侧面更新。",
    "HOLD": "侧面绑定无法可靠确定。",
}


def authority_scopes(action: dict, label_status: str) -> tuple[str, str]:
    if label_status == "HOLD_META_DISAGREEMENT":
        return "NONE", "NONE"
    if action["target_binding"] not in {"DIRECT_TARGET", "TARGET_FACET"} or action["subject"] != "patient":
        return "NONE", "NONE"

    temporality = action["temporality"]
    existence = action["existence_effect"]
    facet = action["facet_effect"]
    existence_scope = "NONE"
    if existence == "MOVE_TO_HISTORY" and temporality in {"current", "historical"}:
        existence_scope = "CURRENT_TO_HISTORY"
    elif existence not in {"HOLD", "NO_EXISTENCE_CHANGE"}:
        if temporality == "current":
            existence_scope = "CURRENT"
        elif temporality == "historical":
            existence_scope = "HISTORY"

    facet_scope = "NONE"
    if facet == "UPDATE_CURRENT_FACET" and temporality == "current":
        facet_scope = "CURRENT"
    elif facet == "UPDATE_HISTORICAL_FACET" and temporality == "historical":
        facet_scope = "HISTORY"
    return existence_scope, facet_scope


def compact_packet(row: dict) -> dict:
    source = row["input"]
    return {
        "packet_id": row["packet_id"],
        "target_concept": source["target_concept"],
        "target_surface": source["target_surface"],
        "doctor_question": source["doctor_question"],
        "patient_answer": source["patient_answer"],
        "previous_target_mentions": source["previous_target_mentions"],
        "local_context": source["local_context"],
    }


def system_prompt(reverse: bool = False) -> str:
    enums = "\n".join(f"{name}={list(values)}" for name, values in FIELD_ENUMS.items())
    existence_order = tuple(reversed(EXISTENCE_EFFECTS)) if reverse else EXISTENCE_EFFECTS
    facet_order = tuple(reversed(FACET_EFFECTS)) if reverse else FACET_EFFECTS
    existence = "\n".join(f"- {name}: {EXISTENCE_GUIDE[name]}" for name in existence_order)
    facets = "\n".join(f"- {name}: {FACET_GUIDE[name]}" for name in facet_order)
    return f"""你是儿科多轮问诊中的结构化语义解析器，不做诊断，也不补充医学常识。
patient指被问诊的孩子；caregiver指代答者本人。医生提问只提供语境，绝不能单独证明患者状态。

对每个packet，联合阅读局部上下文、医生问题、患者回答和既往同概念证据，判断患者回答实际许可的状态操作。
关键边界：
1. 存在层和侧面层不是互斥分类；同一回答可以同时SET_PRESENT并UPDATE_CURRENT_FACET。
2. 否定问题中的频率、程度、性状、诱因或伴随属性，不等于否定症状存在。
3. 短回答必须绑定医生问题；仍无法唯一确定时选择HOLD。
4. 诊断猜测、他人症状、医生建议和未被患者回答支持的内容不得写入患者状态。
5. 不得从常识推导原文未陈述的症状、因果、治疗或严重程度。
6. updated_facets只能从facet枚举选择，可同时包含多个值；没有更新则为空数组。
7. question_quote与answer_quote必须分别是doctor_question和patient_answer中的精确连续子串，无法引用可为空字符串。
8. SET_ABSENT只用于回答否定目标症状本身；如果回答否定被问属性却确认目标存在，必须KEEP_PRESENT并更新侧面。
9. 药物、食物或动作是否诱发症状属于trigger；治疗后症状是否改善属于treatment_response。

context_requirement严格定义：
- ANSWER_SUFFICIENT：只看patient_answer也能知道目标、立场和更新内容。
- QUESTION_REQUIRED：只看回答不够，但doctor_question+patient_answer足够。
- LONG_CONTEXT_REQUIRED：问题与回答仍不够，必须依赖更早对话。
- INSUFFICIENT：读完所给上下文仍不能唯一授权。

结构示例（只学习绑定逻辑，不套用医学词）：
- 问“这个症状多久了”，答“三天”：SET_PRESENT或KEEP_PRESENT取决于既往证据，同时UPDATE_CURRENT_FACET(duration)，QUESTION_REQUIRED。
- 问“是清水样吗”，答“不是，是黄而黏”：不能SET_ABSENT；应KEEP_PRESENT并UPDATE_CURRENT_FACET(character)，QUESTION_REQUIRED。
- 问“之前吃某药会出现这个症状吗”，答“不会”：NO_EXISTENCE_CHANGE并UPDATE_HISTORICAL_FACET(trigger)，QUESTION_REQUIRED。
- 问目标A，回答只谈无关目标B：existence_effect与facet_effect均HOLD，context_requirement=INSUFFICIENT。

封闭字段：
{enums}

存在层效果（本轮显示顺序）：
{existence}

侧面层效果（本轮显示顺序）：
{facets}

输出严格JSON，不解释：
{{"items":[{{"packet_id":"...","target_binding":"...","question_facet":"...","answer_stance":"...","subject":"...","temporality":"...","context_requirement":"...","existence_effect":"...","facet_effect":"...","updated_facets":[],"question_quote":"...","answer_quote":"...","confidence":0.0}}]}}
confidence为0到1；必须为每个输入packet输出且不得改写packet_id。"""


def qwen_response_format() -> dict:
    properties = {
        "packet_id": {"type": "string"},
        **{name: {"type": "string", "enum": list(values)} for name, values in FIELD_ENUMS.items()},
        "updated_facets": {"type": "array", "items": {"type": "string", "enum": list(FACETS[:-1])}},
        "question_quote": {"type": "string"}, "answer_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    item = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_teacher_annotations", "strict": True,
            "schema": {"type": "object", "properties": {"items": {"type": "array", "items": item}}, "required": ["items"], "additionalProperties": False},
        },
    }


def validate_annotation(item: dict, row: dict) -> list[str]:
    errors = [name for name, values in FIELD_ENUMS.items() if item.get(name) not in values]
    source = row["input"]
    if item.get("packet_id") != row["packet_id"]:
        errors.append("packet_id")
    for field, text_key in (("question_quote", "doctor_question"), ("answer_quote", "patient_answer")):
        quote = item.get(field)
        if not isinstance(quote, str) or (quote and quote not in source[text_key]):
            errors.append(field)
    updated = item.get("updated_facets")
    if not isinstance(updated, list) or any(x not in FACETS or x == "unclear" for x in updated):
        errors.append("updated_facets")
    confidence = item.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("confidence")
    return errors


def parse_items(payload: dict) -> dict[str, dict]:
    return {str(item.get("packet_id")): item for item in payload.get("items", [])}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def deepseek_chat(system: str, payload: dict) -> dict:
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    key = os.environ["DEEPSEEK_API_KEY"]
    retries = int(os.environ.get("MAX_RETRY", "2"))
    body = json.dumps({
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "temperature": 0,
        "max_tokens": int(os.environ.get("MAX_TOKENS", "4096")),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, body, {
                "Content-Type": "application/json", "Authorization": f"Bearer {key}",
            })
            with urllib.request.urlopen(request, timeout=240) as response:
                content = json.loads(response.read())["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
