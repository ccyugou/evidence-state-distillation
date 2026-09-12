from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.common.semantic_distillation_common import (
    BINDINGS, EXISTENCE_EFFECTS, EXISTENCE_GUIDE, FACETS, FACET_EFFECTS, FACET_GUIDE,
    SUBJECTS, TEMPORALITIES,
)


ROOT = Path(__file__).resolve().parents[2]
TEACHER_OUTPUT = Path(os.environ.get(
    "EVIDENCE_STATE_TEACHER_OUTPUT",
    ROOT / "outputs" / "teacher_distillation",
))
OUTPUT = Path(os.environ.get(
    "EVIDENCE_STATE_STUDENT_OUTPUT",
    ROOT / "outputs" / "student_distillation",
))
STUDENT_FIELDS = (
    "target_binding", "subject", "temporality", "existence_effect", "facet_effect",
    "updated_facets", "question_quote", "answer_quote",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def student_input(source: dict) -> dict:
    return {k: source[k] for k in (
        "target_concept", "target_surface", "doctor_question", "patient_answer",
        "previous_target_mentions", "local_context",
    )}


def student_target(action: dict) -> dict:
    return {k: action[k] for k in STUDENT_FIELDS}


def system_prompt() -> str:
    existence = "\n".join(f"- {x}: {EXISTENCE_GUIDE[x]}" for x in EXISTENCE_EFFECTS)
    facets = "\n".join(f"- {x}: {FACET_GUIDE[x]}" for x in FACET_EFFECTS)
    return f"""你是儿科多轮问诊的结构化语义解析器，不做诊断，不补充常识。
联合阅读局部上下文、医生问题、患者回答与既往同概念证据，只解析目标概念的状态效果。
存在层和侧面层可同时更新；否定频率、程度、性状或诱因不等于否定症状存在。
医生提问不能单独证明患者状态；短回答必须绑定问题；不确定时输出HOLD。
他人症状、诊断猜测和原文不支持的内容不得写入患者状态。
字段必须取自封闭枚举，question_quote与answer_quote必须是对应原文中的精确连续子串，无法引用则为空字符串。
target_binding={list(BINDINGS)}，其中DIRECT_TARGET直接回答目标，TARGET_FACET只回答目标侧面，其他值不得冒充患者当前目标。
subject={list(SUBJECTS)}；temporality={list(TEMPORALITIES)}。
存在层操作：
{existence}
侧面层操作：
{facets}
updated_facets只能取自{list(FACETS[:-1])}；没有更新则为空数组。
只输出JSON，不解释。"""


def response_schema() -> dict:
    properties = {
        "target_binding": {"type": "string", "enum": list(BINDINGS)},
        "subject": {"type": "string", "enum": list(SUBJECTS)},
        "temporality": {"type": "string", "enum": list(TEMPORALITIES)},
        "existence_effect": {"type": "string", "enum": list(EXISTENCE_EFFECTS)},
        "facet_effect": {"type": "string", "enum": list(FACET_EFFECTS)},
        "updated_facets": {"type": "array", "items": {"type": "string", "enum": list(FACETS[:-1])}},
        "question_quote": {"type": "string"},
        "answer_quote": {"type": "string"},
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "student_semantic_action", "strict": True,
            "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
        },
    }
