from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "30_student_distillation_v1"
FIELDS = ("target_binding", "subject", "temporality", "existence_effect", "facet_effect")


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def exact(row: dict) -> bool:
    return not row["errors"] and all(row["prediction"][f] == row["target"][f] for f in FIELDS) \
        and set(row["prediction"]["updated_facets"]) == set(row["target"]["updated_facets"])


def paired_bootstrap(values: dict[str, list[float]], iterations: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(20260831)
    groups = list(values)
    estimates = np.empty(iterations)
    for i in range(iterations):
        sampled = rng.choice(groups, len(groups), replace=True)
        pooled = [v for group in sampled for v in values[group]]
        estimates[i] = np.mean(pooled)
    return tuple(float(x) for x in np.percentile(estimates, (2.5, 97.5)))


def main() -> None:
    zero = {x["packet_id"]: x for x in read(OUTPUT / "dev_qwen3_4b_zero_shot_v2.jsonl")}
    student = {x["packet_id"]: x for x in read(OUTPUT / "dev_qwen3_4b_qlora_v1.jsonl")}
    if zero.keys() != student.keys():
        raise SystemExit("paired packet contract failed")
    deltas: dict[str, list[float]] = defaultdict(list)
    wins = losses = ties_correct = ties_wrong = 0
    field_delta = {field: [] for field in FIELDS}
    for packet_id, base in zero.items():
        tuned = student[packet_id]
        z, s = exact(base), exact(tuned)
        deltas[base["case_group_id"]].append(float(s) - float(z))
        wins += s and not z
        losses += z and not s
        ties_correct += s and z
        ties_wrong += not s and not z
        for field in FIELDS:
            field_delta[field].append(float(tuned["prediction"][field] == tuned["target"][field])
                                      - float(base["prediction"][field] == base["target"][field]))
    discordant = wins + losses
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2 ** discordant)
    report = {
        "schema_version": "imcs21_qwen3_4b_distillation_comparison_v1",
        "paired_packets": len(zero), "case_groups": len(deltas),
        "zero_shot_full_exact": sum(map(exact, zero.values())) / len(zero),
        "student_full_exact": sum(map(exact, student.values())) / len(student),
        "paired_delta": np.mean([x for values in deltas.values() for x in values]),
        "case_bootstrap_95ci": paired_bootstrap(deltas),
        "discordance": {"student_only_correct": wins, "zero_only_correct": losses,
                        "both_correct": ties_correct, "both_wrong": ties_wrong,
                        "mcnemar_exact_two_sided_p": min(1.0, 2 * tail)},
        "field_accuracy_delta": {field: float(np.mean(values)) for field, values in field_delta.items()},
        "scientific_scope": {
            "supported": "Silver-supervised semantic distillation improves common structured state operations on isolated Dev.",
            "not_supported": "Clinical expert accuracy or reliable generalization for rare operations such as REOPEN.",
        },
    }
    (OUTPUT / "qwen3_4b_distillation_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
