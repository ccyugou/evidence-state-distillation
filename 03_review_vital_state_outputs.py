#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only reviewer for the deterministic 03 vital state layer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

OFFICIAL_FIELDS = {"heart_rate", "respiratory_rate", "spo2"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/03_vital_state_outputs_v1_full688"))
    parser.add_argument("--review-dir", type=Path, default=Path("outputs/03_vital_state_review_v1_full688"))
    args = parser.parse_args()
    rows = read_jsonl(args.output_dir / "03_vital_state_outputs.jsonl")
    audit = json.loads((args.output_dir / "03_vital_state_audit.json").read_text(encoding="utf-8"))
    suspicious = []
    signals = []
    quarantined_measurement_count = 0
    invalid_measurement_referenced_count = 0
    for row in rows:
        if row.get("official_danger_zone_signal_present"): signals.append(row)
        if row.get("step_d_application_status") == "signal_present_not_yet_applied" and not row.get("official_danger_zone_flags"):
            suspicious.append({"instance_id": row.get("instance_id"), "reason": "signal_without_flag"})
        for flag in row.get("safety_context_flags", []):
            if flag.get("official_step_d_trigger_eligible") is not False:
                suspicious.append({"instance_id": row.get("instance_id"), "reason": "context_marked_official"})
        for flag in row.get("official_danger_zone_flags", []):
            if flag.get("measurement_type") not in OFFICIAL_FIELDS:
                suspicious.append({"instance_id": row.get("instance_id"), "reason": "nonofficial_field_signal"})
            if not flag.get("supporting_evidence_id"):
                suspicious.append({"instance_id": row.get("instance_id"), "reason": "signal_without_evidence"})
        observation_ids = {item.get("evidence_id") for item in row.get("vital_observations", []) if item.get("evidence_id")}
        invalid_ids = {item.get("evidence_id") for item in row.get("quarantined_measurements", []) if item.get("evidence_id")}
        quarantined_measurement_count += len(invalid_ids)
        referenced_ids = observation_ids | set(row.get("supporting_evidence_ids", []))
        leaked_invalid = referenced_ids & invalid_ids
        if leaked_invalid:
            invalid_measurement_referenced_count += len(leaked_invalid)
            suspicious.append({"instance_id": row.get("instance_id"), "reason": "invalid_measurement_referenced_as_clinical_evidence", "evidence_ids": sorted(leaked_invalid)})
        for link in row.get("vital_reveal_links", []):
            if link.get("counted_as_clinical_observation") is not False:
                suspicious.append({"instance_id": row.get("instance_id"), "reason": "dialogue_reveal_counted_as_observation"})
    summary = {"schema_version": audit.get("schema_version"), "output_rows": len(rows), "official_signal_rows": len(signals), "signal_field_counts": dict(Counter(flag.get("measurement_type") for row in signals for flag in row.get("official_danger_zone_flags", []) if flag.get("triggered"))), "missing_required_vital_rows": sum(bool(row.get("missing_vital_flags")) for row in rows), "quarantined_rows": sum(bool(row.get("quarantined_measurements")) for row in rows), "quarantined_measurement_count": quarantined_measurement_count, "invalid_measurement_referenced_count": invalid_measurement_referenced_count, "suspicious_count": len(suspicious), "audit_release_gate_passed": audit.get("release_gate_passed") is True, "hard_gate_passed": not suspicious and audit.get("release_gate_passed") is True}
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.review_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.review_dir / "suspicious_outputs.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in suspicious), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
