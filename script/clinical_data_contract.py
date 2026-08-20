"""Prediction-safe contracts shared by dataset audit and evidence extraction.

This module contains rules only. It intentionally has no labels, dataset
statistics, audit results, or output paths.
"""

from __future__ import annotations

from typing import Any


PAIN_MIN = 0.0
PAIN_MAX = 10.0
PAIN_CANONICAL_UNIT = "0_to_10_scale"

VITAL_CONTRACTS: dict[str, dict[str, Any]] = {
    "temperature": {
        "canonical_name": "temperature",
        "raw_unit": "fahrenheit",
        "canonical_unit": "fahrenheit",
        "plausible_min": 80.0,
        "plausible_max": 110.0,
    },
    "heartrate": {
        "canonical_name": "heart_rate",
        "raw_unit": "beats_per_minute",
        "canonical_unit": "beats_per_minute",
        "plausible_min": 20.0,
        "plausible_max": 260.0,
    },
    "resprate": {
        "canonical_name": "respiratory_rate",
        "raw_unit": "breaths_per_minute",
        "canonical_unit": "breaths_per_minute",
        "plausible_min": 4.0,
        "plausible_max": 80.0,
    },
    "o2sat": {
        "canonical_name": "spo2",
        "raw_unit": "percent",
        "canonical_unit": "percent",
        "plausible_min": 40.0,
        "plausible_max": 100.0,
    },
    "sbp": {
        "canonical_name": "systolic_blood_pressure",
        "raw_unit": "mmHg",
        "canonical_unit": "mmHg",
        "plausible_min": 40.0,
        "plausible_max": 300.0,
    },
}

VITAL_NAME_ALIASES = {
    "temperature": "temperature",
    "temp": "temperature",
    "heartrate": "heartrate",
    "heart_rate": "heartrate",
    "hr": "heartrate",
    "pulse": "heartrate",
    "resprate": "resprate",
    "respiratory_rate": "resprate",
    "rr": "resprate",
    "o2sat": "o2sat",
    "spo2": "o2sat",
    "oxygen_saturation": "o2sat",
    "sbp": "sbp",
    "systolic_bp": "sbp",
    "systolic_blood_pressure": "sbp",
}


def normalize_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def classify_vital_contract(raw_name: str, value: Any) -> dict[str, Any]:
    raw_key = str(raw_name or "").strip().lower()
    key = VITAL_NAME_ALIASES.get(raw_key)
    contract = VITAL_CONTRACTS.get(key or "")
    numeric = normalize_numeric(value)
    result = {
        "raw_name": raw_name,
        "canonical_name": contract.get("canonical_name") if contract else None,
        "raw_value": value,
        "raw_unit": contract.get("raw_unit") if contract else None,
        "normalized_value": numeric,
        "canonical_unit": contract.get("canonical_unit") if contract else None,
        "value_plausibility": "missing_or_non_numeric",
        "quarantine_reason": "missing_or_non_numeric",
        "usable_for_clinical_reasoning": False,
        "evidence_use_policy": {"allowed": False, "support_only": True},
    }
    if contract is None:
        result["quarantine_reason"] = "unknown_vital_name"
        return result
    if numeric is None:
        return result
    low, high = contract["plausible_min"], contract["plausible_max"]
    if numeric < low or numeric > high:
        result["value_plausibility"] = "implausible"
        result["quarantine_reason"] = f"{key}_outside_{low:g}_{high:g}"
        return result
    result.update(
        value_plausibility="plausible",
        quarantine_reason=None,
        usable_for_clinical_reasoning=True,
        evidence_use_policy={"allowed": True, "support_only": True},
    )
    return result


def classify_pain_contract(value: Any) -> dict[str, Any]:
    numeric = normalize_numeric(value)
    result = {
        "raw_value": value,
        "raw_unit": PAIN_CANONICAL_UNIT,
        "normalized_value": numeric,
        "canonical_unit": PAIN_CANONICAL_UNIT,
        "value_validity": "missing_or_non_numeric",
        "value_plausibility": "missing_or_non_numeric",
        "quarantine_reason": "missing_or_non_numeric",
        "usable_for_clinical_reasoning": False,
        "evidence_use_policy": {"allowed": False, "support_only": True},
    }
    if numeric is None:
        return result
    if numeric < PAIN_MIN or numeric > PAIN_MAX:
        result["value_validity"] = "out_of_range"
        result["value_plausibility"] = "out_of_range"
        result["quarantine_reason"] = f"pain_outside_{PAIN_MIN:g}_{PAIN_MAX:g}"
        return result
    result.update(
        value_validity="valid_0_10",
        value_plausibility="plausible",
        quarantine_reason=None,
        usable_for_clinical_reasoning=True,
        evidence_use_policy={"allowed": True, "support_only": True},
    )
    return result
