"""Versioned ESI v5 Decision Point D vital-sign thresholds.

Source: Emergency Severity Index Handbook, 5th edition (2023),
Figure 6-1. This module contains policy constants only.
"""

MANIFEST_VERSION = "esi_v5_decision_point_d_vital_manifest_v1.0"
SOURCE = {
    "title": "Emergency Severity Index Handbook, 5th Edition",
    "year": 2023,
    "location": "Chapter 6, Figure 6-1, High-Risk Vital Signs",
}

# Bounds are [min_age_years, max_age_years). None means open-ended.
AGE_BANDS = (
    {"id": "lt_1_month", "min": 0.0, "max": 1.0 / 12.0, "hr_gt": 190.0, "rr_gt": 60.0},
    {"id": "1_to_12_months", "min": 1.0 / 12.0, "max": 1.0, "hr_gt": 180.0, "rr_gt": 55.0},
    {"id": "1_to_3_years", "min": 1.0, "max": 3.0, "hr_gt": 140.0, "rr_gt": 40.0},
    {"id": "3_to_5_years", "min": 3.0, "max": 5.0, "hr_gt": 120.0, "rr_gt": 35.0},
    {"id": "5_to_12_years", "min": 5.0, "max": 12.0, "hr_gt": 120.0, "rr_gt": 30.0},
    {"id": "12_to_18_years", "min": 12.0, "max": 18.0, "hr_gt": 100.0, "rr_gt": 20.0},
    {"id": "adult", "min": 18.0, "max": None, "hr_gt": 100.0, "rr_gt": 20.0},
)

RULE_IDS = {
    "heart_rate": "ESI5_DP_D_AGE_SPECIFIC_HEART_RATE",
    "respiratory_rate": "ESI5_DP_D_AGE_SPECIFIC_RESPIRATORY_RATE",
    "spo2": "ESI5_DP_D_SPO2_LT_92",
}

SPO2_THRESHOLD = 92.0
SPO2_OPERATOR = "<"
SPO2_CONTEXT_REQUIREMENT = "potential_respiratory_compromise"

