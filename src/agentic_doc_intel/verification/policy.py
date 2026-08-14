"""Verification policy for evidence-grounded candidate extraction.

This module intentionally avoids candidate-specific hardcoding. It classifies
verification findings into product-facing categories:

- hard_conflict: likely wrong or unsafe to auto-resolve.
- soft_conflict: explainable difference between sources.
- acceptable_unknown: unknown is expected or non-blocking.
- needs_human_review: cannot safely resolve automatically.

The goal is not to replace model reasoning. The goal is to keep deterministic,
cheap, and auditable policy outside prompts.
"""

from __future__ import annotations

import re
from typing import Any


BOOLEAN_TRUE = {"yes", "true", "y", "present"}
BOOLEAN_FALSE = {"no", "false", "n", "not present", "none"}


def canonical_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    text = text.strip(" :：,.;")
    if text in BOOLEAN_TRUE:
        return "yes"
    if text in BOOLEAN_FALSE:
        return "no"
    return text


def canonical_identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def values_equivalent(left: Any, right: Any) -> bool:
    left_text = canonical_value(left)
    right_text = canonical_value(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True

    left_identity = canonical_identity(left_text)
    right_identity = canonical_identity(right_text)
    if left_identity and right_identity:
        return (
            left_identity == right_identity
            or left_identity in right_identity
            or right_identity in left_identity
        )

    return False


def classify_unknown_field(item: dict[str, Any]) -> dict[str, Any]:
    doc_type = item.get("document_type")
    field = item.get("field")

    if doc_type == "english_language":
        reason = "No English-language qualification evidence was provided for this candidate."
        category = "acceptable_unknown"
    else:
        reason = "The current evidence does not contain a supported value for this field."
        category = "needs_review_unknown"

    return {
        **item,
        "category": category,
        "reason": reason,
        "needs_human_review": category != "acceptable_unknown",
        "label": f"{doc_type}.{field}",
    }


def classify_conflict(item: dict[str, Any]) -> dict[str, Any]:
    doc_type = item.get("document_type")
    field = item.get("field")
    values = item.get("values", [])
    canonical_values = sorted({canonical_value(value) for value in values if canonical_value(value)})

    if len(canonical_values) <= 1:
        return {
            **item,
            "category": "resolved_equivalent",
            "severity": "none",
            "reason": "Values are equivalent after normalization.",
            "needs_human_review": False,
        }

    if doc_type == "transcript" and field == "end or expected end year":
        has_academic_year = any("/" in str(value) for value in values)
        has_single_year = any(re.fullmatch(r"\d{4}", str(value).strip()) for value in values)
        if has_academic_year and has_single_year:
            return {
                **item,
                "category": "soft_conflict",
                "severity": "low",
                "reason": "One source appears to give an academic year, while another gives an issue/current year.",
                "needs_human_review": False,
            }

    if doc_type == "transcript" and field == "validity check":
        return {
            **item,
            "category": "soft_conflict",
            "severity": "low",
            "reason": "Different transcript-like files can have different source-level validity checks.",
            "needs_human_review": False,
        }

    high_risk_fields = {
        "passport number",
        "date of birth",
        "name",
        "nationality",
        "pass status",
        "overall average",
        "math score",
        "english score",
    }
    needs_human_review = field in high_risk_fields
    return {
        **item,
        "category": "hard_conflict" if needs_human_review else "unresolved_conflict",
        "severity": "high" if needs_human_review else "medium",
        "reason": "Values differ after normalization and no deterministic policy resolved the conflict.",
        "needs_human_review": needs_human_review,
    }


def classify_unsupported_field(item: dict[str, Any]) -> dict[str, Any]:
    value = canonical_value(item.get("value"))
    if value in {"yes", "no"}:
        return {
            **item,
            "category": "weak_boolean_support",
            "severity": "low",
            "reason": "Boolean value is plausible, but the reader did not preserve a source quote.",
            "needs_human_review": False,
        }

    return {
        **item,
        "category": "unsupported_field",
        "severity": "medium",
        "reason": "Final value has no supporting evidence quote.",
        "needs_human_review": True,
    }


def apply_verification_policy(report: dict[str, Any]) -> dict[str, Any]:
    unknown_fields = report.get("unknown_fields") or report.get("missing_fields") or []
    conflicts = report.get("conflicts", [])
    unsupported_fields = report.get("unsupported_fields", [])

    classified_unknowns = [classify_unknown_field(item) for item in unknown_fields]
    classified_conflicts = [classify_conflict(item) for item in conflicts]
    classified_unsupported = [classify_unsupported_field(item) for item in unsupported_fields]

    hard_conflicts = [
        item for item in classified_conflicts if item["category"] in {"hard_conflict", "unresolved_conflict"}
    ]
    soft_conflicts = [
        item for item in classified_conflicts if item["category"] in {"soft_conflict", "resolved_equivalent"}
    ]
    acceptable_unknowns = [
        item for item in classified_unknowns if item["category"] == "acceptable_unknown"
    ]
    review_unknowns = [
        item for item in classified_unknowns if item["category"] != "acceptable_unknown"
    ]
    weak_supported_fields = [
        item for item in classified_unsupported if item["category"] == "weak_boolean_support"
    ]
    unsupported_needing_review = [
        item for item in classified_unsupported if item["needs_human_review"]
    ]

    certificate_mismatches = report.get("certificate_identity_mismatches", [])
    needs_human_review = bool(
        [item for item in hard_conflicts if item.get("needs_human_review")]
        or review_unknowns
        or unsupported_needing_review
        or certificate_mismatches
    )

    return {
        **report,
        "classified_unknown_fields": classified_unknowns,
        "classified_conflicts": classified_conflicts,
        "classified_unsupported_fields": classified_unsupported,
        "hard_conflicts": hard_conflicts,
        "soft_conflicts": soft_conflicts,
        "acceptable_unknowns": acceptable_unknowns,
        "review_unknowns": review_unknowns,
        "weak_supported_fields": weak_supported_fields,
        "unsupported_needing_review": unsupported_needing_review,
        "needs_human_review": needs_human_review,
        "needs_retry": needs_human_review,
    }
