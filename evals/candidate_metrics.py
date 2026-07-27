"""Compute candidate-level end-to-end field metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_CASES = Path("evals/private/candidates.local.jsonl")
DEFAULT_RESULTS = Path("evals/results/candidate_baseline_results.local.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_value(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_match(actual, expected) -> bool:
    expected_text = normalize_value(expected)
    actual_text = normalize_value(actual)

    if expected_text in {"", "unknown"}:
        return actual_text in {"", "unknown"}

    return actual_text == expected_text


def score_case(case: dict, result: dict) -> dict:
    ground_truth = case.get("ground_truth", {}) or {}
    prediction = result.get("prediction", {}) or {}

    lower_prediction = {
        str(key).strip().lower(): value
        for key, value in prediction.items()
    }

    total = 0
    correct = 0
    missing = 0
    wrong_fields = []

    for field, expected in ground_truth.items():
        if expected in ("", None):
            continue

        total += 1
        actual = lower_prediction.get(str(field).strip().lower())

        if actual in ("", None):
            missing += 1
            wrong_fields.append(field)
            continue

        if is_match(actual, expected):
            correct += 1
        else:
            wrong_fields.append(field)

    return {
        "id": case["id"],
        "total_fields": total,
        "correct_fields": correct,
        "missing_fields": missing,
        "wrong_fields": wrong_fields,
        "has_error": bool(result.get("error")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    results = {row["id"]: row for row in load_jsonl(args.results)}

    if not results:
        print(f"No result file found or result file is empty: {args.results}")
        print("Run baseline first, for example: python evals/run_candidate_baseline.py --limit 2")
        print(f"Candidates: {len(cases)}")
        return

    scored = [
        score_case(case, results.get(case["id"], {"prediction": {}, "error": "missing_result"}))
        for case in cases
    ]

    scored_with_gt = [row for row in scored if row["total_fields"] > 0]
    total_fields = sum(row["total_fields"] for row in scored_with_gt)
    correct_fields = sum(row["correct_fields"] for row in scored_with_gt)
    missing_fields = sum(row["missing_fields"] for row in scored_with_gt)
    error_cases = sum(1 for row in scored if row["has_error"])

    if total_fields == 0:
        print("No ground truth fields found. Fill candidate ground truth first.")
        print(f"Candidates: {len(cases)}")
        print(f"Result errors: {error_cases}")
        return

    print(f"Candidates with ground truth: {len(scored_with_gt)}")
    print(f"Field accuracy: {correct_fields / total_fields:.3f}")
    print(f"Missing-field rate: {missing_fields / total_fields:.3f}")
    print(f"Result errors: {error_cases}")

    print("Failure cases:")
    for row in scored_with_gt:
        if row["wrong_fields"] or row["has_error"]:
            print(f"  {row['id']}: wrong={row['wrong_fields']}, error={row['has_error']}")


if __name__ == "__main__":
    main()

