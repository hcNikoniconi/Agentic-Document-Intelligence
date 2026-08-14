"""Compute candidate-level end-to-end field metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_CASES = Path("evals/private/candidates.local.jsonl")
DEFAULT_RESULTS = Path("evals/results/candidate_baseline_results.local.jsonl")

UNKNOWN_VALUES = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "not found",
    "not available",
    "not specified",
}

NOT_SCORED_VALUES = {
    "not_scored",
    "__not_scored__",
    "exclude",
    "excluded",
}

MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


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


def normalize_raw_text(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_unknown(value) -> bool:
    return normalize_raw_text(value) in UNKNOWN_VALUES


def is_not_scored(value) -> bool:
    return normalize_raw_text(value) in NOT_SCORED_VALUES


def normalize_date(value) -> str | None:
    text = normalize_raw_text(value)
    if text in UNKNOWN_VALUES:
        return "unknown"

    match = re.fullmatch(r"(\d{4})[-/.\s](\d{1,2})[-/.\s](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.fullmatch(r"(\d{1,2})[-/.\s](\d{1,2})[-/.\s](\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.fullmatch(r"(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})", text)
    if match:
        day, month_name, year = match.groups()
        month = MONTHS.get(month_name)
        if month:
            return f"{year}-{month}-{int(day):02d}"

    match = re.fullmatch(r"([a-z]{3,9})\s+(\d{1,2}),?\s*(\d{4})", text)
    if match:
        month_name, day, year = match.groups()
        month = MONTHS.get(month_name)
        if month:
            return f"{year}-{month}-{int(day):02d}"

    compact = re.sub(r"[\s,./_-]+", "", text).upper()
    match = re.fullmatch(r"(\d{1,2})([A-Z]{3,9})(\d{4})", compact)
    if match:
        day, month_name, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{year}-{month}-{int(day):02d}"

    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", compact)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"

    return None


def normalize_gender(value) -> str:
    text = normalize_raw_text(value)
    compact = re.sub(r"[^a-z]", "", text)
    if compact in {"f", "pf", "p", "female", "perempuan", "woman", "girl"}:
        return "female"
    if compact in {"m", "lm", "l", "male", "lakilaki", "man", "boy"}:
        return "male"
    return text


def normalize_yes_no(value) -> str:
    text = normalize_raw_text(value)
    if text in {"yes", "y", "true", "present", "available", "found"}:
        return "yes"
    if text in {"no", "n", "false", "absent", "not present", "missing"}:
        return "no"
    return text


def normalize_passport_type(value) -> str:
    text = normalize_raw_text(value)
    if text in {"p", "passport", "paspor", "ordinary passport"}:
        return "p"
    return text


def normalize_exam_type(value) -> str:
    text = normalize_raw_text(value)
    if "ielts" in text:
        return "ielts"
    if "toefl" in text:
        return "toefl"
    if "duolingo" in text or text == "det":
        return "duolingo"
    if "pte" in text:
        return "pte"
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_person_name(value) -> str:
    text = normalize_raw_text(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    return "".join(sorted(tokens))


def normalize_qualification_type(value) -> str:
    text = normalize_raw_text(value)
    compact = re.sub(r"[^a-z0-9]+", "", text)

    if text in UNKNOWN_VALUES:
        return "unknown"
    if "ib" in text:
        return "ib"
    if "bachelor" in text and "information" in text and "technology" in text:
        return "bachelorinformationtechnology"
    if "foundation" in text:
        return "foundationyear"
    if "diploma" in text and "school" in text:
        return "highschooldiploma"
    if "high" in text and "school" in text and "transcript" in text:
        return "highschooltranscript"
    if "semester" in text and "report" in text:
        return "semesterreport"

    return compact


def normalize_value(value, field: str = "") -> str:
    field_text = normalize_raw_text(field)
    text = normalize_raw_text(value)

    if text in UNKNOWN_VALUES:
        return "unknown"

    if "date" in field_text:
        normalized_date = normalize_date(value)
        if normalized_date:
            return normalized_date

    if "gender" in field_text or "sex" in field_text:
        return normalize_gender(value)

    if "passport type" in field_text:
        return normalize_passport_type(value)

    if "exam type" in field_text:
        return normalize_exam_type(value)

    if "qualification type" in field_text:
        return normalize_qualification_type(value)

    if field_text.endswith("name") and "institution" not in field_text and "school" not in field_text:
        return normalize_person_name(value)

    if (
        "present" in field_text
        or "official document" in field_text
        or "stamp" in field_text
        or "signature" in field_text
    ):
        return normalize_yes_no(value)

    return re.sub(r"[^a-z0-9]+", "", text)


def is_match(actual, expected, field: str) -> bool:
    return normalize_value(actual, field) == normalize_value(expected, field)


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
    not_scored = 0

    for field, expected in ground_truth.items():
        if expected in ("", None):
            continue
        if is_not_scored(expected):
            not_scored += 1
            continue

        total += 1
        actual = lower_prediction.get(str(field).strip().lower())

        if actual in ("", None):
            missing += 1
            wrong_fields.append(field)
            continue

        if is_match(actual, expected, field):
            correct += 1
        else:
            wrong_fields.append(field)

    return {
        "id": case["id"],
        "total_fields": total,
        "correct_fields": correct,
        "missing_fields": missing,
        "not_scored_fields": not_scored,
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
    not_scored_fields = sum(row["not_scored_fields"] for row in scored)
    error_cases = sum(1 for row in scored if row["has_error"])

    if total_fields == 0:
        print("No ground truth fields found. Fill candidate ground truth first.")
        print(f"Candidates: {len(cases)}")
        print(f"Result errors: {error_cases}")
        return

    print(f"Candidates with ground truth: {len(scored_with_gt)}")
    print(f"Field accuracy: {correct_fields / total_fields:.3f}")
    print(f"Missing-field rate: {missing_fields / total_fields:.3f}")
    print(f"Not-scored fields: {not_scored_fields}")
    print(f"Result errors: {error_cases}")

    print("Failure cases:")
    for row in scored_with_gt:
        if row["wrong_fields"] or row["has_error"]:
            print(f"  {row['id']}: wrong={row['wrong_fields']}, error={row['has_error']}")


if __name__ == "__main__":
    main()
