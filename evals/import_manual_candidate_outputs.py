"""Convert manually downloaded candidate outputs into benchmark JSONL.

Expected local layout:

evals/results/manual_candidate_outputs/
├── UG000000_Example Applicant/
│   ├── *_combined_result.txt
│   └── *_report.html
└── UG000001_Another Applicant/
    ├── *_combined_result.txt
    └── *_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CASES = Path("evals/private/candidates.v0.local.jsonl")
DEFAULT_INPUT_DIR = Path("evals/results/manual_candidate_outputs")
DEFAULT_OUTPUT = Path("evals/results/manual_candidate_results.local.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_combined_result(path: Path) -> dict[str, str]:
    prediction = {}
    current_doc_type = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            current_doc_type = line.lstrip("#").strip()
            continue

        if not current_doc_type or ":" not in line:
            continue

        field, value = line.split(":", 1)
        field = field.strip()
        value = value.strip()

        if field:
            prediction[f"{current_doc_type}.{field}"] = value

    return prediction


def find_one(pattern: str, folder: Path) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def import_case(case: dict, input_dir: Path) -> dict:
    candidate_dir = input_dir / case["id"]
    combined_file = find_one("*_combined_result.txt", candidate_dir)
    report_file = find_one("*_report.html", candidate_dir)

    if combined_file is None:
        return {
            "id": case["id"],
            "display_name": case.get("display_name", ""),
            "application_id": case.get("application_id", ""),
            "prediction": {},
            "combined_output_file": None,
            "report_file": str(report_file) if report_file else None,
            "error": f"Missing combined result under {candidate_dir}",
        }

    return {
        "id": case["id"],
        "display_name": case.get("display_name", ""),
        "application_id": case.get("application_id", ""),
        "prediction": parse_combined_result(combined_file),
        "combined_output_file": str(combined_file),
        "report_file": str(report_file) if report_file else None,
        "error": None,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    rows = [import_case(case, args.input_dir) for case in cases]
    write_jsonl(args.output, rows)

    error_count = sum(1 for row in rows if row["error"])
    print(f"Imported results: {args.output}")
    print(f"Candidates: {len(rows)}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
