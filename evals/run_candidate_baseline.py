"""Run the current folder-level extraction pipeline on candidate cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_v_0_5 import process_folder


DEFAULT_CASES = Path("evals/private/candidates.local.jsonl")
DEFAULT_OUTPUT = Path("evals/results/candidate_baseline_results.local.jsonl")
DEFAULT_ARTIFACT_DIR = Path("evals/results/candidate_outputs")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_results_by_type(results_by_type: dict) -> dict:
    prediction = {}
    for doc_type, result in results_by_type.items():
        chat_data = result.get("chat_data", {}) or {}
        for field_name, value in chat_data.items():
            prediction[f"{doc_type}.{field_name}"] = value
    return prediction


def run_case(case: dict, artifact_dir: Path) -> dict:
    started_at = time.time()
    try:
        combined_output_file, report_file, results_by_type, logs = process_folder(
            case["folder_path"],
            output_root=str(artifact_dir),
            output_base_name=case["id"],
        )
        latency_seconds = time.time() - started_at
        return {
            "id": case["id"],
            "display_name": case.get("display_name", ""),
            "application_id": case.get("application_id", ""),
            "prediction": flatten_results_by_type(results_by_type),
            "doc_types_found": sorted(results_by_type.keys()),
            "combined_output_file": combined_output_file,
            "report_file": report_file,
            "logs": logs,
            "latency_seconds": round(latency_seconds, 3),
            "error": None,
        }
    except Exception as exc:
        latency_seconds = time.time() - started_at
        return {
            "id": case["id"],
            "display_name": case.get("display_name", ""),
            "application_id": case.get("application_id", ""),
            "prediction": {},
            "doc_types_found": [],
            "combined_output_file": None,
            "report_file": None,
            "logs": [],
            "latency_seconds": round(latency_seconds, 3),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case.get('source_folder_name', '')}")
        rows.append(run_case(case, args.artifact_dir))
        write_jsonl(args.output, rows)

    error_count = sum(1 for row in rows if row["error"])
    print(f"Saved results: {args.output}")
    print(f"Candidates: {len(rows)}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
