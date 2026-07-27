"""Run the current extraction pipeline on a local benchmark manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_v_0_5 import process_one_file


DEFAULT_CASES = Path("evals/private/cases.local.jsonl")
DEFAULT_OUTPUT = Path("evals/results/baseline_results.local.jsonl")
DEFAULT_ARTIFACT_DIR = Path("evals/results/baseline_outputs")


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_case(case: dict, artifact_dir: Path) -> dict:
    started_at = time.time()
    try:
        output_file, result = process_one_file(
            case["file_path"],
            output_root=str(artifact_dir),
        )
        latency_seconds = time.time() - started_at
        return {
            "id": case["id"],
            "candidate_id": case["candidate_id"],
            "expected_document_type": case["document_type"],
            "predicted_document_type": result.get("doc_type"),
            "prediction": result.get("chat_data", {}),
            "output_file": output_file,
            "latency_seconds": round(latency_seconds, 3),
            "error": None,
        }
    except Exception as exc:
        latency_seconds = time.time() - started_at
        return {
            "id": case["id"],
            "candidate_id": case.get("candidate_id"),
            "expected_document_type": case.get("document_type"),
            "predicted_document_type": None,
            "prediction": {},
            "output_file": None,
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

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} ({case['document_type']})")
        rows.append(run_case(case, args.artifact_dir))
        write_jsonl(args.output, rows)

    error_count = sum(1 for row in rows if row["error"])
    print(f"Saved results: {args.output}")
    print(f"Cases: {len(rows)}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()

