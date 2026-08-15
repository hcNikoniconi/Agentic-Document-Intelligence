#!/usr/bin/env python3
"""Batch runner for v2 VLM + Text Agent.

Safe default:
    This script does NOT call any model unless --run is explicitly passed.

Default behavior creates a local batch plan from candidate folders so we can
inspect routing and file coverage without spending API tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_doc_intel.pipelines.vlm_text_agent import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    build_manifest,
    candidate_name_from_dir,
    run_candidate_folder,
)


DEFAULT_INPUT_ROOT = ROOT / "data" / "documents-export-2026-01-30"
DEFAULT_ENV_FILE = ROOT / ".env"


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def iter_candidate_dirs(input_root: Path, candidate_filter: str | None) -> list[Path]:
    candidates = [path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    candidates.sort(key=lambda path: path.name.lower())
    if candidate_filter:
        lowered = candidate_filter.lower()
        candidates = [path for path in candidates if lowered in path.name.lower()]
    return candidates


def output_dir_for_candidate(output_root: Path, candidate_dir: Path) -> Path:
    return output_root / candidate_dir.name


def has_completed_run(candidate_output_dir: Path) -> bool:
    required = [
        "document_manifest.json",
        "document_evidence.jsonl",
        "candidate_result.json",
        "verification_report.json",
        "summary_report.md",
    ]
    return all((candidate_output_dir / name).exists() for name in required)


def manifest_summary(candidate_dir: Path) -> dict[str, Any]:
    candidate = candidate_name_from_dir(candidate_dir)
    manifest = build_manifest(candidate_dir, candidate)

    reader_counts: dict[str, int] = {}
    doc_type_counts: dict[str, int] = {}
    for item in manifest:
        reader = item.reading_decision.get("primary_tool", "unknown")
        reader_counts[reader] = reader_counts.get(reader, 0) + 1
        doc_type_counts[item.routed_doc_type] = doc_type_counts.get(item.routed_doc_type, 0) + 1

    return {
        "candidate": candidate,
        "candidate_dir": str(candidate_dir),
        "files_scanned": len(manifest),
        "reader_counts": reader_counts,
        "doc_type_counts": doc_type_counts,
        "manifest": [asdict(item) for item in manifest],
    }


def read_run_summary(candidate_output_dir: Path) -> dict[str, Any]:
    verification_path = candidate_output_dir / "verification_report.json"
    evidence_path = candidate_output_dir / "document_evidence.jsonl"
    manifest_path = candidate_output_dir / "document_manifest.json"

    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {}
    evidence_count = 0
    if evidence_path.exists():
        evidence_count = len([line for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    files_scanned = 0
    if manifest_path.exists():
        files_scanned = len(json.loads(manifest_path.read_text(encoding="utf-8")))

    return {
        "files_scanned": files_scanned,
        "evidence_count": evidence_count,
        "needs_human_review": verification.get("needs_human_review"),
        "needs_retry": verification.get("needs_retry"),
        "hard_conflicts": len(verification.get("hard_conflicts", [])),
        "soft_conflicts": len(verification.get("soft_conflicts", [])),
        "review_unknowns": len(verification.get("review_unknowns", [])),
        "acceptable_unknowns": len(verification.get("acceptable_unknowns", [])),
        "certificate_identity_mismatches": len(verification.get("certificate_identity_mismatches", [])),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    columns = list(rows[0].keys())
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_batch_plan(candidate_dirs: list[Path], output_root: Path) -> dict[str, Any]:
    candidates = []
    rows = []
    for candidate_dir in candidate_dirs:
        summary = manifest_summary(candidate_dir)
        candidate_output_dir = output_dir_for_candidate(output_root, candidate_dir)
        summary["run_dir"] = str(candidate_output_dir)
        summary["already_completed"] = has_completed_run(candidate_output_dir)
        candidates.append(summary)
        rows.append(
            {
                "candidate": summary["candidate"],
                "files_scanned": summary["files_scanned"],
                "pdf_text_layer_files": summary["reader_counts"].get("pdf_text_layer", 0),
                "vlm_page_reader_files": summary["reader_counts"].get("vlm_page_reader", 0),
                "human_review_files": summary["reader_counts"].get("human_review", 0),
                "already_completed": summary["already_completed"],
                "run_dir": summary["run_dir"],
            }
        )
    return {
        "mode": "plan_only_no_model_calls",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--candidate", default=None, help="Optional candidate substring filter.")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Actually call model APIs. Without this, only a safe plan is generated.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip candidates with existing completed v2 outputs.")
    parser.add_argument("--deterministic-aggregate", action="store_true")
    args = parser.parse_args()

    candidate_dirs = iter_candidate_dirs(args.input_root, args.candidate)
    if args.max_candidates is not None:
        candidate_dirs = candidate_dirs[: args.max_candidates]

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.run:
        plan = build_batch_plan(candidate_dirs, output_root)
        plan_json = output_root / "batch_plan.json"
        plan_csv = output_root / "batch_plan.csv"
        plan_md = output_root / "batch_plan.md"
        write_json(plan_json, plan)
        write_csv(plan_csv, plan["rows"])
        write_markdown(plan_md, title="v2 Batch Plan (No Model Calls)", rows=plan["rows"])
        print(json.dumps({"mode": "plan_only_no_model_calls", "batch_plan": str(plan_json)}, ensure_ascii=False, indent=2))
        return 0

    load_env_file(args.env_file)
    batch_rows: list[dict[str, Any]] = []
    for candidate_dir in candidate_dirs:
        started = time.time()
        candidate = candidate_dir.name
        candidate_output_dir = output_dir_for_candidate(output_root, candidate_dir)

        if args.skip_existing and has_completed_run(candidate_output_dir):
            run_summary = read_run_summary(candidate_output_dir)
            batch_rows.append(
                {
                    "candidate": candidate,
                    "status": "skipped_existing",
                    "elapsed_seconds": 0,
                    "run_dir": str(candidate_output_dir),
                    "summary_report": str(candidate_output_dir / "summary_report.md"),
                    **run_summary,
                }
            )
            continue

        try:
            result = run_candidate_folder(
                candidate_dir,
                output_root=output_root,
                use_text_model_aggregator=not args.deterministic_aggregate,
            )
            elapsed = round(time.time() - started, 2)
            # Summary report generation is intentionally imported lazily so the
            # batch plan mode has no side effects.
            from build_summary_report import build_report

            summary_report = build_report(Path(result["run_dir"]))
            run_summary = read_run_summary(Path(result["run_dir"]))
            batch_rows.append(
                {
                    "candidate": candidate,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "run_dir": result["run_dir"],
                    "summary_report": str(summary_report),
                    **run_summary,
                }
            )
        except Exception as exc:
            elapsed = round(time.time() - started, 2)
            batch_rows.append(
                {
                    "candidate": candidate,
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "run_dir": str(candidate_output_dir),
                    "summary_report": "",
                    "files_scanned": 0,
                    "evidence_count": 0,
                    "needs_human_review": "",
                    "needs_retry": "",
                    "hard_conflicts": "",
                    "soft_conflicts": "",
                    "review_unknowns": "",
                    "acceptable_unknowns": "",
                    "certificate_identity_mismatches": "",
                    "error": str(exc),
                }
            )

    batch_json = output_root / "batch_summary.json"
    batch_csv = output_root / "batch_summary.csv"
    batch_md = output_root / "batch_summary.md"
    write_json(batch_json, batch_rows)
    write_csv(batch_csv, batch_rows)
    write_markdown(batch_md, title="v2 Batch Summary", rows=batch_rows)
    print(json.dumps({"mode": "run", "batch_summary": str(batch_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
