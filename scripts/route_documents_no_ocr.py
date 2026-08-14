#!/usr/bin/env python3
"""Generate a no-OCR routing report for local candidate documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_doc_intel.routing.reading_policy import (  # noqa: E402
    TextLayerStats,
    choose_reading_policy,
    stats_from_pdf_text_layer,
)


PROJECT_ROOT = ROOT.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "documents-export-2026-01-30"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "routing-reports" / "no_ocr_routing_report.md"
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def stats_for_file(path: Path) -> TextLayerStats:
    if path.suffix.lower() == ".pdf":
        return stats_from_pdf_text_layer(path)
    return TextLayerStats(page_count=1, text_chars=0, word_count=0, table_signal=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", default=None)
    args = parser.parse_args()

    candidate_dirs = [path for path in args.input_root.iterdir() if path.is_dir()]
    candidate_dirs.sort(key=lambda path: path.name.lower())
    if args.candidate:
        lowered = args.candidate.lower()
        candidate_dirs = [path for path in candidate_dirs if lowered in path.name.lower()]

    rows: list[dict] = []
    for candidate_dir in candidate_dirs:
        files = [
            path
            for path in candidate_dir.iterdir()
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        files.sort(key=lambda path: path.name.lower())
        for file_path in files:
            stats = stats_for_file(file_path)
            decision = choose_reading_policy(file_name=file_path.name, stats=stats)
            rows.append(
                {
                    "candidate": candidate_dir.name,
                    "file": file_path.name,
                    "stats": stats.__dict__,
                    "decision": decision.to_dict(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-OCR Routing Report",
        "",
        "This report does not use OCR. It decides between PDF text layer, VLM page reading, and human review.",
        "",
        "| Candidate | File | Pages | Text chars | Doc type | Primary tool | Auxiliary tools | Pages | Confidence | Reason |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        decision = row["decision"]
        stats = row["stats"]
        selected_pages = decision["selected_pages"]
        if isinstance(selected_pages, list):
            selected_pages_text = ", ".join(str(page) for page in selected_pages)
        else:
            selected_pages_text = selected_pages
        lines.append(
            "| "
            f"{row['candidate']} | "
            f"{row['file']} | "
            f"{stats['page_count']} | "
            f"{stats['text_chars']} | "
            f"{decision['doc_type']} | "
            f"{decision['primary_tool']} | "
            f"{', '.join(decision['auxiliary_tools']) or '-'} | "
            f"{selected_pages_text} | "
            f"{decision['confidence']} | "
            f"{'; '.join(decision['reasons'])} |"
        )

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
