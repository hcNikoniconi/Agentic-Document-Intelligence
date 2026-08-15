#!/usr/bin/env python3
"""Run v2 VLM + Text Agent for one candidate folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_doc_intel.pipelines.vlm_text_agent import run_candidate_folder  # noqa: E402


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


def resolve_candidate(candidate: str) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return candidate_path

    matches = [
        path
        for path in DEFAULT_INPUT_ROOT.iterdir()
        if path.is_dir() and candidate.lower() in path.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(f"No candidate folder matched: {candidate}")
    if len(matches) > 1:
        names = "\n".join(f"- {path}" for path in matches)
        raise ValueError(f"Multiple candidate folders matched:\n{names}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="Candidate folder path or substring, e.g. Michelle")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Only build manifest/routing outputs; no model calls.")
    parser.add_argument(
        "--deterministic-aggregate",
        action="store_true",
        help="Skip text-model aggregation and use a simple first-evidence aggregate.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    candidate_dir = resolve_candidate(args.candidate)
    result = run_candidate_folder(
        candidate_dir,
        dry_run=args.dry_run,
        use_text_model_aggregator=not args.deterministic_aggregate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
