"""Build a local, ignored candidate-level benchmark manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_OUTPUT = Path("evals/private/candidates.local.jsonl")


def make_candidate_id(index: int) -> str:
    return f"candidate_{index:03d}"


def parse_application_id(folder_name: str) -> str:
    match = re.search(r"UG\d{6,}", folder_name, flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def parse_display_name(folder_name: str) -> str:
    text = re.sub(r"\bUG\d{6,}\b", "", folder_name, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_candidates(data_root: Path) -> list[dict]:
    rows = []
    candidate_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())

    for index, folder in enumerate(candidate_dirs, start=1):
        rows.append(
            {
                "id": make_candidate_id(index),
                "folder_path": str(folder),
                "source_folder_name": folder.name,
                "application_id": parse_application_id(folder.name),
                "display_name": parse_display_name(folder.name),
                "ground_truth": {},
                "status": "needs_ground_truth",
            }
        )

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    rows = build_candidates(data_root)
    write_jsonl(args.output, rows)

    print(f"Manifest: {args.output}")
    print(f"Candidates: {len(rows)}")


if __name__ == "__main__":
    main()
