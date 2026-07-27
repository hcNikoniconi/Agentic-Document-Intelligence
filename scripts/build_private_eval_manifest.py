"""Build a local, ignored benchmark manifest from private applicant folders.

The generated manifest intentionally lives under evals/private/ and must not be
committed. It stores local file paths so evaluation scripts can run against the
private documents on this machine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_OUTPUT = Path("evals/private/cases.local.jsonl")


SUPPORTED_EXTENSIONS = {
    ".pdf",
}


IGNORE_KEYWORDS = {
    "photo",
    "personal statement",
    "recommendation",
    "reference letter",
    "conditional offer",
    "unconditional offer",
    "scholarship",
    "cv",
    "syllabus",
    "supporting document",
    "grading",
}


def classify_document(filename: str) -> tuple[str | None, str]:
    name = filename.lower()

    if any(keyword in name for keyword in IGNORE_KEYWORDS):
        return None, "ignored_by_filename"

    if "application form" in name:
        return "application_form", "filename_contains_application_form"

    if "passport" in name:
        return "passport", "filename_contains_passport"

    if any(keyword in name for keyword in ("ielts", "toefl", "pte", "duolingo", "det")):
        return "english_language", "filename_contains_english_exam"

    if any(keyword in name for keyword in ("transcript", "school report", "academic transcript", "final report", "bachelor transcript")):
        return "transcript", "filename_contains_transcript_signal"

    if any(keyword in name for keyword in ("certificate", "diploma", "ijazah", "pre graduation")):
        return "diploma_certificate", "filename_contains_certificate_signal"

    return None, "unclassified"


def iter_candidate_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def build_cases(data_root: Path) -> tuple[list[dict], dict]:
    cases = []
    ignored = []
    per_candidate = defaultdict(Counter)

    for candidate_index, candidate_dir in enumerate(iter_candidate_dirs(data_root), start=1):
        candidate_id = f"candidate_{candidate_index:03d}"

        for file_path in sorted(candidate_dir.iterdir()):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            doc_type, reason = classify_document(file_path.name)

            if suffix not in SUPPORTED_EXTENSIONS or doc_type is None:
                ignored.append(
                    {
                        "candidate_id": candidate_id,
                        "file_path": str(file_path),
                        "reason": reason if suffix in SUPPORTED_EXTENSIONS else "unsupported_file_type",
                    }
                )
                continue

            per_candidate[candidate_id][doc_type] += 1
            doc_index = per_candidate[candidate_id][doc_type]
            case_id = f"{candidate_id}_{doc_type}_{doc_index:02d}"

            cases.append(
                {
                    "id": case_id,
                    "candidate_id": candidate_id,
                    "document_type": doc_type,
                    "file_path": str(file_path),
                    "classification_reason": reason,
                    "ground_truth": {},
                    "status": "needs_ground_truth",
                }
            )

    summary = {
        "candidate_count": len(iter_candidate_dirs(data_root)),
        "case_count": len(cases),
        "ignored_count": len(ignored),
        "document_type_counts": Counter(case["document_type"] for case in cases),
        "ignored_reason_counts": Counter(item["reason"] for item in ignored),
    }

    return cases, summary


def write_manifest(cases: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")


def print_summary(summary: dict, output_path: Path) -> None:
    print(f"Manifest: {output_path}")
    print(f"Candidates: {summary['candidate_count']}")
    print(f"Benchmark cases: {summary['case_count']}")
    print(f"Ignored files: {summary['ignored_count']}")
    print("Document types:")
    for doc_type, count in sorted(summary["document_type_counts"].items()):
        print(f"  {doc_type}: {count}")
    print("Ignored reasons:")
    for reason, count in sorted(summary["ignored_reason_counts"].items()):
        print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    cases, summary = build_cases(data_root)
    write_manifest(cases, args.output)
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()
