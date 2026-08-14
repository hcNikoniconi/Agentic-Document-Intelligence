#!/usr/bin/env python3
"""Build a human-readable v2 candidate summary report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    Path(__file__).resolve().parents[1]
    / "output"
    / "v2_vlm_text_agent"
    / "UG26001270_Michelle Nicole Pramudji"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def basename(path: str) -> str:
    return Path(path).name


def evidence_count_by_file(evidence: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for item in evidence:
        counter[basename(item["source_file"])] += 1
    return counter


def evidence_by_doc_and_field(evidence: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        grouped[(item["document_type"], item["field"])].append(item)
    return grouped


def value_is_unknown(value: Any) -> bool:
    return not str(value or "").strip() or str(value).strip().lower() == "unknown"


def compact(value: Any, max_len: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def first_supporting_evidence(
    grouped_evidence: dict[tuple[str, str], list[dict[str, Any]]],
    doc_type: str,
    field: str,
    final_value: Any,
) -> dict[str, Any] | None:
    items = grouped_evidence.get((doc_type, field), [])
    if not items:
        return None

    final_norm = compact(final_value).lower()
    non_unknown = [item for item in items if not value_is_unknown(item.get("value"))]
    if not non_unknown:
        return items[0]

    for item in non_unknown:
        if compact(item.get("value")).lower() == final_norm:
            return item
    return non_unknown[0]


def build_report(run_dir: Path) -> Path:
    manifest = load_json(run_dir / "document_manifest.json")
    evidence = load_jsonl(run_dir / "document_evidence.jsonl")
    result = load_json(run_dir / "candidate_result.json")
    verification = load_json(run_dir / "verification_report.json")
    trace = load_json(run_dir / "trace.json")

    evidence_counter = evidence_count_by_file(evidence)
    grouped_evidence = evidence_by_doc_and_field(evidence)

    candidate = manifest[0]["candidate"] if manifest else run_dir.name
    started = trace[0]["timestamp"] if trace else None
    ended = trace[-1]["timestamp"] if trace else None
    elapsed = round(ended - started, 2) if started and ended else "unknown"

    lines: list[str] = [
        f"# v2 Candidate Summary: {candidate}",
        "",
        "## Run Overview",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Files scanned: `{len(manifest)}`",
        f"- Evidence items: `{len(evidence)}`",
        f"- Elapsed seconds: `{elapsed}`",
        f"- Needs retry: `{verification.get('needs_retry')}`",
        "",
        "## Files Processed",
        "",
        "| File | Routed type | Reader | Planned pages | Evidence count | Status |",
        "|---|---|---|---|---:|---|",
    ]

    trace_by_file = {}
    for item in trace:
        if item.get("step") == "read_file":
            data = item.get("data") or {}
            if data.get("file"):
                trace_by_file[data["file"]] = item

    for item in manifest:
        decision = item["reading_decision"]
        planned_pages = decision.get("planned_initial_pages", decision.get("selected_pages", "all"))
        if isinstance(planned_pages, list):
            planned_pages_text = ", ".join(str(page) for page in planned_pages)
        else:
            planned_pages_text = str(planned_pages)
        file_name = item["file_name"]
        trace_item = trace_by_file.get(file_name, {})
        status = trace_item.get("status", item.get("status", "planned"))
        lines.append(
            "| "
            f"{file_name} | "
            f"{item['routed_doc_type']} | "
            f"{decision['primary_tool']} | "
            f"{planned_pages_text} | "
            f"{evidence_counter[file_name]} | "
            f"{status} |"
        )

    lines.extend(
        [
            "",
            "## Final Result With Evidence",
            "",
        ]
    )

    for doc_type, fields in result.items():
        lines.extend([f"### {doc_type}", ""])
        if not isinstance(fields, dict):
            lines.extend(["Invalid result section.", ""])
            continue
        lines.extend(
            [
                "| Field | Final value | Source file | Page | Evidence |",
                "|---|---|---|---|---|",
            ]
        )
        for field, value in fields.items():
            support = first_supporting_evidence(grouped_evidence, doc_type, field, value)
            if support:
                source = basename(support.get("source_file", ""))
                page = support.get("page", "unknown")
                evidence_text = compact(support.get("evidence", ""))
            else:
                source = "-"
                page = "-"
                evidence_text = ""
            lines.append(
                "| "
                f"{field} | "
                f"{compact(value)} | "
                f"{source} | "
                f"{page} | "
                f"{evidence_text} |"
            )
        lines.append("")

    lines.extend(["## Verification", ""])

    unknown_fields = verification.get("unknown_fields") or verification.get("missing_fields") or []
    hard_conflicts = verification.get("hard_conflicts", [])
    soft_conflicts = verification.get("soft_conflicts", [])
    acceptable_unknowns = verification.get("acceptable_unknowns", [])
    review_unknowns = verification.get("review_unknowns", [])
    weak_supported_fields = verification.get("weak_supported_fields", [])
    unsupported_needing_review = verification.get("unsupported_needing_review", [])
    lines.extend(
        [
            f"- Unknown fields: `{len(unknown_fields)}`",
            f"- Acceptable unknowns: `{len(acceptable_unknowns)}`",
            f"- Review unknowns: `{len(review_unknowns)}`",
            f"- Weakly supported fields: `{len(weak_supported_fields)}`",
            f"- Unsupported fields needing review: `{len(unsupported_needing_review)}`",
            f"- Hard conflicts: `{len(hard_conflicts)}`",
            f"- Soft conflicts: `{len(soft_conflicts)}`",
            f"- Certificate identity mismatches: `{len(verification.get('certificate_identity_mismatches', []))}`",
            f"- Needs human review: `{verification.get('needs_human_review')}`",
            "",
        ]
    )

    if acceptable_unknowns:
        lines.extend(["### Acceptable Unknowns", ""])
        for item in acceptable_unknowns:
            lines.append(f"- `{item['label']}`: {item.get('reason', '')}")
        lines.append("")

    if review_unknowns:
        lines.extend(["### Unknowns Needing Review", ""])
        for item in review_unknowns:
            label = item.get("label", f"{item['document_type']}.{item['field']}")
            lines.append(f"- `{label}`: {item.get('reason', '')}")
        lines.append("")

    if hard_conflicts:
        lines.extend(["### Hard Conflicts", ""])
        for item in hard_conflicts:
            values = ", ".join(f"`{value}`" for value in item.get("values", []))
            lines.append(f"- `{item['document_type']}.{item['field']}`: {values}. {item.get('reason', '')}")
        lines.append("")

    if soft_conflicts:
        lines.extend(["### Soft Conflicts / Explainable Differences", ""])
        for item in soft_conflicts:
            values = ", ".join(f"`{value}`" for value in item.get("values", []))
            lines.append(f"- `{item['document_type']}.{item['field']}`: {values}. {item.get('reason', '')}")
        lines.append("")

    if weak_supported_fields:
        lines.extend(["### Weakly Supported Fields", ""])
        for item in weak_supported_fields:
            lines.append(f"- `{item['document_type']}.{item['field']}` = `{item.get('value')}`. {item.get('reason', '')}")
        lines.append("")

    if unsupported_needing_review:
        lines.extend(["### Unsupported Fields Needing Review", ""])
        for item in unsupported_needing_review:
            lines.append(f"- `{item['document_type']}.{item['field']}` = `{item.get('value')}`. {item.get('reason', '')}")
        lines.append("")

    if verification.get("certificate_identity_checks"):
        lines.extend(["### Certificate Identity Checks", ""])
        lines.extend(["| Source file | Certificate name | Passed | Evidence |", "|---|---|---|---|"])
        for item in verification["certificate_identity_checks"]:
            lines.append(
                "| "
                f"{basename(item['source_file'])} | "
                f"{item['certificate_student_name']} | "
                f"{item['passed']} | "
                f"{compact(item.get('evidence', ''))} |"
            )
        lines.append("")

    if verification.get("source_consistency_checks"):
        lines.extend(["### Source Consistency Checks", ""])
        lines.extend(
            [
                "| Document type | Left source | Right source | Compatible | Shared signals | Reason |",
                "|---|---|---|---|---|---|",
            ]
        )
        for item in verification["source_consistency_checks"]:
            shared = "; ".join(
                f"{signal['field']}: {signal['left']} ≈ {signal['right']}"
                for signal in item.get("shared_signals", [])
            )
            lines.append(
                "| "
                f"{item['document_type']} | "
                f"{basename(item['left_source_file'])} | "
                f"{basename(item['right_source_file'])} | "
                f"{item['compatible']} | "
                f"{compact(shared, 220)} | "
                f"{item.get('reason', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommended Next Action",
            "",
        ]
    )
    if verification.get("certificate_identity_mismatches"):
        lines.append("- Certificate identity mismatch exists. Human review or targeted reread is needed.")
    elif hard_conflicts:
        lines.append("- Hard conflicts remain. Do not benchmark this candidate until reviewed.")
    elif review_unknowns or unsupported_needing_review:
        lines.append("- Some fields need review, but the extraction pipeline itself ran successfully.")
    elif soft_conflicts:
        lines.append("- Only soft conflicts remain. These are explainable source differences, not immediate extraction failures.")
    elif acceptable_unknowns:
        lines.append("- Only acceptable unknowns remain. Result can proceed to benchmark if the benchmark policy ignores unavailable fields.")
    else:
        lines.append("- Result is ready for benchmark comparison.")

    report_path = run_dir / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    report_path = build_report(args.run_dir)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
