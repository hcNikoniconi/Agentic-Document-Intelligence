"""v2 pipeline: candidate folder -> evidence -> aggregate result.

This version treats the applicant folder as the unit of work. It does not drop
duplicate document types. Every relevant file can contribute evidence, then a
text model aggregates evidence into the final candidate result.

Current active readers:
- PDF text layer reader: cheap local text source for text-heavy PDFs.
- VLM page reader: visual reader for passports, certificates, transcripts, and
  selected application-form pages.

Traditional OCR is not part of this active pipeline. OCR-VL outputs can be used
later as a fallback/cache tool, but the first milestone keeps the product flow
focused on text layer + VLM.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_doc_intel.document_rendering import image_file_to_data_url
from agentic_doc_intel.model_client import chat_completion_content
from agentic_doc_intel.pipelines.vlm_direct import (
    build_image_content_item,
    extract_json_object,
)
from agentic_doc_intel.routing.reading_policy import (
    ReadingDecision,
    TextLayerStats,
    choose_reading_policy,
    stats_from_pdf_text_layer,
)
from agentic_doc_intel.template_registry import (
    DOC_TYPE_ORDER,
    build_json_skeleton,
    choose_template_path,
    load_template,
    parse_field_meta,
)
from agentic_doc_intel.verification.policy import apply_verification_policy


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "vlm-text-agent" / "output" / "v2_vlm_text_agent"
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class DocumentManifestEntry:
    candidate: str
    source_file: str
    file_name: str
    suffix: str
    template_doc_type: str | None
    routed_doc_type: str
    text_layer_stats: dict[str, Any]
    reading_decision: dict[str, Any]
    status: str = "planned"
    error: str | None = None


@dataclass
class DocumentEvidence:
    candidate: str
    document_type: str
    source_file: str
    page: int | str
    field: str
    value: str
    evidence: str
    reader: str
    confidence: float | None = None
    notes: str | None = None


@dataclass
class PipelineTraceEvent:
    step: str
    status: str
    message: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-() \u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:180] or "unnamed"


def candidate_name_from_dir(input_dir: Path) -> str:
    return input_dir.name


def build_output_base_name(input_dir: Path, requested_output_base_name: str | None) -> str:
    return safe_name(requested_output_base_name or input_dir.name)


def load_template_for_doc_type(doc_type: str) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    template_dir = Path(__file__).resolve().parents[3] / "templates"
    for template_path in sorted(template_dir.glob("*.json")):
        template = load_template(template_path)
        if template.get("doc_type") == doc_type:
            return template, parse_field_meta(template)
    return None, []


def template_doc_type_for_file(path: Path) -> str | None:
    template_path = choose_template_path(path)
    if template_path is None:
        return None
    return str(load_template(template_path).get("doc_type") or "unknown")


def iter_candidate_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    files.sort(key=lambda path: path.name.lower())
    return files


def stats_for_file(path: Path) -> TextLayerStats:
    if path.suffix.lower() == ".pdf":
        return stats_from_pdf_text_layer(path)
    return TextLayerStats(page_count=1, text_chars=0, word_count=0, table_signal=0)


def build_manifest(input_dir: Path, candidate: str) -> list[DocumentManifestEntry]:
    manifest: list[DocumentManifestEntry] = []
    for file_path in iter_candidate_files(input_dir):
        try:
            stats = stats_for_file(file_path)
            decision = choose_reading_policy(file_name=file_path.name, stats=stats)
            template_doc_type = template_doc_type_for_file(file_path)
            decision_dict = decision.to_dict()
            planning_doc_type = template_doc_type or decision.doc_type
            if decision.primary_tool == "vlm_page_reader" and planning_doc_type in set(DOC_TYPE_ORDER):
                _, field_meta = load_template_for_doc_type(planning_doc_type)
                page_count = stats.page_count or 1
                fallback_pages = normalize_selected_pages(decision.selected_pages, page_count=page_count)
                planned_pages = select_pages_from_text_layer(
                    file_path=file_path,
                    doc_type=planning_doc_type,
                    field_meta=field_meta,
                    fallback_pages=fallback_pages,
                    max_pages=int(os.getenv("V2_MAX_INITIAL_VLM_PAGES", "4")),
                )
                decision_dict["planned_initial_pages"] = planned_pages
                decision_dict["adaptive_retry"] = {
                    "enabled": True,
                    "condition": "retry additional pages when required fields remain unknown after initial read",
                    "max_retry_pages": int(os.getenv("V2_MAX_RETRY_VLM_PAGES", "4")),
                }
            manifest.append(
                DocumentManifestEntry(
                    candidate=candidate,
                    source_file=str(file_path),
                    file_name=file_path.name,
                    suffix=file_path.suffix.lower(),
                    template_doc_type=template_doc_type,
                    routed_doc_type=decision.doc_type,
                    text_layer_stats=asdict(stats),
                    reading_decision=decision_dict,
                )
            )
        except Exception as exc:
            manifest.append(
                DocumentManifestEntry(
                    candidate=candidate,
                    source_file=str(file_path),
                    file_name=file_path.name,
                    suffix=file_path.suffix.lower(),
                    template_doc_type=template_doc_type_for_file(file_path),
                    routed_doc_type="unknown",
                    text_layer_stats=asdict(TextLayerStats()),
                    reading_decision=ReadingDecision(
                        doc_type="unknown",
                        primary_tool="human_review",
                        confidence="low",
                        reasons=[f"failed to inspect file: {exc}"],
                    ).to_dict(),
                    status="failed_to_plan",
                    error=str(exc),
                )
            )
    return manifest


def extract_pdf_text_pages(path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF text-layer reading.") from exc

    pages: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            pages.append(
                {
                    "page": page_index,
                    "text": (page.get_text("text") or "").strip(),
                }
            )
    return pages


def pdf_page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF page counting.") from exc

    with fitz.open(path) as document:
        return len(document)


def normalize_selected_pages(selected_pages: list[int] | str, *, page_count: int) -> list[int] | str:
    if selected_pages == "all":
        return "all"
    pages = []
    for page in selected_pages:
        if 1 <= int(page) <= page_count and int(page) not in pages:
            pages.append(int(page))
    return pages or ([1] if page_count >= 1 else [])


APPLICATION_FORM_PAGE_KEYWORDS = {
    "application id": ["application id", "submitted date", "proposed programme", "programme choice"],
    "name": ["personal details", "first name", "family name", "passport"],
    "date of birth": ["date of birth", "country of birth", "nationality"],
    "gender": ["gender", "marital status", "occupation"],
    "nationality": ["nationality", "father's nationality", "mother's nationality"],
    "passport number": ["passport number", "passport no"],
    "military, or a diplomatic passport?": ["military", "diplomatic passport"],
    "father's nationality": ["father's nationality"],
    "mother's nationality": ["mother's nationality"],
    "Are you currently studying or living in Mainland China?": ["mainland china", "studying or living"],
    "Have you ever been convicted of a crime?": ["criminal convictions", "convicted of a crime"],
    "work experience": ["work experience", "employer", "employment"],
}


def select_pages_from_text_layer(
    *,
    file_path: Path,
    doc_type: str,
    field_meta: list[dict[str, str]],
    fallback_pages: list[int] | str,
    max_pages: int,
) -> list[int] | str:
    if file_path.suffix.lower() != ".pdf":
        return fallback_pages
    if doc_type != "application_form":
        return fallback_pages

    pages = extract_pdf_text_pages(file_path)
    page_scores: list[tuple[int, int]] = []
    target_keywords: list[str] = []
    for item in field_meta:
        field_name = item["name"]
        target_keywords.extend(APPLICATION_FORM_PAGE_KEYWORDS.get(field_name, [field_name]))

    for page in pages:
        text = re.sub(r"\s+", " ", page["text"].lower())
        score = 0
        for keyword in target_keywords:
            if keyword.lower() in text:
                score += 1
        page_scores.append((page["page"], score))

    selected = [page for page, score in sorted(page_scores, key=lambda item: (-item[1], item[0])) if score > 0]
    if not selected:
        return fallback_pages

    # Always keep page 1 for application forms because headers/application IDs
    # are often important even if keyword scoring is noisy.
    if 1 not in selected:
        selected.insert(0, 1)
    selected = selected[:max_pages]
    return sorted(selected)


def render_pdf_selected_pages(path: Path, selected_pages: list[int] | str, *, zoom: float = 2.0) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for VLM PDF rendering.") from exc

    rendered: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        if selected_pages == "all":
            page_numbers = list(range(1, len(document) + 1))
        else:
            page_numbers = [page for page in selected_pages if 1 <= page <= len(document)]

        max_pages = int(os.getenv("V2_MAX_VLM_PAGES_PER_FILE", "8"))
        page_numbers = page_numbers[:max_pages]
        matrix = fitz.Matrix(zoom, zoom)
        for page_number in page_numbers:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            encoded = pixmap.tobytes("png")
            import base64

            data_url = f"data:image/png;base64,{base64.b64encode(encoded).decode('ascii')}"
            rendered.append((page_number, data_url))
    return rendered


def missing_fields_from_evidence(
    *,
    field_meta: list[dict[str, str]],
    evidence: list[DocumentEvidence],
) -> list[str]:
    found: set[str] = set()
    for item in evidence:
        if item.value.strip() and item.value.strip().lower() != "unknown":
            found.add(item.field)
    return [item["name"] for item in field_meta if item["name"] not in found]


def additional_pages_for_retry(
    *,
    file_path: Path,
    selected_pages: list[int] | str,
    max_extra_pages: int,
) -> list[int]:
    if file_path.suffix.lower() != ".pdf" or selected_pages == "all":
        return []
    page_count = pdf_page_count(file_path)
    already = set(selected_pages)
    remaining = [page for page in range(1, page_count + 1) if page not in already]
    return remaining[:max_extra_pages]


def evidence_prompt(
    *,
    candidate: str,
    document_type: str,
    source_file: str,
    field_meta: list[dict[str, str]],
    reader: str,
) -> str:
    field_names = [item["name"] for item in field_meta]
    return "\n".join(
        [
            "You are extracting field-level evidence from an applicant document.",
            "Return JSON only. Do not output markdown or explanations.",
            "Every non-unknown value must include short evidence copied from the source.",
            "If a field is not visible or cannot be confirmed, set value to \"unknown\" and evidence to \"\".",
            "Do not invent values.",
            "",
            f"Candidate folder: {candidate}",
            f"Document type: {document_type}",
            f"Source file: {source_file}",
            f"Reader: {reader}",
            "",
            "Fields to extract:",
            *[f"- {item['name']}: {item.get('note', '')}".rstrip() for item in field_meta],
            "",
            "Return this exact JSON shape:",
            json.dumps(
                {
                    "items": [
                        {
                            "field": field_names[0] if field_names else "field name",
                            "value": "unknown",
                            "evidence": "",
                            "page": "unknown",
                            "confidence": 0.0,
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )


def parse_evidence_items(raw_output: str) -> list[dict[str, Any]]:
    parsed = extract_json_object(raw_output)
    if isinstance(parsed.get("items"), list):
        return parsed["items"]
    return []


def normalize_evidence_item(
    *,
    candidate: str,
    document_type: str,
    source_file: str,
    reader: str,
    raw_item: dict[str, Any],
) -> DocumentEvidence:
    value = raw_item.get("value", "unknown")
    if value is None or str(value).strip() == "":
        value = "unknown"
    confidence = raw_item.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    return DocumentEvidence(
        candidate=candidate,
        document_type=document_type,
        source_file=source_file,
        page=raw_item.get("page", "unknown"),
        field=str(raw_item.get("field", "")).strip(),
        value=str(value).strip(),
        evidence=str(raw_item.get("evidence", "") or "").strip(),
        reader=reader,
        confidence=confidence_value,
    )


def extract_evidence_with_pdf_text(
    *,
    candidate: str,
    file_path: Path,
    document_type: str,
    field_meta: list[dict[str, str]],
) -> tuple[list[DocumentEvidence], str]:
    pages = extract_pdf_text_pages(file_path)
    text = "\n\n".join(f"# Page {page['page']}\n{page['text']}" for page in pages)
    prompt = evidence_prompt(
        candidate=candidate,
        document_type=document_type,
        source_file=file_path.name,
        field_meta=field_meta,
        reader="pdf_text_layer",
    )
    raw_output = chat_completion_content(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\nSource text:\n{text}",
            }
        ],
        temperature=0,
        max_tokens=int(os.getenv("V2_TEXT_READER_MAX_TOKENS", "4096")),
    )
    return [
        normalize_evidence_item(
            candidate=candidate,
            document_type=document_type,
            source_file=str(file_path),
            reader="pdf_text_layer",
            raw_item=item,
        )
        for item in parse_evidence_items(raw_output)
    ], raw_output


def extract_evidence_with_vlm(
    *,
    candidate: str,
    file_path: Path,
    document_type: str,
    field_meta: list[dict[str, str]],
    selected_pages: list[int] | str,
) -> tuple[list[DocumentEvidence], str]:
    prompt = evidence_prompt(
        candidate=candidate,
        document_type=document_type,
        source_file=file_path.name,
        field_meta=field_meta,
        reader="vlm_page_reader",
    )

    if file_path.suffix.lower() == ".pdf":
        rendered_pages = render_pdf_selected_pages(file_path, selected_pages)
    else:
        rendered_pages = [(1, image_file_to_data_url(file_path))]

    page_note = "Rendered pages: " + ", ".join(str(page_number) for page_number, _ in rendered_pages)
    content: list[dict[str, Any]] = [{"type": "text", "text": f"{prompt}\n\n{page_note}"}]
    for page_number, data_url in rendered_pages:
        content.append({"type": "text", "text": f"Page {page_number}"})
        content.append(build_image_content_item(data_url))

    raw_output = chat_completion_content(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        max_tokens=int(os.getenv("V2_VLM_READER_MAX_TOKENS", "4096")),
    )
    return [
        normalize_evidence_item(
            candidate=candidate,
            document_type=document_type,
            source_file=str(file_path),
            reader="vlm_page_reader",
            raw_item=item,
        )
        for item in parse_evidence_items(raw_output)
    ], raw_output


def reader_doc_type(manifest_entry: DocumentManifestEntry) -> str | None:
    doc_type = manifest_entry.template_doc_type or manifest_entry.routed_doc_type
    if doc_type in set(DOC_TYPE_ORDER):
        return doc_type
    return None


def extract_file_evidence(
    *,
    manifest_entry: DocumentManifestEntry,
    candidate: str,
) -> tuple[list[DocumentEvidence], dict[str, Any]]:
    file_path = Path(manifest_entry.source_file)
    decision = manifest_entry.reading_decision
    primary_tool = decision["primary_tool"]
    doc_type = reader_doc_type(manifest_entry)
    raw_outputs: dict[str, Any] = {}

    if doc_type is None:
        return [], {"skipped": "no matching extraction template"}

    _, field_meta = load_template_for_doc_type(doc_type)
    if not field_meta:
        return [], {"skipped": f"no field template for doc_type={doc_type}"}

    if primary_tool == "human_review":
        return [], {"skipped": "human_review"}

    if primary_tool == "pdf_text_layer":
        evidence, raw_output = extract_evidence_with_pdf_text(
            candidate=candidate,
            file_path=file_path,
            document_type=doc_type,
            field_meta=field_meta,
        )
        raw_outputs["pdf_text_layer"] = raw_output
        return evidence, raw_outputs

    if primary_tool == "vlm_page_reader":
        page_count = pdf_page_count(file_path) if file_path.suffix.lower() == ".pdf" else 1
        fallback_selected_pages = normalize_selected_pages(
            decision.get("selected_pages", "all"),
            page_count=page_count,
        )
        max_initial_pages = int(os.getenv("V2_MAX_INITIAL_VLM_PAGES", "4"))
        selected_pages = select_pages_from_text_layer(
            file_path=file_path,
            doc_type=doc_type,
            field_meta=field_meta,
            fallback_pages=fallback_selected_pages,
            max_pages=max_initial_pages,
        )
        evidence, raw_output = extract_evidence_with_vlm(
            candidate=candidate,
            file_path=file_path,
            document_type=doc_type,
            field_meta=field_meta,
            selected_pages=selected_pages,
        )
        missing_fields = missing_fields_from_evidence(field_meta=field_meta, evidence=evidence)
        raw_outputs["vlm_page_reader_initial"] = {
            "selected_pages": selected_pages,
            "missing_fields_after_read": missing_fields,
            "raw_output": raw_output,
        }

        retry_threshold = int(os.getenv("V2_RETRY_IF_MISSING_FIELD_COUNT_AT_LEAST", "1"))
        if missing_fields and len(missing_fields) >= retry_threshold:
            retry_pages = additional_pages_for_retry(
                file_path=file_path,
                selected_pages=selected_pages,
                max_extra_pages=int(os.getenv("V2_MAX_RETRY_VLM_PAGES", "4")),
            )
            if retry_pages:
                retry_evidence, retry_raw_output = extract_evidence_with_vlm(
                    candidate=candidate,
                    file_path=file_path,
                    document_type=doc_type,
                    field_meta=field_meta,
                    selected_pages=retry_pages,
                )
                evidence.extend(retry_evidence)
                raw_outputs["vlm_page_reader_retry"] = {
                    "selected_pages": retry_pages,
                    "missing_fields_before_retry": missing_fields,
                    "raw_output": retry_raw_output,
                }
        return evidence, raw_outputs

    return [], {"skipped": f"unsupported primary tool: {primary_tool}"}


def build_candidate_result_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for doc_type in DOC_TYPE_ORDER:
        _, field_meta = load_template_for_doc_type(doc_type)
        schema[doc_type] = build_json_skeleton(field_meta)
    return schema


def aggregate_evidence_with_text_model(
    *,
    candidate: str,
    evidence: list[DocumentEvidence],
) -> tuple[dict[str, Any], str]:
    schema = build_candidate_result_schema()
    evidence_payload = [asdict(item) for item in evidence]
    prompt = "\n".join(
        [
            "You are aggregating extracted evidence for one applicant.",
            "Return JSON only. Do not output markdown or explanations.",
            "Use the exact output schema.",
            "For each field, choose the best supported value from evidence.",
            "If no reliable evidence exists, return \"unknown\".",
            "Normalize obvious case and date formatting, but do not invent values.",
            "If evidence conflicts, choose the value with stronger source support and mention the conflict in _verification.",
            "",
            f"Candidate: {candidate}",
            "",
            "Output schema:",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "Evidence:",
            json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        ]
    )

    raw_output = chat_completion_content(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=int(os.getenv("V2_AGGREGATOR_MAX_TOKENS", "4096")),
    )
    return repair_candidate_result_schema(extract_json_object(raw_output)), raw_output


def normalize_schema_key(key: str) -> str:
    normalized = str(key).strip().lower()
    normalized = normalized.rstrip(":：").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def repair_candidate_result_schema(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Force model output back into the exact benchmark/template schema.

    Aggregators sometimes return visually similar keys such as
    "passport number:" instead of "passport number". This repair step keeps the
    final result compatible with benchmark scripts and avoids fake errors.
    """

    schema = build_candidate_result_schema()
    repaired: dict[str, Any] = {}
    for doc_type, expected_fields in schema.items():
        raw_fields = raw_result.get(doc_type, {})
        if not isinstance(raw_fields, dict):
            raw_fields = {}

        raw_by_normalized_key = {
            normalize_schema_key(raw_key): value for raw_key, value in raw_fields.items()
        }
        repaired[doc_type] = {}
        for expected_key in expected_fields:
            value = raw_by_normalized_key.get(normalize_schema_key(expected_key), "unknown")
            if value is None or str(value).strip() == "":
                value = "unknown"
            repaired[doc_type][expected_key] = value

    return repaired


def deterministic_aggregate(evidence: list[DocumentEvidence]) -> dict[str, Any]:
    result = build_candidate_result_schema()
    for item in evidence:
        if item.document_type not in result:
            continue
        if item.field not in result[item.document_type]:
            continue
        current = str(result[item.document_type].get(item.field, "") or "").strip().lower()
        value = item.value.strip()
        if not value or value.lower() == "unknown":
            if not current:
                result[item.document_type][item.field] = "unknown"
            continue
        if not current or current == "unknown":
            result[item.document_type][item.field] = value
    for doc_type, fields in result.items():
        for field_name, value in fields.items():
            if not str(value).strip():
                fields[field_name] = "unknown"
    return result


def evidence_by_source(evidence: list[DocumentEvidence]) -> dict[tuple[str, str], list[DocumentEvidence]]:
    grouped: dict[tuple[str, str], list[DocumentEvidence]] = {}
    for item in evidence:
        grouped.setdefault((item.document_type, item.source_file), []).append(item)
    return grouped


def source_field_value(items: list[DocumentEvidence], field: str) -> str | None:
    for item in items:
        if item.field == field and item.value and item.value.lower() != "unknown":
            return item.value
    return None


def values_look_equivalent(left: str, right: str) -> bool:
    left_norm = normalized_identity(left)
    right_norm = normalized_identity(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def build_source_consistency_checks(evidence: list[DocumentEvidence]) -> list[dict[str, Any]]:
    """Check whether files of the same doc_type can be treated as complementary."""

    checks: list[dict[str, Any]] = []
    grouped = evidence_by_source(evidence)
    doc_types = sorted({doc_type for doc_type, _ in grouped})

    for doc_type in doc_types:
        sources = sorted(source for source_doc_type, source in grouped if source_doc_type == doc_type)
        if len(sources) <= 1:
            continue

        for index, left_source in enumerate(sources):
            for right_source in sources[index + 1 :]:
                left_items = grouped[(doc_type, left_source)]
                right_items = grouped[(doc_type, right_source)]

                shared_signals: list[dict[str, str]] = []
                for field_name in ["student name", "name", "institution name", "institution country"]:
                    left_value = source_field_value(left_items, field_name)
                    right_value = source_field_value(right_items, field_name)
                    if left_value and right_value and values_look_equivalent(left_value, right_value):
                        shared_signals.append(
                            {
                                "field": field_name,
                                "left": left_value,
                                "right": right_value,
                            }
                        )

                compatible = bool(shared_signals)
                checks.append(
                    {
                        "document_type": doc_type,
                        "left_source_file": left_source,
                        "right_source_file": right_source,
                        "compatible": compatible,
                        "shared_signals": shared_signals,
                        "reason": (
                            "Files share identity/institution signals and can be used as complementary evidence."
                            if compatible
                            else "No shared identity/institution signal found; do not merge fields automatically."
                        ),
                    }
                )

    return checks


def evidence_can_complement_result(
    *,
    doc_type: str,
    candidate: DocumentEvidence,
    all_evidence: list[DocumentEvidence],
    consistency_checks: list[dict[str, Any]],
) -> bool:
    same_doc_type_sources = {
        item.source_file for item in all_evidence if item.document_type == doc_type
    }
    if len(same_doc_type_sources) <= 1:
        return True

    for check in consistency_checks:
        if check["document_type"] != doc_type:
            continue
        if not check["compatible"]:
            continue
        if candidate.source_file in {check["left_source_file"], check["right_source_file"]}:
            return True
    return False


def fill_unknowns_from_evidence(
    *,
    candidate_result: dict[str, Any],
    evidence: list[DocumentEvidence],
    consistency_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evidence-grounded repair for aggregator omissions.

    If the final result says unknown but the evidence layer contains a concrete
    value for the same doc_type.field, use the value only when the source is
    safe to merge. For multi-file doc types, sources must first pass a
    consistency check, e.g. same institution or same student identity.
    """

    consistency_checks = consistency_checks or build_source_consistency_checks(evidence)

    evidence_index: dict[tuple[str, str], list[DocumentEvidence]] = {}
    for item in evidence:
        if item.value and item.value.lower() != "unknown":
            evidence_index.setdefault((item.document_type, item.field), []).append(item)

    for doc_type, fields in candidate_result.items():
        if not isinstance(fields, dict):
            continue
        for field_name, value in fields.items():
            if str(value or "").strip().lower() != "unknown":
                continue

            candidates = evidence_index.get((doc_type, field_name), [])
            if not candidates:
                continue
            candidates = [
                item
                for item in candidates
                if evidence_can_complement_result(
                    doc_type=doc_type,
                    candidate=item,
                    all_evidence=evidence,
                    consistency_checks=consistency_checks,
                )
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    item.confidence if item.confidence is not None else 0.0,
                    bool(item.evidence),
                ),
                reverse=True,
            )
            fields[field_name] = candidates[0].value

    return candidate_result


def normalized_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def values_look_same_person(left: str, right: str) -> bool:
    left_norm = normalized_identity(left)
    right_norm = normalized_identity(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def reference_identity_names(evidence: list[DocumentEvidence]) -> list[str]:
    names: list[str] = []
    for item in evidence:
        if item.value.lower() == "unknown":
            continue
        if item.document_type == "passport" and item.field == "name":
            names.append(item.value)
        if item.document_type == "application_form" and item.field == "name":
            names.append(item.value)
    return names


def verify_certificate_identity(evidence: list[DocumentEvidence]) -> list[dict[str, Any]]:
    reference_names = reference_identity_names(evidence)
    checks: list[dict[str, Any]] = []
    if not reference_names:
        return checks

    for item in evidence:
        if item.document_type != "diploma_certificate":
            continue
        if item.field != "student name":
            continue
        if not item.value or item.value.lower() == "unknown":
            continue

        matched = any(values_look_same_person(item.value, reference) for reference in reference_names)
        checks.append(
            {
                "source_file": item.source_file,
                "certificate_student_name": item.value,
                "reference_names": reference_names,
                "passed": matched,
                "evidence": item.evidence,
            }
        )
    return checks


def verify_candidate_result(
    *,
    candidate_result: dict[str, Any],
    evidence: list[DocumentEvidence],
) -> dict[str, Any]:
    evidence_index: dict[tuple[str, str], list[DocumentEvidence]] = {}
    for item in evidence:
        evidence_index.setdefault((item.document_type, item.field), []).append(item)

    missing: list[dict[str, str]] = []
    unsupported: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    evidence_doc_types = {item.document_type for item in evidence}
    certificate_identity_checks = verify_certificate_identity(evidence)
    source_consistency_checks = build_source_consistency_checks(evidence)

    for doc_type in DOC_TYPE_ORDER:
        fields = candidate_result.get(doc_type, {})
        for field_name, value in fields.items():
            normalized_value = str(value or "").strip()
            items = evidence_index.get((doc_type, field_name), [])
            non_unknown_items = [item for item in items if item.value and item.value.lower() != "unknown"]
            distinct_values = sorted({item.value for item in non_unknown_items})

            if not normalized_value or normalized_value.lower() == "unknown":
                if doc_type in evidence_doc_types:
                    missing.append({"document_type": doc_type, "field": field_name})
                continue

            if not any(item.evidence for item in non_unknown_items):
                unsupported.append({"document_type": doc_type, "field": field_name, "value": normalized_value})

            if len(distinct_values) > 1:
                if doc_type == "diploma_certificate":
                    # Multiple certificates can legitimately have different
                    # schools, years, signatures, or stamp status. We validate
                    # certificate identity separately against passport/app form.
                    continue
                conflicts.append(
                    {
                        "document_type": doc_type,
                        "field": field_name,
                        "values": distinct_values,
                    }
                )

    base_report = {
        "unknown_fields": missing,
        "missing_fields": missing,
        "unsupported_fields": unsupported,
        "conflicts": conflicts,
        "certificate_identity_checks": certificate_identity_checks,
        "source_consistency_checks": source_consistency_checks,
        "certificate_identity_mismatches": [
            item for item in certificate_identity_checks if not item["passed"]
        ],
        "needs_retry": bool(
            unsupported
            or conflicts
            or [item for item in certificate_identity_checks if not item["passed"]]
        ),
    }
    return apply_verification_policy(base_report)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_combined_result(candidate_result: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for doc_type in DOC_TYPE_ORDER:
            file.write(f"#{doc_type}\n")
            fields = candidate_result.get(doc_type, {})
            if not fields:
                file.write("未提供或未识别到该类文件\n\n")
                continue
            for key, value in fields.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")


def run_candidate_folder(
    input_dir: str | Path,
    output_root: str | Path | None = None,
    output_base_name: str | None = None,
    *,
    dry_run: bool = False,
    use_text_model_aggregator: bool = True,
) -> dict[str, Any]:
    """Run v2 for one applicant folder.

    In dry-run mode, the pipeline only scans files and writes the routing
    manifest/trace. It does not call any model.
    """

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a valid applicant folder: {input_dir}")

    candidate = candidate_name_from_dir(input_dir)
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    run_dir = output_root / build_output_base_name(input_dir, output_base_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    trace: list[PipelineTraceEvent] = []
    trace.append(PipelineTraceEvent(step="start", status="ok", message=f"candidate={candidate}"))

    manifest = build_manifest(input_dir, candidate)
    save_json(run_dir / "document_manifest.json", [asdict(item) for item in manifest])
    trace.append(
        PipelineTraceEvent(
            step="manifest",
            status="ok",
            message=f"planned {len(manifest)} file(s)",
        )
    )

    if dry_run:
        empty_result = build_candidate_result_schema()
        verification_report = {
            "dry_run": True,
            "missing_fields": [],
            "unsupported_fields": [],
            "conflicts": [],
            "needs_retry": False,
        }
        save_jsonl(run_dir / "document_evidence.jsonl", [])
        save_json(run_dir / "candidate_result.json", empty_result)
        save_combined_result(empty_result, run_dir / "combined_result.txt")
        save_json(run_dir / "verification_report.json", verification_report)
        save_json(run_dir / "trace.json", [asdict(item) for item in trace])
        return {
            "pipeline": "v2_vlm_text_agent",
            "dry_run": True,
            "run_dir": str(run_dir),
            "document_manifest": str(run_dir / "document_manifest.json"),
            "document_evidence": str(run_dir / "document_evidence.jsonl"),
            "candidate_result": str(run_dir / "candidate_result.json"),
            "combined_result": str(run_dir / "combined_result.txt"),
            "verification_report": str(run_dir / "verification_report.json"),
            "trace": str(run_dir / "trace.json"),
            "logs": [item.message for item in trace],
        }

    evidence: list[DocumentEvidence] = []
    raw_outputs: dict[str, Any] = {}

    for entry in manifest:
        if entry.status != "planned":
            trace.append(PipelineTraceEvent(step="read_file", status="skipped", message=entry.file_name, data=entry.to_dict() if hasattr(entry, "to_dict") else asdict(entry)))
            continue

        try:
            file_evidence, file_raw_outputs = extract_file_evidence(
                manifest_entry=entry,
                candidate=candidate,
            )
            evidence.extend(file_evidence)
            raw_outputs[entry.file_name] = file_raw_outputs
            trace.append(
                PipelineTraceEvent(
                    step="read_file",
                    status="ok",
                    message=f"{entry.file_name}: {len(file_evidence)} evidence item(s)",
                    data={
                        "file": entry.file_name,
                        "primary_tool": entry.reading_decision.get("primary_tool"),
                        "doc_type": reader_doc_type(entry),
                    },
                )
            )
        except Exception as exc:
            trace.append(
                PipelineTraceEvent(
                    step="read_file",
                    status="failed",
                    message=f"{entry.file_name}: {exc}",
                    data={"file": entry.file_name},
                )
            )

    save_jsonl(run_dir / "document_evidence.jsonl", [asdict(item) for item in evidence])
    save_json(run_dir / "raw_model_outputs.json", raw_outputs)

    if use_text_model_aggregator and evidence:
        try:
            candidate_result, aggregator_raw_output = aggregate_evidence_with_text_model(
                candidate=candidate,
                evidence=evidence,
            )
            save_json(run_dir / "aggregator_raw_output.json", {"raw_output": aggregator_raw_output})
            trace.append(PipelineTraceEvent(step="aggregate", status="ok", message="text model aggregator completed"))
        except Exception as exc:
            candidate_result = deterministic_aggregate(evidence)
            trace.append(
                PipelineTraceEvent(
                    step="aggregate",
                    status="fallback",
                    message=f"text model aggregator failed, used deterministic aggregate: {exc}",
                )
            )
    else:
        candidate_result = deterministic_aggregate(evidence)
        trace.append(PipelineTraceEvent(step="aggregate", status="ok", message="deterministic aggregate completed"))

    source_consistency_checks = build_source_consistency_checks(evidence)
    candidate_result = fill_unknowns_from_evidence(
        candidate_result=candidate_result,
        evidence=evidence,
        consistency_checks=source_consistency_checks,
    )
    trace.append(PipelineTraceEvent(step="repair", status="ok", message="filled unknown fields from evidence when possible"))

    verification_report = verify_candidate_result(candidate_result=candidate_result, evidence=evidence)

    save_json(run_dir / "candidate_result.json", candidate_result)
    save_combined_result(candidate_result, run_dir / "combined_result.txt")
    save_json(run_dir / "verification_report.json", verification_report)
    save_json(run_dir / "trace.json", [asdict(item) for item in trace])

    return {
        "pipeline": "v2_vlm_text_agent",
        "dry_run": False,
        "run_dir": str(run_dir),
        "document_manifest": str(run_dir / "document_manifest.json"),
        "document_evidence": str(run_dir / "document_evidence.jsonl"),
        "candidate_result": str(run_dir / "candidate_result.json"),
        "combined_result": str(run_dir / "combined_result.txt"),
        "verification_report": str(run_dir / "verification_report.json"),
        "trace": str(run_dir / "trace.json"),
        "evidence_count": len(evidence),
        "needs_retry": verification_report["needs_retry"],
        "logs": [item.message for item in trace],
    }
