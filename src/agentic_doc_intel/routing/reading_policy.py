"""No-OCR document reading policy.

The current product direction is VLM-first and Agent-driven. OCR is not part of
the active reading toolchain in this version.

Allowed reading tools:
- pdf_text_layer: cheap local text extraction from selectable PDFs.
- vlm_page_reader: render selected pages and send page images to a VLM.
- human_review: stop when the file cannot be routed safely.

PaddleOCR-VL outputs may exist in local data folders for research comparison,
but they are intentionally not used here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


ReadingTool = Literal["pdf_text_layer", "vlm_page_reader", "human_review"]
DocType = Literal[
    "application_form",
    "passport",
    "transcript",
    "diploma_certificate",
    "english_language",
    "personal_statement",
    "conditional_offer",
    "recommendation_letter",
    "photo",
    "unknown",
]


@dataclass
class TextLayerStats:
    page_count: int = 0
    text_chars: int = 0
    word_count: int = 0
    table_signal: int = 0


@dataclass
class ReadingDecision:
    doc_type: DocType
    primary_tool: ReadingTool
    auxiliary_tools: list[ReadingTool] = field(default_factory=list)
    selected_pages: list[int] | Literal["all"] = "all"
    confidence: str = "medium"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def infer_doc_type_from_filename(file_name: str) -> DocType:
    name = file_name.lower()

    if re.search(r"passport", name):
        return "passport"
    if re.search(r"application form|application_form", name):
        return "application_form"
    if re.search(r"transcript|report|grade|academic record|school report", name):
        return "transcript"
    if re.search(r"ielts|toefl|duolingo|pte|english", name):
        return "english_language"
    if re.search(r"certificate|diploma|graduation|pre graduation|ijazah|passing grade", name):
        return "diploma_certificate"
    if re.search(r"personal statement|motivation", name):
        return "personal_statement"
    if re.search(r"offer", name):
        return "conditional_offer"
    if re.search(r"recommendation|reference", name):
        return "recommendation_letter"
    if re.search(r"photo|pasphoto", name):
        return "photo"
    return "unknown"


def text_layer_is_strong(stats: TextLayerStats) -> bool:
    if stats.page_count <= 0:
        return False
    if stats.text_chars < 300:
        return False
    if stats.word_count < 60:
        return False
    return True


def choose_reading_policy(
    *,
    file_name: str,
    stats: TextLayerStats,
    max_pages_for_direct_vlm: int = 4,
) -> ReadingDecision:
    doc_type = infer_doc_type_from_filename(file_name)
    reasons: list[str] = []

    if doc_type == "photo":
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="human_review",
            confidence="high",
            reasons=["photo is not part of the current extraction target"],
        )

    if doc_type in {"passport", "diploma_certificate", "english_language"}:
        reasons.append(f"{doc_type} requires visual evidence or layout-sensitive reading")
        if doc_type == "passport" and stats.page_count > max_pages_for_direct_vlm:
            reasons.append("multi-page passport-like file; start with first two pages and expand if fields are missing")
            selected_pages: list[int] | Literal["all"] = [1, 2]
        else:
            selected_pages = "all" if stats.page_count <= max_pages_for_direct_vlm else [1, 2]
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="vlm_page_reader",
            selected_pages=selected_pages,
            confidence="high",
            reasons=reasons,
        )

    if doc_type == "transcript":
        reasons.append("transcript extraction depends on table/grade structure")
        if not text_layer_is_strong(stats):
            reasons.append("PDF text layer is weak, so use VLM page reading")
        else:
            reasons.append("text layer can help locate content, but VLM should read grade pages")
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="vlm_page_reader",
            auxiliary_tools=["pdf_text_layer"] if text_layer_is_strong(stats) else [],
            selected_pages="all" if stats.page_count <= max_pages_for_direct_vlm else "all",
            confidence="high",
            reasons=reasons,
        )

    if doc_type == "application_form":
        if text_layer_is_strong(stats):
            reasons.append("application form has usable text layer for locating fields")
            reasons.append("VLM still reads selected pages because table structure matters")
            return ReadingDecision(
                doc_type=doc_type,
                primary_tool="vlm_page_reader",
                auxiliary_tools=["pdf_text_layer"],
                selected_pages="all" if stats.page_count <= max_pages_for_direct_vlm else [1, 2, 3],
                confidence="medium",
                reasons=reasons,
            )
        reasons.append("application form text layer is weak")
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="vlm_page_reader",
            selected_pages="all" if stats.page_count <= max_pages_for_direct_vlm else [1, 2, 3],
            confidence="medium",
            reasons=reasons,
        )

    if doc_type in {"personal_statement", "conditional_offer", "recommendation_letter"}:
        if text_layer_is_strong(stats):
            return ReadingDecision(
                doc_type=doc_type,
                primary_tool="pdf_text_layer",
                selected_pages="all",
                confidence="high",
                reasons=[f"{doc_type} is mostly text and PDF text layer is strong"],
            )
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="vlm_page_reader",
            selected_pages="all" if stats.page_count <= max_pages_for_direct_vlm else [1, 2],
            confidence="medium",
            reasons=[f"{doc_type} is expected to be text-like, but text layer is weak"],
        )

    if text_layer_is_strong(stats):
        return ReadingDecision(
            doc_type=doc_type,
            primary_tool="pdf_text_layer",
            selected_pages="all",
            confidence="low",
            reasons=["unknown file type, but PDF text layer is usable"],
        )

    return ReadingDecision(
        doc_type=doc_type,
        primary_tool="vlm_page_reader",
        selected_pages="all" if stats.page_count <= max_pages_for_direct_vlm else [1, 2],
        confidence="low",
        reasons=["unknown file type and weak/unknown text layer"],
    )


def stats_from_pdf_text_layer(pdf_path: Path) -> TextLayerStats:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for local PDF text-layer stats.") from exc

    text_chars = 0
    word_count = 0
    table_signal = 0
    with fitz.open(pdf_path) as document:
        for page in document:
            text = page.get_text("text") or ""
            text_chars += len(text.strip())
            word_count += len(re.findall(r"\S+", text))
            table_signal += text.lower().count("table")
        return TextLayerStats(
            page_count=len(document),
            text_chars=text_chars,
            word_count=word_count,
            table_signal=table_signal,
        )
