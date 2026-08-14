"""v1 pipeline: original document images/PDF pages -> VLM direct extraction."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agentic_doc_intel.document_rendering import document_to_data_urls
from agentic_doc_intel.model_client import chat_completion_content
from agentic_doc_intel.template_registry import (
    DOC_TYPE_ORDER,
    build_json_skeleton,
    choose_template_path,
    load_template,
    parse_field_meta,
)


SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def build_vlm_prompt(template: dict, field_meta: list[dict[str, str]]) -> str:
    doc_type = template.get("doc_type", "").strip()
    skeleton = build_json_skeleton(field_meta)

    lines = [
        "You are extracting structured information from applicant documents.",
        "Read the provided document page images directly.",
        "Return JSON only.",
        "Do not output markdown.",
        "Do not output explanations.",
        "Do not output <think> tags.",
        "Use exactly the requested field names as JSON keys.",
        "If a field is not visible or cannot be confirmed, return \"unknown\".",
        "Do not invent values.",
        "Prefer values supported by visible evidence in the document image.",
        "",
        f"Document type: {doc_type}",
        "",
    ]

    if doc_type == "transcript":
        lines.extend(
            [
                "Transcript-specific rules:",
                "- Read tables carefully. Preserve year, grade, semester, or subject labels when extracting score summaries.",
                "- Extract math score, physics score, english score, each year average, grading scale, grade, and validity check when visible.",
                "- For grade, provide a compact but complete score summary, not just one number.",
                "- For validity check, return exactly three comma-separated values: document clear/complete, school/formal information present, student information comparable with passport.",
                "",
            ]
        )
    elif doc_type == "diploma_certificate":
        lines.extend(
            [
                "Diploma/certificate-specific rules:",
                "- Only use graduation, diploma, completion, degree, foundation completion, or official school completion evidence.",
                "- Ignore unrelated awards, music certificates, competitions, or participation certificates.",
                "- If this image is not an academic completion/graduation document, return unknown for fields.",
                "",
            ]
        )

    lines.append("Fields:")
    for item in field_meta:
        name = item.get("name", "").strip()
        note = item.get("note", "").strip()
        section = item.get("section", "").strip()
        hints = []
        if section:
            hints.append(f"section={section}")
        if note:
            hints.append(f"note={note}")
        if hints:
            lines.append(f"- {name}: " + "; ".join(hints))
        else:
            lines.append(f"- {name}")

    lines.extend(
        [
            "",
            "Return JSON only in this exact shape:",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(lines)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"VLM did not return a JSON object: {raw_text[:300]}")
    return json.loads(text[start : end + 1])


def normalize_chat_data(parsed: dict[str, Any], field_meta: list[dict[str, str]]) -> dict[str, str]:
    chat_data: dict[str, str] = {}
    for item in field_meta:
        key = item["name"]
        value = parsed.get(key, "")
        if value is None:
            value = ""
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        chat_data[key] = str(value).strip()
    return chat_data


def build_image_content_item(image_url: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": image_url},
    }

    min_pixels = os.getenv("VLM_MIN_PIXELS", "").strip()
    max_pixels = os.getenv("VLM_MAX_PIXELS", "").strip()

    if min_pixels:
        item["min_pixels"] = int(min_pixels)
    if max_pixels:
        item["max_pixels"] = int(max_pixels)

    return item


def extract_one_file(input_file: str | Path) -> dict[str, Any]:
    input_file = Path(input_file)
    template_path = choose_template_path(input_file)
    if template_path is None:
        raise ValueError(f"No template matched file name: {input_file.name}")

    template = load_template(template_path)
    field_meta = parse_field_meta(template)
    prompt = build_vlm_prompt(template, field_meta)
    image_urls = document_to_data_urls(
        input_file,
        max_pages=int(os.getenv("VLM_MAX_PAGES_PER_FILE", "4")),
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_url in image_urls:
        content.append(build_image_content_item(image_url))

    raw_output = chat_completion_content(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        max_tokens=int(os.getenv("VLM_MAX_TOKENS", "4096")),
    )

    parsed = extract_json_object(raw_output)
    chat_data = normalize_chat_data(parsed, field_meta)

    return {
        "doc_type": template.get("doc_type", "unknown"),
        "template_file": str(template_path),
        "template": template,
        "field_meta": field_meta,
        "chat_data": chat_data,
        "input_file": str(input_file),
        "llm_raw_output": raw_output,
        "page_images_sent": len(image_urls),
    }


def score_doc_type_candidate(doc_type: str, input_file: str | Path) -> int:
    filename = Path(input_file).name.lower()
    normalized = re.sub(r"[_\-\s]+", " ", filename)

    if doc_type == "diploma_certificate":
        terms = [
            ("pre graduation", 120),
            ("graduation", 110),
            ("diploma", 105),
            ("ijazah", 105),
            ("completion", 85),
            ("foundation", 75),
            ("certificate", 20),
            ("yamaha", -140),
            ("music", -120),
            ("award", -90),
            ("competition", -90),
        ]
    elif doc_type == "transcript":
        terms = [
            ("academic transcript", 130),
            ("transcript", 120),
            ("school report", 100),
            ("report card", 95),
            ("final report", 90),
            ("semester", 55),
            ("certificate", -50),
            ("diploma", -40),
        ]
    else:
        terms = []

    score = 0
    for term, weight in terms:
        if term in normalized or term in filename:
            score += weight
    return score


def build_output_base_name(input_dir: str | Path, output_base_name: str | None) -> str:
    if output_base_name:
        return output_base_name
    return Path(input_dir).name.replace("/", "_").replace(" ", "_")


def save_combined_result(results_by_type: dict[str, dict[str, Any]], output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for doc_type in DOC_TYPE_ORDER:
            file.write(f"#{doc_type}\n")
            result = results_by_type.get(doc_type)
            if not result:
                file.write("未提供或未识别到该类文件\n\n")
                continue

            for item in result.get("field_meta", []):
                key = item.get("name", "")
                value = result.get("chat_data", {}).get(key, "")
                file.write(f"{key}: {value}\n")
            file.write("\n")


def run_candidate_folder(
    input_dir: str | Path,
    output_root: str | Path | None = None,
    output_base_name: str | None = None,
) -> dict[str, Any]:
    """Run VLM direct extraction for one applicant folder."""

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a valid applicant folder: {input_dir}")

    output_root = Path(output_root or "output/v1_vlm_direct")
    output_root.mkdir(parents=True, exist_ok=True)

    results_by_type: dict[str, dict[str, Any]] = {}
    result_scores_by_type: dict[str, int] = {}
    logs: list[str] = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if choose_template_path(path) is None:
            logs.append(f"[跳过] {path.name}: filename does not match any template")
            continue

        try:
            result = extract_one_file(path)
            doc_type = result["doc_type"]
            candidate_score = score_doc_type_candidate(doc_type, path)

            if doc_type not in results_by_type:
                results_by_type[doc_type] = result
                result_scores_by_type[doc_type] = candidate_score
                logs.append(f"[成功] {path.name} -> {doc_type} (score={candidate_score})")
            else:
                current_score = result_scores_by_type.get(doc_type, 0)
                if candidate_score > current_score:
                    previous_file = Path(results_by_type[doc_type]["input_file"]).name
                    results_by_type[doc_type] = result
                    result_scores_by_type[doc_type] = candidate_score
                    logs.append(
                        f"[替换] {path.name} -> {doc_type} (score={candidate_score})，"
                        f"替换 {previous_file} (score={current_score})"
                    )
                else:
                    logs.append(
                        f"[重复] {path.name} -> {doc_type} (score={candidate_score})，"
                        f"保留已有同类型文件 (score={current_score})"
                    )
        except Exception as exc:
            logs.append(f"[失败] {path.name}: {exc}")

    base_name = build_output_base_name(input_dir, output_base_name)
    combined_output_file = output_root / f"{base_name}_combined_result.txt"
    save_combined_result(results_by_type, combined_output_file)
    logs.append(f"[输出] {combined_output_file}")

    return {
        "pipeline": "v1_vlm_direct",
        "combined_output_file": str(combined_output_file),
        "report_file": None,
        "results_by_type": results_by_type,
        "logs": logs,
    }
