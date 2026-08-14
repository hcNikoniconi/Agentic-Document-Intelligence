"""Lightweight template helpers shared by non-OCR pipelines."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "templates"

TEMPLATE_MAP = {
    "passport": "passport.json",
    "application form": "application_form.json",
    "application": "application_form.json",
    "transcript": "transcript.json",
    "ielts": "english_language.json",
    "toefl": "english_language.json",
    "pte": "english_language.json",
    "duolingo": "english_language.json",
    "det": "english_language.json",
    "certificate": "diploma_certificate.json",
    "diploma": "diploma_certificate.json",
}

DOC_TYPE_ORDER = [
    "passport",
    "application_form",
    "transcript",
    "diploma_certificate",
    "english_language",
]


def choose_template_path(input_file: str | Path) -> Path | None:
    filename = Path(input_file).name.lower()
    for keyword, template_name in TEMPLATE_MAP.items():
        if keyword in filename:
            return TEMPLATE_DIR / template_name
    return None


def load_template(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_field_meta(template: dict) -> list[dict[str, str]]:
    raw_fields = template.get("fields", [])
    field_meta: list[dict[str, str]] = []

    for item in raw_fields:
        if isinstance(item, str):
            name = item.strip()
            note = ""
            section = ""
        else:
            name = str(item.get("name", "")).strip()
            note = str(item.get("note", "")).strip()
            section = str(item.get("section", "")).strip()

        if name:
            field_meta.append({"name": name, "note": note, "section": section})

    return field_meta


def build_json_skeleton(field_meta: list[dict[str, str]]) -> dict[str, str]:
    return {item["name"]: "" for item in field_meta}

