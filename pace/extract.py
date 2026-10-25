"""Auditable deterministic extraction helpers; no inferred field is silently committed."""
from __future__ import annotations

import re
from typing import Any

COURSE_CODE = re.compile(r"\b[A-Z]{2,5}\s?-?\s?\d{2,4}[A-Z]?\b", re.I)  # filtered for grade/percentage false positives below
DATE = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
PERCENT = re.compile(r"(?<!\w)(\d{1,3}(?:\.\d+)?)\s*%")
CREDITS = re.compile(r"\b(\d+(?:\.\d+)?)[ \t]*(?:semester[ \t]+)?credits?\b", re.I)
GRADE_TOKEN = r"[A-F](?:[+−]|-(?![ \t]*[-–—−][ \t]*[A-F]))?"
GRADE_CONTEXT = re.compile(
    rf"\b(?:final[ \t]+)?(?:grades?(?:[ \t]+scale)?|grading[ \t]+scale)[ \t]*"
    rf"(?::|=|\bis\b|\bare\b|\bwas\b|\bwere\b)?[ \t]*"
    rf"(?:\([ \t]*)?({GRADE_TOKEN}"
    rf"(?:[ \t]*(?:,[ \t]*(?:(?:and|or)\b[ \t]*)?|/|\band\b|\bor\b)[ \t]*{GRADE_TOKEN})*)"
    rf"(?:[ \t]*[-–—−][ \t]*({GRADE_TOKEN})|[ \t]+(?:to|through)[ \t]+({GRADE_TOKEN}))?[ \t]*\)?(?![A-Za-z+−-])",
    re.I,
)
GRADE_VALUE = re.compile(rf"(?<![A-Za-z]){GRADE_TOKEN}(?![A-Za-z+−-])", re.I)


def extract_candidates(text: str) -> dict[str, list[dict[str, Any]]]:
    """Return source-grounded spans with offsets; extraction does not assert semantic roles."""
    patterns = {"course_code": COURSE_CODE, "date": DATE, "percentage": PERCENT,
                "credit_hours": CREDITS}
    result = {}
    for kind, pattern in patterns.items():
        matches = []
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "course_code":
                # A percent quantity or letter grade is not a course identifier.
                if value.casefold().startswith("worth ") or value.upper() in {"A", "B", "C", "D", "F"}:
                    continue
            # A value is written on one line; a match that runs into the next row is
            # two separate values, not one.
            if "\n" in value or "\r" in value:
                continue
            matches.append({"text": value, "start": match.start(), "end": match.end(),
                            "confidence": 1.0, "evidence": "literal_pattern_match", "requires_review": True})
        result[kind] = matches
    grade_matches = []
    for context in GRADE_CONTEXT.finditer(text):
        for group in (1, 2, 3):
            grade_list = context.group(group)
            if not grade_list:
                continue
            for grade in GRADE_VALUE.finditer(grade_list):
                start = context.start(group) + grade.start()
                end = context.start(group) + grade.end()
                if "\n" in text[start:end] or "\r" in text[start:end]:
                    continue
                grade_matches.append({"text": text[start:end], "start": start, "end": end,
                                      "confidence": 1.0, "evidence": "literal_pattern_match",
                                      "requires_review": True})
    result["grade"] = grade_matches
    return result


def validate_academic_ir(ir: dict[str, Any]) -> list[str]:
    """Validate structural constraints; this cannot verify source truth."""
    errors = []
    if ir.get("schema_version") != "1.0": errors.append("schema_version must be 1.0")
    if not isinstance(ir.get("document_type"), dict): errors.append("document_type must be a proposal object")
    if not isinstance(ir.get("sections"), list): errors.append("sections must be a list")
    for i, section in enumerate(ir.get("sections", [])):
        if not isinstance(section, dict): errors.append(f"sections[{i}] must be an object"); continue
        if not isinstance(section.get("confidence"), (int, float)) or not 0 <= section["confidence"] <= 1:
            errors.append(f"sections[{i}].confidence must be in [0,1]")
        if not section.get("source_text"): errors.append(f"sections[{i}] missing source_text evidence")
    if ir.get("commit_policy") != "proposals_only_user_confirmation_required":
        errors.append("commit_policy must require user confirmation")
    return errors
