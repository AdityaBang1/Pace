"""Model-independent text preparation for academic documents."""
from __future__ import annotations

import re
from typing import Any


def split_sections(text: str) -> list[dict[str, Any]]:
    """Split long OCR text into heading-aware sections, then bounded paragraphs."""
    lines = [line.strip() for line in text.splitlines()]
    sections, title, body = [], "Document", []
    heading = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 /&()-]{1,65}:?$")
    known_headings = {"course overview", "grading", "schedule", "assignments", "prerequisites",
                      "learning outcomes", "policies", "contact", "credits", "course description",
                      "instructor", "grading policy", "course objectives"}
    for line in lines:
        if not line:
            continue
        is_heading = heading.match(line) and (line.rstrip(":").casefold() in known_headings or line.endswith(":"))
        if is_heading and body:
            sections.append({"title": title, "text": "\n".join(body).strip()}); title, body = line.rstrip(":"), []
        elif is_heading and not body and title == "Document":
            title = line.rstrip(":")
        else:
            body.append(line)
    if body:
        sections.append({"title": title, "text": "\n".join(body).strip()})
    if not sections and text.strip(): sections = [{"title": "Document", "text": text.strip()}]
    result = []
    for section in sections:
        words = section["text"].split()
        # 120 words per chunk keeps wordpiece length under the model's 160-token budget.
        for start in range(0, len(words), 120):
            chunk = " ".join(words[start:start+120])
            result.append({"title": section["title"], "text": chunk})
    return result
