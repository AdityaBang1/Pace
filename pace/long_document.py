"""Split a long document into pieces the reader can actually see.

`detect_document_sections` is line-anchored: it finds headings because they are
alone on a line. That is exactly right for a document whose line structure
survived, and useless for one that did not. Measured on the corpus's two longest
syllabi, a 397 k-character document returned **zero** sections after HTML
stripping, so the reader saw its first 127 tokens and called the whole thing
``other`` at 99.8%.

This module keeps the heading path first, and falls back to paragraph boundaries
— and then to sentence-aligned windows with overlap — only when the heading scan
finds nothing. Every piece records how it was produced, so a caller can tell a
real section from a window and can show the user which one a field came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .field_corpus import detect_document_sections

#: Paragraphs shorter than this are boilerplate or page furniture.
MIN_PARAGRAPH_CHARS = 80
#: A window is the smallest unit that still carries a sentence of context.
WINDOW_CHARS = 600
OVERLAP_CHARS = 80


@dataclass
class Piece:
    text: str
    method: str            # "heading" | "paragraph" | "window"
    heading: str | None = None
    start: int = 0
    end: int = 0
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "method": self.method,
                "heading": self.heading, "start": self.start, "end": self.end}


def _windows(text: str, size: int = WINDOW_CHARS, overlap: int = OVERLAP_CHARS) -> list[Piece]:
    """Sentence-aligned windows, so a window does not cut a fact in half."""
    raw = re.split(r"(?<=[.!?;])\s+", text)
    # Flattened OCR often has no sentence breaks at all, which would otherwise
    # leave one enormous "sentence" and silently drop most of the document.
    sentences: list[str] = []
    for sentence in raw:
        if len(sentence) <= size:
            sentences.append(sentence)
            continue
        sentences.extend(sentence[offset:offset + size]
                         for offset in range(0, len(sentence), size))
    pieces: list[Piece] = []
    buffer = ""
    start = 0
    cursor = 0
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > size:
            pieces.append(Piece(buffer.strip(), "window", start=start, end=cursor,
                                index=len(pieces)))
            keep = buffer[-overlap:]
            buffer = f"{keep} {sentence}"
            start = cursor - len(keep)
        else:
            buffer = f"{buffer} {sentence}".strip()
        cursor += len(sentence) + 1
    if buffer.strip():
        pieces.append(Piece(buffer.strip(), "window", start=start, end=len(text),
                            index=len(pieces)))
    return pieces


def split_document(text: str, min_paragraph: int = MIN_PARAGRAPH_CHARS) -> list[Piece]:
    """Heading sections if the document still has lines, otherwise safe fallback."""
    text = text or ""
    if not text.strip():
        return []

    sections = detect_document_sections(text)
    if sections:
        pieces: list[Piece] = []
        for position, section in enumerate(sections):
            body_start = section["end"]
            body_end = sections[position + 1]["start"] if position + 1 < len(sections) else len(text)
            body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
            heading = (section.get("text") or "").strip()
            combined = f"{heading}. {body}" if body else heading
            if not combined.strip():
                continue
            pieces.append(Piece(combined, "heading", heading=heading or None,
                                start=section.get("start", 0), end=body_end,
                                index=len(pieces)))
        if pieces:
            return pieces

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text)]
    usable = [chunk for chunk in paragraphs if len(chunk) >= min_paragraph]
    if len(usable) >= 3:
        return [Piece(chunk, "paragraph", start=0, end=len(chunk), index=index)
                for index, chunk in enumerate(usable)]

    return _windows(re.sub(r"\s+", " ", text).strip())


def classify_document(engine, text: str, max_chars: int = 400,
                      min_confidence: float = 0.0) -> dict[str, Any]:
    """Read a long document as pieces and aggregate, never as one truncated blob."""
    pieces = split_document(text)
    rows = []
    for piece in pieces:
        result = engine.classify(piece.text[:max_chars], top_k=1)
        if result["confidence"] < min_confidence:
            continue
        rows.append({**piece.as_dict(), "field": result["label"],
                     "percent": result["percent"], "confidence": result["confidence"]})
    covered = [row for row in rows if row["field"] != "other"]
    return {
        "characters": len(text),
        "pieces": len(pieces),
        "methods": sorted({piece.method for piece in pieces}),
        "classified": len(covered),
        "coverage": (len(covered) / len(rows)) if rows else 0.0,
        "fields": rows,
    }
