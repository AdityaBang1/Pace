"""A long document with no line structure must still reach the reader.

The bug this covers was measured: a 397 k-character syllabus produced zero
sections, so the reader classified its first 127 tokens and called the whole
document `other`. The fallback is only allowed to fire when heading detection
finds nothing, and it must not silently drop text.
"""
from __future__ import annotations

from pace.long_document import classify_document, split_document

HEADED = """MATH 2210 Linear Algebra
Fall 2026

Instructor
Dr. Allison Johnson

Grading
Homework 30%, Midterms 40%, Final 30%

Office Hours
Wednesdays 2-4pm in Sexton 204
"""

FLATTENED = ("course description this course covers linear systems and matrices "
             "and eigenvalues with applications to differential equations. "
             "grading homework counts thirty percent and the final exam counts "
             "thirty percent and two midterms make up the rest. ") * 40


def test_a_document_with_headings_keeps_its_sections():
    pieces = split_document(HEADED)

    assert pieces
    assert all(piece.method == "heading" for piece in pieces)
    assert any(piece.heading and "Grading" in piece.heading for piece in pieces)


def test_a_flattened_document_falls_back_to_windows():
    pieces = split_document(FLATTENED)

    assert pieces
    assert all(piece.method == "window" for piece in pieces)
    assert len(pieces) > 3
    # Windows may overlap, but no window may be enormous and nothing is dropped.
    assert all(len(piece.text) <= 900 for piece in pieces)
    covered = sum(len(piece.text) for piece in pieces)
    assert covered >= len(FLATTENED)


def test_text_with_no_sentence_breaks_is_still_split():
    blob = "x " * 5000
    pieces = split_document(blob)

    assert len(pieces) > 1
    assert all(len(piece.text) <= 900 for piece in pieces)


def test_an_empty_document_produces_nothing():
    assert split_document("") == []
    assert split_document("   ") == []


class _Engine:
    """A stand-in reader: long text is `other`, anything with a keyword is a field."""

    def classify(self, text, top_k=1):
        if "grading" in text.lower():
            return {"label": "grading_policy", "confidence": 0.9, "percent": 90.0}
        return {"label": "other", "confidence": 0.8, "percent": 80.0}


def test_classify_document_aggregates_pieces_and_keeps_provenance():
    result = classify_document(_Engine(), HEADED)

    assert result["pieces"] >= 1
    assert result["methods"] == ["heading"]
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["classified"] >= 1
    for row in result["fields"]:
        assert {"method", "start", "end", "field", "percent"} <= set(row)
