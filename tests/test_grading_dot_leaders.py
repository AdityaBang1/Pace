"""Dot-leader grade tables.

A printed syllabus usually aligns its weights with a run of periods:

    Participation ............ 15%

The reader originally required a colon or dash between category and percentage, so
this — the commonest layout in a photographed handout — produced zero categories
and the whole breakdown was lost. These tests pin the rule that fixed it, and pin
the trap that must not come with it: a sentence about losing 10% per day is not a
category called "late work".
"""
from __future__ import annotations

from pace.api import understand_academic_document

SYLLABUS = """Honors World Literature
Instructor: Ms. Priya Raman

Grading Policy
Participation ............ 15%
Close Reading Essays ..... 30%
Comparative Paper ........ 25%
Final Portfolio .......... 30%

Late Work
Assignments lose 10% per day, to a maximum of 50%.
"""


def test_dot_leaders_are_read_as_weights():
    record = understand_academic_document(SYLLABUS)
    grading = {row["category"]: row["weight"] for row in record["grading"]}
    assert grading == {
        "Participation": 15,
        "Close Reading Essays": 30,
        "Comparative Paper": 25,
        "Final Portfolio": 30,
    }
    assert record["grading_total_percent"] == 100
    assert record["grading_total_consistent"] is True


def test_a_penalty_sentence_is_not_a_category():
    record = understand_academic_document(SYLLABUS)
    for row in record["grading"]:
        assert "late" not in row["category"].casefold()
        assert row["weight"] != 10


def test_every_value_is_still_grounded():
    record = understand_academic_document(SYLLABUS)
    text = record["document"].get("cleaned_text")
    for row in record["grading"]:
        evidence = row["evidence"]
        assert 0 <= evidence["start"] < evidence["end"]
        if text is not None:
            assert text[evidence["start"]:evidence["end"]]
