"""The record an application reads must keep its promises.

The point of these tests is not that the reader is accurate -- that is measured
elsewhere, on held-out documents. It is that whatever the reader returns is
shaped the same way every time, that every value points at text that is really
in the source, and that it never claims a document type it cannot support.
"""
from __future__ import annotations

import pytest

from pace.api import (BANDS, SCHEMA_VERSION, confidence_band,
                      understand_academic_document)
from pace.ir_schema import migrate, validate

SYLLABUS = """BIOL 101: Foundations of Biology
Fall 2026

Instructor: Dr. Ada Chen
Office Hours: Tuesdays 2-4pm in Room 312

Meeting Times
TR 9:30-10:45 in Science 210

Grading
Homework 30%
Midterm Exam 30%
Final Exam 40%

Attendance Policy
Students may miss two classes without penalty.
"""


@pytest.fixture(scope="module")
def record():
    return understand_academic_document(SYLLABUS)


def test_the_contract_has_one_shape(record):
    assert record["schema_version"] == SCHEMA_VERSION
    for key in ("document", "course", "grading", "meetings", "policies", "sections",
                "evidence", "confidence", "model", "warnings"):
        assert key in record, key
    assert record["requires_confirmation"] is True
    assert record["commit_policy"] == "proposals_only_user_confirmation_required"


def test_a_clean_syllabus_is_read_end_to_end(record):
    assert record["course"]["course_title"] == "BIOL 101: Foundations of Biology"
    assert record["course"]["course_code"] == "BIOL 101"
    assert record["course"]["instructor"] == "Dr. Ada Chen"
    assert [(item["category"], item["weight"]) for item in record["grading"]] == [
        ("Homework", 30), ("Midterm Exam", 30), ("Final Exam", 40)]
    assert record["grading_total_percent"] == 100
    assert record["grading_total_consistent"] is True
    meeting = record["meetings"][0]
    assert meeting["days"] == ["TU", "TH"]
    assert (meeting["start_time"], meeting["end_time"]) == ("09:30", "10:45")
    assert meeting["room"] == "Science 210"


def test_every_value_points_at_text_that_is_really_there(record):
    assert record["evidence"]
    for item in record["evidence"]:
        assert item["grounded_in_source"] is True, item
        assert item["requires_confirmation"] is True
        assert 0 <= item["start"] < item["end"] <= len(SYLLABUS)


def test_a_value_read_by_a_rule_is_not_given_a_probability_band(record):
    # Section labels are model probabilities; everything else is a rule's own
    # number, and calling that a probability would be a lie the UI would show.
    for item in record["evidence"]:
        assert item["confidence_source"] == "rule"
        assert item["band"] is None
    for section in record["sections"]:
        assert section["confidence_source"] == "model"
        assert section["band"] in {name for _, name in BANDS}


def test_a_non_academic_document_is_refused():
    # The audit's worst failure was a bank statement read as a transcript at 0.9975,
    # and then a rule still returning PERIOD 01 as a course code at 0.93.
    bank = understand_academic_document(
        "STATEMENT PERIOD 01/09/2026 - 30/09/2026\nOpening balance 1,204.55\n"
        "09/03 Direct deposit 2,100.00\n09/05 Rent -1,450.00\nClosing balance 1,342.15")
    document_type = bank["document"]["document_type"]
    if document_type is None:  # the optional artifact is absent in a bare checkout
        pytest.skip("document-type checkpoint not built")
    assert document_type["academic"] is False
    assert document_type["label"] == "other"
    assert any("does not look like an academic document" in warning
               for warning in bank["warnings"])
    # The gate must gate, not merely warn.
    assert bank["document"]["gate"] == "blocked"
    assert bank["course"] == {}
    assert bank["grading"] == []
    assert bank["meetings"] == []
    assert bank["entities"] == []
    assert bank["sections"] == []
    assert bank["evidence"] == []
    assert any(item["reason"] == "document_type_not_academic"
               for item in bank["suppressed"])


def test_a_clean_record_satisfies_its_own_schema(record):
    assert validate(record) == []
    assert record["document"]["gate"] == "open"
    assert record["document"]["id"].startswith("doc_")


def test_ids_are_stable_across_reads_and_unique_within_one(record):
    again = understand_academic_document(SYLLABUS)
    assert again["document"]["id"] == record["document"]["id"]
    assert [item["id"] for item in again["evidence"]] == [
        item["id"] for item in record["evidence"]]
    ids = [item["id"] for collection in ("sections", "entities", "relations", "evidence")
           for item in record[collection]]
    assert len(ids) == len(set(ids))


def test_every_entity_points_at_its_own_evidence(record):
    evidence_ids = {item["id"] for item in record["evidence"]}
    assert record["entities"]
    for entity in record["entities"]:
        assert entity["evidence_id"] in evidence_ids
        assert entity["requires_confirmation"] is True
    kinds = {entity["kind"] for entity in record["entities"]}
    assert {"course", "instructor", "grade_category", "meeting"} <= kinds


def test_structural_relations_are_parsed_not_predicted(record):
    weight_relations = [rel for rel in record["relations"] if rel["type"] == "WEIGHT_OF"]
    assert len(weight_relations) == len(record["grading"])
    assert {rel["weight"] for rel in weight_relations} == {30, 40}
    taught = [rel for rel in record["relations"] if rel["type"] == "TAUGHT_BY"]
    assert len(taught) == 1
    assert all(rel["predicted"] is False for rel in record["relations"])


def test_a_02_record_migrates_to_the_current_version():
    legacy = understand_academic_document(SYLLABUS)
    for key in ("schema", "entities", "relations"):
        legacy.pop(key, None)
    legacy["schema_version"] = "pace.academic-ir/0.2"
    legacy.pop("document", None)
    legacy.pop("sections")
    migrated = migrate(legacy)
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["document"]["id"].startswith("doc_")
    assert migrated["evidence"][0]["id"].startswith("ev_")


def test_a_syllabus_is_recognised_as_one(record):
    document_type = record["document"]["document_type"]
    if document_type is None:
        pytest.skip("document-type checkpoint not built")
    assert document_type["academic"] is True
    assert document_type["label"] == "syllabus"
    assert document_type["label"] in {"syllabus", "course_catalog", "schedule"}


def test_weights_that_do_not_add_up_are_warned_about():
    text = SYLLABUS.replace("Final Exam 40%", "Final Exam 10%")
    result = understand_academic_document(text)
    assert result["grading_total_consistent"] is False
    assert any("not 100" in warning for warning in result["warnings"])


def test_empty_or_non_text_input_is_refused():
    with pytest.raises(ValueError):
        understand_academic_document("   \n  ")
    with pytest.raises(TypeError):
        understand_academic_document(None)  # type: ignore[arg-type]


def test_bands_follow_the_measured_floors():
    assert confidence_band(0.99) == "high"
    assert confidence_band(0.97) == "high"
    assert confidence_band(0.90) == "medium"
    assert confidence_band(0.85) == "medium"
    assert confidence_band(0.60) == "low"
