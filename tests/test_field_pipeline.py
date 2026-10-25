"""Tests for the real-text academic field classifier.

These cover the parts that decide whether the model is honest: the taxonomy,
the corpus builders, the leak-free splits, and the calibrated inference path.
Model weights are only exercised when a trained artifact is present.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from pace.field_corpus import (FIELD_TYPES, SYLLABUS_SECTION_MAP, clean_html,
                               detect_document_sections, exclude_intrinsically_ambiguous,
                               load_usc_field_examples, normalize_value, window_text)
from pace.field_model import FIELD_LABELS, FIELD_TASKS
from pace.field_splits import _split_by_group, choose_holdout_schools

ARTIFACT = Path("artifacts/field-classifier-minilm")


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_field_vocabulary_matches_the_model_head_exactly():
    assert set(FIELD_LABELS) == set(FIELD_TYPES)
    assert set(FIELD_TASKS) == set(FIELD_TYPES)


def test_every_section_mapping_targets_a_real_field_type():
    unknown = {heading: label for heading, label in SYLLABUS_SECTION_MAP.items()
               if label not in FIELD_TYPES}
    assert unknown == {}


def test_every_field_type_belongs_to_a_task_group():
    assert all(FIELD_TASKS[label] for label in FIELD_TYPES)
    assert all(not label.startswith("task:") for label in FIELD_TYPES)


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def test_clean_html_removes_markup_but_keeps_words():
    text, _ = clean_html("<p>Office&nbsp;Hours</p><p>Mon 10-12</p>")
    assert "Office" in text and "Hours" in text and "Mon 10-12" in text
    assert "<p>" not in text and "&nbsp;" not in text


def test_headings_are_only_trusted_when_the_whole_line_is_the_heading():
    document = (
        "CS 101 Introduction to Computing\n"
        "This course meets on Monday.\n"
        "Office Hours: Mon 2-4pm\n"
        "We also discuss office hours during lecture.\n"
    )
    sections = detect_document_sections(document)
    labels = [section["label"] for section in sections]
    assert "office_hours" in labels
    assert "course_description" not in labels
    inline = [s for s in sections if s["label"] == "office_hours"][0]
    assert inline["inline"] == "Mon 2-4pm"


def test_numbered_headings_are_normalised():
    sections = detect_document_sections("1. Course Description\n2. Grading Policy\n")
    assert [s["label"] for s in sections] == ["course_description", "grading_policy"]


def test_long_values_are_split_into_sentence_aligned_windows():
    text = " ".join(["Sentence number %d about the course." % i for i in range(40)])
    windows = window_text(text, window_chars=200, max_windows=3)
    assert 1 < len(windows) <= 3
    assert all(len(window) <= 260 for window in windows)
    assert windows[0] + " " in text or text.startswith(windows[0])


def test_short_values_stay_whole():
    assert window_text("Lecture") == ["Lecture"]


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def test_usc_loader_keeps_the_natural_cell_value(tmp_path: Path):
    source = tmp_path / "tiny.csv"
    fields = ["Program Code", "Class ID", "Class Name", "Catalogue", "Units",
              "Prerequisites", "Restrictions", "Time", "Days", "Class Type", "Instructor"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Program Code": "BIO", "Class ID": "BIO 101", "Class Name": "Intro Biology",
            "Catalogue": "Study of living systems.", "Units": "(4.0 units)",
            "Prerequisites": "Not available", "Restrictions": "Not available",
            "Time": "10:00am", "Days": "Mon", "Class Type": "Lecture",
            "Instructor": "Person Name"})

    rows = load_usc_field_examples(source)
    texts = {row["label"]: row["text"] for row in rows}
    assert texts["course_title"] == "Intro Biology"
    assert texts["credits"] == "(4.0 units)"
    assert texts["meeting_days"] == "Mon"
    assert texts["instructor"] == "Person Name"
    assert "prerequisites" not in texts  # missing values are skipped, not invented
    assert all(row["group_id"] == "usc-subject:BIO" for row in rows)


def test_conflicting_identical_values_are_excluded():
    rows = [{"text": "TBA", "label": "meeting_days", "group_id": "a"},
            {"text": " tba ", "label": "meeting_time", "group_id": "b"},
            {"text": "Monday", "label": "meeting_days", "group_id": "a"}]
    clean, ambiguous = exclude_intrinsically_ambiguous(rows)
    assert ambiguous == {"tba": ["meeting_days", "meeting_time"]}
    assert [row["text"] for row in clean] == ["Monday"]


def test_normalize_value_is_case_and_space_insensitive():
    assert normalize_value("  Mon,   Wed ") == "mon, wed"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _fake_rows() -> list[dict]:
    rows = []
    for label in ("course_title", "credits", "drop_add_deadline"):
        for group in range(12):
            rows.append({"text": f"{label} value {group}", "label": label,
                         "group_id": f"g-{group}", "source": "x/y",
                         "source_key": "x", "source_field": label})
    return rows


def test_group_split_keeps_groups_disjoint():
    splits = _split_by_group(_fake_rows(), seed=1)
    train_groups = {row["group_id"] for row in splits["train"]}
    test_groups = {row["group_id"] for row in splits["test"]}
    assert not (train_groups & test_groups)
    assert sum(len(part) for part in splits.values()) == len(_fake_rows())


def test_every_field_type_keeps_a_training_group():
    splits = _split_by_group(_fake_rows(), seed=3)
    trained = {row["label"] for row in splits["train"]}
    assert trained == {"course_title", "credits", "drop_add_deadline"}


def test_holdout_schools_cover_a_share_of_documents():
    rows = []
    for school in ("alpha", "beta", "gamma", "delta"):
        for index in range(4):
            rows.append({"group_id": f"syllabus-doc:{school}-{index}", "label": "term",
                         "meta": {"school": school}})
    holdout = choose_holdout_schools(rows, target_fraction=0.5)
    assert len(holdout) >= 2
    assert set(holdout) <= {"alpha", "beta", "gamma", "delta"}


# ---------------------------------------------------------------------------
# Trained artifact
# ---------------------------------------------------------------------------

def _checkpoint_dir() -> Path | None:
    if (ARTIFACT / "model.pt").exists():
        return ARTIFACT
    for candidate in sorted(ARTIFACT.glob("seed-*")):
        if (candidate / "model.pt").exists():
            return candidate
    return None


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_field_engine_returns_calibrated_percentages():
    from pace.field_model import FieldEngine

    engine = FieldEngine(_checkpoint_dir(), device="cpu")
    result = engine.classify("Mon, Wed")
    assert result["label"] in FIELD_TYPES
    assert result["task"] == FIELD_TASKS[result["label"]]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["percent"] == round(result["confidence"] * 100, 2)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
    assert len(result["top"]) == 5


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_field_engine_classifies_many_consistently():
    from pace.field_model import FieldEngine

    engine = FieldEngine(_checkpoint_dir(), device="cpu")
    rows = engine.classify_many(["Mon, Wed", "3.0 units", "Lecture"])
    assert len(rows) == 3
    assert all(row["label"] in FIELD_TYPES for row in rows)


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_classify_field_cli_reports_a_label(tmp_path: Path, capsys):
    from pace.cli import main

    main(["classify-field", "Office Hours: Mon 2-4pm",
          "--checkpoint", str(_checkpoint_dir()), "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] in FIELD_TYPES
    assert payload["percent"] > 0


@pytest.mark.skipif(not Path("artifacts/field-ood/ood_report.json").exists(),
                    reason="OOD report is not present")
def test_ood_report_covers_multiple_sources():
    payload = json.loads(Path("artifacts/field-ood/ood_report.json").read_text(encoding="utf-8"))
    assert len(payload["per_source"]) >= 3
    for entry in payload["per_source"].values():
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert entry["rows"] > 0


# ---------------------------------------------------------------------------
# Meeting parsing and the held-out scoring harness
# ---------------------------------------------------------------------------

class _StubEngine:
    """Enough of FieldEngine for the meeting reader: any line, fixed confidence."""

    def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
        return {"probabilities": {"meeting_time": 0.9, "course_schedule": 0.5}}


def test_spelled_and_compact_days_resolve_to_the_same_week():
    from pace.field_extract import _days_in

    assert _days_in("MWF 9:10-10:00") == ["MO", "WE", "FR"]
    assert _days_in("Monday through Friday, 9 to 10:20 AM") == ["MO", "TU", "WE", "TH", "FR"]
    assert _days_in("M, W, F 2:25") == ["MO", "WE", "FR"]
    assert _days_in("TueThu 12:30-2pm") == ["TU", "TH"]
    assert _days_in("Tuesday and Thursdays; 10:45 - 12:50 pm") == ["TU", "TH"]


def test_room_is_read_without_the_word_room():
    from pace.field_extract import _room_from

    assert _room_from("Lecture: MWF 12:30pm -- 1:20pm MM 103") == "MM 103"
    assert _room_from("Time and Place: MWF 1 pm-1:50 pm in JGB 2.216") == "JGB 2.216"
    assert _room_from("17 Thurs: Acheulean Handaxes, Hewlett 201") in {None, "Hewlett 201"}
    assert _room_from("Fall 2019 schedule") is None


def test_meetings_skip_office_hours_and_exam_lines():
    from pace.field_extract import extract_meetings

    text = "\n".join([
        "O\ufb03ce Hours: Tuesday and Thursday 10:30 AM to 12 PM",
        "Final Exams: 38760, MWF 11-12:  Thursday, 5/16, 9 am - 12 noon",
        "Monday - Thursday 2-7pm and Friday 2pm-5pm (lab open hours)",
        "Lecture: MWF 11-12, WCH 1.120",
    ])
    meetings = extract_meetings(text, _StubEngine())
    assert [(m["days"], m["start_time"], m["end_time"], m["room"]) for m in meetings] == [
        (["MO", "WE", "FR"], "11:00", "12:00", "WCH 1.120")]


def test_meeting_without_a_clock_time_keeps_its_place():
    from pace.field_extract import extract_meetings

    meetings = extract_meetings("Uniques: 49970 and 49975 · TTh in WEL 2.224", _StubEngine())
    assert meetings and meetings[0]["days"] == ["TU", "TH"]
    assert meetings[0]["start_time"] is None
    assert meetings[0]["room"].startswith("WEL 2")


def test_section_span_scoring_ignores_fields_the_gold_does_not_cover():
    from pace.field_extract_eval import evaluate_document

    record = {
        "course": {
            "course_title": {"value": "Chemistry 1A", "confidence": 0.99,
                             "evidence": {"text": "Chemistry 1A: General Chemistry"}},
            "credits": {"value": 4.0, "confidence": 0.99,
                        "evidence": {"text": "4 credit hours"}},
        },
        "grading": [], "meetings": [], "policies": {},
    }
    gold = [{"label": "course_title", "text": "Chemistry 1A: General Chemistry"},
            {"label": "course_title", "text": "Chemistry 1B: Organic"}]
    judged = evaluate_document(record, gold)
    title = next(value for value in judged if value["field"] == "course_title")
    credits = next(value for value in judged if value["field"] == "credits")
    assert title["judged"] and title["matched"]
    # No credits gold in this document, so the value is emitted but not scored.
    assert not credits["judged"]


def test_hand_checked_scoring_separates_days_times_and_rooms():
    from pace.field_extract_eval import score_hand_gold

    records = {"doc": {"course": {"credits": {"value": 4.0}}, "meetings": [
        {"days": ["MO", "WE", "FR"], "start_time": "18:00", "end_time": "19:50",
         "room": "Hewlett 201"}]}}
    hand = {"documents": [{
        "id": "doc", "credits": 4,
        "meetings": [{"days": ["Monday", "Wednesday", "Friday"], "start": "18:00",
                      "end": "19:50", "room": "Hewlett 201"}],
    }]}
    score = score_hand_gold(records, hand)
    assert score["meetings"]["exact"] == 1
    assert score["meetings"]["rooms_correct"] == 1
    assert score["credits"]["correct"] == 1
    assert score["failures"] == []


def test_hand_checked_scoring_counts_a_wrong_room_as_a_failure():
    from pace.field_extract_eval import score_hand_gold

    records = {"doc": {"course": {}, "meetings": [
        {"days": ["MO"], "start_time": "09:00", "end_time": "10:00", "room": None}]}}
    hand = {"documents": [{"id": "doc", "meetings": [
        {"days": ["Monday"], "start": "09:00", "end": "10:00", "room": "MM 103"}]}]}
    score = score_hand_gold(records, hand)
    assert score["meetings"]["exact"] == 0
    assert score["meetings"]["days_correct"] == 1
    assert score["meetings"]["times_correct"] == 1
    assert score["meetings"]["rooms_correct"] == 0
    assert score["failures"][0]["field"] == "meetings"


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_extract_command_fills_a_record_from_a_sample_syllabus(capsys):
    """The documented command, run the way a user runs it, on committed text."""
    from pace.field_extract import main

    main(["data/samples/mock_syllabus.txt", "--checkpoint", str(_checkpoint_dir()),
          "--device", "cpu"])
    record = json.loads(capsys.readouterr().out)

    course = {name: entry["value"] for name, entry in record["course"].items()}
    assert course["course_code"] == "CHEM 1A"
    assert "General Chemistry" in course["course_title"]
    assert course["credits"] == 4.0
    assert "Okonkwo" in course["instructor"]

    weights = {row["category"]: row["weight"] for row in record["grading"]}
    assert weights == {"Homework": 20, "Laboratory": 25, "Midterm Exam": 25, "Final Exam": 30}
    assert record["grading_total_percent"] == 100
    assert record["grading_total_consistent"] is True

    meetings = {(tuple(row["days"]), row["start_time"], row["end_time"], row["room"])
                for row in record["meetings"]}
    assert (("MO", "WE", "FR"), "09:10", "10:00", "Pimentel Hall") in meetings
    assert (("TU", "TH"), "14:00", "14:50", "Latimer 120") in meetings

    for policy in ("office_hours", "attendance_policy", "academic_integrity",
                   "disability_services"):
        assert record["policies"][policy]["value"]
    # The disability statement stops before the next section's materials text.
    assert "Central Science" not in record["policies"]["disability_services"]["value"]


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_the_worked_example_in_the_docs_is_the_record_the_extractor_emits(capsys):
    """The example page cannot drift from the extractor it documents."""
    from pace.field_extract import main

    page = Path("docs/example.md").read_text(encoding="utf-8")
    source = re.search(r"```text\n(.*?)```", page, re.S).group(1)
    documented = json.loads(re.search(r"```json\n(.*?)```", page, re.S).group(1))

    sample = Path("data/samples/example_syllabus.txt").read_text(encoding="utf-8")
    assert source.strip("\n") == sample.strip("\n")

    main(["data/samples/example_syllabus.txt", "--checkpoint", str(_checkpoint_dir()),
          "--device", "cpu"])
    emitted = json.loads(capsys.readouterr().out)
    # The echoed paths depend on how the command is invoked; the record does not.
    for key in ("source", "checkpoint"):
        documented.pop(key)
        emitted.pop(key)
    assert documented == emitted


# ---------------------------------------------------------------------------
# Grading written as prose, and titles that must exist in the document
# ---------------------------------------------------------------------------

class _GradingEngine:
    def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
        return {"probabilities": {"grading_policy": 0.9}}


def test_grading_reads_a_sentence_of_parenthesised_weights():
    from pace.field_extract import extract_grading

    text = "Grades will be based on homework (20%), quizzes (10%), and the final exam (70%)."
    rows = extract_grading(text, _GradingEngine())
    assert {(row["category"].lower(), row["weight"]) for row in rows} == {
        ("homework", 20), ("quizzes", 10), ("final exam", 70)}
    assert rows[0]["weights_sum"] == 100


def test_grading_reads_a_percent_first_list():
    from pace.field_extract import extract_grading

    text = "Your grade is based on: 15% Homework, 10% Quizzes, 25% Exam 1, 50% Final Exam"
    rows = extract_grading(text, _GradingEngine())
    assert {(row["category"].lower(), row["weight"]) for row in rows} == {
        ("homework", 15), ("quizzes", 10), ("exam 1", 25), ("final exam", 50)}


def test_grading_reads_a_table_flattened_to_two_lines():
    from pace.field_extract import extract_grading

    text = "Homework\n20%\nMidterm\n30%\nFinal Exam\n50%"
    rows = extract_grading(text, _GradingEngine())
    assert {(row["category"], row["weight"]) for row in rows} == {
        ("Homework", 20), ("Midterm", 30), ("Final Exam", 50)}


def test_grading_strips_a_leading_article_from_a_category():
    from pace.field_extract import extract_grading

    text = "Homework: 25%\nEach of two tests 25%\nThe Final: 25%"
    rows = extract_grading(text, _GradingEngine())
    assert {(row["category"], row["weight"]) for row in rows} == {
        ("Homework", 25), ("tests", 25), ("Final", 25)}


def test_grading_ignores_percentages_that_are_not_weights():
    from pace.field_extract import extract_grading

    text = ("Attendance is 80% required and participation counts 20% toward the total.\n"
            "You need 90% to earn an A.")
    assert extract_grading(text, _GradingEngine()) == []


def test_nested_sub_weights_are_components_not_categories():
    from pace.field_extract import extract_grading

    text = ("WebAssign 15% (Lecture-quizzes 5% + Homework 10%)\n"
            "Lecture-quizzes (5%): answer the questions after each video.\n"
            "Homework (10%): assigned for each lecture.\n"
            "Midterm Exams 50%\n"
            "Final Exam 35%")
    rows = extract_grading(text, _GradingEngine())
    weights = {row["category"].lower(): row["weight"] for row in rows}
    assert weights.get("webassign") == 15
    assert "lecture-quizzes" not in weights and "homework" not in weights
    assert sum(weights.values()) == 100


def test_grading_ignores_range_endpoints():
    from pace.field_extract import extract_grading

    text = ("Extra Credit 0-1%\n"
            "A 90-100%\nB 80-89%\n"
            "Homework 40%\nFinal Exam 60%")
    rows = extract_grading(text, _GradingEngine())
    assert {(row["category"], row["weight"]) for row in rows} == {
        ("Homework", 40), ("Final Exam", 60)}


def test_course_code_ignores_an_office_or_room_number():
    from pace.field_extract import extract_course_code

    text = ("ECO 3704\nPrinciples of Economics\n"
            "Instructor: Dr. Smith\nOffice: MAT 328\n")
    code = extract_course_code(text, _GradingEngine())
    assert code is not None and code["value"] == "ECO 3704"

    room_first = "Room CRR 0521\nCourse Code: PCB 3402\n"
    code = extract_course_code(room_first, _GradingEngine())
    assert code is not None and code["value"] == "PCB 3402"


def test_policy_value_rejects_a_placeholder_and_the_next_section():
    from pace.field_extract import _policy_value

    assert _policy_value("office_hours", "TBD Wednesday periods 2-4", False) is None
    assert _policy_value("office_hours",
                         "Class: Thursday E1-E3 9:00-11:00 Screening: Wednesday",
                         False) is None
    assert _policy_value("office_hours", "Monday 2-4pm in WEL 5.239", False) is not None


class _DocumentEngine:
    """A tiny stand-in for the section classifier: headings say what they are."""

    def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
        if text.lower().startswith("office hours"):
            return {"label": "office_hours", "confidence": 0.99,
                    "probabilities": {"office_hours": 0.99}}
        return {"label": "course_title", "confidence": 0.4,
                "probabilities": {"course_title": 0.4}}


def test_the_policy_record_keeps_the_section_it_overwrote():
    from pace.field_extract import extract_record

    text = ("Office Hours\nM 2-4pm in the lab.\n\n"
            "Office Hours\nW 3-5pm in the lab.\n")
    record = extract_record(text, _DocumentEngine())
    entry = record["policies"]["office_hours"]
    assert entry["value"].startswith("W 3-5")
    assert [candidate["status"] for candidate in entry["candidates"]] == [
        "overwritten", "chosen"]
    assert any(item["field"] == "office_hours" and "M 2-4" in str(item["value"])
               for item in record["suppressed"])


def test_a_rejected_policy_body_is_visible_with_its_reason():
    from pace.field_extract import extract_record

    text = ("Office Hours\nTBD\n\n"
            "Office Hours\nBy appointment, M 2-4pm; email me.\n")
    record = extract_record(text, _DocumentEngine())
    entry = record["policies"]["office_hours"]
    rejected = [candidate for candidate in entry["candidates"]
                if candidate["status"] == "rejected"]
    assert rejected and "placeholder" in rejected[0]["reason"]


class _ScheduleTypedEngine:
    """Types a class-time heading as office hours, the way the model can."""

    def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
        if text.lower().startswith("class times"):
            return {"label": "office_hours", "confidence": 0.6,
                    "probabilities": {"office_hours": 0.6}}
        return {"label": "course_title", "confidence": 0.4,
                "probabilities": {"course_title": 0.4}}


def test_a_class_time_heading_is_rejected_even_when_typed_office_hours():
    from pace.field_extract import extract_record

    text = ("Class Times\n"
            "Online Section ONL: No assigned meeting times. Watch lecture videos.\n")
    record = extract_record(text, _ScheduleTypedEngine())
    assert "office_hours" not in record["policies"]
    losses = [item for item in record["suppressed"] if item["field"] == "office_hours"]
    assert [item["reason"] for item in losses] == [
        "a class-time heading, not office hours"]


def test_an_office_hours_body_must_open_with_an_availability():
    from pace.field_extract import extract_record

    # Course prose after the heading is the next section's text, not a value:
    # it names no time, day or way to meet in its opening sentence.
    prose = extract_record(
        "Office Hours\nAs part of the undergraduate curriculum, this course is "
        "designed to equip you to participate in civic discourse.\n",
        _DocumentEngine())
    assert "office_hours" not in prose["policies"]
    losses = [item for item in prose["suppressed"] if item["field"] == "office_hours"]
    assert [item["reason"] for item in losses] == [
        "body opens without naming when the instructor is available"]

    # The same shape is a true statement when its opening names the office hour.
    narrative = extract_record(
        "Office Hours\nEach week, I will hold regularly scheduled office hours "
        "set aside for helping you with issues related to the course.\n",
        _DocumentEngine())
    assert narrative["policies"]["office_hours"]["value"].startswith("Each week")

    # A title abbreviation is not the end of the opening sentence.
    titled = extract_record(
        "Office Hours\nDr. Ortiz holds office hours Tuesdays 2-4 in Evans 330.\n",
        _DocumentEngine())
    assert titled["policies"]["office_hours"]["value"].startswith("Dr. Ortiz")


def test_suppressed_grading_rows_keep_their_reason():
    from pace.field_extract import extract_record

    text = ("Homework 20%\n"
            "Homework 20%\n"
            "Exams 30% (Midterm 10% + Final 20%)\n"
            "Midterm 10%\n"
            "Participation 10%\n"
            "Final Exam 40%")
    record = extract_record(text, _DocumentEngine())
    reasons = [item["reason"] for item in record["suppressed"]
               if item["field"] == "grading"]
    assert any("component of 'Exams'" in reason for reason in reasons)
    assert any("duplicate" in reason for reason in reasons)


def test_a_grading_row_too_long_to_read_is_a_visible_loss_and_the_record_says_so():
    """The audit's case: a printed 10% row vanished and a 90% sum was called consistent."""
    from pace.field_extract import extract_record

    text = ("Grading\n"
            "Homework 20%\n"
            "Class participation and weekly written reflection journal entries 10%\n"
            "Exams 30% (Midterm 10% + Final 20%)\n"
            "Final Exam 40%\n")
    record = extract_record(text, _DocumentEngine())
    assert sum(row["weight"] for row in record["grading"]) == 90
    assert record["grading_total_consistent"] is False
    assert all(row["weights_sum_consistent"] is False for row in record["grading"])
    lost = [item for item in record["suppressed"] if item["field"] == "grading"]
    assert any("10%" in str(item["value"]) and "no category name" in item["reason"]
               for item in lost)


def test_the_no_category_note_is_only_for_label_shaped_text():
    """CSS, grade scales and prose are not called a missing category name."""
    from pace.field_extract import extract_grading
    from pace.field_ledger import Ledger

    text = ("#activities-schedule-table {margin: 0 10%; padding: 1 100%}\n"
            "April 1: 90.0% = A-, 89.9-87.0% = B+\n"
            "You may miss 5% of the sessions without penalty.\n")
    ledger = Ledger()
    extract_grading(text, _DocumentEngine(), ledger=ledger)
    assert not [item for item in ledger.losses()
                if "no category name" in item["reason"]]


def test_no_cap_hides_a_rejected_code_or_title():
    from pace.field_extract import extract_course_code, extract_course_name
    from pace.field_ledger import Ledger

    rooms = "CHEM 101: General Chemistry\n" + "".join(
        f"Office: WEL {5 + index}.{100 + index} Hall\n" for index in range(20))
    code_losses = Ledger()
    extract_course_code(rooms, _DocumentEngine(), ledger=code_losses)
    assert len(code_losses) == 20

    terms = "Syllabus\n" + "".join(
        f"Meeting time line {index} Fall 2011\n" for index in range(20))
    title_losses = Ledger()
    extract_course_name(terms, _DocumentEngine(), [], ledger=title_losses)
    # The 20 term lines plus the document's own heading; the old cap stopped at 8.
    assert len(title_losses) >= 20


def test_meetings_note_each_line_they_drop():
    from pace.field_extract import extract_meetings
    from pace.field_ledger import Ledger

    text = "\n".join([
        "O\ufb03ce Hours: Tuesday and Thursday 10:30 AM to 12 PM",
        "Final Exams: 38760, MWF 11-12:  Thursday, 5/16, 9 am - 12 noon",
        "Lecture: MWF 11-12, WCH 1.120",
        "Lecture: MWF 11-12, WCH 1.120",
    ])
    ledger = Ledger()
    meetings = extract_meetings(text, _DocumentEngine(), ledger=ledger)
    assert len(meetings) == 1
    reasons = [item["reason"] for item in ledger.losses()]
    assert any("office-hours" in reason for reason in reasons)
    assert any("exam line" in reason for reason in reasons)
    assert any("duplicate" in reason for reason in reasons)


def test_one_named_rule_governs_every_sum_check():
    from pace.field_extract import GRADING_SUM_TOLERANCE, grading_sum_consistent

    assert GRADING_SUM_TOLERANCE == 5
    assert [grading_sum_consistent(total) for total in (94, 95, 100, 105, 106)] == [
        False, True, True, True, False]


def test_suppressed_values_is_the_public_ledger():
    from pace.field_extract import extract_record
    from pace.field_ledger import suppressed_values

    text = ("Office Hours\nM 2-4pm in the lab.\n\n"
            "Office Hours\nW 3-5pm in the lab.\n")
    record = extract_record(text, _DocumentEngine())
    ledger = suppressed_values(record)
    assert ledger == [{"field": item["field"], "value": item["value"],
                       "reason": item["reason"]} for item in record["suppressed"]]
    assert ledger and all(item["field"] and item["reason"] for item in ledger)


def test_ledger_notes_one_shape_and_truncates_evidence():
    from pace.field_ledger import EVIDENCE_CHARS, Ledger

    ledger = Ledger()
    ledger.note("course_title", "Grading", "a section heading", "x" * 500)
    ledger.extend("grading", [{"value": "Homework 20%", "reason": "duplicate of 'Homework'"}])
    assert len(ledger) == 2
    first, second = ledger.losses()
    assert set(first) == {"field", "value", "reason", "evidence"}
    assert len(first["evidence"]) == EVIDENCE_CHARS
    assert second == {"field": "grading", "value": "Homework 20%",
                      "reason": "duplicate of 'Homework'", "evidence": ""}


def test_a_generic_heading_is_not_the_course_title():
    from pace.field_extract import extract_course_name

    class Engine:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.99}}

    text = ("General Information\n"
            "Department of Chemistry\n"
            "CHEM 101: General Chemistry\n")
    value = extract_course_name(text, Engine(), [])
    assert value is None or value["value"].lower() not in {
        "general information", "department of chemistry"}


def test_title_rejects_a_pasted_file_name_and_a_sentence():
    from pace.field_extract import title_plausible

    assert not title_plausible("CH301 - McCord - Syllabus - Spring 2011")
    assert not title_plausible("Chemistry 130 is a general introduction to the principles of")
    assert not title_plausible("CHEM_101_syllabus.pdf")
    assert title_plausible("CHEM 1A: General Chemistry")
    assert title_plausible("CS103 Course Syllabus")


def test_document_labels_are_rejected_with_reasons():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    clean = "Chemistry 1A: General Chemistry"
    for label in ("CS 180 -- Course Syllabus", "Syllabus | Math 140", "Page 1 of 4",
                  "SPAN 1002 Syllabus", "English Composition II | Syllabus"):
        ledger = Ledger()
        value = extract_course_name(f"{label}\n{clean}\n", _DocumentEngine(), [],
                                    ledger=ledger)
        assert value is not None and value["value"] == clean
        reasons = {item["value"]: item["reason"] for item in ledger.losses()}
        assert reasons[label] == "a document label, not a course name"


def test_a_title_that_starts_with_general_is_not_a_section_heading():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    text = ("CHEMISTRY 130\n"
            "General Chemistry: Macroscopic Investigations and Reaction Principles\n"
            "Fall 2003 Syllabus\n")
    value = extract_course_name(text, _DocumentEngine(), [])
    assert value is not None
    assert value["value"].startswith("General Chemistry: Macroscopic")

    heading = Ledger()
    extract_course_name("General Course Information\nCHEM 101: General Chemistry\n",
                        _DocumentEngine(), [], ledger=heading)
    reasons = {item["value"]: item["reason"] for item in heading.losses()}
    assert reasons["General Course Information"] == "a section heading"


def test_a_rhetorical_heading_and_a_meeting_slot_are_rejected_with_reasons():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    text = ("CS10 Spring 2021\n"
            "Collaboration, Not Competition\n"
            "MWF 1-2 pm, O. Hallatschek\n"
            "Physics 7A: Mechanics\n")
    ledger = Ledger()
    value = extract_course_name(text, _DocumentEngine(), [], ledger=ledger)
    assert value is not None and value["value"] == "Physics 7A: Mechanics"
    reasons = {item["value"]: item["reason"] for item in ledger.losses()}
    assert reasons["Collaboration, Not Competition"] == "a heading, not a course name"
    assert reasons["MWF 1-2 pm, O. Hallatschek"] == "a contact line"


def test_a_labelled_attribute_is_metadata_but_a_subtitle_is_a_title():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    text = ("Skills and Experience Flag: Quantitative Reasoning\n"
            "Course Format: Online, Self-Paced\n"
            "Canvas: CS 10\n"
            "General Chemistry: Macroscopic Investigations and Reaction Principles\n")
    ledger = Ledger()
    value = extract_course_name(text, _DocumentEngine(), [], ledger=ledger)
    assert value is not None and value["value"].startswith("General Chemistry: Macroscopic")
    reasons = {item["value"]: item["reason"] for item in ledger.losses()}
    assert reasons["Skills and Experience Flag: Quantitative Reasoning"] == \
        "a labelled attribute, not a course name"
    assert reasons["Course Format: Online, Self-Paced"] == \
        "a labelled attribute, not a course name"
    assert reasons["Canvas: CS 10"] == "a labelled link, not a course name"


def test_a_sole_candidate_below_the_floor_is_a_reasoned_loss_and_an_abstention():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    class Weak:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.1}}

    ledger = Ledger()
    assert extract_course_name("Introduction to American Government\n", Weak(), [],
                               ledger=ledger) is None
    losses = ledger.losses()
    assert losses[0]["value"] == "Introduction to American Government"
    assert losses[0]["reason"] == "below the plausibility floor"
    # The candidate's rejection is not the field's answer: the record says the
    # field came back empty and why, rather than reporting a silent `None`.
    assert losses[1]["value"] is None
    assert losses[1]["reason"] == "no course name: the best line is not a printed title"
    assert "Introduction to American Government" in losses[1]["evidence"]


def test_a_page_whose_only_title_shaped_lines_are_labels_abstains_with_the_label():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    ledger = Ledger()
    value = extract_course_name("CS 101 Syllabus\nMay 16-June 24\n",
                                _DocumentEngine(), [], ledger=ledger)
    assert value is None
    losses = ledger.losses()
    note = [item for item in losses if item["value"] is None]
    assert len(note) == 1
    assert note[0]["reason"] == ("no course name: every title-shaped line names "
                                 "something other than the course")
    assert note[0]["evidence"] == "CS 101 Syllabus"
    reasons = {item["value"]: item["reason"] for item in losses if item["value"]}
    assert reasons["May 16-June 24"] == "a date or week line, not a course name"


def test_a_page_with_no_title_shaped_line_says_so_instead_of_naming_a_label():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    ledger = Ledger()
    value = extract_course_name("\n\n{\n}\n\n", _DocumentEngine(), [], ledger=ledger)
    assert value is None
    note = [item for item in ledger.losses() if item["value"] is None]
    assert note and note[0]["reason"] == "no course name: nothing on the page is shaped like a title"


def test_abstention_evidence_is_a_printed_line_not_page_script():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    script = "(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':1});});"
    ledger = Ledger()
    value = extract_course_name(
        f"Department of Biological Sciences - Purdue University\n{script}\n",
        _DocumentEngine(), [], ledger=ledger)
    assert value is None
    note = [item for item in ledger.losses() if item["value"] is None][0]
    assert "function" not in note["evidence"] and "Purdue" in note["evidence"]


def test_a_title_glued_to_page_script_is_still_read():
    """A raw page prints its heading on the same line as a script block."""
    from pace.field_extract import extract_course_name

    class Engine:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.8}}

    text = ('   Introductory Sociology        '
            'window.__SITE_ENV_CONFIG__ = {"siteId":"x"};\n'
            "Reading: Chapters 1-3\n")
    value = extract_course_name(text, Engine(), [])
    assert value is not None and value["value"] == "Introductory Sociology"
    assert value["grounded"] and value["value"] in text


def test_a_one_letter_department_code_vouches_for_its_own_line():
    """The classifier scores a packed header line low; the code is the evidence."""
    from pace.field_extract import extract_course_name

    class Weak:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.05}}

    value = extract_course_name("M 408C: Differential and Integral Calculus\n",
                                Weak(), [])
    assert value is not None and value["value"] == "M 408C: Differential and Integral Calculus"
    assert value["confidence"] == 0.86


def test_a_numbered_list_marker_is_not_a_catalogue_code():
    """"1. Read the syllabus" is a bullet, not a course number.

    The marker once counted as a code, which let a page of policy bullets emit
    fabricated titles at code-bearing confidence: the marker took the score bonus,
    skipped the plausibility floor and satisfied the promotion gate.
    """
    from pace.field_extract import (TITLE_ABSTENTION, catalogue_code,
                                    code_names_a_course, extract_course_name)
    from pace.field_ledger import Ledger

    assert catalogue_code("1. Read the syllabus before class") is None
    assert catalogue_code("2. Come prepared every day") is None
    assert not code_names_a_course("2. Come prepared every day")

    class Weak:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.05}}

    ledger = Ledger()
    value = extract_course_name(
        "1. Read the syllabus before class\n2. Come prepared every day\n"
        "3. Ask questions early\n", Weak(), [], ledger=ledger)
    assert value is None
    notes = [item for item in ledger.losses() if item["value"] is None]
    assert len(notes) == 1 and notes[0]["reason"] in TITLE_ABSTENTION.values()


def test_one_predicate_answers_whether_a_line_carries_a_course_code():
    """The score, the floor, the promotion gate and the label check share it."""
    from pace.field_extract import catalogue_code, code_names_a_course

    for line in ("MATH 1A: Calculus",
                 "Chemistry 1A: General Chemistry",
                 "M 408C: Differential and Integral Calculus",
                 "P.S. 101 INTRODUCTION TO AMERICAN GOVERNMENT",
                 "Hist. 151: American History to 1877",
                 "General Chemistry 103",
                 "21-120 : Differential and Integral Calculus",
                 "BIOL 11: Introductory Biology I"):
        assert code_names_a_course(line), line
    for line in ("Set 3",
                 "V 1.4 Release Notes",
                 "Uniques: 49970 and 49975 · TTh in WEL 2.224",
                 "1. Read the syllabus before class",
                 "HIS 315k External",
                 "May 16-June 24",
                 "WEL 2.224"):
        assert not code_names_a_course(line), line
    # A version is not a code at all, and a delivery mode is not a name.
    assert catalogue_code("V 1.4 Release Notes") is None
    assert catalogue_code("HIS 315k External") is not None


def test_a_code_without_a_name_abstains_at_the_floor_it_does_not_skip():
    from pace.field_extract import TITLE_ABSTENTION, extract_course_name
    from pace.field_ledger import Ledger

    class Weak:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.05}}

    ledger = Ledger()
    assert extract_course_name("HIS 315k External\n", Weak(), [], ledger=ledger) is None
    notes = [item for item in ledger.losses() if item["value"] is None]
    assert [item["reason"] for item in notes] == [TITLE_ABSTENTION["floor"]]


def test_every_title_abstention_uses_the_shared_reason_vocabulary():
    from pace.field_extract import TITLE_ABSTENTION, extract_course_name
    from pace.field_ledger import Ledger

    class Weak:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.05}}

    for text in ("CS 101 Syllabus\nMay 16-June 24\n",
                 "\n\n{\n}\n\n",
                 "HIS 315k External\n"):
        ledger = Ledger()
        assert extract_course_name(text, Weak(), [], ledger=ledger) is None
        notes = [item for item in ledger.losses() if item["value"] is None]
        assert notes and all(item["reason"] in TITLE_ABSTENTION.values()
                             for item in notes), text


def test_a_speaker_with_an_affiliation_is_not_a_course_name():
    from pace.field_extract import extract_course_name
    from pace.field_ledger import Ledger

    ledger = Ledger()
    extract_course_name("Syllabus for PSCI 310L\nPaul Light (Brookings Institution)\n",
                        _DocumentEngine(), [], ledger=ledger)
    reasons = {item["value"]: item["reason"] for item in ledger.losses() if item["value"]}
    assert reasons["Paul Light (Brookings Institution)"] == "a person's name, not a course name"


def test_title_value_is_text_that_exists_in_the_document():
    from pace.field_extract import extract_course_name

    class Engine:
        def classify(self, text: str, top_k: int = 1, **kwargs) -> dict:
            return {"probabilities": {"course_title": 0.92}}

    text = ("Syllabus\n"
            "Chemistry 130 is a general introduction to the principles of chemistry.\n"
            "CHEM 130: General Chemistry\n")
    value = extract_course_name(text, Engine(), [])
    assert value is None or (value["grounded"] and value["value"] in text)


# ---------------------------------------------------------------------------
# The evaluation protocol: dev and test institutions stay separate
# ---------------------------------------------------------------------------

def test_every_institution_belongs_to_exactly_one_side_of_the_split():
    from pace.field_extract_eval import DEV_SCHOOLS, TEST_SCHOOLS, split_of

    assert not set(DEV_SCHOOLS) & set(TEST_SCHOOLS)
    for school in DEV_SCHOOLS:
        assert split_of(school) == "dev"
    for school in TEST_SCHOOLS:
        assert split_of(school) == "test"
    assert split_of("University of California, Berkeley") is None


def test_split_artifact_records_documents_and_exclusions():
    from pace.field_extract_eval import DEV_SCHOOLS, TEST_SCHOOLS

    path = Path("data/processed/field/extract_split.json")
    if not path.exists():
        pytest.skip("split artifact is not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["development_schools"]) == set(DEV_SCHOOLS)
    assert set(TEST_SCHOOLS) <= set(payload["test_schools"])
    assert payload["document_counts"]["dev"] > 0
    assert payload["document_counts"]["test"] > 0
    assert sum(payload["documents"]["dev"].values()) == payload["document_counts"]["dev"]
    gold = payload.get("test_gold_documents") or {"documents": 0, "schools": []}
    corpus_test = sum(payload["documents"]["test"].values())
    assert payload["document_counts"]["test"] == corpus_test + gold["documents"]
    if gold["documents"]:
        assert set(gold["schools"]) <= set(payload["test_schools"])
        assert not set(gold["schools"]) & set(payload["development_schools"])
        assert sum(entry["documents"] for entry in gold["by_school"].values()) == \
            gold["documents"]


def test_published_report_lists_every_cause_including_abstentions(tmp_path):
    """The per-cause table is complete: no reason is folded into "other"."""
    from pace.field_extract_eval import write_protocol_report

    paths = [Path("data/processed/field/extract_split.json"),
             Path("artifacts/field-extract/extract_eval_dev.json"),
             Path("artifacts/field-extract/extract_eval_test.json")]
    if not all(path.exists() for path in paths):
        pytest.skip("recorded runs are not present")
    split, dev, test = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    target = tmp_path / "EXTRACTION_EVAL.md"
    write_protocol_report(split, dev, test, target)
    text = target.read_text(encoding="utf-8")
    assert "for other reasons" not in text
    for field, reasons in dev["coverage_reasons_by_field"].items():
        for reason, count in reasons.items():
            assert f"| `{field}` | {reason} | {count} |" in text, (field, reason)
    abstentions = [reason for reasons in dev["coverage_reasons_by_field"].values()
                   for reason in reasons if reason.startswith("no course name:")]
    assert abstentions, "the development run has no abstention to render"
    for reason in abstentions:
        assert f"| {reason} |" in text


def test_published_report_quotes_the_test_split(tmp_path):
    """The published table must lead with the test number, not the tuned one."""
    from pace.field_extract_eval import write_protocol_report

    paths = [Path("data/processed/field/extract_split.json"),
             Path("artifacts/field-extract/extract_eval_dev.json"),
             Path("artifacts/field-extract/extract_eval_test.json")]
    if not all(path.exists() for path in paths):
        pytest.skip("recorded runs are not present")
    split, dev, test = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    target = tmp_path / "EXTRACTION_EVAL.md"
    write_protocol_report(split, dev, test, target)
    text = target.read_text(encoding="utf-8")
    headline = text.split("## Development split")[0]
    final = test["overall"]
    gold = test.get("field_gold") or {}
    if gold:
        overall = gold["overall"]
        assert f"{overall['matched']}/{overall['judged']}" in headline
        assert f"precision {overall['precision']:.3f}" in headline
        assert f"{overall['precision_lower_95']:.3f}" in headline
        assert "## Reliability" in text and "## What a default threshold would emit" in text
        if gold["operating_point"]["threshold"] is None:
            assert "No threshold certifies" in text
    else:
        assert f"precision {final['precision']:.3f}" in headline
    assert "development" in text and "test" in text
    assert "scored once" in text


# ---------------------------------------------------------------------------
# Value-level gold: absent fields are false positives, not silence
# ---------------------------------------------------------------------------

def test_value_gold_covers_the_fetched_corpus_exactly():
    """Every gold document exists, and every fetched document is labelled."""
    from pace.field_gold import FIELDS, load_gold_payload

    gold_path = Path("data/processed/field/gold_test_fields.json")
    corpus_path = Path("data/raw/field-eval/syllabi.jsonl")
    if not gold_path.exists() or not corpus_path.exists():
        pytest.skip("value-level gold is not present")
    payload = load_gold_payload(gold_path)
    # Iterate the file rather than splitting on newlines: PDF text can carry
    # U+2028 or a form feed, which str.splitlines() would treat as a line break
    # in the middle of a JSON record.
    with corpus_path.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    gold_ids = [document["id"] for document in payload["documents"]]
    assert len(gold_ids) == len(set(gold_ids))
    assert set(gold_ids) == {document["id"] for document in documents}
    for document in payload["documents"]:
        named = [field for field in FIELDS if document.get(field)]
        assert named, document["school"]
        assert not set(named) & set(document.get("absent") or [])
        assert set(document.get("absent") or []) <= set(FIELDS)
        for field in ("meetings", "grading"):
            for item in document.get(field) or []:
                assert item


def test_value_gold_documents_are_not_in_the_classifier_corpus():
    """The only documents where neither the rules nor the encoder saw the text."""
    corpus_path = Path("data/processed/field/corpus.jsonl")
    corpus_text = Path("data/raw/field-eval/syllabi.jsonl")
    if not corpus_path.exists() or not corpus_text.exists():
        pytest.skip("corpus is not present")
    with corpus_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    haystack = " ".join(" ".join(str(row.get("text", "")).split()).lower() for row in rows)
    with corpus_text.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    for document in documents:
        words = " ".join(document["text"].split()).lower().split()
        probe = " ".join(words[len(words) // 3: len(words) // 3 + 9])
        assert len(probe.split()) == 9
        assert probe not in haystack, document["url"]


def test_absent_field_makes_an_emitted_value_a_false_positive():
    from pace.field_gold import judge

    gold = {"absent": ["credits"]}
    judged, matched = judge({"field": "credits", "text": "4 credits"}, gold)
    assert judged is True and matched is False
    judged, matched = judge({"field": "course_title", "text": "Anything"}, gold)
    assert judged is False and matched is False


def test_gold_matching_accepts_slot_orders_codes_and_weights():
    from pace.field_gold import judge

    gold = {"meetings": [{"days": ["Tuesday", "Thursday"], "start": "19:00",
                          "end": "20:30"}]}
    assert judge({"field": "meetings", "days": ["TH", "TU"], "start_time": "19:00",
                  "end_time": "20:30"}, gold) == (True, True)
    assert judge({"field": "meetings", "days": ["TU"], "start_time": "19:00",
                  "end_time": "20:30"}, gold) == (True, False)
    codes = {"course_code": ["AMH 4373"]}
    assert judge({"field": "course_code", "text": "AMH4373"}, codes)[1] is True
    grading = {"grading": [{"category": "Final Exam", "weight": 35}]}
    assert judge({"field": "grading", "text": "final exams", "weight": 35}, grading)[1] is True
    assert judge({"field": "grading", "text": "Final Exam", "weight": 30}, grading)[1] is False


def test_sweep_reports_coverage_and_a_lower_bound_that_shrinks_with_errors():
    from pace.field_gold import sweep

    values = ([{"confidence": 0.995, "judged": True, "matched": True}] * 30
              + [{"confidence": 0.995, "judged": True, "matched": False}]
              + [{"confidence": 0.4, "judged": True, "matched": True}] * 5
              + [{"confidence": 0.4, "judged": False, "matched": False}] * 10)
    rows = {row["threshold"]: row for row in sweep(values)}
    low, high = rows[0.0], rows[0.99]
    assert low["coverage"] == 1.0 and low["accepted"] == 36
    assert high["coverage"] < low["coverage"]
    assert high["errors"] <= low["errors"]
    assert 0.0 < high["precision_lower_95"] < high["precision"]
    assert rows[0.995]["accepted"] == 31


def test_coverage_reasons_add_up_to_the_suppressed_count():
    """A field's reasons must account for every value it lost, or the ledger is short."""
    from pace.field_ledger import reason_group, summarize

    assert reason_group("duplicate of 'Homework'") == "duplicate of an earlier row"
    assert reason_group("duplicate of 'font-size'") == "duplicate of an earlier row"
    assert reason_group("component of 'Exams'") == "a component of the parent weight"
    assert reason_group("body is not about the field") == "body is not about the field"
    folded = summarize([{"field": "grading", "reason": "duplicate of 'Homework'"},
                        {"field": "grading", "reason": "duplicate of 'Exams'"},
                        {"field": "grading", "reason": "component of 'Exams'"}])
    assert folded == {"grading": {"duplicate of an earlier row": 2,
                                  "a component of the parent weight": 1}}
    path = Path("artifacts/field-extract/extract_eval_dev.json")
    if not path.exists():
        pytest.skip("the recorded development run is not present")
    result = json.loads(path.read_text(encoding="utf-8"))
    coverage, by_field = result["coverage"], result["coverage_reasons_by_field"]
    for field, stats in coverage.items():
        if not stats["suppressed"]:
            continue
        listed = sum((by_field.get(field) or {}).values())
        assert listed == stats["suppressed"], (field, listed, stats["suppressed"])


def test_readability_gates_reject_unreadable_documents_only():
    from scripts.fetch_eval_syllabi import readable

    good = "Syllabus for Economics 1. " * 80
    assert readable(good)[0] is True
    assert readable("too short")[0] is False
    assert readable("\x00\x01\x02" * 500)[0] is False
    assert readable("1234 5678 90 " * 200)[0] is False


def test_default_threshold_is_the_measured_one():
    """Changing it is a re-measurement, not a tweak."""
    from pace.field_extract import DEFAULT_MIN_CONFIDENCE

    assert DEFAULT_MIN_CONFIDENCE == 0.0


def test_default_threshold_is_the_one_the_recorded_runs_were_scored_at():
    from pace.field_extract import DEFAULT_MIN_CONFIDENCE

    for name in ("dev", "test"):
        path = Path(f"artifacts/field-extract/extract_eval_{name}.json")
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["overall"]["min_confidence"] == DEFAULT_MIN_CONFIDENCE
        if payload.get("field_gold"):
            assert payload["field_gold"]["overall"]["min_confidence"] == DEFAULT_MIN_CONFIDENCE
            sweep_rows = payload["field_gold"]["sweep"]
            assert sweep_rows[0]["threshold"] == 0.0


def test_an_absent_gold_room_does_not_have_to_be_matched():
    """`compact(None)` used to be "none", so a roomless gold meeting never matched."""
    from pace.field_gold import compact, judge

    assert compact(None) == ""
    gold = {"meetings": [{"days": ["Monday", "Wednesday", "Friday"], "start": "13:00",
                          "end": "14:00"}]}
    assert judge({"field": "meetings", "days": ["MO", "WE", "FR"], "start_time": "13:00",
                  "end_time": "14:00", "room": "atschek"}, gold) == (True, True)


# ---------------------------------------------------------------------------
# The held-out set: measurement on documents no rule was tuned on

def test_heldout_gold_matches_its_frozen_hash_and_selection():
    from pace.field_extract_eval import DEV_SCHOOLS, TEST_SCHOOLS
    from pace.field_gold import FIELDS, load_gold
    from pace.field_heldout_eval import (GOLD, SELECTION, check_freeze, load_documents,
                                         load_selection, sha256)

    if not GOLD.exists() or not SELECTION.exists():
        pytest.skip("the held-out gold is not present")
    selection = load_selection(SELECTION)
    assert selection["gold_sha256"] == sha256(GOLD)
    check_freeze(selection, GOLD)  # must not raise
    gold = load_gold(GOLD)
    ids = [entry["id"] for entry in selection["documents"]]
    assert list(gold) == ids
    assert [document["id"] for document in load_documents(selection)] == ids
    for entry in selection["documents"]:
        assert entry["school"] not in DEV_SCHOOLS + TEST_SCHOOLS, entry["school"]
        document = gold[entry["id"]]
        named = [field for field in FIELDS if document.get(field)]
        assert named, entry["id"]
        assert not set(named) & set(document.get("absent") or [])


def test_heldout_scoring_refuses_a_changed_gold(tmp_path):
    from pace.field_heldout_eval import GOLD, SELECTION, check_freeze, load_selection

    if not GOLD.exists() or not SELECTION.exists():
        pytest.skip("the held-out gold is not present")
    selection = load_selection(SELECTION)
    edited = tmp_path / "gold.json"
    edited.write_text(GOLD.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        check_freeze(selection, edited)
    check_freeze(selection, edited, allow_changed_gold=True)


def test_heldout_bound_arithmetic_matches_the_decision():
    """149 clean values bound 0.98; 61 cannot, however perfect they are."""
    from pace.field_calibration import error_upper_bound
    from pace.field_heldout_eval import needed_for, perfect_bound

    assert needed_for() == 149
    assert perfect_bound(148) < 0.98 <= perfect_bound(149)
    assert perfect_bound(61) == pytest.approx(0.9521, abs=5e-4)
    assert 1 - error_upper_bound(0, 149, 0.95) == pytest.approx(perfect_bound(149))


def test_heldout_run_is_internally_consistent():
    from pace.field_calibration import error_upper_bound
    from pace.field_gold import CONFIDENCE
    from pace.field_heldout_eval import GOLD, perfect_bound, sha256

    path = Path("artifacts/field-extract/heldout_eval.json")
    if not path.exists() or not GOLD.exists():
        pytest.skip("the held-out run is not present")
    result = json.loads(path.read_text(encoding="utf-8"))
    overall, certification = result["overall"], result["certification"]
    assert result["protocol"]["gold_sha256"] == sha256(GOLD)
    assert overall["judged"] == overall["matched"] + overall["errors"]
    assert overall["errors"] == len(result["errors"])
    assert overall["precision"] == pytest.approx(overall["matched"] / overall["judged"])
    bound = error_upper_bound(overall["errors"], overall["judged"], CONFIDENCE)
    assert overall["precision_lower_95"] == pytest.approx(1 - bound)
    assert sum(row["judged"] for row in result["by_field"].values()) == overall["judged"]
    assert sum(row["matched"] for row in result["by_field"].values()) == overall["matched"]
    assert certification["judged"] == overall["judged"]
    assert certification["perfect_lower_95"] == pytest.approx(perfect_bound(overall["judged"]))
    assert certification["needed_for_target"] == 149
    assert certification["certifies_target"] is False
    assert result["losses"]["emitted"] == overall["emitted"]
    assert result["losses"]["suppressed"] == sum(result["losses"]["suppressed_by_field"].values())
    assert result["protocol"]["truncated"] is False
    assert result["protocol"]["documents"] == result["protocol"]["documents_selected"] == 20


def test_heldout_values_account_for_every_emitted_value():
    """The artifact's headline must be a fold of its own per-value rows."""
    from collections import Counter

    from pace.field_calibration import error_upper_bound
    from pace.field_gold import CONFIDENCE

    path = Path("artifacts/field-extract/heldout_eval.json")
    if not path.exists():
        pytest.skip("the held-out run is not present")
    result = json.loads(path.read_text(encoding="utf-8"))
    rows, overall = result["values"], result["overall"]
    decisions = Counter(row["decision"] for row in rows)
    assert len(rows) == overall["emitted"]
    assert decisions["matched"] == overall["matched"]
    assert decisions["wrong"] == overall["errors"]
    assert decisions["matched"] + decisions["wrong"] == overall["judged"]
    assert decisions["matched"] / (decisions["matched"] + decisions["wrong"]) == \
        pytest.approx(overall["precision"])
    assert 1 - error_upper_bound(decisions["wrong"], decisions["matched"] + decisions["wrong"],
                                 CONFIDENCE) == pytest.approx(overall["precision_lower_95"])
    assert {row["document"] for row in rows} <= {document["id"] for document in result["documents"]}
    assert sum(document["emitted"] for document in result["documents"]) == len(rows)
    assert len({(row["document"], row["field"], row["text"]) for row in rows}) == len(rows)
    for row in rows:
        assert row["field"] and row["rule"] and row["document"]
        assert row["decision"] in {"matched", "wrong", "unjudged"}
        if row["decision"] == "matched":
            assert row["gold_entry"] is not None and row["compared_against"] is None
        elif row["decision"] == "wrong":
            assert row["gold_entry"] is None and row["compared_against"] is not None
        else:
            assert row["gold_entry"] is None and row["why_unjudged"]
    assert len(result["errors"]) == overall["errors"]


def test_heldout_artifact_pins_the_published_numbers():
    """If these move, the published report must move with them."""
    path = Path("artifacts/field-extract/heldout_eval.json")
    if not path.exists():
        pytest.skip("the held-out run is not present")
    result = json.loads(path.read_text(encoding="utf-8"))
    overall = result["overall"]
    assert overall["documents"] == 20
    assert (overall["emitted"], overall["judged"], overall["matched"], overall["errors"]) == \
        (73, 61, 50, 11)
    assert round(overall["precision"], 3) == 0.820
    assert round(overall["precision_lower_95"], 3) == 0.719
    assert result["losses"]["suppressed"] == 245
    assert result["certification"]["certifies_target"] is False


def test_the_second_touch_is_labelled_and_the_first_touch_stays_frozen():
    first_path = Path("artifacts/field-extract/heldout_eval.json")
    second_path = Path("artifacts/field-extract/heldout_eval_second_touch.json")
    if not first_path.exists() or not second_path.exists():
        pytest.skip("the held-out runs are not present")
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    # The published artifact carries no note and keeps the number pinned elsewhere.
    assert first["protocol"].get("note") is None
    assert round(first["overall"]["precision"], 3) == 0.820
    # A second look must say what it is: same documents, same gold, after a fix
    # written knowing the first touch's errors.
    note = second["protocol"]["note"].lower()
    assert "second touch" in note and "contaminated" in note
    assert "not an independent estimate" in note
    assert second["protocol"]["gold_sha256"] == first["protocol"]["gold_sha256"]
    assert second["protocol"]["documents"] == first["protocol"]["documents"] == 20
    assert second["protocol"]["truncated"] is False
    assert second["overall"]["judged"] == second["overall"]["matched"] + \
        second["overall"]["errors"]
    text = Path("REPORT.md").read_text(encoding="utf-8")
    start = text.index("### The second touch")
    section = text[start: text.index("### What the record throws away", start)]
    assert f"{second['overall']['matched']} of {second['overall']['judged']} judged values" \
        in section
    assert f"{second['overall']['precision']:.3f}" in section
    assert f"{first['overall']['precision']:.3f}" in section
    assert "contaminated by construction" in section


def test_truncated_run_marks_itself_and_cannot_overwrite_the_published_output():
    from pace.field_heldout_eval import (DEFAULT_OUTPUT, GOLD, load_selection, main,
                                         run_protocol, score)

    if not GOLD.exists():
        pytest.skip("the held-out gold is not present")
    selection = load_selection()
    assert run_protocol(selection)["truncated"] is False
    short = run_protocol(selection, limit=2)
    assert short["truncated"] is True
    assert (short["documents"], short["documents_selected"], short["limit"]) == (2, 20, 2)
    fake_selection = {"documents": [{"id": "a", "school": "S"},
                                     {"id": "b", "school": "S"}]}
    fake_record = {"course": {}, "grading": [], "meetings": [], "policies": {},
                   "suppressed": []}
    empty = score({"a": fake_record}, {"a": {}, "b": {}}, fake_selection, limit=1)
    assert empty["protocol"]["truncated"] is True
    assert (empty["protocol"]["documents"], empty["protocol"]["documents_selected"]) == (1, 2)
    assert empty["overall"]["documents"] == 1
    before = DEFAULT_OUTPUT.read_bytes() if DEFAULT_OUTPUT.exists() else None
    with pytest.raises(SystemExit):
        main(["--limit", "1"])
    after = DEFAULT_OUTPUT.read_bytes() if DEFAULT_OUTPUT.exists() else None
    assert after == before
    # A full re-run does not silently replace the published artifact either: the
    # first touch is evidence, and a second look needs its own path.
    with pytest.raises(SystemExit):
        main([])
    assert DEFAULT_OUTPUT.read_bytes() == before


def test_certification_uses_the_same_confidence_as_the_bound():
    from pace.field_calibration import error_upper_bound
    from pace.field_gold import CONFIDENCE
    from pace.field_heldout_eval import needed_for, perfect_bound

    assert perfect_bound(61) == pytest.approx(1 - error_upper_bound(0, 61, CONFIDENCE))
    assert perfect_bound(61, 0.99) == pytest.approx(0.01 ** (1 / 61))
    assert needed_for(0.98, 0.99) > needed_for(0.98, CONFIDENCE)
    path = Path("artifacts/field-extract/heldout_eval.json")
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["certification"]["confidence"] == CONFIDENCE


def test_gold_loader_has_one_owner_and_validates(tmp_path):
    import pace.field_heldout_eval as heldout
    from pace.field_gold import load_gold

    assert heldout.load_gold is load_gold
    for payload in ({"protocol": []},
                    {"documents": [{"id": "a"}, {"id": "a"}]},
                    {"documents": [{"school": "nowhere"}]}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold(path)
    for gold in (Path("data/processed/field/gold_heldout_fields.json"),
                 Path("data/processed/field/gold_test_fields.json")):
        if gold.exists():
            assert load_gold(gold)


def test_report_heldout_section_matches_the_artifact():
    """The prose numbers are the artifact's numbers, not a hand-typed copy."""
    path = Path("artifacts/field-extract/heldout_eval.json")
    report = Path("REPORT.md")
    if not path.exists() or not report.exists():
        pytest.skip("the held-out run or report is not present")
    result = json.loads(path.read_text(encoding="utf-8"))
    overall = result["overall"]
    text = report.read_text(encoding="utf-8")
    start = text.index("### The held-out measurement")
    section = text[start: text.index("### What the record throws away", start)]
    assert f"{overall['matched']} of {overall['judged']} judged values correct" in section
    assert f"precision {overall['precision']:.3f}" in section
    assert f"lower bound {overall['precision_lower_95']:.3f}" in section
    assert f"emitted {overall['emitted']} values" in section
    assert f"{result['losses']['suppressed']} values were read and suppressed" in section
    assert "protocol.truncated: false" in section
    assert "values" in section and "recomputed" in section


def test_layer_wise_decay_indexes_every_encoder_family():
    from pace.field_train import _layer_id

    assert _layer_id("encoder.layer.0.attention.self.query.weight", 12) == 1
    assert _layer_id("layers.5.mlp.Wi.weight", 22) == 6
    assert _layer_id("embeddings.tok_embeddings.weight", 22) == 0
    assert _layer_id("classifier.weight", 22) == 24
