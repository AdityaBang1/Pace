"""Safety and reproducibility tests for the Pace pipeline."""
import json

import pytest

from pace.data import generate_examples, split_by_group
from pace.extract import extract_candidates, validate_academic_ir
from pace.labels import TASK_LABELS
from pace.schedule_extract import extract_schedule_candidates
from pace.text_structure import split_sections


def test_cli_reports_missing_and_empty_input_as_usage_errors(tmp_path, capsys):
    from pace.cli import main

    missing = tmp_path / "missing.txt"
    with pytest.raises(SystemExit) as missing_error:
        main(["extract", str(missing)])
    assert missing_error.value.code == 2
    error = capsys.readouterr().err
    assert "cannot read" in error and "missing.txt" in error

    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    with pytest.raises(SystemExit) as empty_error:
        main(["extract", str(empty)])
    assert empty_error.value.code == 2
    assert "is empty" in capsys.readouterr().err


def test_schedule_scanner_extracts_recurring_and_dated_events_with_exact_offsets():
    text = (
        "Class meetings: Mondays and Wednesdays, 10:30 AM to 11:45 AM, Science 2.\n"
        "Final Exam — October 20, 2026 at 2:00 PM.\n"
    )
    result = extract_schedule_candidates(text)
    assert result["schema_version"] == "pace-schedule/0.1"
    assert len(result["events"]) == 2
    recurring, dated = result["events"]
    assert recurring["event_type"] == "recurring_class_candidate"
    assert recurring["normalized"] == {
        "date": None, "weekdays": ["MO", "WE"], "start_time": "10:30", "end_time": "11:45"
    }
    assert dated["event_type"] == "dated_event_candidate"
    assert dated["normalized"]["date"]["iso_date"] == "2026-10-20"
    assert dated["date_association_status"] == "resolved"
    assert [item["normalized"]["iso_date"] for item in dated["field_evidence"]["dates"]] == ["2026-10-20"]
    assert dated["field_evidence"]["date"]["text"] == "October 20, 2026"
    assert dated["normalized"]["start_time"] == "14:00"
    for event in result["events"]:
        evidence = event["evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["text"]
        for field in event["field_evidence"].values():
            entries = field if isinstance(field, list) else [field]
            for item in entries:
                assert text[item["start"]:item["end"]] == item["text"]
        assert event["requires_review"] is True
        assert event["confidence"] is None
        assert event["confidence_calibrated"] is False
    assert result["calibration"] == {
        "status": "uncalibrated",
        "confidence_calibrated": False,
        "reason": "no_independent_human_labeled_schedule_evaluation",
    }


def test_schedule_scanner_reports_missing_ambiguous_and_invalid_values():
    text = "Quiz 3/4, 9:30\nClass meets Monday 1:00 pm\n"
    events = extract_schedule_candidates(text)["events"]
    dated, recurring = events
    assert dated["normalized"]["date"]["year"] is None
    assert "year_missing" in dated["ambiguities"]
    assert "numeric_date_order_locale_assumed_month_first" in dated["ambiguities"]
    assert recurring["normalized"]["weekdays"] == ["MO"]
    assert recurring["normalized"]["start_time"] == "13:00"
    assert recurring["normalized"]["end_time"] is None
    assert "end_time_missing" in recurring["ambiguities"]

    invalid = extract_schedule_candidates("Final exam: February 30, 2026")["events"][0]
    assert invalid["normalized"]["date"]["iso_date"] is None
    assert "invalid_calendar_date" in invalid["ambiguities"]

    unrelated_time = extract_schedule_candidates(
        "Class meets Monday at 9:00 AM. Exam October 20, 2026."
    )["events"]
    recurring, dated_event = unrelated_time
    assert recurring["normalized"]["start_time"] == "09:00"
    assert "date_time_association_unconfirmed" not in dated_event["ambiguities"]
    assert "times" not in dated_event["field_evidence"]
    assert dated_event["normalized"]["start_time"] is None

    no_meridiem = extract_schedule_candidates("Class meets Monday 9:30")["events"][0]
    assert no_meridiem["normalized"]["start_time"] is None
    assert "meridiem_missing" in no_meridiem["ambiguities"]
    assert "end_time_missing" in no_meridiem["ambiguities"]
    assert "9:30" == no_meridiem["field_evidence"]["time"]["text"]
    evidence = no_meridiem["field_evidence"]["time"]
    assert "Class meets Monday 9:30"[evidence["start"]:evidence["end"]] == evidence["text"]


def test_schedule_scanner_requires_explicit_date_or_weekday_evidence():
    assert extract_schedule_candidates("Office hours are available by appointment.")["events"] == []
    entries = extract_schedule_candidates("Office hours Tuesday 2:00 PM")["events"]
    assert len(entries) == 1
    assert "end_time_missing" in entries[0]["ambiguities"]


def test_schedule_scanner_marks_unconfirmed_schedule_context_and_multiple_dates():
    unrelated = extract_schedule_candidates("The report mentions Monday 2:00 PM.")["events"][0]
    assert "schedule_context_unconfirmed" in unrelated["ambiguities"]
    dated = extract_schedule_candidates("Course runs 2026-01-05 through 2026-04-20.")["events"]
    assert len(dated) == 2
    assert all("event_context_unconfirmed" in event["ambiguities"] for event in dated)
    assert all("multiple_dates_on_line" not in event["ambiguities"] for event in dated)
    assert all("multiple_dates_in_clause" in event["ambiguities"] for event in dated)
    assert [event["normalized"]["date"]["iso_date"] for event in dated] == ["2026-01-05", "2026-04-20"]


def test_schedule_scanner_does_not_mistake_iso_date_hyphens_for_time_ranges():
    text = "Exam 2026-10-20 and 2026-11-02, 14:00."
    event, = extract_schedule_candidates(text)["events"]
    assert event["normalized"]["date"] is None
    assert event["normalized"]["start_time"] is None
    assert event["date_association_status"] == "ambiguous"
    assert "event_date_association_ambiguous" in event["ambiguities"]
    assert [item["text"] for item in event["field_evidence"]["dates"]] == [
        "2026-10-20", "2026-11-02"
    ]
    assert [item["text"] for item in event["field_evidence"]["times"]] == ["14:00"]
    for value in event["field_evidence"].values():
        for item in value if isinstance(value, list) else [value]:
            assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_pairs_multiple_weekday_slots_with_nearest_day():
    text = "Tuesday 9:30-10:45am class, Thursday 11:00-12:15pm lab."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert [event["normalized"]["weekdays"] for event in events] == [["TU"], ["TH"]]
    assert [event["normalized"]["start_time"] for event in events] == ["09:30", "11:00"]
    assert [event["normalized"]["end_time"] for event in events] == ["10:45", "12:15"]
    assert all("weekday_time_association_inferred" in event["ambiguities"] for event in events)
    for event in events:
        for field in event["field_evidence"].values():
            for item in field if isinstance(field, list) else [field]:
                assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_handles_named_date_ranges_without_time_collision():
    text = "Exam October 20-22, 2026 at 2:00 PM."
    event = extract_schedule_candidates(text)["events"][0]
    assert event["normalized"]["date"]["iso_date"] == "2026-10-20"
    assert event["normalized"]["date"]["range_end"]["iso_date"] == "2026-10-22"
    assert event["normalized"]["start_time"] == "14:00"
    assert "date_range_candidate" in event["ambiguities"]
    assert event["field_evidence"]["time"]["text"] == "2:00 PM"
    assert text[event["field_evidence"]["time"]["start"]:event["field_evidence"]["time"]["end"]] == "2:00 PM"


def test_schedule_scanner_associates_individual_dates_and_times_in_order():
    text = "Final exams: October 20, 2026 at 2:00 PM; November 2, 2026 at 3:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert [event["normalized"]["date"]["iso_date"] for event in events] == ["2026-10-20", "2026-11-02"]
    assert [event["normalized"]["start_time"] for event in events] == ["14:00", "15:00"]
    assert [event["date_association_status"] for event in events] == ["resolved", "resolved"]
    assert [[item["normalized"]["iso_date"] for item in event["field_evidence"]["dates"]] for event in events] == [
        ["2026-10-20"], ["2026-11-02"]
    ]
    assert all("time_date_association_inferred" in event["ambiguities"] for event in events)
    for event in events:
        date = event["field_evidence"]["date"]
        time = event["field_evidence"]["time"]
        assert text[date["start"]:date["end"]] == date["text"]
        assert text[time["start"]:time["end"]] == time["text"]


def test_schedule_scanner_scopes_weekdays_to_each_dated_event_clause():
    text = (
        "Final exam Monday October 20, 2026 at 2:00 P.M.; "
        "quiz Tuesday November 2, 2026 at 3:00 P.M."
    )
    events = extract_schedule_candidates(text)["events"]
    assert [event["normalized"]["date"]["iso_date"] for event in events] == [
        "2026-10-20", "2026-11-02"
    ]
    assert [event["normalized"]["weekdays"] for event in events] == [["MO"], ["TU"]]
    assert [event["normalized"]["start_time"] for event in events] == ["14:00", "15:00"]
    assert [event["field_evidence"]["weekdays"][0]["days"] for event in events] == [["MO"], ["TU"]]
    for event in events:
        for item in event["field_evidence"]["weekdays"]:
            assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_does_not_treat_period_after_abbreviation_as_clause_boundary():
    text = "Course meets Mon. Final exam Tuesday October 20, 2026 at 2:00 PM."
    event = extract_schedule_candidates(text)["events"][0]
    assert event["normalized"]["weekdays"] == ["TU"]
    weekday = event["field_evidence"]["weekdays"][0]
    assert weekday["text"] == "Tuesday"
    assert text[weekday["start"]:weekday["end"]] == weekday["text"]


def test_schedule_scanner_associates_weekdays_by_nearest_date_when_grouped():
    text = "Monday October 20, 2026 and Tuesday November 2, 2026 exams."
    events = extract_schedule_candidates(text)["events"]
    assert [event["normalized"]["weekdays"] for event in events] == [["MO"], ["TU"]]
    assert all("weekday_date_association_inferred" in event["ambiguities"] for event in events)


def test_schedule_scanner_leaves_non_unique_weekday_date_pairs_unassigned():
    text = "Monday and Tuesday dates October 20, 2026, November 2, 2026, and December 1, 2026."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 3
    assert all(event["normalized"]["weekdays"] == [] for event in events)
    assert all("weekday_date_association_ambiguous" in event["ambiguities"] for event in events)
    assert all(
        text[item["start"]:item["end"]] == item["text"]
        for event in events
        for item in event["field_evidence"].get("weekdays", [])
    )


def test_schedule_scanner_parses_noon_midnight_and_overnight_boundaries():
    text = "Class Tuesday 12:00-1:00pm; lab Wednesday 11:00pm-1:00am."
    events = extract_schedule_candidates(text)["events"]
    assert events[0]["normalized"]["start_time"] == "12:00"
    assert events[0]["normalized"]["end_time"] == "13:00"
    assert events[1]["normalized"]["start_time"] == "23:00"
    assert events[1]["normalized"]["end_time"] == "01:00"
    assert "meridiem_inferred_for_crossing_noon_range" not in events[0]["ambiguities"]
    assert events[1]["field_evidence"]["time_range"]["text"] == "11:00pm-1:00am"
    midnight = extract_schedule_candidates("Class Monday 12:00-1:00am")["events"][0]
    assert midnight["normalized"]["start_time"] == "00:00"
    assert midnight["normalized"]["end_time"] == "01:00"


def test_schedule_scanner_preserves_separate_time_clauses():
    text = "Class starts 9:00 AM and ends 10:00 AM Monday."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert all(event["normalized"]["start_time"] is None for event in events)
    assert all(event["normalized"]["weekdays"] == [] for event in events)
    assert all("weekday_time_association_ambiguous" in event["ambiguities"] for event in events)
    assert all("weekday_time_association_unconfirmed" in event["ambiguities"] for event in events)


def test_schedule_scanner_summarizes_dated_events_by_clause():
    text = "Final exam October 20, 2026 at 2:00 PM; quiz November 2, 2026 at 3:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert [event["summary"] for event in events] == [
        "Final exam October 20, 2026 at 2:00 PM",
        "quiz November 2, 2026 at 3:00 PM",
    ]
    assert all("multiple_dates_in_clause" not in event["ambiguities"] for event in events)
    for event in events:
        evidence = event["evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]


def test_schedule_scanner_summarizes_sentence_clauses_without_terminal_period():
    text = "Final exam October 20, 2026 at 2:00 PM. Quiz November 2, 2026 at 3:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert [event["summary"] for event in events] == [
        "Final exam October 20, 2026 at 2:00 PM",
        "Quiz November 2, 2026 at 3:00 PM",
    ]
    assert all("multiple_dates_in_clause" not in event["ambiguities"] for event in events)
    for event in events:
        evidence = event["evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]


def test_schedule_scanner_splits_distinct_event_cues_within_one_clause():
    text = "Final exam October 20, 2026 and quiz November 2, 2026 at 2:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert [event["summary"] for event in events] == [
        "Final exam October 20, 2026",
        "quiz November 2, 2026 at 2:00 PM",
    ]
    assert [event["date_association_status"] for event in events] == ["resolved", "resolved"]
    assert all(len(event["field_evidence"]["dates"]) == 1 for event in events)
    assert [event["normalized"]["start_time"] for event in events] == [None, "14:00"]
    assert all("multiple_dates_in_clause" not in event["ambiguities"] for event in events)
    assert "date_time_association_unconfirmed" in events[0]["ambiguities"]
    assert "date_time_association_ambiguous" not in events[1]["ambiguities"]
    for event in events:
        evidence = event["evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]
        for value in event["field_evidence"].values():
            for span in value if isinstance(value, list) else [value]:
                assert text[span["start"]:span["end"]] == span["text"]

    reverse_text = "Quiz November 2, 2026 at 3:00 PM and final exam October 20, 2026 at 2:00 PM."
    reverse_events = extract_schedule_candidates(reverse_text)["events"]
    assert [event["summary"] for event in reverse_events] == [
        "Quiz November 2, 2026 at 3:00 PM",
        "final exam October 20, 2026 at 2:00 PM",
    ]
    assert [event["normalized"]["start_time"] for event in reverse_events] == ["15:00", "14:00"]
    for event in reverse_events:
        evidence = event["evidence"]
        assert reverse_text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]


def test_schedule_scanner_uses_pre_date_cues_as_local_summaries_without_pairing_dates():
    text = "Final exam and quiz are both on October 20, 2026 and November 2, 2026 at 2:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert [event["summary"] for event in events] == ["Final exam", "quiz"]
    assert all(event["normalized"]["date"] is None for event in events)
    assert all(event["date_association_status"] == "ambiguous" for event in events)
    assert all(event["normalized"]["start_time"] is None for event in events)
    assert all("event_date_association_ambiguous" in event["ambiguities"] for event in events)
    assert all("date_time_association_ambiguous" in event["ambiguities"] for event in events)
    for event in events:
        summary = event["evidence"]
        cue = event["field_evidence"]["event_cue"]
        dates = event["field_evidence"]["dates"]
        times = event["field_evidence"]["times"]
        assert text[summary["start"]:summary["end"]] == summary["text"] == event["summary"]
        assert text[cue["start"]:cue["end"]] == cue["text"] == event["summary"]
        assert [item["text"] for item in dates] == ["October 20, 2026", "November 2, 2026"]
        assert "date" not in event["field_evidence"]
        assert [text[item["start"]:item["end"]] for item in dates] == [item["text"] for item in dates]
        assert [item["text"] for item in times] == ["2:00 PM"]
        assert [text[item["start"]:item["end"]] for item in times] == [item["text"] for item in times]


def test_schedule_scanner_emits_one_ambiguous_candidate_for_one_cue_and_multiple_dates():
    text = "Exam dates October 20, 2026 and November 2, 2026 at 2:00 PM."
    event, = extract_schedule_candidates(text)["events"]
    assert event["summary"] == "Exam dates"
    assert event["normalized"]["date"] is None
    assert event["normalized"]["start_time"] is None
    assert event["date_association_status"] == "ambiguous"
    assert "event_date_association_ambiguous" in event["ambiguities"]
    assert "date_time_association_ambiguous" in event["ambiguities"]
    assert "multiple_dates_in_clause" in event["ambiguities"]
    dates = event["field_evidence"]["dates"]
    assert [item["normalized"]["iso_date"] for item in dates] == [
        "2026-10-20", "2026-11-02"
    ]
    assert [item["text"] for item in dates] == ["October 20, 2026", "November 2, 2026"]
    assert [item["text"] for item in event["field_evidence"]["times"]] == ["2:00 PM"]
    evidence = event["evidence"]
    assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]
    for value in event["field_evidence"].values():
        for item in value if isinstance(value, list) else [value]:
            assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_emits_one_event_for_multiple_dates_with_explicit_group_heading():
    text = "Final exams are October 20, 2026 and November 2, 2026."
    event, = extract_schedule_candidates(text)["events"]
    assert event["summary"] == "Final exams"
    assert event["normalized"]["date"] is None
    assert "event_date_association_ambiguous" in event["ambiguities"]
    assert [item["text"] for item in event["field_evidence"]["dates"]] == [
        "October 20, 2026", "November 2, 2026"
    ]
    evidence = event["evidence"]
    assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]


def test_schedule_scanner_does_not_pair_pre_date_cues_and_dates_by_matching_counts():
    text = "Final exam and quiz are October 20, 2026 at 2:00 PM, November 2, 2026 at 3:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert [event["summary"] for event in events] == ["Final exam", "quiz"]
    assert all(event["normalized"]["date"] is None for event in events)
    assert all(event["date_association_status"] == "ambiguous" for event in events)
    assert all(event["normalized"]["start_time"] is None for event in events)
    assert all("event_date_association_ambiguous" in event["ambiguities"] for event in events)
    assert all("date_time_association_ambiguous" in event["ambiguities"] for event in events)
    for event in events:
        assert [item["text"] for item in event["field_evidence"]["dates"]] == [
            "October 20, 2026", "November 2, 2026"
        ]
        assert [item["text"] for item in event["field_evidence"]["times"]] == [
            "2:00 PM", "3:00 PM"
        ]
        for value in event["field_evidence"].values():
            for item in value if isinstance(value, list) else [value]:
                assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_groups_multi_date_event_once_without_assigning_its_time():
    text = "Exam dates October 20, 2026 and November 2, 2026 at 2:00 PM; quiz December 1, 2026 at 3:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    first, quiz = events
    assert first["summary"] == "Exam dates"
    assert first["normalized"]["date"] is None
    assert first["normalized"]["start_time"] is None
    assert "event_date_association_ambiguous" in first["ambiguities"]
    assert "date_time_association_ambiguous" in first["ambiguities"]
    assert "multiple_dates_in_clause" in first["ambiguities"]
    assert [item["normalized"]["iso_date"] for item in first["field_evidence"]["dates"]] == [
        "2026-10-20", "2026-11-02"
    ]
    assert [item["text"] for item in first["field_evidence"]["times"]] == ["2:00 PM"]
    assert quiz["summary"] == "quiz December 1, 2026 at 3:00 PM"
    assert quiz["normalized"]["start_time"] == "15:00"
    assert quiz["date_association_status"] == "resolved"
    assert len(quiz["field_evidence"]["dates"]) == 1
    assert "date_time_association_ambiguous" not in quiz["ambiguities"]
    for event in events:
        evidence = event["evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["text"] == event["summary"]
        for value in event["field_evidence"].values():
            for span in value if isinstance(value, list) else [value]:
                assert text[span["start"]:span["end"]] == span["text"]


def test_schedule_scanner_does_not_pair_across_unrelated_clauses():
    text = "Course runs from October 20-22, 2026; final exam on October 21, 2026 at 2:00 PM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    first, second = events
    assert "date_time_association_unconfirmed" not in first["ambiguities"]
    assert "times" not in first["field_evidence"]
    assert first["normalized"]["start_time"] is None
    assert second["normalized"]["start_time"] == "14:00"
    assert "date_time_association_ambiguous" not in second["ambiguities"]
    assert "date_time_association_unconfirmed" not in second["ambiguities"]

    shared_time = extract_schedule_candidates("Course runs 2026-10-20; final exam 2026-11-02 at 2:00 PM.")["events"]
    assert shared_time[0]["normalized"]["start_time"] is None
    assert "date_time_association_unconfirmed" not in shared_time[0]["ambiguities"]
    assert "times" not in shared_time[0]["field_evidence"]

    distinct_events = extract_schedule_candidates(
        "Final exam October 20, 2026 and quiz November 2, 2026 at 2:00 PM."
    )["events"]
    assert [event["normalized"]["start_time"] for event in distinct_events] == [None, "14:00"]
    assert "time_shared_across_multiple_dates" not in distinct_events[1]["ambiguities"]

    dates = extract_schedule_candidates(
        "Final exam October 20, 2026 at 2:00 PM; quiz November 2, 2026 at 3:00 PM."
    )["events"]
    assert [event["normalized"]["start_time"] for event in dates] == ["14:00", "15:00"]

    shared_time = extract_schedule_candidates("Exam 2026-10-20 and 2026-11-02, 14:00.")["events"]
    assert len(shared_time) == 1
    assert shared_time[0]["normalized"]["date"] is None
    assert shared_time[0]["normalized"]["start_time"] is None
    assert "event_date_association_ambiguous" in shared_time[0]["ambiguities"]
    assert "date_time_association_ambiguous" in shared_time[0]["ambiguities"]


def test_schedule_scanner_does_not_assign_across_mismatched_weekday_slots():
    text = "Tuesday and Thursday 9:00 AM; Friday 10:00 AM."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert events[0]["normalized"]["weekdays"] == []
    assert events[0]["normalized"]["start_time"] is None
    assert "weekday_time_association_ambiguous" in events[0]["ambiguities"]
    assert "weekday_time_association_unconfirmed" in events[0]["ambiguities"]
    assert events[1]["normalized"]["weekdays"] == ["FR"]
    assert events[1]["normalized"]["start_time"] == "10:00"


def test_schedule_scanner_keeps_grouped_days_and_times_ambiguous():
    text = "Monday and Tuesday 9:00 AM, 10:00 AM classes."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert all(event["normalized"]["weekdays"] == [] for event in events)
    assert all(event["normalized"]["start_time"] is None for event in events)
    assert all("weekday_time_association_ambiguous" in event["ambiguities"] for event in events)
    assert all("weekday_time_association_unconfirmed" in event["ambiguities"] for event in events)


def test_schedule_scanner_marks_non_unique_date_time_assignment_ambiguous():
    text = "Exams October 20, 2026 at 2:00 PM, November 2, 2026 at 3:00 PM, December 1, 2026."
    event, = extract_schedule_candidates(text)["events"]
    assert event["summary"] == "Exams"
    assert event["normalized"]["date"] is None
    assert event["normalized"]["start_time"] is None
    assert event["date_association_status"] == "ambiguous"
    assert "event_date_association_ambiguous" in event["ambiguities"]
    assert "date_time_association_ambiguous" in event["ambiguities"]
    assert [item["text"] for item in event["field_evidence"]["dates"]] == [
        "October 20, 2026", "November 2, 2026", "December 1, 2026"
    ]
    assert [item["text"] for item in event["field_evidence"]["times"]] == [
        "2:00 PM", "3:00 PM"
    ]
    for value in event["field_evidence"].values():
        for item in value if isinstance(value, list) else [value]:
            assert text[item["start"]:item["end"]] == item["text"]


def test_schedule_scanner_reports_calendar_invalidity_and_exact_time_spans():
    text = "Exam February 29, 2025 at 9:00 AM.\r\nExam 2/30/2026 at 14:00."
    events = extract_schedule_candidates(text)["events"]
    assert len(events) == 2
    assert events[0]["normalized"]["date"]["iso_date"] is None
    assert "invalid_calendar_date" in events[0]["ambiguities"]
    assert events[1]["normalized"]["date"]["iso_date"] is None
    assert "invalid_calendar_date" in events[1]["ambiguities"]
    for event in events:
        time = event["field_evidence"]["time"]
        assert text[time["start"]:time["end"]] == time["text"]
        assert time["text"][-1] not in ".,;!?"


def test_schedule_cli_emits_json_and_rejects_missing_input(tmp_path, capsys):
    from pace.cli import main

    fixture = tmp_path / "schedule.txt"
    text = "BIO 210 meets Tuesdays and Thursdays 9:00-10:15am.\n"
    fixture.write_text(text, encoding="utf-8")
    main(["scan-schedule", str(fixture)])
    result = json.loads(capsys.readouterr().out)
    assert result["events"][0]["normalized"]["weekdays"] == ["TU", "TH"]
    assert result["calibration"]["confidence_calibrated"] is False

    mixed = tmp_path / "mixed-schedule.txt"
    text = (
        "Class meets Tuesdays and Thursdays, 9:00-10:15 AM. "
        "Final exam October 20, 2026 at 2:00 PM."
    )
    mixed.write_text(text, encoding="utf-8")
    main(["scan-schedule", str(mixed)])
    result = json.loads(capsys.readouterr().out)
    recurring, dated = result["events"]
    assert [event["event_type"] for event in result["events"]] == [
        "recurring_class_candidate", "dated_event_candidate"
    ]
    assert recurring["normalized"]["weekdays"] == ["TU", "TH"]
    assert recurring["normalized"]["start_time"] == "09:00"
    assert recurring["normalized"]["end_time"] == "10:15"
    assert dated["normalized"]["start_time"] == "14:00"
    assert "date_time_association_unconfirmed" not in dated["ambiguities"]
    for event in result["events"]:
        for evidence in [event["evidence"], *[
            span
            for value in event["field_evidence"].values()
            for span in (value if isinstance(value, list) else [value])
        ]]:
            assert text[evidence["start"]:evidence["end"]] == evidence["text"]

    same_clause = tmp_path / "same-clause-schedule.txt"
    text = "Class meets Monday 9:00 AM, final exam October 20, 2026 at 2:00 PM."
    same_clause.write_text(text, encoding="utf-8")
    main(["scan-schedule", str(same_clause)])
    result = json.loads(capsys.readouterr().out)
    recurring, dated = result["events"]
    assert [event["event_type"] for event in result["events"]] == [
        "recurring_class_candidate", "dated_event_candidate"
    ]
    assert recurring["summary"] == "Class meets Monday 9:00 AM"
    assert recurring["normalized"]["weekdays"] == ["MO"]
    assert recurring["normalized"]["start_time"] == "09:00"
    assert dated["summary"] == "final exam October 20, 2026 at 2:00 PM"
    assert dated["normalized"]["date"]["iso_date"] == "2026-10-20"
    assert dated["normalized"]["start_time"] == "14:00"
    for event in result["events"]:
        spans = [event["evidence"]]
        for value in event["field_evidence"].values():
            spans.extend(value if isinstance(value, list) else [value])
        assert all(text[span["start"]:span["end"]] == span["text"] for span in spans)

    reverse_clause = tmp_path / "reverse-same-clause-schedule.txt"
    text = "Final exam October 20, 2026 at 2:00 PM, class meets Monday 9:00 AM."
    reverse_clause.write_text(text, encoding="utf-8")
    main(["scan-schedule", str(reverse_clause)])
    result = json.loads(capsys.readouterr().out)
    dated, recurring = result["events"]
    assert [event["event_type"] for event in result["events"]] == [
        "dated_event_candidate", "recurring_class_candidate"
    ]
    assert dated["normalized"]["start_time"] == "14:00"
    assert recurring["normalized"]["start_time"] == "09:00"

    with pytest.raises(SystemExit) as missing:
        main(["scan-schedule", str(tmp_path / "missing.txt")])
    assert missing.value.code == 2
    error = capsys.readouterr().err
    assert "cannot read" in error and "missing.txt" in error


def test_generation_is_deterministic_and_covers_labels():
    first = generate_examples(8400, 7)
    assert first == generate_examples(8400, 7)
    for task, labels in TASK_LABELS.items():
        assert {row["label"] for row in first if row["task"] == task} == set(labels)
    splits = split_by_group(first, 7)
    assert sum(map(len, splits.values())) == len(first)


def test_template_groups_do_not_leak_between_splits():
    splits = split_by_group(generate_examples(2100, 123), 123)
    group_sets = [{row["group_id"] for row in splits[name]} for name in splits]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    assert all(splits[name] for name in splits)
    for name in splits:
        for task, labels in TASK_LABELS.items():
            assert {row["label"] for row in splits[name] if row["task"] == task} == set(labels)


def test_extraction_returns_literal_evidence_offsets():
    text = "CS 201 requires MATH 115. Project due October 22; worth 15%."
    extracted = extract_candidates(text)
    assert [item["text"] for item in extracted["course_code"]] == ["CS 201", "MATH 115"]
    for items in extracted.values():
        for item in items:
            assert text[item["start"]:item["end"]] == item["text"]
            assert item["requires_review"] is True


def test_credit_candidates_never_span_across_lines():
    text = "Schedule: Room 2\n\nCredits: 3.0 credits"
    spans = extract_candidates(text)["credit_hours"]
    assert [item["text"] for item in spans] == ["3.0 credits"]
    assert all("\n" not in item["text"] and "\r" not in item["text"] for item in spans)


def test_credit_candidate_does_not_duplicate_or_intersect_entity_proposal():
    from pace.synthetic_doc_model import SyntheticDocumentEngine

    fixture = "Syllabus\n\nCredits: 3.0 credits\n"
    result = SyntheticDocumentEngine(device="cpu").analyze_document(fixture)
    credit_span = next(e for e in result["entities"] if e["type"] == "CREDIT")
    credit_candidates = [e for e in result["entities"] if e["type"] == "LITERAL_CANDIDATE"]
    assert fixture[credit_span["start"]:credit_span["end"]] == "3.0"
    assert all((e["start"], e["end"]) != (credit_span["start"], credit_span["end"]) for e in credit_candidates)
    assert all(not (e["start"] < credit_span["end"] and credit_span["start"] < e["end"]) for e in credit_candidates)
    assert not any("\n" in e["text"] or "\r" in e["text"] for e in credit_candidates)


def test_credit_candidates_do_not_bridge_a_real_line_break():
    text = "Credits: 3.0\ncredits"
    assert extract_candidates(text)["credit_hours"] == []


def test_grade_candidates_require_explicit_grade_context():
    assert extract_candidates("A student should attend office hours.")["grade"] == []
    assert extract_candidates("A grade is assigned at the end of term.")["grade"] == []
    assert extract_candidates("Grades are assigned weekly.")["grade"] == []
    assert extract_candidates("Room B is next to section C.")["grade"] == []
    assert [item["text"] for item in extract_candidates("Grade: A")["grade"]] == ["A"]
    assert [item["text"] for item in extract_candidates("Final grade: B+")["grade"]] == ["B+"]
    assert [item["text"] for item in extract_candidates("Grades: A, B, C, D, F")["grade"]] == ["A", "B", "C", "D", "F"]
    assert [item["text"] for item in extract_candidates("Grades are A, B and C")["grade"]] == ["A", "B", "C"]
    assert [item["text"] for item in extract_candidates("Grades are A, B, and C")["grade"]] == ["A", "B", "C"]
    assert [item["text"] for item in extract_candidates("The final grades were A and B.")["grade"]] == ["A", "B"]
    assert [item["text"] for item in extract_candidates("The grade was C+.")["grade"]] == ["C+"]
    assert [item["text"] for item in extract_candidates("Grades: A, B, C, D, and F")["grade"]] == ["A", "B", "C", "D", "F"]
    assert [item["text"] for item in extract_candidates("Grades (A/B/C)")["grade"]] == ["A", "B", "C"]
    assert [item["text"] for item in extract_candidates("The grading scale is A through F.")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grading scale: A-F")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grade: B−")["grade"]] == ["B−"]
    assert extract_candidates("Grade: B−−")["grade"] == []
    assert [item["text"] for item in extract_candidates("Grade scale A-F")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A–F")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A—F")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A+-F")["grade"]] == ["A+", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A+–F")["grade"]] == ["A+", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A+—F")["grade"]] == ["A+", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale A−-F")["grade"]] == ["A−", "F"]
    assert [item["text"] for item in extract_candidates("Grade scale F-A+")["grade"]] == ["F", "A+"]
    assert [item["text"] for item in extract_candidates("Grade scale A-F+")["grade"]] == ["A", "F+"]
    assert extract_candidates("Grade scale A--F")["grade"] == []
    assert [item["text"] for item in extract_candidates("Grade: A to F")["grade"]] == ["A", "F"]
    assert [item["text"] for item in extract_candidates("Grade: A-")["grade"]] == ["A-"]
    assert extract_candidates("Grade: A--")["grade"] == []
    assert extract_candidates("Grade: Failing")["grade"] == []
    assert extract_candidates("Grades are assigned.")["grade"] == []


def test_hierarchical_text_is_bounded_and_keeps_headings():
    sections = split_sections("Course Overview\nBIO 101 studies life.\n\nGrading\nProjects 50%.\n")
    assert sections[0]["title"] == "Course Overview"
    assert sections[1]["title"] == "Grading"
    assert all(len(section["text"].split()) <= 120 for section in sections)


def test_ir_validator_rejects_unsafe_or_malformed_record():
    valid = {"schema_version": "1.0", "document_type": {"confidence": 0.8},
             "sections": [{"confidence": 0.7, "source_text": "grading"}],
             "commit_policy": "proposals_only_user_confirmation_required"}
    assert validate_academic_ir(valid) == []
    invalid = {**valid, "commit_policy": "automatic"}
    assert "commit_policy must require user confirmation" in validate_academic_ir(invalid)
    invalid["sections"] = [{"confidence": 1.3}]
    assert len(validate_academic_ir(invalid)) == 3
