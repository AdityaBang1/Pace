"""Growing the corpus must not disturb a single row the numbers were earned on.

Two properties are load-bearing here: a newly downloaded catalog maps its own
columns to our labels (and keeps the heading a value was printed under), and the
extension only ever *appends* -- the base corpus, its labels and its
train/validation/test group assignment come through untouched. If either broke,
every previously reported test score would stop being comparable.
"""
from __future__ import annotations

import csv
import json

from pace.field_corpus_ext import extend_corpus
from pace.field_mix import build_mix
from pace.field_sources_ext import load_columbia_examples, load_coursera_examples

COLUMBIA_HEADER = ["course_code", "course_title", "course_descr", "instructor",
                   "scheduled_time_start", "scheduled_time_end", "location", "points",
                   "prerequisites", "department", "method_of_instruction", "type",
                   "open_to", "class_id", "scheduled_days"]


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_columbia_columns_become_the_fields_they_are(tmp_path):
    _write(tmp_path / "2020-Fall.csv", COLUMBIA_HEADER, [[
        "COMS W3134", "Data Structures", "Lists, trees and hash tables.", "Ada Lovelace",
        "9:00am", "10:15am", "310 Mudd", "3", "COMS W1004", "Computer Science",
        "In Person", "LECTURE", "Barnard", "W3134-20203-001", "TR"]])

    rows = load_columbia_examples(tmp_path)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    assert by_label["course_code"][0]["text"] == "COMS W3134"
    assert by_label["meeting_days"][0]["text"] == "TR"
    assert by_label["meeting_time"][0]["text"] == "9:00am - 10:15am"
    assert by_label["credits"][0]["text"] == "3"
    assert by_label["class_type"][0]["text"] == "LECTURE"
    assert by_label["term"][0]["text"] == "Fall 2020"
    # The heading a value was published under travels with the row.
    assert by_label["meeting_days"][0]["source_field"] == "Days"
    assert all(row["meta"]["school"] == "Columbia University" for row in rows)


def test_coursera_lists_are_printed_as_they_read(tmp_path):
    path = _write(tmp_path / "coursera.csv",
                  ["title", "Instructor", "Organization", "Description", "Level",
                   "Schedule", "Modules/Courses"],
                  [["Machine Learning", "Andrew Ng", "Stanford", "A broad introduction.",
                    "Beginner level", "2 months", "['Week 1', 'Week 2']"]])

    rows = load_coursera_examples(path)
    by_label = {row["label"]: row["text"] for row in rows}

    assert by_label["course_title"] == "Machine Learning"
    assert by_label["program"] == "Stanford"
    assert by_label["course_level"] == "Beginner level"
    assert by_label["course_schedule"] == "2 months"
    assert by_label["course_content"] == "Week 1 Week 2"


def _row(text, label, group, source="Test/Source", field="Field", school="Base University"):
    return {"text": text, "label": label, "group_id": group, "source": source,
            "source_field": field, "provenance": "test", "source_id": f"{group}:1",
            "text_sha256": "0" * 64, "meta": {"school": school}}


def test_extension_appends_and_leaves_the_base_alone():
    base = [_row("Class Name: Intro Biology", "course_title", "usc-subject:BIO"),
            _row("Units: 3.0", "credits", "usc-subject:BIO")]
    # "Units: 3.0" already means credits; the new source may not relabel it.
    extra = [_row("Units: 3.0", "capacity", "new-course:1"),
             _row("Instructor: Grace Hopper", "instructor", "new-course:1",
                  school="New University"),
             _row("Instructor: Grace Hopper", "instructor", "new-course:1",
                  school="New University")]

    rows, report = extend_corpus(base, extra)

    assert rows[:2] == base
    assert report["new_rows_kept"] == 1
    assert report["dropped"]["contradicts_base"] == 1
    assert report["dropped"]["duplicate"] == 1
    appended = rows[2]
    # The extension attaches the heading the base builder would have attached.
    assert appended["text"] == "Field: Instructor: Grace Hopper"
    assert appended["bare_text"] == "Instructor: Grace Hopper"
    assert appended["source_key"] == "Test"


def _mix_rows():
    rows = []
    for index, (label, value) in enumerate([
            ("meeting_days", "MWF"), ("meeting_days", "TR"),
            ("meeting_time", "9:00am - 9:50am"), ("meeting_time", "2:00pm - 3:15pm"),
            ("class_meeting_room", "310 Mudd"), ("class_meeting_room", "Room 214"),
            ("course_title", "Data Structures"), ("course_title", "Algorithms"),
            ("credits", "3"), ("credits", "4")]):
        rows.append({"text": f"Heading: {value}", "bare_text": value, "heading": "Heading",
                     "label": label, "group_id": f"group:{index}", "source": "Test/Source",
                     "source_key": "Test", "source_field": "Field", "provenance": "test",
                     "source_id": f"id:{index}", "text_sha256": "0" * 64,
                     "meta": {"school": "Test University"}})
    return rows


def test_schedule_family_prints_days_and_times_together(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    rows = _mix_rows()
    for name in ("train", "validation", "test"):
        part = rows if name == "train" else rows[-2:]
        with (split_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in part:
                stream.write(json.dumps(row) + "\n")

    splits, stats = build_mix(rows, split_dir)

    schedule = [row for row in splits["train"] if row["family"] == "schedule"]
    times = [row for row in schedule if row["label"] == "meeting_time"]
    days = [row for row in schedule if row["label"] == "meeting_days"]
    assert times and days
    # A day token followed by a real clock span is a meeting time, not a code.
    assert any(row["text"].startswith(("MWF ", "TR ")) and " - " in row["text"]
               for row in times)
    # The days on their own stay meeting_days, printed as bare day strings.
    assert {row["text"] for row in days} <= {"MWF", "TR"}
    assert all("Heading" not in row["text"] for row in schedule)
    assert stats["schedule_days"] == 2
