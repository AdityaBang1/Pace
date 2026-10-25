"""The SIS adapter is the only way a district's export becomes training text.

These tests fix the two header tables and the rules that keep a wrong label out:
a column is mapped only when it is in a table, placeholder cells produce nothing,
and every row of a course carries the same group id so a split cannot leak.
"""
from __future__ import annotations

import csv
import json

from pace.sis_import import HEADER_ALIASES, ONEROSTER_LABELS, load_sis_examples, main

COURSES_HEADER = ["sourcedId", "status", "dateLastModified", "schoolYearSourcedId",
                  "title", "courseCode", "grades", "orgSourcedId", "subjects",
                  "subjectCodes"]
CLASSES_HEADER = ["sourcedId", "status", "dateLastModified", "title", "grades",
                  "courseSourcedId", "classCode", "classType", "location",
                  "schoolSourcedId", "termSourcedIds", "subjects", "subjectCodes",
                  "periods"]


def _write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _oneroster_dir(tmp_path):
    _write(tmp_path / "courses.csv", COURSES_HEADER, [
        ["c1", "active", "2026-01-01", "sy1", "Algebra I", "MATH-101", "09;10",
         "org1", "Mathematics", "MATH"],
        ["c2", "active", "2026-01-01", "sy1", "Geometry", "MATH-101", "10",
         "org1", "Mathematics", "MATH"],
    ])
    _write(tmp_path / "classes.csv", CLASSES_HEADER, [
        ["cl1", "active", "2026-01-01", "Algebra I", "09", "c1", "MATH-101-01",
         "homeroom", "Room 12", "sch1", "Fall 2026", "Mathematics", "MATH",
         "1st period"],
    ])
    return tmp_path


def test_oneroster_courses_and_classes_map_to_our_labels(tmp_path):
    rows = load_sis_examples(_oneroster_dir(tmp_path), school="Lincoln High")

    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    expected = {"course_title", "course_code", "course_level", "program",
                "class_section", "class_type", "class_meeting_room", "meeting_time",
                "term"}
    assert expected <= set(by_label)
    assert "Fall 2026" in [row["text"] for row in by_label["term"]]
    assert "MATH-101-01" in [row["text"] for row in by_label["class_section"]]
    assert all(row["meta"]["school"] == "Lincoln High" for row in rows)
    assert all(row["group_id"] for row in rows)


def test_a_course_directory_is_read_in_full(tmp_path):
    rows = load_sis_examples([_oneroster_dir(tmp_path)])
    assert {row["meta"]["sis_file"] for row in rows} == {"courses", "classes"}


def test_a_section_shares_the_group_of_its_course_and_not_another(tmp_path):
    rows = load_sis_examples(_oneroster_dir(tmp_path))
    titles = [row for row in rows if row["label"] == "course_title"]
    # The catalog row (courses.csv, c1) and the section row (classes.csv, whose
    # courseSourcedId is c1) are the same course: one group, so a split cannot
    # put the course in train and its own section in test.
    algebra = [row for row in titles if row["text"] == "Algebra I"]
    geometry = [row for row in titles if row["text"] == "Geometry"]
    assert algebra and geometry
    assert len({row["group_id"] for row in algebra}) == 1
    assert {row["group_id"] for row in algebra} != {row["group_id"] for row in geometry}


def test_infinite_campus_adhoc_headers_map_regardless_of_case_and_spacing(tmp_path):
    path = _write(tmp_path / "course-master.csv",
                  ["Course Number", "Course Name", "Section Number", "Teacher Name",
                   "Room", "Period", "Prerequisites", "Credits", "Max Enrollment"],
                  [["MATH-101", "Algebra I", "01", "Ada Lovelace", "Room 12",
                    "1st period", "MATH-090", "3", "30"]])
    rows = load_sis_examples([path], school="Lincoln High", source="Infinite Campus")

    labels = {row["label"]: row["text"] for row in rows}
    assert labels["course_code"] == "MATH-101"
    assert labels["course_title"] == "Algebra I"
    assert labels["class_section"] == "01"
    assert labels["instructor"] == "Ada Lovelace"
    assert labels["class_meeting_room"] == "Room 12"
    assert labels["meeting_time"] == "1st period"
    assert labels["prerequisites"] == "MATH-090"
    assert labels["credits"] == "3"  # a one-character cell is still a real value
    assert labels["capacity"] == "30"
    assert all(row["source"] == "Infinite Campus" for row in rows)


def test_a_column_outside_the_tables_is_ignored_rather_than_guessed(tmp_path):
    path = _write(tmp_path / "export.csv",
                  ["Course Number", "Vendor Only Field"],
                  [["MATH-101", "some private string"]])
    rows = load_sis_examples([path])
    assert {row["label"] for row in rows} == {"course_code"}
    assert all(row["source_field"] != "Vendor Only Field" for row in rows)


def test_placeholder_cells_produce_no_rows(tmp_path):
    path = _write(tmp_path / "export.csv",
                  ["Course Number", "Course Name", "Section Number", "Credits"],
                  [["MATH-101", "N/A", "-", "TBA"], ["", "Geometry", "  ", "3"]])
    rows = load_sis_examples([path])
    texts = {row["text"] for row in rows}
    assert texts == {"MATH-101", "Geometry", "3"}
    assert all(row["text"].casefold() not in {"n/a", "-", "tba", ""} for row in rows)


def test_the_cli_writes_jsonl_and_prints_a_summary(tmp_path, capsys):
    path = _write(tmp_path / "course-master.csv",
                  ["Course Number", "Course Name", "Credits"],
                  [["MATH-101", "Algebra I", "3"]])
    output = tmp_path / "sis_corpus.jsonl"

    main(["--input", str(path), "--output", str(output), "--school", "Lincoln High"])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 3
    assert {row["label"] for row in rows} == {"course_code", "course_title", "credits"}
    assert rows[0]["meta"]["school"] == "Lincoln High"
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    summary = json.loads("\n".join(printed))
    assert summary["rows"] == 3
    assert summary["by_label"]["course_code"] == 1


def test_the_mapping_tables_only_name_taxonomy_labels():
    from pace.field_corpus import FIELD_TYPES

    known = set(FIELD_TYPES)
    assert set(HEADER_ALIASES.values()) <= known
    for table in ONEROSTER_LABELS.values():
        assert set(table.values()) <= known
