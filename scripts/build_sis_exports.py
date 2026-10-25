"""Compose 1EdTech OneRoster exports -- the format Infinite Campus writes -- out of
real course-catalog values.

Why compose instead of download: a real Infinite Campus Ad Hoc export is student
data, and there is no public corpus of them. What *is* real is the vocabulary the
columns carry (``courseMasterNumber``, ``classCode``, ``periods``, ``termSourcedIds``),
and that vocabulary is already fixed by ``pace.sis_import``. So this script takes
values that a real institution actually printed -- USC, Columbia, UW-Madison, MIT,
Gothenburg, Coursera listings -- and writes them into the sanctioned interchange
layout, grouped so one course stays one course.

The output is read back by ``python -m pace.sis_import``, which is the same code
path a district's own export takes: nothing here bypasses the adapter, and every
row keeps the column it came from.

    python scripts/build_sis_exports.py --train data/processed/transfer/syllabus-docs/train.jsonl \
        --output-dir data/processed/sis/export --max-courses 1500
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

#: Which corpus labels feed which OneRoster column. A group missing a column
#: simply leaves it blank, which is what a real export does.
COURSE_COLUMNS = {
    "title": "course_title",
    "courseCode": "course_code",
    "grades": "course_level",
    "subjects": "program",
}
CLASS_COLUMNS = {
    "title": "course_title",
    "grades": "course_level",
    "classCode": "class_section",
    "classType": "class_type",
    "location": "class_meeting_room",
    "termSourcedIds": "term",
    "periods": "meeting_time",
}
LINEITEM_COLUMNS = {
    "title": "assignments",
    "description": "course_content",
    "dueDate": "due_date",
}
RESULT_COLUMNS = {"score": "grades", "comment": "additional_information"}

COURSES_HEADER = ["sourcedId", "status", "dateLastModified", "schoolYearSourcedId",
                  "title", "courseCode", "grades", "orgSourcedId", "subjects",
                  "subjectCodes"]
CLASSES_HEADER = ["sourcedId", "status", "dateLastModified", "title", "grades",
                  "courseSourcedId", "classCode", "classType", "location",
                  "schoolSourcedId", "termSourcedIds", "subjects", "subjectCodes",
                  "periods"]
LINEITEMS_HEADER = ["sourcedId", "status", "dateLastModified", "title",
                    "description", "assignDate", "dueDate", "classSourcedId",
                    "categorySourcedId", "resultValueMin", "resultValueMax"]
RESULTS_HEADER = ["sourcedId", "status", "dateLastModified", "lineItemSourcedId",
                  "studentSourcedId", "score", "scoreStatus", "comment"]

#: A cell that is really a heading or a sentence is not a value; keep the
#: exports to short printed values, as the interchange format expects.
_SPACE = re.compile(r"\s+")


def _clean(text: Any) -> str:
    return _SPACE.sub(" ", str(text or "")).strip()


def _usable(text: str, limit: int = 160) -> bool:
    return bool(text) and len(text) <= limit and any(c.isalnum() for c in text)


def load_groups(path: Path) -> dict[str, dict[str, str]]:
    """group_id -> {label: first usable value}, from real (non-noise) rows."""
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "noise" in str(row.get("family", "")):
                continue
            value = _clean(row.get("bare_text") or "")
            if not _usable(value):
                continue
            groups[row["group_id"]].setdefault(row["label"], value)
    return groups


def build(groups: dict[str, dict[str, str]], max_courses: int, seed: int
          ) -> dict[str, list[list[str]]]:
    usable = [key for key, fields in groups.items()
              if "course_title" in fields and len(fields) >= 3]
    usable.sort()
    random.Random(seed).shuffle(usable)
    usable = usable[:max_courses]

    courses: list[list[str]] = []
    classes: list[list[str]] = []
    lineitems: list[list[str]] = []
    results: list[list[str]] = []
    for index, key in enumerate(usable):
        # Anything longer than a printed cell is prose; an interchange format
        # would not carry it, so it must not reach a training row either.
        fields = {label: value for label, value in groups[key].items()
                  if _usable(value)}
        course_id = f"c{index:05d}"
        class_id = f"cl{index:05d}"
        courses.append([
            course_id, "active", "2026-08-01", "sy2026",
            fields.get("course_title", ""), fields.get("course_code", ""),
            fields.get("course_level", ""), "org1", fields.get("program", ""), "",
        ])
        classes.append([
            class_id, "active", "2026-08-01", fields.get("course_title", ""),
            fields.get("course_level", ""), course_id,
            fields.get("class_section", ""), fields.get("class_type", ""),
            fields.get("class_meeting_room", ""), "sch1",
            fields.get("term", ""), fields.get("program", ""), "",
            fields.get("meeting_time", ""),
        ])
        if "assignments" in fields or "due_date" in fields:
            lineitems.append([
                f"li{index:05d}", "active", "2026-08-01",
                fields.get("assignments", ""), fields.get("course_content", "")[:160],
                "", fields.get("due_date", ""), class_id, "", "", "",
            ])
        if "grades" in fields:
            results.append([
                f"r{index:05d}", "active", "2026-08-01", f"li{index:05d}",
                f"s{index:05d}", fields["grades"], "fully graded", "",
            ])
    return {"courses": courses, "classes": classes,
            "lineitems": lineitems, "results": results}


def write_tables(tables: dict[str, list[list[str]]], output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    headers = {"courses": COURSES_HEADER, "classes": CLASSES_HEADER,
               "lineitems": LINEITEMS_HEADER, "results": RESULTS_HEADER}
    written: dict[str, int] = {}
    for kind, rows in tables.items():
        if not rows:
            continue
        path = output / f"{kind}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers[kind])
            writer.writerows(rows)
        written[kind] = len(rows)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compose OneRoster exports from real values")
    parser.add_argument("--train", default="data/processed/transfer/syllabus-docs/train.jsonl")
    parser.add_argument("--output-dir", default="data/processed/sis/export")
    parser.add_argument("--max-courses", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)

    groups = load_groups(Path(args.train))
    tables = build(groups, args.max_courses, args.seed)
    written = write_tables(tables, Path(args.output_dir))
    print(json.dumps({"groups_seen": len(groups), "written": written,
                      "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
