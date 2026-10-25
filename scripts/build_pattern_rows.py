"""Mechanical value shapes the reader gets wrong when a heading is missing.

A value like ``Dr. Johnson`` or ``Room 303`` is unambiguous to a person because
the shape alone carries the field: an honorific is a person, ``Room`` plus a
number is a place, a day plus a time range is when a class meets. The model reads
these as ``course_code`` at 99%, because the corpus it trained on almost never
prints them without a heading in front.

The rows here are generated shapes, not invented facts: each one is a form the
`pace.sis_import` header table already maps to that label, written the way a
district export or a syllabus prints it. They are added to the training corpus so
the shape is covered, then the model is re-measured on the same three benchmarks
-- generated coverage that costs a real benchmark point is not worth having.

    python scripts/build_pattern_rows.py --output data/processed/field_patterns/rows.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

SURNAMES = ["Johnson", "Okafor", "Nguyen", "Alvarez", "Kowalski", "Patel", "Rossi",
            "Haddad", "Silva", "Brown", "Chen", "Dubois", "Ibrahim", "Novak",
            "Oyelaran", "Reyes", "Sato", "Weber", "Young", "Zimmer"]
FIRST = ["Allison", "Marcus", "Priya", "Daniel", "Sofia", "Wei", "Amara", "Noah",
         "Elena", "Tomas", "Aisha", "Jonah", "Mei", "Omar", "Ruth"]
HONORIFICS = ["Dr.", "Prof.", "Professor", "Mr.", "Mrs.", "Ms.", "Mx."]
BUILDINGS = ["Sexton", "Kemper", "Latimer", "Whitman", "Barrows", "Haviland",
             "Bishop", "Cory", "Evans", "LeConte", "Wheeler", "Dwinelle"]
ROOM_WORDS = ["Room", "Rm.", "Rm", "Office"]
DAYS = ["MWF", "TTh", "TR", "MW", "Mon/Wed", "Tue/Thu", "Wed", "Fri", "M-F"]
TERMS = ["Fall 2026", "Spring 2027", "Summer 2026", "Winter 2027", "Autumn 2026"]
EXAM_WORDS = ["Final Examination", "Final Exam", "Midterm Exam", "Midterm",
              "Final", "Unit Exam", "Comprehensive Final"]
MONTHS = ["January", "February", "March", "April", "September", "October",
          "November", "December"]


def _row(text: str, label: str, index: int) -> dict:
    digest = hashlib.sha256(f"{label}:{text}".encode()).hexdigest()[:16]
    return {
        "text": text,
        "label": label,
        "bare_text": text,
        "family": "pattern",
        "group_id": f"pattern:{label}:{index}",
        "source": "pattern",
        "source_field": label,
        "source_id": f"pattern:{digest}",
        "source_key": "pattern",
        "provenance": "generated value shape for a field the SIS header table already maps",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def rows() -> list[dict]:
    out: list[dict] = []
    index = 0

    def add(text: str, label: str) -> None:
        nonlocal index
        out.append(_row(text, label, index))
        index += 1

    for honorific, surname in itertools.product(HONORIFICS, SURNAMES):
        add(f"{honorific} {surname}", "instructor")
    for first, surname in itertools.product(FIRST, SURNAMES[:8]):
        add(f"{first} {surname}", "instructor")
    for honorific, first, surname in itertools.product(["Dr.", "Prof."], FIRST[:6],
                                                       SURNAMES[:6]):
        add(f"{honorific} {first} {surname}", "instructor")
        add(f"{first} {surname}, {honorific.rstrip('.')}", "instructor")
    for word, number in itertools.product(ROOM_WORDS, [f"{n}" for n in range(101, 460, 7)]):
        add(f"{word} {number}", "class_meeting_room")
    for building, number in itertools.product(BUILDINGS, range(110, 420, 23)):
        add(f"{building} {number}", "class_meeting_room")
        add(f"{building} Hall, Room {number}", "class_meeting_room")
    for days, start, end in itertools.product(
            DAYS, ["8:00", "9:30", "10:00", "11:00", "1:00", "2:00"],
            ["9:15", "10:45", "11:50", "12:15", "2:15", "3:30"]):
        add(f"{days} {start}-{end}", "meeting_time")
        add(f"{days} {start} AM - {end} PM", "meeting_time")
        add(f"{days}, {start} - {end}", "meeting_time")
    # Whole-hour ranges with a bare suffix are how most syllabi print them.
    for days, start, end, suffix in itertools.product(
            DAYS, ["2", "3", "4", "9", "10", "11", "1"],
            ["3", "4", "5", "10", "11", "12", "2"], ["pm", "am", ""]):
        add(f"{days} {start}-{end}{suffix}", "meeting_time")
        add(f"{days} {start}:00-{end}:00{suffix}", "meeting_time")
    for period in list(range(1, 9)):
        add(f"Period {period}", "meeting_time")
        add(f"{period}{'st' if period == 1 else 'nd' if period == 2 else 'rd' if period == 3 else 'th'} period", "meeting_time")
    for term in TERMS:
        add(term, "term")
        add(term.replace(" ", " "), "term")
    for credits in ["1", "2", "3", "4", "5", "0.5", "1.0", "3.0"]:
        add(f"{credits} credits", "credits")
        add(f"{credits} units", "credits")
        add(f"{credits} credit hours", "credits")
    for word, month, day in itertools.product(EXAM_WORDS, MONTHS, [3, 7, 12, 16, 21, 28]):
        add(f"{word} {month} {day}", "exam_schedule")
        add(f"{word}: {month} {day}", "exam_schedule")
        add(f"{word} - {month} {day}", "exam_schedule")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate mechanical value-shape rows")
    parser.add_argument("--output", default="data/processed/field_patterns/rows.jsonl")
    args = parser.parse_args(argv)

    generated = rows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in generated:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(json.dumps({"output": str(output), "rows": len(generated),
                      "by_label": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
