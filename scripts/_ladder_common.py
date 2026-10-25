"""The work both transcript ladders share: gold, comparison, tallies, guarantee.

The two harnesses measure different things — one degrades a photograph, the other
downscales a clean render — but they answer the same question about every rung:
does each value match the fixture, and when it does not, is it visibly unread
rather than wrong? That comparison used to be copied into both scripts, and so was
the guarantee summation. The copy that forgot the user capture is how
``guarantee.wrong_values`` came to undercount: arithmetic written twice is
arithmetic nobody owns. Everything here is one implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

FIELDS = ("course_code", "mark", "weight", "credit")


def gold_rows(fixture: Path) -> list[dict[str, Any]]:
    """The fixture's course rows, reduced to the four verified fields."""
    from pace.transcript_extract import parse_transcript

    parsed = parse_transcript(fixture.read_text(encoding="utf-8"))
    return [{field: record[field] for field in FIELDS}
            for record in parsed["course_records"]]


def same_value(field: str, left: Any, right: Any) -> bool:
    """Is a measured value the fixture's value for this field?"""
    if left is None or right is None:
        return False
    if field in {"weight", "credit"}:
        return abs(float(left) - float(right)) < 1e-9
    return str(left).strip().lower() == str(right).strip().lower()


def field_tally(rows: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-field outcomes, the wrong values themselves, and the exactness flag.

    Rows are matched by course code, not by index: a lost row must not shift every
    row after it and be reported as a pile of wrong values.
    """
    tally = {field: {"correct": 0, "unread": 0, "wrong": 0, "missing": 0}
             for field in FIELDS}
    by_code = {str(row.get("course_code") or "").strip().lower(): row for row in rows}
    expected_codes = {str(expected["course_code"]).lower() for expected in gold}
    for extra in by_code:
        if extra and extra not in expected_codes:
            tally["course_code"]["wrong"] += 1
    for expected in gold:
        code = str(expected["course_code"]).lower()
        row = by_code.get(code)
        if row is None:
            for field in FIELDS:
                tally[field]["missing"] += 1
            continue
        tally["course_code"]["correct"] += 1
        for field in ("mark", "weight", "credit"):
            value = row.get(field)
            if value is None:
                tally[field]["unread"] += 1
            elif same_value(field, value, expected[field]):
                tally[field]["correct"] += 1
            else:
                tally[field]["wrong"] += 1
    wrong_values = [
        {"field": field, "expected": expected[field],
         "got": by_code.get(str(expected["course_code"]).lower(), {}).get(field)}
        for expected in gold for field in FIELDS
        if (row := by_code.get(str(expected["course_code"]).lower())) is not None
        and row.get(field) is not None
        and not same_value(field, row.get(field), expected[field])
    ]
    # A code read into a different string is a wrong value even though no expected
    # row can be matched to it; counting it as "missing" only would let a
    # confidently wrong identifier pass the guarantee.
    wrong_values += [
        {"field": "course_code", "expected": None,
         "got": by_code[extra].get("course_code")}
        for extra in by_code if extra and extra not in expected_codes
    ]
    return {
        "fields": tally,
        "wrong_values": wrong_values,
        "wrong_total": len(wrong_values),
        "missing_total": sum(tally[field]["missing"] for field in FIELDS),
        "exact_rows": len(rows) == len(gold) and all(
            tally[field]["correct"] == len(gold) for field in FIELDS),
    }


def unverified_counts(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Values this measurement emitted that nothing re-read, counted per field.

    The record marks them one by one (``record["unverified"][field]``); the ladder
    counts the marks so a reader of the artifact cannot take a title or a GPA for a
    checked value just because it appeared.
    """
    counts: dict[str, dict[str, Any]] = {}

    def note(container: dict[str, Any]) -> None:
        for field, reason in (container.get("unverified") or {}).items():
            entry = counts.setdefault(field, {"values": 0, "checked": 0,
                                              "reason": reason})
            entry["values"] += 1

    for record in parsed.get("course_records") or []:
        note(record)
    note(parsed.get("academic_standing") or {})
    note(parsed.get("header") or {})
    for record in parsed.get("assessment_records") or []:
        note(record)
    return counts


GUARANTEE_STATEMENT = ("Every value in the verified fields is either equal to the "
                       "fixture or null with a reason; no rung may differ.")
UNVERIFIED_STATEMENT = ("Every other value these measurements emitted — titles, "
                        "term text, GPA, printed total credits, assessment values — "
                        "was never checked against the fixture. Each such value is "
                        "marked `unverified` on the record that carries it, and is "
                        "counted here under its field.")


def guarantee_block(measurements: list[dict[str, Any]], *, rungs: int) -> dict[str, Any]:
    """The guarantee over every measurement, the user capture included.

    ``wrong_values`` sums each measurement's ``wrong_total`` — the one number
    ``field_tally`` derives from the wrong values it found. A guarantee that sums
    a subset is a claim about a subset, which is how the capture came to be
    missing from it.
    """
    unverified: dict[str, dict[str, Any]] = {}
    for entry in measurements:
        for field, detail in entry["unverified"].items():
            total = unverified.setdefault(
                field, {"values": 0, "checked": 0, "reason": detail["reason"]})
            total["values"] += detail["values"]
    return {
        "statement": GUARANTEE_STATEMENT,
        "wrong_values": sum(entry["wrong_total"] for entry in measurements),
        "measurements": len(measurements),
        "rungs": rungs,
        "includes_user_capture": len(measurements) > rungs,
        "unverified": {
            "statement": UNVERIFIED_STATEMENT,
            "checked": 0,
            "fields": unverified,
        },
    }
