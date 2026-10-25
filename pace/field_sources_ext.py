"""Two more real catalogs, chosen because they print fields we had almost none of.

``Columbia University Directory of Classes`` (soid/columbia-catalog-data)
    37 terms of the public directory, 2015 Summer to 2027 Summer. Every row is
    one section, and the columns are already the field we want: the meeting
    days are a day string, the start and end times are clock times, the room is
    a building and room number, the section type is LECTURE/SEMINAR/LAB. This is
    the source that turns ``meeting_days`` (20 training rows), ``class_type``
    (18) and ``class_meeting_room`` from a handful of examples into thousands,
    without paraphrasing anything: each value is printed as the registrar
    printed it.

``Coursera course catalog`` (azrai99/coursera-course-dataset, Apache-2.0)
    Online-course listings with an organisation, a level, a schedule and a
    description. It is not a university catalog, which is the point: the field
    vocabulary has to survive a second layout where "level" is a word and the
    schedule is a sentence rather than a time.

Both loaders return rows in the same schema as the other sources so
``pace.field_corpus_ext`` can concatenate them and the existing split logic,
ambiguity exclusion and heading pools keep working unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from .field_corpus import _record, normalize_value


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _usable(value: str, minimum: int = 1) -> bool:
    # One character is a real value here: Columbia prints a 1-credit course as
    # ``Points: 1`` and a single-day meeting as ``Days: R``.
    if len(value) < minimum:
        return False
    if value.casefold() in {"none", "null", "nan", "-", "n/a", "tba", "not available"}:
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))


def _term_label(stem: str) -> str:
    """``2020-Fall`` is printed the way a catalog prints a term: ``Fall 2020``."""
    match = re.match(r"^(\d{4})-([A-Za-z]+)$", stem)
    if not match:
        return _clean(stem.replace("-", " "))
    return f"{match.group(2)} {match.group(1)}"


# --------------------------------------------------------------------------- #
# Columbia University directory of classes
# --------------------------------------------------------------------------- #

#: Column -> (label, heading as printed). The printed heading is what the row
#: carries into the context corpus, so it must read like a catalog column, not
#: like a database name: ``Points`` not ``points``.
COLUMBIA_COLUMNS: dict[str, tuple[str, str]] = {
    "course_code": ("course_code", "Course Code"),
    "course_title": ("course_title", "Course Title"),
    "course_descr": ("course_description", "Description"),
    "instructor": ("instructor", "Instructor"),
    "scheduled_days": ("meeting_days", "Days"),
    "location": ("class_meeting_room", "Location"),
    "points": ("credits", "Points"),
    "prerequisites": ("prerequisites", "Prerequisites"),
    "department": ("program", "Department"),
    "method_of_instruction": ("teaching_form", "Method of Instruction"),
    "type": ("class_type", "Type"),
    "open_to": ("restrictions", "Open To"),
    "class_id": ("class_section", "Class Section"),
}

COLUMBIA_SOURCE = "Columbia/Columbia-Directory-of-Classes"
COLUMBIA_SCHOOL = "Columbia University"


def _meeting_time(start: str, end: str) -> str:
    """A clock span printed the way a schedule prints it: ``9:00am - 12:10pm``."""
    start, end = _clean(start), _clean(end)
    if start and end:
        return f"{start} - {end}"
    return start or end


def load_columbia_examples(root: str | Path = "data/raw/columbia") -> list[dict[str, Any]]:
    """Every printed cell of every term of Columbia's public directory."""
    directory = Path(root)
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in sorted(directory.glob("*.csv")):
        term = _term_label(path.stem)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                course = _clean(row.get("course_code"))
                section = _clean(row.get("class_id")) or _clean(row.get("section_key"))
                group = f"columbia-course:{course or section or index}"
                source_id = f"columbia:{path.stem}:{section or index}"
                meta = {"school": COLUMBIA_SCHOOL, "term": term}
                for column, (label, printed) in COLUMBIA_COLUMNS.items():
                    value = _clean(row.get(column))
                    if not _usable(value):
                        continue
                    key = (normalize_value(value), label, path.stem)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(_record(value, label, group, COLUMBIA_SOURCE, printed,
                                        f"real directory cell, {term} term", source_id, dict(meta)))
                meeting = _meeting_time(row.get("scheduled_time_start"),
                                        row.get("scheduled_time_end"))
                if _usable(meeting):
                    key = (normalize_value(meeting), "meeting_time", path.stem)
                    if key not in seen:
                        seen.add(key)
                        rows.append(_record(meeting, "meeting_time", group, COLUMBIA_SOURCE,
                                            "Time",
                                            f"real directory cell, {term} term", source_id,
                                            dict(meta)))
                if _usable(term):
                    rows.append(_record(term, "term", group, COLUMBIA_SOURCE, "Term",
                                        "term printed by the directory", source_id, dict(meta)))
    return rows


# --------------------------------------------------------------------------- #
# Coursera catalog
# --------------------------------------------------------------------------- #

COURSERA_COLUMNS: dict[str, tuple[str, str]] = {
    "title": ("course_title", "Course Title"),
    "Instructor": ("instructor", "Instructor"),
    "Organization": ("program", "Organization"),
    "Description": ("course_description", "Description"),
    "Level": ("course_level", "Level"),
    "Schedule": ("course_schedule", "Schedule"),
    "Modules/Courses": ("course_content", "Modules"),
}

COURSERA_SOURCE = "Coursera/Coursera-Catalog"
COURSERA_SCHOOL = "Coursera"


def _coursera_value(value: str) -> str:
    """Module lists arrive as ``['a', 'b']``; print them as a list reads."""
    text = _clean(value)
    if text.startswith("[") and text.endswith("]"):
        items = [_clean(part.strip(" '\"")) for part in text[1:-1].split(",")]
        return " ".join(item for item in items if item)
    return text


def load_coursera_examples(path: str | Path = "data/raw/coursera/coursera_course_2024.csv"
                           ) -> list[dict[str, Any]]:
    """Online-course listings: a second layout for the same academic fields."""
    source_path = Path(path)
    if not source_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            title = _clean(row.get("title"))
            digest = hashlib.sha1(title.casefold().encode("utf-8")).hexdigest()[:16]
            group = f"coursera-course:{digest}"
            source_id = f"coursera:{index}:{digest}"
            for column, (label, printed) in COURSERA_COLUMNS.items():
                value = _coursera_value(row.get(column))
                if not _usable(value):
                    continue
                key = (normalize_value(value), label)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(_record(value[:600], label, group, COURSERA_SOURCE, printed,
                                    "real online-course listing cell", source_id,
                                    {"school": COURSERA_SCHOOL}))
    return rows


def load_extension_examples(columbia_root: str | Path = "data/raw/columbia",
                            coursera_path: str | Path = "data/raw/coursera/coursera_course_2024.csv"
                            ) -> list[dict[str, Any]]:
    """Both new catalogs, or whichever one is on disk."""
    return [*load_columbia_examples(columbia_root), *load_coursera_examples(coursera_path)]
