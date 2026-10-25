"""Read student-information-system exports the way a district actually has them.

Infinite Campus is the SIS most US districts run, and there is no single fixed
CSV header to learn: an Ad Hoc report exports whichever columns the user ticked
(``courseMasterNumber``, ``courseMasterName``, ``sectionNumber``, ...), while the
sanctioned interchange format is 1EdTech **OneRoster 1.1** (``courses.csv``,
``classes.csv``, ``lineItems.csv``, ``results.csv``). So this module maps both:

``OneRoster`` files by their standard header
    ``courses.title`` -> ``course_title``, ``courses.courseCode`` ->
    ``course_code``, ``courses.grades`` -> ``course_level`` (grade levels are
    what a K-12 course is offered at), ``courses.subjects`` -> ``program``.
    ``classes.classCode`` -> ``class_section``, ``classes.classType`` ->
    ``class_type``, ``classes.location`` -> ``class_meeting_room``,
    ``classes.periods`` -> ``meeting_time``, ``classes.termSourcedIds`` ->
    ``term``. ``lineItems.title`` -> ``assignments``,
    ``lineItems.description`` -> ``course_content``,
    ``lineItems.dueDate`` -> ``due_date``. ``results.score`` -> ``grades``
    (a score is the printed grade value), ``results.comment`` ->
    ``additional_information``.

``Ad Hoc`` reports by header alias
    Lowercased and stripped of spaces/underscores, so ``Course Number``,
    ``course_number`` and ``CourseNumber`` all reach ``course_code``.

Nothing is guessed: a column that is not in one of the two tables is skipped,
because a wrong label is worse than a missing one. Values are printed exactly as
the export prints them (``3`` credits stays ``3``), and the record schema matches
``pace.field_corpus`` so an export can be concatenated with the public corpus.

    python -m pace.sis_import --input exports/ --school "Lincoln High" \\
        --output data/processed/sis/sis_corpus.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import save_jsonl
from .field_corpus import _record

#: OneRoster 1.1 columns -> our field labels, per file kind. Only columns whose
#: OneRoster meaning is the field itself are mapped; ``sourcedId``, ``status``
#: and ``dateLastModified`` are plumbing and are never a label.
ONEROSTER_LABELS: dict[str, dict[str, str]] = {
    "courses": {
        "title": "course_title",
        "coursecode": "course_code",
        "grades": "course_level",
        "subjects": "program",
    },
    "classes": {
        "title": "course_title",
        "classcode": "class_section",
        "classtype": "class_type",
        "location": "class_meeting_room",
        "periods": "meeting_time",
        "termsourcedids": "term",
        "grades": "course_level",
    },
    "lineitems": {
        "title": "assignments",
        "description": "course_content",
        "duedate": "due_date",
    },
    "results": {
        "score": "grades",
        "comment": "additional_information",
    },
}

#: Columns that identify the course or section a row belongs to, in priority
#: order. Used only for the group id, so one course never spans two splits.
ONEROSTER_KEYS: dict[str, tuple[str, ...]] = {
    # A course is identified by its sourcedId and a section by the course it
    # belongs to, so the two rows land in the same group.
    "courses": ("sourcedid", "coursecode"),
    "classes": ("coursesourcedid", "sourcedid", "classcode"),
    "lineitems": ("classsourcedid", "sourcedid"),
    "results": ("lineitemsourcedid", "sourcedid"),
}

#: Infinite Campus Ad Hoc / district export headers -> our field labels. Keys
#: are normalised with :func:`_norm` before lookup, so spacing and case do not
#: matter. Every target is a label in ``pace.field_corpus.FIELD_TYPES``.
HEADER_ALIASES: dict[str, str] = {
    "course number": "course_code",
    "course code": "course_code",
    "coursenumber": "course_code",
    "coursecode": "course_code",
    "coursemasternumber": "course_code",
    "coursemasterno": "course_code",
    "course name": "course_title",
    "course title": "course_title",
    "coursename": "course_title",
    "coursetitle": "course_title",
    "coursemastername": "course_title",
    "description": "course_description",
    "course description": "course_description",
    "coursedescription": "course_description",
    "section number": "class_section",
    "sectionnumber": "class_section",
    "section": "class_section",
    "teacher name": "instructor",
    "teachername": "instructor",
    "teacher": "instructor",
    "instructor": "instructor",
    "instructorname": "instructor",
    "room": "class_meeting_room",
    "roomname": "class_meeting_room",
    "roomnumber": "class_meeting_room",
    "location": "class_meeting_room",
    "period": "meeting_time",
    "periodname": "meeting_time",
    "periods": "meeting_time",
    "meetingtime": "meeting_time",
    "days": "meeting_days",
    "meetingdays": "meeting_days",
    "term": "term",
    "termname": "term",
    "termcode": "term",
    "prerequisite": "prerequisites",
    "prerequisites": "prerequisites",
    "credits": "credits",
    "units": "credits",
    "points": "credits",
    "capacity": "capacity",
    "maxenrollment": "capacity",
    "enrollmentcapacity": "capacity",
    "gradelevel": "course_level",
    "grade level": "course_level",
    "grades": "course_level",
    "department": "program",
    "subject": "program",
    "subjectarea": "program",
    "coursetype": "class_type",
    "classtype": "class_type",
    "scheduletype": "class_type",
}

#: Cell contents that mean "this field was left blank" in a real export.
MISSING = {"", "-", "--", "n/a", "na", "none", "null", "nan", "tba", "not available", "."}

_SPACE = re.compile(r"\s+")
_SLUG = re.compile(r"[^a-z0-9]+")


def _norm(header: str) -> str:
    """A header key that ignores case, spaces, underscores and punctuation."""
    return _SLUG.sub("", str(header or "").lower())


#: Case/spacing-insensitive lookup for :data:`HEADER_ALIASES`.
_ALIAS_LOOKUP = {_norm(key): label for key, label in HEADER_ALIASES.items()}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE.sub(" ", str(value)).strip()


def _usable(text: str) -> bool:
    """False for blanks, placeholder markers and punctuation-only cells."""
    if not text or text.casefold() in MISSING:
        return False
    return any(character.isalnum() for character in text)


def _slug(text: str) -> str:
    return _SLUG.sub("-", str(text or "").strip().lower()).strip("-") or "export"


def _expand(paths: Iterable[str | Path] | str | Path) -> list[Path]:
    # One export is the common case, so a bare path is accepted as well as a list.
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            files.append(path)
    return files


def _oneroster_kind(path: Path) -> str | None:
    stem = _norm(path.stem)
    return stem if stem in ONEROSTER_LABELS else None


def _column_map(fieldnames: Sequence[str], kind: str | None) -> dict[str, str]:
    """column name -> label, using the OneRoster table or the Ad Hoc aliases."""
    mapping: dict[str, str] = {}
    for name in fieldnames:
        if not name:
            continue
        key = _norm(name)
        label = ONEROSTER_LABELS[kind].get(key) if kind else _ALIAS_LOOKUP.get(key)
        if label:
            mapping[name] = label
    return mapping


def _group_key(record: dict[str, str], mapping: dict[str, str],
               kind: str | None) -> str:
    """The course or section this row belongs to, so splits never split it."""
    by_label: dict[str, str] = {}
    for column, label in mapping.items():
        value = _clean(record.get(column))
        if value and label not in by_label:
            by_label[label] = value
    if kind:
        for column in ONEROSTER_KEYS[kind]:
            for name in (record or {}):
                if _norm(name) == column and _clean(record.get(name)):
                    return _clean(record.get(name))
    if "course_code" in by_label:
        return by_label["course_code"]
    if "class_section" in by_label:
        return by_label["class_section"]
    for name, value in (record or {}).items():
        if "id" in _norm(name) and _clean(value):
            return _clean(value)
    return ""


def load_sis_examples(paths: Sequence[str | Path] | str | Path, school: str = "Unspecified",
                      source: str = "Infinite Campus") -> list[dict[str, Any]]:
    """Turn OneRoster or Infinite Campus Ad Hoc CSVs into labelled field rows."""
    rows: list[dict[str, Any]] = []
    for path in _expand(paths):
        kind = _oneroster_kind(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                continue
            mapping = _column_map(fieldnames, kind)
            if not mapping:
                continue
            for index, record in enumerate(reader):
                key = _group_key(record, mapping, kind)
                group = f"{_slug(source)}-group:{key.casefold()}" if key else \
                    f"{_slug(source)}-file:{path.stem}:{index}"
                source_id = f"{_slug(source)}:{path.stem}:{index}:{key or 'row'}"
                for column, label in mapping.items():
                    text = _clean(record.get(column))
                    if not _usable(text):
                        continue
                    rows.append(_record(
                        text, label, group, source, column,
                        f"column of a real {source} export ({path.name})", source_id,
                        meta={"school": school, "sis_file": path.stem}))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import a SIS export as labelled field rows")
    parser.add_argument("--input", nargs="+", required=True,
                        help="OneRoster files, Ad Hoc CSVs, or directories of them")
    parser.add_argument("--output", default="data/processed/sis/sis_corpus.jsonl")
    parser.add_argument("--school", default="Unspecified",
                        help="School the export belongs to (used by transfer folds)")
    parser.add_argument("--source", default="Infinite Campus")
    args = parser.parse_args(argv)

    rows = load_sis_examples(args.input, school=args.school, source=args.source)
    save_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(json.dumps({"output": args.output, "rows": len(rows),
                      "by_label": dict(sorted(counts.items())),
                      "by_source": {args.source: len(rows)}}, indent=2))


if __name__ == "__main__":
    main()
