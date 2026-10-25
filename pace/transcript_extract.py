"""Deterministic, source-grounded academic transcript parsing.

This module performs **no model inference**. Every value it returns is copied from the
input text and carries character offsets, so a reviewer can verify each field against the
source. It targets the common secondary/post-secondary transcript table layout:

    DATE       GL  TITLE                  MARK  CREDIT
    ELECTIVE
    03/20/04   09  Elective Intensive    A     0.250

Columns are found by whitespace gaps, so the parser also survives a row-wise OCR of a
two-column page, where both page columns land on one text line:

    ELECTIVE   SOCIAL STUDIES
    03/20/04  09 Elective Intensive A 0.250   12/20/03  09 AK Studies B 0.500

It also reads the term-block layout, where a banner such as
``2025-2026 Grade 09   Term 2`` sits above rows of COURSE / MARK / WEIGHT / CREDIT with
percentage marks and no date column, a long course title wraps onto the next printed
line, and the footer prints ``Credit: 7.500   GPA: 4.13333``:

    2025-2026 Grade 09   Term 2
    220725y Accel Adv        94  1.0000  1
    Alg:Concept&Connect

It also parses the identity header (student, school, contact), subject-area section
headings, state assessment rows that carry a score and a performance level, and the
academic-standing footer (as-of date, total credits, GPA). Structural checks such as
"does the listed credit sum equal the reported total" are reported rather than hidden.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

SCHEMA_VERSION = "pace-transcript-ir/0.1"

# Marks are ordered longest-first so "A+" wins over "A" in the alternation.
MARK_VALUES = (
    "CR-", "NC", "CR", "AUD", "PASS", "FAIL",
    "A+", "A-", "A", "B+", "B-", "B", "C+", "C-", "C", "D+", "D-", "D",
    "F", "P", "W", "I",
)
MARK_ALTERNATION = "|".join(re.escape(mark) for mark in MARK_VALUES)
MARK_SET = frozenset(MARK_VALUES)

# Loose (single-space) fallbacks for transcripts that were never column aligned.
COURSE_RECORD_RE = re.compile(
    r"(?P<date>(?<![\d/])\d{1,2}/\d{1,2}/\d{2,4})"
    r"[ \t]+(?P<grade_level>\d{1,2})"
    r"[ \t]+(?P<title>[^\n\t]*?\S)"
    r"[ \t]+(?<![A-Za-z])(?P<mark>" + MARK_ALTERNATION + r")(?![A-Za-z])"
    r"[ \t]+(?P<credit>\d+(?:\.\d{1,3})?)(?![\d.])"
)
ASSESSMENT_RECORD_RE = re.compile(
    r"(?P<date>(?<![\d/])\d{1,2}/\d{1,2}/\d{2,4})"
    r"[ \t]+(?P<grade_level>\d{1,2})"
    r"[ \t]+(?P<title>[^\n\t]*?\S)"
    r"[ \t]+(?P<score>\d{2,4})(?!\d|\.\d)"
)
PERFORMANCE_RE = re.compile(
    r"Performance\s+Level\s*:?[ \t]*(?P<level>\d+)[ \t]+(?P<label>[A-Za-z][A-Za-z]*(?:[ \t]+[A-Za-z]+)*)",
    re.IGNORECASE,
)

DATE_CELL_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
GRADE_LEVEL_CELL_RE = re.compile(r"^\d{1,2}$")
# Transcripts print credits as 0.250, 0.25, 0.5, 0.75 or 1, so a credit cell may be a
# whole number as well as a decimal with up to three places.
CREDIT_CELL_RE = re.compile(r"^\d+(?:\.\d{1,3})?$")
SCORE_CELL_RE = re.compile(r"^\d{2,4}$")

IDENTITY_LABELS = (
    "sex", "dob", "ssn", "entry date", "exit date", "graduation",
    "parent/ guardian", "parent/guardian", "phone",
)
LABEL_RES = tuple(
    (label, re.compile(rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])", re.IGNORECASE))
    for label in IDENTITY_LABELS
)

STUDENT_NAME_RE = re.compile(r"^[A-Z][A-Za-z'’\-]+,\s*[A-Z][A-Za-z'’\-]*$")
STREET_RE = re.compile(r"^\d+\s+\S+")
CITY_STATE_ZIP_RE = re.compile(r"^[A-Za-z .'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$")
SECTION_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 &/,.()'\-]{1,60}$")

NON_SECTION_HEADINGS = frozenset({
    "ACADEMIC HISTORY", "ACADEMIC STANDING", "DATE GL TITLE MARK CREDIT",
    "DATE GL TITLE MARK CREDIT TERM", "GRADE POINT AVERAGE", "TRANSCRIPT",
    "CUMULATIVE GPA", "STUDENT RECORD",
})
TABLE_HEADER_CELLS = frozenset({
    "DATE", "GL", "TITLE", "MARK", "CREDIT", "TERM", "SCORE", "GRADE",
    "GRADE LEVEL", "PERFORMANCE", "SUBJECT", "COURSE", "PERIOD", "HOURS",
})
# A term block names its term above the rows instead of dating every row.
TERM_BANNER_RE = re.compile(
    r"(?<![0-9])(?P<year>\d{4}\s*-\s*\d{4})\s+Grade\s+(?P<grade>\d{1,2})\s+Term\s+(?P<term>\d{1,2})",
    re.IGNORECASE,
)
NUMERIC_CELL_RE = re.compile(r"^\d+(?:\.\d{1,6})?$")
# The column header says which numbers follow the title, so the same reader handles
# MARK / WEIGHT / CREDIT, MARK / CREDIT, or a district's own names for them.
NUMERIC_COLUMN_NAMES = {
    "MARK": "mark", "GRADE": "mark",
    "WEIGHT": "weight",
    "CREDIT": "credit", "CREDITS": "credit", "UNITS": "credit",
}
# A local course code printed in front of the title: 220725y, 42075y, 708325y. The
# trailing letter is required so a year such as 2025 is never read as a code, and a
# hyphenated catalogue number such as 21-120 is left to the syllabus reader.
COURSE_CODE_PREFIX_RE = re.compile(r"^(?P<code>\d{3,6}[A-Za-z]{1,2})\s+")
# --- course-block grade report ---------------------------------------------------
# A grade report prints no date, no credit and no weight. It prints one block per
# course: a header naming the grade column (S1, Q2, MARK), a row with the local
# course code, the title, the instructor in brackets, the period and the grade, then
# one indented row per graded task with the marks that task earned. The grade column
# of the last block is the report's own name for those marks.
GRADE_REPORT_TASK_LABEL = "TASK"
GRADE_REPORT_COURSE_LABELS = frozenset({"COURSE", "COURSES"})
# "9 WEEK", "9 Weeks", "Q1", "S1", "Sem 2", "Term 1".
PERIOD_PATTERN = r"\d{1,2}\s*Weeks?|Weeks?\s*\d{1,2}|[QS][1-4]|Sem(?:ester)?\s*\d|[Tt]erm\s*\d{1,2}"
PERIOD_TAIL_RE = re.compile(rf"(?<![A-Za-z0-9])(?P<period>{PERIOD_PATTERN})\s*$", re.IGNORECASE)
# The grade a course block prints: a percentage, or a letter grade on letter layouts.
GRADE_TAIL_RE = re.compile(r"^(?P<middle>.*?)\s*(?P<grade>\d{1,3}|[A-F][+-]?)$")
# The local course code of a grade-report row keeps its section after a hyphen:
# 10145y-101, 220745y-302. The letter suffix separates it from a bare year.
# A scan that lost its column gaps glues the title to the section number, so a
# sectioned code may be followed directly by its title; a code with no section must
# end at a gap, or "32145y" and the title's first letter would read as one code.
COURSE_ROW_HEAD_RE = re.compile(
    r"^(?P<code>\d{3,6}[A-Za-z]{1,2})(?:(?:\s*-\s*)(?P<section>\d{1,4})(?![0-9])|(?=\s))")
# A code printed with a delimiter — its section or a gap — so a line whose gaps were
# lost can still be split at the right place instead of mid-word.
DELIMITED_CODE_RE = re.compile(
    r"(?<![\dA-Za-z])(?P<code>\d{3,6}[A-Za-z]{1,2})(?=\s*-\s*\d|\s)")
# The bracketed instructor is peeled before the title, wherever it sits on the row:
# the period label beside it can be too damaged to read and still leave the name.
INSTRUCTOR_CLOSED_RE = re.compile(r"\[(?P<name>[^\[\]]+)\]")
# The instructor name and the grade can both wrap onto the next printed line, leaving
# the bracket open: "[de la Rosa," then "Jacquelyn Gabriell 9 WEEK 97".
INSTRUCTOR_OPEN_RE = re.compile(r"\[(?P<name>[^\[\]]*)$")
# The attendance table above a grade report: COURSE / ABSENT / TARDY.
ATTENDANCE_COLUMNS = ("ABSENT", "TARDY")

# Totals printed under the last row instead of in an ACADEMIC STANDING block.
FOOTER_TOTALS_RE = re.compile(r"(?<![A-Za-z])Credits?\s*:\s*(?P<credits>\d+(?:\.\d{1,4})?)", re.IGNORECASE)
FOOTER_GPA_RE = re.compile(r"(?<![A-Za-z])GPA\s*:\s*(?P<gpa>\d+(?:\.\d{1,6})?)", re.IGNORECASE)
# A line that reads as "Label: value" is never the rest of a wrapped course title.
LABELLED_LINE_RE = re.compile(r"[A-Za-z]\s*:\s+\S")
DOCUMENT_TITLE_HINTS = ("HISTORY", "TRANSCRIPT", "ACADEMIC RECORD", "STUDENT RECORD")
# A heading that names an institution is a school banner, not a subject area.
INSTITUTION_WORDS = ("HIGH SCHOOL", "UNIVERSITY", "COLLEGE", "DISTRICT", "ACADEMY")

_COLUMN_SEPARATOR_RE = re.compile(r"[ ]*\t+[ ]*|[ ]{2,}|[ ]*\|[ ]*")


def _line_spans(text: str) -> list[dict[str, Any]]:
    """Return every line with its character offset so evidence stays exact."""
    lines = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        stripped = raw.rstrip("\r\n")
        lines.append({"text": stripped, "start": offset, "end": offset + len(stripped)})
        offset += len(raw)
    return lines


def _span(text: str, start: int, end: int) -> dict[str, Any]:
    return {"text": text[start:end], "start": start, "end": end}


def _strip_span(text: str, start: int, end: int) -> dict[str, Any] | None:
    """Return the offset-correct span of the stripped text inside [start, end)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return _span(text, start, end)


def _cells(body: str) -> list[dict[str, Any]]:
    """Split a line into column cells on whitespace gaps, keeping exact offsets."""
    cells: list[dict[str, Any]] = []
    position = 0
    for match in _COLUMN_SEPARATOR_RE.finditer(body):
        if match.start() > position:
            cells.append({"text": body[position:match.start()], "start": position, "end": match.start()})
        position = match.end()
    if position < len(body):
        cells.append({"text": body[position:], "start": position, "end": len(body)})
    trimmed: list[dict[str, Any]] = []
    for cell in cells:
        raw = cell["text"]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        text = raw.strip()
        if not text:
            continue
        trimmed.append({"text": text, "start": cell["start"] + lead, "end": cell["end"] - trail})
    return trimmed


def _label_hits(line: str) -> list[tuple[str, int, int]]:
    hits: list[tuple[str, int, int]] = []
    for label, pattern in LABEL_RES:
        for match in pattern.finditer(line):
            hits.append((label, match.start(), match.end()))
    hits.sort(key=lambda hit: (hit[1], hit[2]))
    return hits


def _parse_identity(text: str, lines: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse identity fields, leftover header text and the school/contact block."""
    fields: dict[str, Any] = {}
    leftovers: list[dict[str, Any]] = []
    school_lines: list[dict[str, Any]] = []
    identity_started = False

    for line in lines:
        body = line["text"]
        if not body.strip():
            continue
        hits = _label_hits(body)
        if not hits:
            if identity_started:
                span = _strip_span(text, line["start"], line["end"])
                if span:
                    leftovers.append({"value": span["text"], **span, "role": "unassigned"})
            else:
                school_lines.append(line)
            continue
        identity_started = True
        prefix = _strip_span(text, line["start"], line["start"] + hits[0][1])
        if prefix:
            leftovers.append({"value": prefix["text"], **prefix, "role": "unassigned"})
        for index, (label, label_start, label_end) in enumerate(hits):
            value_start = label_end
            value_end = hits[index + 1][1] if index + 1 < len(hits) else len(body)
            key = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()
            if key in fields:
                continue
            span = _strip_span(text, line["start"] + value_start, line["start"] + value_end)
            if span is not None and span["text"].startswith(":"):
                span = _strip_span(text, span["start"] + 1, span["end"])
            if span is None:
                fields[key] = {"value": None, "start": None, "end": None, "empty_label": True}
            else:
                fields[key] = {"value": span["text"], **span}
    return fields, leftovers, school_lines


def _classify_leftovers(leftovers: list[dict[str, Any]]) -> dict[str, Any]:
    """Sort leftover header text into student/address fields without guessing identity."""
    student_name = None
    street = None
    city = None
    unassigned = []
    for item in leftovers:
        value = item["text"]
        if student_name is None and STUDENT_NAME_RE.match(value):
            student_name = {"value": value, **item}
        elif street is None and STREET_RE.match(value):
            street = {"value": value, **item}
        elif city is None and CITY_STATE_ZIP_RE.match(value):
            city = {"value": value, **item}
        else:
            unassigned.append(item)
    return {"student_name": student_name, "street_address": street, "city_state_zip": city, "unassigned": unassigned}


def _parse_academic_standing(text: str, lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the academic-standing footer: as-of date, total credits and GPA."""
    standing: dict[str, Any] = {"as_of": None, "total_credits": None, "gpa": None,
                                "heading": None, "source": None}
    heading_index = None
    for index, line in enumerate(lines):
        if line["text"].strip().upper() == "ACADEMIC STANDING":
            heading_index = index
            standing["heading"] = _span(text, line["start"], line["end"])
            standing["source"] = "academic_standing"
            break
    if heading_index is None:
        # Some transcripts print the totals under the last row, labelled, with no
        # block heading at all: "Credit: 7.500   GPA: 4.13333". The labels are the
        # evidence, so the number is read and pointed at rather than inferred.
        credits = FOOTER_TOTALS_RE.search(text)
        gpa = FOOTER_GPA_RE.search(text)
        if credits is None and gpa is None:
            return standing
        standing["source"] = "labelled_footer"
        if credits is not None:
            standing["total_credits"] = {
                "value": float(credits.group("credits")),
                **_span(text, credits.start("credits"), credits.end("credits")),
            }
        if gpa is not None:
            standing["gpa"] = {
                "value": float(gpa.group("gpa")),
                **_span(text, gpa.start("gpa"), gpa.end("gpa")),
            }
        return standing
    for line in lines[heading_index + 1:]:
        body = line["text"]
        if not body.strip():
            continue
        if "AS OF" in body.upper():
            match = re.search(r"(?<![\d/])\d{1,2}/\d{1,2}/\d{2,4}", body)
            if match:
                standing["as_of"] = _span(text, line["start"] + match.start(), line["start"] + match.end())
            continue
        values = re.search(
            r"(?P<date>(?<![\d/])\d{1,2}/\d{1,2}/\d{2,4})?"
            r"[ \t]+(?P<credits>\d+(?:\.\d{1,3})?)?"
            r"[ \t]*(?P<gpa>\d+\.\d{1,4})?[ \t]*$",
            body,
        )
        if not values:
            continue
        if not (values.group("date") or values.group("credits") or values.group("gpa")):
            continue
        if values.group("date") and standing["as_of"] is None:
            standing["as_of"] = _span(text, line["start"] + values.start("date"), line["start"] + values.end("date"))
        if values.group("credits") and standing["total_credits"] is None:
            standing["total_credits"] = {
                "value": float(values.group("credits")),
                **_span(text, line["start"] + values.start("credits"), line["start"] + values.end("credits")),
            }
        if values.group("gpa") and standing["gpa"] is None:
            standing["gpa"] = {
                "value": float(values.group("gpa")),
                **_span(text, line["start"] + values.start("gpa"), line["start"] + values.end("gpa")),
            }
        if standing["total_credits"] is not None:
            break
    return standing


def _is_header_run(cells: list[dict[str, Any]]) -> bool:
    """True for a run of cells that names a transcript table header."""
    if len(cells) < 3:
        return False
    texts = [cell["text"].upper() for cell in cells]
    if "DATE" not in texts or "TITLE" not in texts:
        return False
    return all(text in TABLE_HEADER_CELLS for text in texts)


def _header_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Remove a table-header run at either edge of a (possibly merged) row.

    A row-wise OCR of a two-column page puts one page column after the other on the
    same line, so a header row can appear on its own, at the start of the line, or at
    the end of it. Returns the cells that are not part of a header run, or None when
    the line carries no header at all.
    """
    if not cells:
        return None
    if _is_header_run(cells):
        return []
    lead = 0
    while lead < len(cells) and cells[lead]["text"].upper() in TABLE_HEADER_CELLS:
        lead += 1
    if _is_header_run(cells[:lead]):
        remaining = cells[lead:]
        trail = 0
        while trail < len(remaining) and remaining[len(remaining) - 1 - trail]["text"].upper() in TABLE_HEADER_CELLS:
            trail += 1
        if trail and _is_header_run(remaining[len(remaining) - trail:]):
            remaining = remaining[:len(remaining) - trail]
        return remaining
    trail = 0
    while trail < len(cells) and cells[len(cells) - 1 - trail]["text"].upper() in TABLE_HEADER_CELLS:
        trail += 1
    if trail and _is_header_run(cells[len(cells) - trail:]):
        return cells[:len(cells) - trail]
    return None


def _is_heading_text(value: str) -> bool:
    """True for a subject-area heading such as 'LANGUAGE ARTS'."""
    if value in NON_SECTION_HEADINGS or value in MARK_SET:
        return False
    if not SECTION_HEADING_RE.match(value):
        return False
    return not any(word in value for word in INSTITUTION_WORDS)


def _record_from_cells(
    cells: list[dict[str, Any]],
    index: int,
    body: str,
    line_start: int,
    text: str,
) -> tuple[dict[str, Any] | None, int]:
    """Parse one record that starts at `cells[index]`; return (record, next index)."""
    if index + 1 >= len(cells):
        return None, index + 1
    if not GRADE_LEVEL_CELL_RE.match(cells[index + 1]["text"]):
        return None, index + 1
    end = index + 2
    while end < len(cells) and not DATE_CELL_RE.match(cells[end]["text"]):
        end += 1
    region = cells[index + 2:end]
    if not region:
        return None, end

    for position in range(len(region) - 1):
        mark_cell = region[position]
        credit_cell = region[position + 1]
        if mark_cell["text"] not in MARK_SET or not CREDIT_CELL_RE.match(credit_cell["text"]):
            continue
        title_cells = region[:position]
        if not title_cells:
            return None, end
        title_start = title_cells[0]["start"]
        title_end = title_cells[-1]["end"]
        return {
            "kind": "course",
            "date": cells[index]["text"],
            "grade_level": cells[index + 1]["text"],
            "title": body[title_start:title_end],
            "mark": mark_cell["text"],
            "credit": float(credit_cell["text"]),
            "evidence": {
                "date": _span(text, line_start + cells[index]["start"], line_start + cells[index]["end"]),
                "grade_level": _span(text, line_start + cells[index + 1]["start"], line_start + cells[index + 1]["end"]),
                "title": _span(text, line_start + title_start, line_start + title_end),
                "mark": _span(text, line_start + mark_cell["start"], line_start + mark_cell["end"]),
                "credit": _span(text, line_start + credit_cell["start"], line_start + credit_cell["end"]),
            },
            "start": line_start + cells[index]["start"],
            "end": line_start + credit_cell["end"],
        }, index + 2 + position + 2

    score_cell = region[-1]
    if not SCORE_CELL_RE.match(score_cell["text"]):
        return None, end
    title_cells = region[:-1]
    if not title_cells:
        return None, end
    title_start = title_cells[0]["start"]
    title_end = title_cells[-1]["end"]
    return {
        "kind": "assessment",
        "date": cells[index]["text"],
        "grade_level": cells[index + 1]["text"],
        "title": body[title_start:title_end],
        "score": int(score_cell["text"]),
        "evidence": {
            "date": _span(text, line_start + cells[index]["start"], line_start + cells[index]["end"]),
            "grade_level": _span(text, line_start + cells[index + 1]["start"], line_start + cells[index + 1]["end"]),
            "title": _span(text, line_start + title_start, line_start + title_end),
            "score": _span(text, line_start + score_cell["start"], line_start + score_cell["end"]),
        },
        "start": line_start + cells[index]["start"],
        "end": line_start + score_cell["end"],
    }, end


def _loose_record(body: str, line_start: int, text: str) -> dict[str, Any] | None:
    """Fallback for transcripts that were never column aligned."""
    match = COURSE_RECORD_RE.search(body)
    if match:
        return {
            "kind": "course",
            "date": match.group("date"),
            "grade_level": match.group("grade_level"),
            "title": match.group("title").strip(),
            "mark": match.group("mark"),
            "credit": float(match.group("credit")),
            "evidence": {
                "date": _span(text, line_start + match.start("date"), line_start + match.end("date")),
                "grade_level": _span(text, line_start + match.start("grade_level"), line_start + match.end("grade_level")),
                "title": _span(text, line_start + match.start("title"), line_start + match.end("title")),
                "mark": _span(text, line_start + match.start("mark"), line_start + match.end("mark")),
                "credit": _span(text, line_start + match.start("credit"), line_start + match.end("credit")),
            },
            "start": line_start + match.start(),
            "end": line_start + match.end(),
        }
    match = ASSESSMENT_RECORD_RE.search(body)
    if match:
        return {
            "kind": "assessment",
            "date": match.group("date"),
            "grade_level": match.group("grade_level"),
            "title": match.group("title").strip(),
            "score": int(match.group("score")),
            "evidence": {
                "date": _span(text, line_start + match.start("date"), line_start + match.end("date")),
                "grade_level": _span(text, line_start + match.start("grade_level"), line_start + match.end("grade_level")),
                "title": _span(text, line_start + match.start("title"), line_start + match.end("title")),
                "score": _span(text, line_start + match.start("score"), line_start + match.end("score")),
            },
            "start": line_start + match.start(),
            "end": line_start + match.end(),
        }
    return None


def _term_banners(text: str, lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every printed term banner, with the exact span it was read from."""
    banners: list[dict[str, Any]] = []
    for line in lines:
        match = TERM_BANNER_RE.search(line["text"])
        if match is None:
            continue
        year = re.sub(r"\s*-\s*", "-", match.group("year"))
        grade = match.group("grade")
        term = match.group("term")
        banners.append({
            "value": f"{year} Grade {grade} Term {term}",
            "academic_year": year,
            "grade_level": grade,
            "term": term,
            "_span": _span(text, line["start"] + match.start(), line["start"] + match.end()),
            "records": 0,
        })
    for banner in banners:
        span = banner.pop("_span")
        banner.update(span)
    return banners


def _numeric_columns(cells: list[dict[str, Any]]) -> list[str] | None:
    """Name the numeric columns of a COURSE / MARK / WEIGHT / CREDIT header.

    A term block has no date column, so the dated-row reader never fires on it. The
    header is the only place that says which numbers follow a title and in what
    order, so each row is read against this list rather than against fixed columns.
    """
    names = [cell["text"].upper() for cell in cells]
    if "DATE" in names:
        return None
    if "MARK" not in names or "CREDIT" not in names:
        return None
    if not {"COURSE", "COURSES", "TITLE"} & set(names):
        return None
    first = min(index for index, name in enumerate(names) if name in NUMERIC_COLUMN_NAMES)
    if first == 0:
        return None
    if any(name not in NUMERIC_COLUMN_NAMES for name in names[first:]):
        return None
    columns = [NUMERIC_COLUMN_NAMES[name] for name in names[first:]]
    if len(set(columns)) != len(columns):
        return None
    return columns


def _header_columns(cells: list[dict[str, Any]], body: str) -> list[str] | None:
    """Read the term-block header, whether its cells kept their gaps or not."""
    columns = _numeric_columns(cells)
    if columns is not None:
        return columns
    if len(body.split()) < 3:
        return None
    tokens = [{"text": match.group(), "start": match.start(), "end": match.end()}
              for match in re.finditer(r"\S+", body)]
    return _numeric_columns(tokens)


def _numeric_cell(name: str, cell: dict[str, Any]) -> dict[str, Any] | None:
    """Read one cell as the number its column names, or None if it is not one."""
    if not NUMERIC_CELL_RE.match(cell["text"]):
        return None
    value = float(cell["text"])
    if name == "mark":
        if not 0 <= value <= 100:
            return None
    elif value > 100:
        return None
    return {"text": cell["text"], "value": value, "start": cell["start"], "end": cell["end"]}


def _numeric_row(
    cells: list[dict[str, Any]], columns: list[str]
) -> tuple[list[dict[str, Any]] | None, dict[str, dict[str, Any]]]:
    """Split one printed row into its title cells and the numbers the header names."""
    index = len(cells)
    values: dict[str, dict[str, Any]] = {}
    for name in reversed(columns):
        index -= 1
        if index < 0:
            return None, values
        value = _numeric_cell(name, cells[index])
        if value is None:
            return None, values
        values[name] = value
    if index == 0:
        return None, values
    return cells[:index], values


def _loose_numeric_row(
    body: str, columns: list[str]
) -> tuple[int, int, dict[str, dict[str, Any]]] | None:
    """Fallback for a term row whose columns collapsed to single spaces.

    A PDF text layer often loses the wide gaps that mark the columns. The row is
    still a title followed by the named numbers in header order, so it is read from
    the right edge of the line instead of from the grid.
    """
    parts = [r"(?P<title>.+?)"] + [rf"(?P<{name}>\d+(?:\.\d{{1,6}})?)" for name in columns]
    match = re.match(r"^\s*" + r"\s+".join(parts) + r"\s*$", body)
    if match is None:
        return None
    values: dict[str, dict[str, Any]] = {}
    for name in columns:
        cell = {"text": match.group(name), "start": match.start(name), "end": match.end(name)}
        value = _numeric_cell(name, cell)
        if value is None:
            return None
        values[name] = value
    if not body[match.start("title"):match.end("title")].strip():
        return None
    return match.start("title"), match.end("title"), values


def _parse_numeric_table(
    text: str,
    lines: list[dict[str, Any]],
    cells_by_line: list[list[dict[str, Any]]],
    header_index: int,
    end_index: int,
    columns: list[str],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Read a term block whose rows carry numbers instead of dates.

    A row is recognised by its shape - a title followed by exactly one number per
    named column - because there is no date cell to anchor on. A course title that
    was broken across two printed lines is completed from the line below, and the
    evidence span then covers the line break, so the title is still a literal slice
    of the document.
    """
    records: list[dict[str, Any]] = []
    per_line: dict[int, int] = {}
    for line_index in range(header_index + 1, end_index):
        body = lines[line_index]["text"]
        cells = cells_by_line[line_index]
        if not body.strip() or not cells:
            continue
        if TERM_BANNER_RE.search(body):
            continue
        line_start = lines[line_index]["start"]
        title_cells, values = _numeric_row(cells, columns)
        if title_cells:
            title_start = line_start + title_cells[0]["start"]
            title_end = line_start + title_cells[-1]["end"]
        else:
            loose = _loose_numeric_row(body, columns)
            if loose is None:
                continue
            title_start_rel, title_end_rel, values = loose
            title_start = line_start + title_start_rel
            title_end = line_start + title_end_rel
        # The local course code shares the title's first cell in print, so it is split
        # off the text itself, with its own span, rather than from the grid.
        code = None
        code_span = None
        prefix = COURSE_CODE_PREFIX_RE.match(text[title_start:title_end])
        if prefix:
            code = prefix.group("code")
            code_span = _span(text, title_start, title_start + prefix.end("code"))
            title_start += prefix.end()
        if title_start >= title_end:
            continue
        mark = values["mark"]["value"]
        record = {
            "kind": "course",
            "date": None,
            "grade_level": None,
            "term": None,
            "course_code": code,
            "title": " ".join(text[title_start:title_end].split()),
            "mark": values["mark"]["text"],
            "mark_percent": int(mark) if mark == int(mark) else mark,
            "weight": values["weight"]["value"] if "weight" in values else None,
            "credit": float(values["credit"]["value"]),
            "evidence": {
                "course_code": code_span,
                "title": _span(text, title_start, title_end),
                "title_lines": [_span(text, title_start, title_end)],
                "mark": _span(text, line_start + values["mark"]["start"],
                              line_start + values["mark"]["end"]),
                "credit": _span(text, line_start + values["credit"]["start"],
                                line_start + values["credit"]["end"]),
                "weight": (None if "weight" not in values else
                           _span(text, line_start + values["weight"]["start"],
                                 line_start + values["weight"]["end"])),
                "term": None,
            },
            "start": title_start,
            "end": (line_start + cells[-1]["end"]) if title_cells else (line_start + len(body.rstrip())),
            "line_index": line_index,
            "column": 0,
            "title_wrapped": False,
            "section": None,
            "section_confidence": "unassigned",
        }

        # A printed line break inside a course title leaves the rest of the title on
        # the next line, with no numbers of its own. Only a line directly under its
        # own row, inside this table, and not an all-caps subject heading, continues
        # the title; anything else ends it.
        cursor = line_index + 1
        wrapped = 0
        while cursor < end_index and wrapped < 2:
            following_body = lines[cursor]["text"]
            following = cells_by_line[cursor]
            if not following_body.strip() or not following:
                break
            stripped = following_body.strip()
            if stripped.upper() == stripped:
                break
            if TERM_BANNER_RE.search(following_body):
                break
            if FOOTER_TOTALS_RE.search(following_body) or FOOTER_GPA_RE.search(following_body):
                break
            if _numeric_columns(following) is not None:
                break
            if any(NUMERIC_CELL_RE.match(cell["text"]) for cell in following):
                break
            if LABELLED_LINE_RE.search(stripped):
                break
            # The next printed line may itself be a row whose columns collapsed to
            # single spaces; it starts a record, so it cannot be a title fragment.
            if _loose_numeric_row(following_body, columns) is not None:
                break
            extra = following_body[following[0]["start"]:following[-1]["end"]]
            absolute_start = lines[cursor]["start"] + following[0]["start"]
            absolute_end = lines[cursor]["start"] + following[-1]["end"]
            # The title is one value printed on two lines, so it is joined with a
            # single space while each printed slice keeps its own evidence span.
            record["title"] = f"{record['title']} {' '.join(extra.split())}"
            record["evidence"]["title_lines"].append(_span(text, absolute_start, absolute_end))
            record["end"] = max(record["end"], absolute_end)
            record["title_wrapped"] = True
            cursor += 1
            wrapped += 1

        records.append(record)
        per_line[line_index] = per_line.get(line_index, 0) + 1
    return records, per_line


def _grade_report_header(cells: list[dict[str, Any]], body: str) -> str | None:
    """The grade column a course-block header names, or None if it is not one.

    A grade report's header is ``Course  Task  S1``: two fixed labels and one name
    for the mark column. The name is returned rather than assumed, because the same
    page prints S1, S2, Q1 or MARK under different districts. An OCR or a PDF text
    layer can lose the column gaps, so the tokens of the line are tried as well.
    """
    names = [cell["text"].upper() for cell in cells]
    label = _grade_report_label(names)
    if label is not None:
        return label
    tokens = body.split()
    if len(tokens) < 3:
        return None
    return _grade_report_label([token.upper() for token in tokens])


def _grade_report_label(names: list[str]) -> str | None:
    """One grade column named beside COURSE and TASK, or None."""
    if not GRADE_REPORT_COURSE_LABELS & set(names) or GRADE_REPORT_TASK_LABEL not in names:
        return None
    rest = [name for name in names
            if name not in GRADE_REPORT_COURSE_LABELS and name != GRADE_REPORT_TASK_LABEL]
    if len(rest) != 1:
        return None
    return rest[0]


def _known_course_codes(lines: list[dict[str, Any]]) -> set[str]:
    """Course codes the page prints with a delimiter after them.

    A grade report prints the same course code twice: once with a section number in
    the register, once alone above the attendance counts. A scan that loses the
    column gaps glues the code to the title, so the codes that survived with a gap
    or a hyphen are what lets such a line be split at the right character.
    """
    codes: set[str] = set()
    for line in lines:
        for match in DELIMITED_CODE_RE.finditer(line["text"]):
            codes.add(match.group("code"))
    return codes


def _course_row_head(body: str, known_codes: set[str]) -> re.Match[str] | None:
    """Where a grade-report row's code (and section) ends, or None."""
    head = COURSE_ROW_HEAD_RE.match(body)
    if head is not None:
        return head
    for code in sorted(known_codes, key=len, reverse=True):
        if not body.startswith(code) or len(body) == len(code):
            continue
        if body[len(code)].isdigit():
            continue  # a digit here belongs to a longer code, not to the title
        head = re.match(
            rf"^(?P<code>{re.escape(code)})(?:(?:\s*-\s*)(?P<section>\d{{1,4}})(?![0-9]))?",
            body)
        if head is not None:
            return head
    return None


def _row_end(spans: list[dict[str, Any] | None], floor: int) -> int:
    """The end of a record: the furthest printed cell that belongs to it."""
    return max([floor] + [span["end"] for span in spans if span])


def _grade_course_row(
    text: str, body: str, offset: int, line_index: int, grade_column: str | None,
    head: re.Match[str],
) -> tuple[dict[str, Any], bool, bool]:
    """Read one course block header row; say whether its grade was printed here.

    The values are peeled from the right edge of the line in the order the page
    prints them — grade, then period, then the bracketed instructor — because the
    title is free text and may itself contain numbers. Whatever is left is the
    title. A line whose grade and period wrapped onto the next printed line comes
    back incomplete, and the caller completes it from the line below.
    """
    code = head.group("code")
    section = head.group("section")
    code_span = _span(text, offset + head.start("code"), offset + head.end("code"))
    section_span = (None if section is None else
                    _span(text, offset + head.start("section"), offset + head.end("section")))
    rest_start = head.end()

    grade = None
    grade_span = None
    tail = GRADE_TAIL_RE.match(body[rest_start:])
    if tail is not None:
        grade = tail.group("grade")
        grade_span = _span(text, offset + rest_start + tail.start("grade"),
                           offset + rest_start + tail.end("grade"))
        middle = tail.group("middle").rstrip()
    else:
        middle = body[rest_start:].rstrip()

    period = None
    period_span = None
    period_match = PERIOD_TAIL_RE.search(middle)
    if period_match is not None:
        period = " ".join(period_match.group("period").split())
        period_span = _span(text, offset + rest_start + period_match.start("period"),
                            offset + rest_start + period_match.end("period"))
        middle = middle[:period_match.start()].rstrip()

    instructor = None
    instructor_span = None
    instructor_open = False
    closed = INSTRUCTOR_CLOSED_RE.search(middle)
    opened = INSTRUCTOR_OPEN_RE.search(middle)
    if opened is not None and closed is None:
        # The bracket is still open: the name, and often the grade, wrap.
        instructor_open = True
    if closed is not None or opened is not None:
        match = closed if closed is not None else opened
        instructor = " ".join(match.group("name").split())
        instructor_span = _span(text, offset + rest_start + match.start("name"),
                                offset + rest_start + match.end("name"))
        middle = middle[:match.start()].rstrip()

    title_span = _strip_span(text, offset + rest_start, offset + rest_start + len(middle))
    title = " ".join(title_span["text"].split()) if title_span else None
    spans = [code_span, section_span, title_span, instructor_span, period_span, grade_span]
    record = {
        "kind": "course",
        "date": None,
        "grade_level": None,
        "term": None,
        "course_code": code,
        "section_number": section,
        "title": title,
        "instructor": instructor,
        "period": period,
        "grade_column": grade_column,
        "mark": grade,
        "mark_percent": int(grade) if grade and grade.isdigit() else None,
        "weight": None,
        "credit": None,
        "evidence": {
            "course_code": code_span,
            "section_number": section_span,
            "title": title_span,
            "title_lines": [title_span] if title_span else [],
            "instructor": instructor_span,
            # A wrapped instructor name is one value printed on two lines, the same
            # way a wrapped title is: each printed slice keeps its own span.
            "instructor_lines": [instructor_span] if instructor_span else [],
            "period": period_span,
            "mark": grade_span,
            "weight": None,
            "credit": None,
            "term": None,
        },
        "start": code_span["start"],
        "end": _row_end(spans, code_span["end"]),
        "line_index": line_index,
        "column": 0,
        "title_wrapped": False,
        "instructor_wrapped": False,
        "section": None,
        "section_confidence": "unassigned",
    }
    if grade is None:
        # The page prints no grade on this row. That is a value nobody could read,
        # not a zero, so it is refused with its reason where the value would sit.
        record.setdefault("unread", {})["mark"] = {
            "reason": ("the grade for this course is not printed on its own row; "
                       "no grade is emitted rather than one taken from another row"),
            "page_reading": None,
            "focused_readings": [],
            "evidence": None,
        }
    return record, grade is not None, instructor_open


def _grade_fragment_guard(body: str) -> bool:
    """True for a line that is prose or another table, never the rest of a header."""
    if body.upper() == "ACADEMIC STANDING" or body.upper() == body:
        return True
    if LABELLED_LINE_RE.search(body):
        return True
    if FOOTER_TOTALS_RE.search(body) or FOOTER_GPA_RE.search(body):
        return True
    return _grade_report_header([], body) is not None


def _append_grade_fragment(
    text: str, record: dict[str, Any], body: str, offset: int, instructor_open: bool = False
) -> dict[str, Any] | None:
    """Add a printed line that continues the course header above it.

    A wrapped instructor name is completed by the line below, and a wrapped title
    becomes one value printed on two lines. Each printed slice keeps its own
    evidence span, so the joined value is still traceable to the page.
    """
    span = _strip_span(text, offset, offset + len(body.rstrip()))
    if span is None:
        return None
    # The instructor bracket is still open, so this line finishes the name. Evidence
    # stays one span per printed line: a joined value never claims to be a slice.
    if record.get("instructor") is not None and instructor_open:
        cleaned = span["text"].replace("]", " ").strip()
        record["instructor"] = " ".join(f"{record['instructor']} {cleaned}".split())
        record["evidence"]["instructor_lines"].append(span)
        record["instructor_wrapped"] = True
    else:
        record["title"] = " ".join(f"{record['title'] or ''} {span['text']}".split())
        record["evidence"]["title_lines"].append(span)
        record["title_wrapped"] = True
    record["end"] = max(record["end"], span["end"])
    return span


def _grade_task_row(
    text: str, body: str, offset: int, line_index: int, course: dict[str, Any]
) -> dict[str, Any] | None:
    """One graded task printed under a course block, with the course it belongs to.

    A task row carries a label and the mark earned. The label can be lost by the
    optimiser or by OCR and leave the mark on its own line; the mark is still kept
    and the missing label is left null rather than guessed from position alone. A
    task marked with a letter keeps its printed mark and reports no percentage,
    instead of the letter sliding into the course title above it.
    """
    match = GRADE_TAIL_RE.match(body)
    if match is None:
        return None
    printed = match.group("grade")
    score = int(printed) if printed.isdigit() else None
    # A task mark is an earned-mark figure, not a percentage of the course grade.
    if score is not None and score > 100:
        return None
    label_text = match.group("middle")
    label_span = _strip_span(text, offset, offset + len(label_text))
    if label_span is None and score is None:
        return None
    score_span = _span(text, offset + match.start("grade"), offset + match.end("grade"))
    label = " ".join(label_span["text"].split()) if label_span else None
    return {
        "kind": "assessment",
        "date": None,
        "grade_level": None,
        "term": None,
        "title": label,
        "score": score,
        "mark": printed,
        "course_code": course.get("course_code"),
        "period": course.get("period"),
        "grade_column": course.get("grade_column"),
        "evidence": {
            "title": label_span,
            "score": score_span,
            "course_code": course["evidence"].get("course_code"),
        },
        "start": label_span["start"] if label_span else score_span["start"],
        "end": score_span["end"],
        "line_index": line_index,
        "column": 0,
        "section": None,
        "section_confidence": "unassigned",
    }


def _parse_grade_report(
    text: str,
    lines: list[dict[str, Any]],
    header_index: int,
    end_index: int,
    grade_column: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    """Read a course-block grade report: one course header, then its task rows."""
    courses: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    per_line: dict[int, int] = {}
    current: dict[str, Any] | None = None
    awaiting_grade = False
    instructor_open = False
    wrapped = 0
    known_codes = _known_course_codes(lines)
    for line_index in range(header_index + 1, end_index):
        body = lines[line_index]["text"]
        stripped = body.strip()
        if not stripped:
            continue
        offset = lines[line_index]["start"] + (len(body) - len(body.lstrip()))
        head = _course_row_head(stripped, known_codes)
        if head is not None:
            record, complete, opened = _grade_course_row(text, stripped, offset, line_index,
                                                        grade_column, head)
            courses.append(record)
            current = record
            awaiting_grade = not complete
            instructor_open = opened
            wrapped = 0
            per_line[line_index] = per_line.get(line_index, 0) + 1
            continue
        if current is None:
            continue
        if awaiting_grade:
            tail = GRADE_TAIL_RE.match(stripped)
            middle = tail.group("middle").rstrip() if tail is not None else ""
            # The wrapped line carries the period's label as well as the grade. A line
            # that ends in a bare number with no period is a task row, not this
            # course's grade, and reading it as the grade would move a task's mark
            # onto the course.
            period_match = PERIOD_TAIL_RE.search(middle) if middle else None
            if tail is not None and period_match is not None and tail.group("grade").isdigit():
                fragment = middle[:period_match.start()].rstrip()
                if fragment:
                    _append_grade_fragment(text, current, fragment, offset, instructor_open)
                    instructor_open = (instructor_open
                                       and not fragment.rstrip().endswith("]"))
                if period_match:
                    period = " ".join(period_match.group("period").split())
                    period_span = _span(
                        text, offset + period_match.start("period"),
                        offset + period_match.end("period"))
                    current["period"] = period
                    current["evidence"]["period"] = period_span
                    current["end"] = max(current["end"], period_span["end"])
                grade = tail.group("grade")
                grade_span = _span(text, offset + tail.start("grade"),
                                   offset + tail.end("grade"))
                current["mark"] = grade
                current["mark_percent"] = int(grade)
                current["evidence"]["mark"] = grade_span
                current["end"] = max(current["end"], grade_span["end"])
                unread = current.get("unread") or {}
                unread.pop("mark", None)
                if not unread:
                    current.pop("unread", None)
                awaiting_grade = False
                per_line[line_index] = per_line.get(line_index, 0) + 1
                continue
        task = _grade_task_row(text, stripped, offset, line_index, current)
        if task is not None:
            assessments.append(task)
            per_line[line_index] = per_line.get(line_index, 0) + 1
            continue
        if wrapped >= 2 or _grade_fragment_guard(stripped):
            continue
        if _append_grade_fragment(text, current, stripped, offset, instructor_open) is not None:
            wrapped += 1
            if stripped.rstrip().endswith("]"):
                instructor_open = False
    return courses, assessments, per_line


def _attendance_header(cells: list[dict[str, Any]], body: str) -> list[str] | None:
    """The attendance columns of a COURSE / ABSENT / TARDY header, in print order."""
    for names in ([cell["text"].upper() for cell in cells],
                  [token.upper() for token in body.split()]):
        if not set(ATTENDANCE_COLUMNS) <= set(names):
            continue
        if not GRADE_REPORT_COURSE_LABELS & set(names):
            continue
        extra = [name for name in names
                 if name not in GRADE_REPORT_COURSE_LABELS and name not in ATTENDANCE_COLUMNS]
        if extra:
            continue
        return [name.lower() for name in names if name in ATTENDANCE_COLUMNS]
    return None


def _parse_attendance(
    text: str, lines: list[dict[str, Any]], header_index: int, end_index: int,
    columns: list[str],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Read an attendance summary: a course on each row, one count per column.

    The table sits above the grade report and carries no grade at all, so the
    dated-row reader and the grade block both walk past it. Dropping it silently
    would leave six courses of the page unread.
    """
    records: list[dict[str, Any]] = []
    per_line: dict[int, int] = {}
    known_codes = _known_course_codes(lines)
    pattern = re.compile(
        r"^(?P<rest>.*?)\s+" +
        r"\s+".join(rf"(?P<{name}>\d{{1,3}})" for name in columns) + r"\s*$")
    for line_index in range(header_index + 1, end_index):
        body = lines[line_index]["text"]
        stripped = body.strip()
        if not stripped or _attendance_header([], stripped) is not None:
            continue
        head = _course_row_head(stripped, known_codes)
        if head is None:
            continue
        offset = lines[line_index]["start"] + (len(body) - len(body.lstrip()))
        match = pattern.match(stripped[head.end():])
        if match is None:
            continue
        first = min(match.start(name) for name in columns)
        title_span = _strip_span(text, offset + head.end(), offset + head.end() + first)
        if title_span is None:
            continue
        code_span = _span(text, offset + head.start("code"), offset + head.end("code"))
        counts = {name: int(match.group(name)) for name in columns}
        count_spans = {name: _span(text, offset + head.end() + match.start(name),
                                   offset + head.end() + match.end(name))
                       for name in columns}
        records.append({
            "kind": "attendance",
            "course_code": head.group("code"),
            "title": " ".join(title_span["text"].split()),
            **counts,
            "evidence": {
                "course_code": code_span,
                "title": title_span,
                **count_spans,
            },
            "start": code_span["start"],
            "end": _row_end([*count_spans.values()], code_span["end"]),
            "line_index": line_index,
            "column": 0,
            "section": None,
            "section_confidence": "unassigned",
        })
        per_line[line_index] = per_line.get(line_index, 0) + 1
    return records, per_line


def _column_seam(cells_by_line: list[list[dict[str, Any]]]) -> int | None:
    """Estimate the x position where the right page column starts.

    A row-wise OCR of a two-column page prints the table date twice on the same
    line, once per page column, so the second date cell marks the start of the
    right column. The estimate is only accepted when most such lines agree, so a
    single wide table row is never mistaken for a page layout.
    """
    candidates: list[int] = []
    for cells in cells_by_line:
        dates = [cell["start"] for cell in cells if DATE_CELL_RE.match(cell["text"])]
        if len(dates) >= 2:
            candidates.append(dates[1])
    if not candidates:
        return None
    candidates.sort()
    seam = candidates[len(candidates) // 2]
    agreeing = sum(1 for value in candidates if abs(value - seam) <= 2)
    if agreeing * 2 < len(candidates):
        return None
    return seam


def _column_groups(cells: list[dict[str, Any]], seam: int | None) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split one physical line into the page columns it carries.

    On a row-wise OCR of a two-column page both columns land on the same line.
    Splitting them here keeps every row in the column it belongs to, so a
    subject-area heading in the right column never captures the left-column rows
    printed below it.
    """
    if seam is None:
        return [(0, cells)] if cells else []
    split_cells: list[dict[str, Any]] = []
    for cell in cells:
        # A cell that straddles the gutter (a narrow gap between two tables) is
        # split at the seam so each half stays with its own column.
        if cell["start"] < seam < cell["end"]:
            left_part = {**cell, "end": seam}
            right_text = cell["text"][seam - cell["start"]:].strip()
            right_start = cell["end"] - len(cell["text"][seam - cell["start"]:].rstrip())
            right_part = {"text": right_text, "start": right_start, "end": cell["end"]}
            if right_part["text"]:
                split_cells.append(left_part)
                split_cells.append(right_part)
                continue
        split_cells.append(cell)
    left = [cell for cell in split_cells if cell["start"] < seam]
    right = [cell for cell in split_cells if cell["start"] >= seam]
    groups: list[tuple[int, list[dict[str, Any]]]] = []
    if left:
        groups.append((0, left))
    if right:
        groups.append((1, right))
    return groups


def parse_transcript(text: str) -> dict[str, Any]:
    """Parse an academic transcript into source-grounded records.

    Returns a review-only proposal object. Nothing here is committed to a student
    record, and every value points back at the exact source substring.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")

    lines = _line_spans(text)
    cells_by_line = [_cells(line["text"]) if line["text"].strip() else [] for line in lines]
    seam = _column_seam(cells_by_line)
    terms = _term_banners(text, lines)

    # A COURSE / MARK / WEIGHT / CREDIT header opens a term block the dated-row
    # reader cannot see, because there is no date to anchor a row on. A transcript
    # can print several such blocks — one per term, one per page — each with its
    # own header, so every block is found here. Finding only the first would read
    # page one and silently drop every later page.
    numeric_blocks: list[tuple[int, int, list[str]]] = []
    probe = 0
    while probe < len(lines):
        columns = _header_columns(cells_by_line[probe], lines[probe]["text"])
        if columns is None:
            probe += 1
            continue
        end = len(lines)
        for cursor in range(probe + 1, len(lines)):
            body = lines[cursor]["text"]
            if FOOTER_TOTALS_RE.search(body) or FOOTER_GPA_RE.search(body):
                end = cursor
                break
            # The next block's header closes this one even when no footer is printed.
            if _header_columns(cells_by_line[cursor], body) is not None:
                end = cursor
                break
        numeric_blocks.append((probe, end, columns))
        probe = end
    numeric_columns = numeric_blocks[0][2] if numeric_blocks else None

    # A grade report has no dates, no credit and no weight, so none of the readers
    # above can see it. Its blocks are found the same way — a header, then everything
    # up to the next header — and the attendance table above it is read too, because
    # it is a table of courses that would otherwise be dropped without a word.
    attendance_blocks: list[tuple[int, int, list[str]]] = []
    grade_blocks: list[tuple[int, int, str | None]] = []
    probe = 0
    while probe < len(lines):
        columns = _attendance_header(cells_by_line[probe], lines[probe]["text"])
        if columns is None:
            probe += 1
            continue
        end = len(lines)
        for cursor in range(probe + 1, len(lines)):
            body = lines[cursor]["text"]
            if _attendance_header(cells_by_line[cursor], body) is not None:
                end = cursor
                break
            if _grade_report_header(cells_by_line[cursor], body) is not None:
                end = cursor
                break
        attendance_blocks.append((probe, end, columns))
        probe = end
    probe = 0
    while probe < len(lines):
        grade_column = _grade_report_header(cells_by_line[probe], lines[probe]["text"])
        if grade_column is None:
            probe += 1
            continue
        end = len(lines)
        for cursor in range(probe + 1, len(lines)):
            body = lines[cursor]["text"]
            if body.strip().upper() == "ACADEMIC STANDING":
                end = cursor
                break
            if _grade_report_header(cells_by_line[cursor], body) is not None:
                end = cursor
                break
            # An attendance table's rows start with a course code and end in counts,
            # which a grade block would otherwise read as courses and grades.
            if _attendance_header(cells_by_line[cursor], body) is not None:
                end = cursor
                break
            if _numeric_columns(cells_by_line[cursor]) is not None:
                end = cursor
                break
        grade_blocks.append((probe, end, grade_column))
        probe = end

    # Every block's lines are owned by its own reader. Letting them through the
    # line loop would turn the header's grade column name into a subject heading.
    owned_blocks = numeric_blocks + attendance_blocks + grade_blocks

    # --- records ------------------------------------------------------------------
    course_records: list[dict[str, Any]] = []
    assessment_records: list[dict[str, Any]] = []
    heading_candidates: list[dict[str, Any]] = []
    column_header_offset: int | None = None
    standing_offset: int | None = None
    records_per_line: dict[int, int] = {}
    # A "group" is one record and everything that follows it until the next date cell.
    # On a row-wise OCR of a two-column page a group continues across lines, so a
    # performance level printed under an assessment row stays with its own group.
    group_assessments: dict[tuple[int, int], list[dict[str, Any]]] = {}
    performance_cells: list[dict[str, Any]] = []
    current_group = {0: 0, 1: 0}

    for line_index, line in enumerate(lines):
        body = line["text"]
        if not body.strip():
            continue
        if body.strip().upper() == "ACADEMIC STANDING":
            standing_offset = line["start"]
            continue
        if any(start <= line_index < end for start, end, _columns in owned_blocks):
            # The header and every row in a block belong to its own reader;
            # letting them through here would turn column names such as MARK and
            # CREDIT into subject-area headings.
            if line_index == owned_blocks[0][0] and column_header_offset is None:
                column_header_offset = line["start"]
            continue
        line_records: list[dict[str, Any]] = []
        for column, cells in _column_groups(cells_by_line[line_index], seam):
            remaining_cells = _header_cells(cells)
            if remaining_cells is not None:
                if column_header_offset is None:
                    column_header_offset = line["start"]
                cells = remaining_cells
                if not cells:
                    continue
            group_start = cells[0]["start"]
            group_end = cells[-1]["end"]

            group_records: list[dict[str, Any]] = []
            index = 0
            while index < len(cells):
                cell = cells[index]
                if not DATE_CELL_RE.match(cell["text"]):
                    performance = PERFORMANCE_RE.search(cell["text"])
                    if performance:
                        performance_cells.append({
                            "group": (column, current_group[column]),
                            "start": line["start"] + cell["start"] + performance.start(),
                            "end": line["start"] + cell["start"] + performance.end(),
                        })
                    elif _is_heading_text(cell["text"]):
                        heading_candidates.append({
                            "name": cell["text"],
                            "column": column,
                            **_span(text, line["start"] + cell["start"], line["start"] + cell["end"]),
                        })
                    index += 1
                    continue
                current_group[column] += 1
                record, next_index = _record_from_cells(cells, index, body, line["start"], text)
                if record is None:
                    index = next_index
                    continue
                record["column"] = column
                group_records.append(record)
                if record["kind"] == "assessment":
                    group_assessments.setdefault((column, current_group[column]), []).append(record)
                index = next_index

            if not group_records:
                loose = _loose_record(body[group_start:group_end], line["start"] + group_start, text)
                if loose is not None:
                    current_group[column] += 1
                    loose["column"] = column
                    group_records.append(loose)
                    if loose["kind"] == "assessment":
                        group_assessments.setdefault((column, current_group[column]), []).append(loose)
            line_records.extend(group_records)

        if line_records:
            records_per_line[line_index] = len(line_records)
        for record in line_records:
            record["line_index"] = line_index
            if record["kind"] == "course":
                course_records.append(record)
            else:
                assessment_records.append(record)

    for header, end, columns in numeric_blocks:
        numeric_records, numeric_per_line = _parse_numeric_table(
            text, lines, cells_by_line, header, end, columns
        )
        course_records.extend(numeric_records)
        for line_index, count in numeric_per_line.items():
            records_per_line[line_index] = records_per_line.get(line_index, 0) + count
    for header, end, grade_column in grade_blocks:
        grade_courses, grade_tasks, grade_per_line = _parse_grade_report(
            text, lines, header, end, grade_column
        )
        course_records.extend(grade_courses)
        assessment_records.extend(grade_tasks)
        for line_index, count in grade_per_line.items():
            records_per_line[line_index] = records_per_line.get(line_index, 0) + count
    attendance_records: list[dict[str, Any]] = []
    for header, end, columns in attendance_blocks:
        attendance_rows, attendance_per_line = _parse_attendance(
            text, lines, header, end, columns
        )
        attendance_records.extend(attendance_rows)
        for line_index, count in attendance_per_line.items():
            records_per_line[line_index] = records_per_line.get(line_index, 0) + count
    if numeric_blocks or grade_blocks or attendance_blocks:
        course_records.sort(key=lambda record: record["start"])
        assessment_records.sort(key=lambda record: record["start"])

    # Every course record carries the same keys whatever layout it came from, so a
    # consumer never has to guess which layout it is reading.
    for record in course_records:
        record.setdefault("course_code", None)
        record.setdefault("weight", None)
        record.setdefault("mark_percent", None)
        record.setdefault("title_wrapped", False)
        record.setdefault("term", None)
        record.setdefault("instructor", None)
        record.setdefault("instructor_wrapped", False)
        record.setdefault("period", None)
        record.setdefault("section_number", None)
        record.setdefault("grade_column", None)
        record["evidence"].setdefault("course_code", None)
        record["evidence"].setdefault("weight", None)
        record["evidence"].setdefault("term", None)
        record["evidence"].setdefault("instructor", None)
        record["evidence"].setdefault("instructor_lines",
                                       [record["evidence"]["instructor"]]
                                       if record["evidence"].get("instructor") else [])
        record["evidence"].setdefault("period", None)
        record["evidence"].setdefault("section_number", None)
        title_evidence = record["evidence"].get("title")
        record["evidence"].setdefault("title_lines", [title_evidence] if title_evidence else [])

    performances = []
    for match in PERFORMANCE_RE.finditer(text):
        span = _span(text, match.start(), match.end())
        performances.append({
            "level": int(match.group("level")),
            "label": match.group("label").strip(),
            "text": span["text"],
            "start": span["start"],
            "end": span["end"],
            # The value span is the level and its label, without the field label
            # ("Performance Level: 3 Proficient" -> "3 Proficient").
            "value_start": match.start("level"),
            "value_end": match.end("label"),
        })

    # --- header block boundary ---------------------------------------------------
    first_record_offset = min(
        [record["start"] for record in course_records + assessment_records] or [len(text)]
    )
    body_start = min(
        [first_record_offset]
        + ([column_header_offset] if column_header_offset is not None else [len(text)])
    )
    # A subject heading printed above the first record still opens the body. Adopt it
    # only when a record follows within a few lines, so a school banner is not a subject.
    for candidate in sorted(heading_candidates, key=lambda item: -item["start"]):
        if candidate["start"] >= body_start:
            continue
        following = [line for line in lines if line["start"] > candidate["end"] and line["text"].strip()][:3]
        if not any(
            COURSE_RECORD_RE.match(line["text"].strip()) or ASSESSMENT_RECORD_RE.match(line["text"].strip())
            for line in following
        ):
            continue
        body_start = candidate["start"]
        break

    headings = [
        candidate for candidate in heading_candidates
        if candidate["start"] >= body_start
        and (standing_offset is None or candidate["start"] < standing_offset)
    ]
    headings.sort(key=lambda item: item["start"])
    body_line_index = bisect_right([line["start"] for line in lines], body_start) - 1
    header_lines = lines[:max(body_line_index, 0)]
    identity_fields, leftovers, school_lines = _parse_identity(text, header_lines)
    # A term banner above the table names the term, not the school.
    school_lines = [line for line in school_lines if not TERM_BANNER_RE.search(line["text"])]
    header_roles = _classify_leftovers(leftovers)

    document_title = None
    school_name = None
    school_address = []
    remaining_school_lines = list(school_lines)
    if remaining_school_lines:
        first = remaining_school_lines[0]
        candidate = first["text"].strip()
        if candidate.upper() in NON_SECTION_HEADINGS or any(hint in candidate.upper() for hint in DOCUMENT_TITLE_HINTS):
            document_title = _span(text, first["start"], first["start"] + len(candidate))
            remaining_school_lines = remaining_school_lines[1:]
    for index, line in enumerate(remaining_school_lines):
        span = _strip_span(text, line["start"], line["end"])
        if span is None:
            continue
        if index == 0 and school_name is None:
            school_name = {"value": span["text"], **span}
        else:
            school_address.append({"value": span["text"], **span})

    # --- section assignment ------------------------------------------------------
    def section_for(record: dict[str, Any]) -> dict[str, Any] | None:
        """The subject area that owns a record inside its own page column."""
        column = record.get("column", 0)
        chosen = None
        for heading in headings:
            if heading.get("column", 0) != column:
                continue
            if heading["start"] <= record["start"]:
                chosen = heading
            else:
                break
        return chosen

    for record in course_records + assessment_records:
        heading = section_for(record)
        record["section"] = heading["name"] if heading else None
        if heading is None:
            record["section_confidence"] = "unassigned"
        elif seam is not None:
            record["section_confidence"] = "column_order"
        else:
            record["section_confidence"] = "line_order"

    # A term banner owns every record printed below it, in its own page. When the
    # page prints the banner twice (a page header and again above the table), the
    # nearest one above the row is the one that owns it.
    for record in course_records + assessment_records:
        record["term"] = None
        record["evidence"]["term"] = None
        owner = None
        for term in terms:
            if term["start"] <= record["start"]:
                owner = term
            else:
                break
        if owner is None:
            continue
        record["term"] = owner["value"]
        record["evidence"]["term"] = {key: owner[key] for key in ("text", "start", "end")}
        owner["records"] += 1

    # A performance level belongs to the assessment row that precedes it. Prefer the
    # assessment inside the same row group; fall back to the nearest preceding one.
    for record in assessment_records:
        record["performance_level"] = None
        record["performance_label"] = None
        record["evidence"]["performance_level"] = None
    for performance in performances:
        owner = None
        for cell in performance_cells:
            if cell["start"] != performance["start"]:
                continue
            group_rows = group_assessments.get(cell["group"], [])
            if group_rows:
                owner = group_rows[-1]
            break
        if owner is None:
            candidates = [
                record for record in assessment_records
                if record["end"] <= performance["start"] and performance["start"] - record["end"] <= 200
            ]
            if candidates:
                owner = max(candidates, key=lambda record: record["end"])
        if owner is None:
            continue
        owner["performance_level"] = performance["level"]
        owner["performance_label"] = performance["label"]
        owner["evidence"]["performance_level"] = {
            "level": performance["level"],
            "label": performance["label"],
            "start": performance["start"],
            "end": performance["end"],
            "text": performance["text"],
        }
        owner["evidence"]["performance_level_value"] = {
            "level": performance["level"],
            "label": performance["label"],
            "start": performance["value_start"],
            "end": performance["value_end"],
            "text": text[performance["value_start"]:performance["value_end"]],
        }

    # --- interleaved two-column detection ---------------------------------------
    interleaved = any(count > 1 for count in records_per_line.values())

    standing = _parse_academic_standing(text, lines)

    # --- consistency checks --------------------------------------------------------
    # A grade report prints no credit at all. Summing a column of nulls as zero would
    # read as "this transcript awards no credit", which is not what the page says.
    credits = [record["credit"] for record in course_records
               if record.get("credit") is not None]
    computed_credits = round(sum(credits), 3)
    layout = ("course_mark_weight_credit" if numeric_columns is not None else
              "course_task_grade" if grade_blocks else
              "course_attendance" if attendance_blocks else "dated_rows")
    reported_credits = standing["total_credits"]["value"] if standing["total_credits"] else None
    section_counts: dict[str, int] = {}
    for record in course_records:
        key = record["section"] or "unassigned"
        section_counts[key] = section_counts.get(key, 0) + 1
    weighted = [record["weight"] for record in course_records if record.get("weight") is not None]
    checks = {
        "course_record_count": len(course_records),
        "assessment_record_count": len(assessment_records),
        "attendance_record_count": len(attendance_records),
        "layout": layout,
        "grade_report_blocks": len(grade_blocks),
        "marks_numeric": bool(weighted) or any(
            record.get("mark_percent") is not None for record in course_records
        ),
        "computed_total_credits": computed_credits,
        "credits_printed": bool(credits),
        "records_without_grade": sum(1 for record in course_records
                                     if record.get("mark") is None),
        "computed_total_weight": round(sum(weighted), 4) if weighted else None,
        "records_without_weight": (sum(1 for record in course_records if record.get("weight") is None)
                                   if weighted else None),
        "records_without_term": sum(1 for record in course_records if not record.get("term")),
        "reported_total_credits": reported_credits,
        "credit_sum_matches_reported_total": (
            None if reported_credits is None else abs(computed_credits - reported_credits) < 0.001
        ),
        "credit_sum_difference": (
            None if reported_credits is None else round(computed_credits - reported_credits, 3)
        ),
        "records_without_title": sum(1 for record in course_records if not record["title"]),
        "records_without_section": sum(1 for record in course_records if not record["section"]),
        "records_per_subject_area": section_counts,
        "column_interleaving_detected": interleaved,
        "unattached_performance_levels": sum(
            1 for performance in performances
            if not any(
                record.get("performance_level") is not None
                and record["evidence"].get("performance_level")
                and record["evidence"]["performance_level"]["start"] == performance["start"]
                for record in assessment_records
            )
        ),
        "gpa_present": standing["gpa"] is not None,
        # A transcript has rows, a name, or both. A syllabus pointed at this parser
        # has neither, and its title can look like a school banner; saying so keeps
        # an empty result from reading like a confident reading of zero courses.
        "looks_like_transcript": bool(course_records or assessment_records
                                      or header_roles["student_name"]),
    }

    header = {
        "document_title": document_title,
        "school_name": school_name,
        "school_address": school_address,
        "phone": identity_fields.get("PHONE"),
        "student_name": header_roles["student_name"],
        "street_address": header_roles["street_address"],
        "city_state_zip": header_roles["city_state_zip"],
        "sex": identity_fields.get("SEX"),
        "dob": identity_fields.get("DOB"),
        "ssn": identity_fields.get("SSN"),
        "parent_guardian": identity_fields.get("PARENT_GUARDIAN"),
        "entry_date": identity_fields.get("ENTRY_DATE"),
        "exit_date": identity_fields.get("EXIT_DATE"),
        "graduation": identity_fields.get("GRADUATION"),
        "unassigned_header_text": header_roles["unassigned"],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "parser": "deterministic_rule_based_no_model",
        "document_kind": ("academic_history" if checks["looks_like_transcript"]
                          else "unrecognized_text"),
        "layout": layout,
        "header": header,
        "sections": headings,
        "terms": terms,
        "course_records": course_records,
        "assessment_records": assessment_records,
        "attendance_records": attendance_records,
        "academic_standing": standing,
        "checks": checks,
        "source_characters": len(text),
        "requires_review": True,
        "commit_policy": "proposals_only_user_confirmation_required",
    }


def summarize(parsed: dict[str, Any]) -> dict[str, Any]:
    """Compact, human-readable view of a parsed transcript."""
    standing = parsed["academic_standing"]
    header = parsed["header"]
    summary = {
        "document_title": (header.get("document_title") or {}).get("text"),
        "student_name": (header.get("student_name") or {}).get("text"),
        "school": (header.get("school_name") or {}).get("value"),
        "course_records": len(parsed["course_records"]),
        "assessment_records": len(parsed["assessment_records"]),
        "attendance_records": len(parsed.get("attendance_records") or []),
        "sections": [section["name"] for section in parsed["sections"]],
        # A grade report names its mark column itself (S1, Q2, MARK); the name is
        # printed on the page and is reported instead of assumed.
        "grade_columns": sorted({record["grade_column"] for record in parsed["course_records"]
                                 if record.get("grade_column")}),
        "layout": parsed.get("layout"),
        "document_kind": parsed.get("document_kind", "academic_history"),
        "looks_like_transcript": parsed["checks"].get("looks_like_transcript", True),
        "warnings": ([] if parsed["checks"].get("looks_like_transcript", True) else [
            "no course or assessment rows were found; this text does not look like a "
            "transcript table, so the header values may be guesses"
        ]),
        "terms": sorted({term["value"] for term in parsed.get("terms") or []}),
        # A layout with no credit column has nothing to add up. Reporting zero there
        # would read as "this transcript awards no credit" instead of "no column".
        "computed_total_credits": (None if parsed["checks"].get("credits_printed") is False
                                   else parsed["checks"]["computed_total_credits"]),
        "reported_total_credits": standing["total_credits"]["value"] if standing["total_credits"] else None,
        "gpa": standing["gpa"]["value"] if standing["gpa"] else None,
        "column_interleaving_detected": parsed["checks"]["column_interleaving_detected"],
        "credit_sum_matches_reported_total": parsed["checks"]["credit_sum_matches_reported_total"],
    }
    # The compact view flattens the footer GPA and printed total into bare numbers.
    # When they came from a page read by OCR they were never cell-checked, so the
    # marks travel with the summary instead of leaving those two numbers looking
    # as checked as the four verified fields.
    marked: dict[str, str] = {}
    for field, shown_as in (("gpa", "gpa"),
                            ("total_credits", "reported_total_credits")):
        reason = (standing.get("unverified") or {}).get(field)
        if reason:
            marked[shown_as] = reason
    for field, shown_as in (("document_title", "document_title"),
                            ("student_name", "student_name"),
                            ("school_name", "school")):
        reason = (header.get("unverified") or {}).get(field)
        if reason:
            marked[shown_as] = reason
    if marked:
        summary["unverified"] = marked
    return summary
