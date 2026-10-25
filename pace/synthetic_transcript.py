"""Seeded synthetic academic transcript generator.

Transcripts are rendered as real transcript tables: an identity header, subject-area
sections with DATE / GL / TITLE / MARK / CREDIT rows, state assessment rows with a
performance level, and an academic-standing footer. Several layout families are
produced (aligned table, compact single-space rows, a row-wise OCR of a two-column
page, pipe delimited, markdown table) so the model never sees a single fixed grid.

Every entity span is emitted at render time, so the labels are literal substrings of
the generated text. The generated rows are also kept as gold records, which lets the
deterministic parser be regression tested against generated transcripts.
"""
from __future__ import annotations

import random
from typing import Any

TRANSCRIPT_FAMILIES = (
    "transcript_table",
    "transcript_compact",
    "transcript_rowwise",
    "transcript_piped",
    "transcript_markdown",
)

SUBJECT_AREAS = (
    ("LANGUAGE ARTS", ("Readers Workshop", "Writers Workshop", "Content English - Sci",
                       "Content English - Math", "Content English - SS", "CE - Math",
                       "Holocaust Literature", "Survival Stories", "Revision", "Creative Writing")),
    ("MATH", ("Algebra Prep", "Algebra 1", "Geometry", "Precalculus", "Statistics")),
    ("SCIENCE", ("Physical Sci 1", "Life Sci 1", "Astronomy", "Human Physiology",
                 "Environmental Science", "Chemistry 1")),
    ("SOCIAL STUDIES", ("AK Studies", "US Government", "US History", "Global Issues",
                        "Economics", "Civics")),
    ("PHYSICAL EDUCATION", ("Independent PE", "Health", "PE Conditioning", "Team Sports")),
    ("SCHOOL & COMMUNITY RELATIONS", ("Orientation", "Crew", "Student Leadership",
                                      "Community Service")),
    ("ELECTIVE", ("Elective Intensive", "Intro to Programming", "Digital Design", "Yearbook")),
    ("VOCATIONAL", ("Work Experience", "Intro to Welding", "Culinary Arts")),
    ("ART", ("Studio Art", "Ceramics", "Sculpture")),
    ("MUSIC", ("Concert Band", "Choir", "Guitar")),
    ("BUSINESS", ("Business Essentials", "Marketing", "Accounting")),
)

MARKS = ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "NC", "P", "CR", "W")
GPA_POINTS = {"A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
              "C+": 2.3, "C": 2.0, "C-": 1.7, "D": 1.0}
CREDIT_VALUES = (0.25, 0.5, 0.75, 1.0)
GRADE_LEVELS = ("09", "10", "11", "12")
PERFORMANCE_LABELS = ("Below Proficient", "Not Proficient", "Proficient", "Advanced")
ASSESSMENT_AREAS = (
    ("AK HSGQE READING", "Reading"),
    ("AK HSGQE WRITING", "Writing"),
    ("AK HSGQE MATH", "Math"),
    ("AK HSGQE SCIENCE", "Science"),
)
STREET_NAMES = ("Harbor Mtn Road", "Lincoln Street", "Cedar Avenue", "Willow Way",
                "Summit Drive", "Birch Lane", "Ocean Boulevard", "Hillcrest Road")
CITY_STATE_ZIP = ("Sitka, AK 99835", "Juneau, AK 99801", "Anchorage, AK 99501",
                  "Fairbanks, AK 99701", "Ketchikan, AK 99901", "Valdez, AK 99686")
TRANSCRIPT_FIRST_NAMES = ("Arthur", "Maya", "Noah", "Iris", "Caleb", "Nadia", "Eli",
                          "Rosa", "Jonah", "Freya", "Silas", "Wren")
TRANSCRIPT_LAST_NAMES = ("Lepid", "Okafor", "Vasquez", "Nakamura", "Bergstrom",
                         "Delacroix", "Halvorsen", "Marchetti", "Osei", "Ruiz")


def _segments(*pairs: tuple[str, str | None]) -> list[tuple[str, str | None]]:
    return list(pairs)


def _pad(width: int) -> tuple[str, str | None]:
    return (" " * max(width, 1), None)


def _render(lines: list[list[tuple[str, str | None]]]) -> tuple[str, list[dict[str, Any]]]:
    """Render segment lines into text plus literal character-span entities."""
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    offset = 0
    for line in lines:
        for text, entity_type in line:
            if not text:
                continue
            start = offset
            parts.append(text)
            offset += len(text)
            if entity_type:
                entities.append({"type": entity_type, "start": start, "end": offset, "text": text})
        parts.append("\n")
        offset += 1
    return "".join(parts), entities


def _weighted_mark(rng: random.Random) -> str:
    """Pick a mark with a realistic grade distribution."""
    table = (
        ("A", 22), ("A-", 14), ("B+", 16), ("B", 20), ("B-", 9),
        ("C+", 4), ("C", 4), ("C-", 2), ("D", 1), ("NC", 3), ("P", 3), ("CR", 1), ("W", 1),
    )
    population = [mark for mark, weight in table for _ in range(weight)]
    return rng.choice(population)


def _transcript_rows(rng: random.Random) -> dict[str, Any]:
    """Build the canonical academic history: subject areas, rows, assessments, totals."""
    areas = rng.sample(SUBJECT_AREAS, k=rng.randint(3, 6))
    # Some transcripts report no GPA at all: every mark is ungraded, and the
    # academic-standing footer leaves the GPA cell empty.
    no_gpa = rng.random() < 0.3
    courses: list[dict[str, Any]] = []
    for area, titles in areas:
        chosen = rng.sample(titles, k=min(len(titles), rng.randint(1, 5)))
        for title in chosen:
            courses.append({
                "area": area,
                "date": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(3, 8):02d}",
                "grade_level": rng.choice(GRADE_LEVELS),
                "title": title,
                "mark": rng.choice(("NC", "P", "CR", "W")) if no_gpa else _weighted_mark(rng),
                "credit": rng.choice(CREDIT_VALUES),
            })
    assessments: list[dict[str, Any]] = []
    for area, title in rng.sample(ASSESSMENT_AREAS, k=rng.randint(1, 3)):
        level = rng.randint(1, 4)
        assessments.append({
            "area": area,
            "date": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(3, 8):02d}",
            "grade_level": rng.choice(GRADE_LEVELS),
            "title": title,
            "score": rng.randint(180, 400),
            "level": level,
            "performance_label": PERFORMANCE_LABELS[level - 1],
        })
    total_credits = round(sum(course["credit"] for course in courses), 3)
    graded = [course for course in courses if course["mark"] in GPA_POINTS]
    graded_credits = sum(course["credit"] for course in graded)
    gpa = None
    if graded_credits:
        gpa = round(
            sum(GPA_POINTS[course["mark"]] * course["credit"] for course in graded) / graded_credits,
            2,
        )
    return {"courses": courses, "assessments": assessments, "total_credits": total_credits, "gpa": gpa}


def _identity_lines(rng: random.Random, school: str, student: str, street: str, city: str) -> list[list[tuple[str, str | None]]]:
    """Render the transcript identity header."""
    phone = f"({rng.randint(200, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    dob = f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(80, 99):02d}"
    sex = rng.choice(("Male", "Female"))
    name_width = max(len(student), len("DOB  1/1/00"), len("SSN")) + 4
    rows: list[list[tuple[str, str | None]]] = [
        _segments(("ACADEMIC HISTORY", None)),
        _segments(("", None)),
        _segments((school, "SCHOOL")),
        _segments((street, None)),
        _segments((city, None)),
        _segments((f"Phone: {phone}", None)),
        _segments(("", None)),
        _segments((student, "STUDENT"), _pad(name_width), (f"Sex  {sex}", None)),
        _segments((street, None), _pad(name_width), (f"DOB  {dob}", None)),
        _segments((city, None), _pad(name_width), ("SSN", None)),
        _segments(("Parent/ Guardian", None), _pad(name_width), ("Entry Date", None)),
        _segments(("", None), _pad(name_width), ("Exit Date", None)),
        _segments(("", None), _pad(name_width), ("Graduation", None)),
        _segments(("", None)),
    ]
    return rows


def _course_row_lines(rows: list[dict[str, Any]], widths: dict[str, int], family: str) -> list[list[tuple[str, str | None]]]:
    """Render one course row per record for the given layout family."""
    lines: list[list[tuple[str, str | None]]] = []
    for row in rows:
        credit = f"{row['credit']:.3f}"
        if family == "transcript_piped":
            lines.append(_segments(
                ("| ", None), (row["date"], "DATE"), (" | ", None), (row["grade_level"], "GRADE_LEVEL"),
                (" | ", None), (row["title"], "COURSE"), (" | ", None), (row["mark"], "MARK"),
                (" | ", None), (credit, "CREDIT"), (" |", None),
            ))
        elif family == "transcript_markdown":
            lines.append(_segments(
                ("| ", None), (row["date"], "DATE"), (" | ", None), (row["grade_level"], "GRADE_LEVEL"),
                (" | ", None), (row["title"], "COURSE"), (" | ", None), (row["mark"], "MARK"),
                (" | ", None), (credit, "CREDIT"), (" |", None),
            ))
        elif family == "transcript_compact":
            lines.append(_segments(
                (row["date"], "DATE"), (" ", None), (row["grade_level"], "GRADE_LEVEL"), (" ", None),
                (row["title"], "COURSE"), (" ", None), (row["mark"], "MARK"), (" ", None),
                (credit, "CREDIT"),
            ))
        else:
            lines.append(_segments(
                (row["date"], "DATE"), _pad(widths["date"]), (row["grade_level"], "GRADE_LEVEL"),
                _pad(widths["gl"]), (row["title"], "COURSE"), _pad(widths["title"]),
                (row["mark"], "MARK"), _pad(widths["mark"]), (credit, "CREDIT"),
            ))
    return lines


def _assessment_lines(rows: list[dict[str, Any]], widths: dict[str, int], family: str) -> list[list[tuple[str, str | None]]]:
    """Render assessment rows: a score row plus an indented performance level."""
    lines: list[list[tuple[str, str | None]]] = []
    for row in rows:
        score = str(row["score"])
        performance = f"{row['level']} {row['performance_label']}"
        if family in ("transcript_piped", "transcript_markdown"):
            lines.append(_segments(
                ("| ", None), (row["date"], "DATE"), (" | ", None), (row["grade_level"], "GRADE_LEVEL"),
                (" | ", None), (row["title"], "COURSE"), (" | ", None), (score, "SCORE"), (" |", None),
            ))
            lines.append(_segments(("| Performance Level: ", None), (performance, "PERFORMANCE_LEVEL"), (" |", None)))
        elif family == "transcript_compact":
            lines.append(_segments(
                (row["date"], "DATE"), (" ", None), (row["grade_level"], "GRADE_LEVEL"), (" ", None),
                (row["title"], "COURSE"), (" ", None), (score, "SCORE"),
            ))
            lines.append(_segments(("Performance Level: ", None), (performance, "PERFORMANCE_LEVEL")))
        else:
            lines.append(_segments(
                (row["date"], "DATE"), _pad(widths["date"]), (row["grade_level"], "GRADE_LEVEL"),
                _pad(widths["gl"]), (row["title"], "COURSE"), _pad(widths["title"]),
                (score, "SCORE"),
            ))
            lines.append(_segments(("", None), _pad(widths["date"] + widths["gl"]),
                                    ("Performance Level: ", None), (performance, "PERFORMANCE_LEVEL")))
    return lines


def _column_widths(rows: list[dict[str, Any]]) -> dict[str, int]:
    date = max(len(row["date"]) for row in rows) + 2
    gl = max(len(row["grade_level"]) for row in rows) + 2
    title = max(len(row["title"]) for row in rows) + 4
    mark = 6
    return {"date": date, "gl": gl, "title": title, "mark": mark}


def _body_lines(world: dict[str, Any], family: str) -> tuple[list[list[tuple[str, str | None]]], list[dict[str, Any]]]:
    """Render the academic-history body for one layout family."""
    courses = world["courses"]
    assessments = world["assessments"]
    widths = _column_widths(courses + assessments)
    lines: list[list[tuple[str, str | None]]] = []
    gold_records: list[dict[str, Any]] = []

    if family == "transcript_piped":
        lines.append(_segments(("| DATE | GL | TITLE | MARK | CREDIT |", None)))
    elif family == "transcript_markdown":
        lines.append(_segments(("| DATE | GL | TITLE | MARK | CREDIT |", None)))
        lines.append(_segments(("| --- | --- | --- | --- | --- |", None)))
    else:
        header = f"{'DATE':<{widths['date']}}{'GL':<{widths['gl']}}{'TITLE':<{widths['title']}}{'MARK':<{widths['mark']}}CREDIT"
        lines.append(_segments((header, None)))
    lines.append(_segments(("", None)))

    areas: list[str] = []
    for course in courses:
        if course["area"] not in areas:
            areas.append(course["area"])
    for assessment in assessments:
        if assessment["area"] not in areas:
            areas.append(assessment["area"])

    for area in areas:
        lines.append(_segments((area, "SUBJECT_AREA")))
        area_courses = [course for course in courses if course["area"] == area]
        area_assessments = [assessment for assessment in assessments if assessment["area"] == area]
        for row in area_courses:
            gold_records.append({"kind": "course", "section": area, **{k: row[k] for k in ("date", "grade_level", "title", "mark", "credit")}})
        for row in area_assessments:
            gold_records.append({
                "kind": "assessment", "section": area,
                **{k: row[k] for k in ("date", "grade_level", "title", "score")},
                "performance_level": row["level"], "performance_label": row["performance_label"],
            })
        if area_courses:
            lines.extend(_course_row_lines(area_courses, widths, family))
        if area_assessments:
            lines.extend(_assessment_lines(area_assessments, widths, family))
        lines.append(_segments(("", None)))

    return lines, gold_records


def _merge_two_columns(
    left: list[list[tuple[str, str | None]]],
    right: list[list[tuple[str, str | None]]],
) -> list[list[tuple[str, str | None]]]:
    """Merge two page columns the way a row-wise OCR of a two-column page does."""
    left_width = max((sum(len(segment[0]) for segment in line) for line in left), default=0)
    merged: list[list[tuple[str, str | None]]] = []
    for index in range(max(len(left), len(right))):
        left_line = left[index] if index < len(left) else []
        right_line = right[index] if index < len(right) else []
        used = sum(len(segment[0]) for segment in left_line)
        gap = max(3, left_width - used + 3)
        merged.append(left_line + [(" " * gap, None)] + right_line)
    return merged


def render_transcript_document(
    world: dict[str, Any],
    seed: int,
    family: str,
    school: str,
    school_id: str,
    digest,
) -> dict[str, Any]:
    """Render one synthetic transcript document with literal entity spans."""
    if family not in TRANSCRIPT_FAMILIES:
        raise ValueError(f"Unsupported transcript family: {family}")
    rng = random.Random(seed ^ 0x7A11C)
    student = f"{rng.choice(TRANSCRIPT_LAST_NAMES)}, {rng.choice(TRANSCRIPT_FIRST_NAMES)}"
    street = f"{rng.randint(10, 9999)} {rng.choice(STREET_NAMES)}"
    city = rng.choice(CITY_STATE_ZIP)

    content = _transcript_rows(rng)
    transcript_world = {**world, **content}

    header = _identity_lines(rng, school, student, street, city)
    body, gold_records = _body_lines(transcript_world, family)

    if family == "transcript_rowwise":
        split = max(1, len(body) // 2)
        # Each page column carries its own table header, so the right column starts
        # with the header line again.
        header_line = body[0]
        left, right = body[:split], body[split:]
        right = [header_line] + right
        body = _merge_two_columns(left, right)

    lines = list(header) + body

    # Academic standing footer. The credit value is printed under "Total" or under
    # "Credits"; when the student has no GPA the GPA cell is left empty, which is a
    # layout the model must also learn.
    as_of = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(3, 8):02d}"
    lines.append(_segments(("ACADEMIC STANDING", None)))
    lines.append(_segments(("As of:          Total Credits       GPA", None)))
    credit_text = f"{content['total_credits']:.3f}".rstrip("0").rstrip(".")
    credit_column = rng.choice((16, 22))
    segments: list[tuple[str, str | None]] = [
        (as_of, "DATE"),
        (" " * max(1, credit_column - len(as_of)), None),
        (credit_text, "CREDIT"),
    ]
    if content["gpa"] is not None:
        gpa_text = f"{content['gpa']:.2f}"
        segments.append((" " * max(1, credit_column + 12 - (credit_column + len(credit_text))), None))
        segments.append((gpa_text, "GPA"))
    lines.append(_segments(*segments))

    text, entities = _render(lines)
    gold = {
        "student_name": student,
        "school_name": school,
        "course_records": [record for record in gold_records if record["kind"] == "course"],
        "assessment_records": [record for record in gold_records if record["kind"] == "assessment"],
        "total_credits": content["total_credits"],
        "gpa": content["gpa"],
    }
    return {
        "document_id": f"synthetic-transcript-{seed}-{digest(text)[:12]}",
        "document_type": "transcript",
        "template_family": family,
        "seed": seed,
        "world_school_id": school_id,
        "text": text,
        "gold": gold,
        "entities": entities,
        "blocks": [
            {"text": "ACADEMIC HISTORY", "section_type": "unknown", "evidence_sha256": digest("ACADEMIC HISTORY")},
        ],
    }
