"""Real academic field-type corpus built from natural documents and catalogs.

Every example is a piece of real text with a real field label. Three independent
real sources feed one harmonised taxonomy so that the classifier has to learn the
field *semantics* instead of one institution's column names:

* ``USC/USC-Course-Catalog`` -- natural CSV cell values for 13 catalog columns.
* ``anordkvist/gu-course-syllabus`` -- natural syllabus field passages for 11
  European syllabus fields.
* ``Dev-the-dev91/syllabus-cleaned`` -- 90 real university syllabi from 14
  institutions with labelled section headings, entity spans and structured
  document facts.

``iristun/CU-DocBench`` supplies real non-field text for the ``other`` class, so
the model can also say "this is not an academic field value" instead of always
committing to the closest field.

Nothing here is generated or paraphrased. Long passages are cut into windows so
that short cell-like text and long prose-like text are both represented.
"""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .data import document_hash

# ---------------------------------------------------------------------------
# Harmonised field taxonomy
# ---------------------------------------------------------------------------

FIELD_TYPES: list[str] = [
    "additional_information",
    "academic_integrity",
    "assessment",
    "assignments",
    "attendance_policy",
    "capacity",
    "class_meeting_room",
    "class_section",
    "class_type",
    "confirmation",
    "contact_information",
    "course_code",
    "course_content",
    "course_description",
    "course_evaluation",
    "course_level",
    "course_modality",
    "course_policies",
    "course_schedule",
    "course_title",
    "credits",
    "disability_services",
    "drop_add_deadline",
    "due_date",
    "entry_requirements",
    "exam_schedule",
    "grades",
    "grading_policy",
    "instructor",
    "language_of_instruction",
    "learning_outcomes",
    "materials",
    "meeting_days",
    "meeting_time",
    "office_hours",
    "other",
    "prerequisites",
    "program",
    "restrictions",
    "session_dates",
    "teaching_form",
    "term",
    "weekly_readings",
]
FIELD_TO_ID = {name: i for i, name in enumerate(FIELD_TYPES)}

#: USC catalog column -> harmonised field type.
USC_COLUMN_MAP: dict[str, str] = {
    "Class Name": "course_title",
    "Class ID": "course_code",
    "Program Code": "program",
    "Catalogue": "course_description",
    "Units": "credits",
    "Prerequisites": "prerequisites",
    "Restrictions": "restrictions",
    "Class Section": "class_section",
    "Time": "meeting_time",
    "Days": "meeting_days",
    "Class Type": "class_type",
    "Instructor": "instructor",
}

#: GU syllabus column -> harmonised field type.
GU_COLUMN_MAP: dict[str, str] = {
    "course_code": "course_code",
    "Confirmation": "confirmation",
    "Position in the educational system": "course_level",
    "Entry requirements": "entry_requirements",
    "Learning outcomes": "learning_outcomes",
    "Course content": "course_content",
    "Form of teaching": "teaching_form",
    "Assessment": "assessment",
    "Grades": "grades",
    "Course evaluation": "course_evaluation",
    "Additional information": "additional_information",
}

#: Real syllabus section heading -> harmonised field type. The vocabulary covers
#: the section names that actually appear in university syllabi, so the heading
#: text in a real document is the supervision signal.
SYLLABUS_SECTION_MAP: dict[str, str] = {
    # identity
    "course title": "course_title",
    "course name": "course_title",
    "title": "course_title",
    "course code": "course_code",
    "course number": "course_code",
    "catalog number": "course_code",
    "class id": "course_code",
    "course id": "course_code",
    "class section": "class_section",
    "section number": "class_section",
    "class type": "class_type",
    "course type": "class_type",
    "type of course": "class_type",
    "instructional mode": "class_type",
    "modality": "class_type",
    "program": "program",
    "program code": "program",
    "degree program": "program",
    "department": "program",
    "school": "program",
    "faculty": "program",
    "course level": "course_level",
    "position in the educational system": "course_level",
    "educational level": "course_level",
    "level": "course_level",
    # instructor & contact
    "instructor": "instructor",
    "instructors": "instructor",
    "course instructor": "instructor",
    "teaching staff": "instructor",
    "professor": "instructor",
    "lecturer": "instructor",
    "contact information": "contact_information",
    "contact info": "contact_information",
    "contact details": "contact_information",
    "instructor contact information": "contact_information",
    "office hours": "office_hours",
    "office hour": "office_hours",
    "faculty office hours": "office_hours",
    "instructor office hours": "office_hours",
    "office hours information": "office_hours",
    "consultation hours": "office_hours",
    "instructor availability": "office_hours",
    "email": "contact_information",
    "phone": "contact_information",
    # enrollment & structure
    "credits": "credits",
    "credit hours": "credits",
    "units": "credits",
    "course credits": "credits",
    "number of credits": "credits",
    "contact hours": "credits",
    "hours per week": "credits",
    "prerequisites": "prerequisites",
    "prerequisite": "prerequisites",
    "pre-requisites": "prerequisites",
    "pre-requisite": "prerequisites",
    "prior knowledge": "prerequisites",
    "recommended preparation": "prerequisites",
    "corequisites": "prerequisites",
    "co-requisites": "prerequisites",
    "entry requirements": "entry_requirements",
    "admission requirements": "entry_requirements",
    "requirements for admission": "entry_requirements",
    "general entrance requirements": "entry_requirements",
    "eligibility": "entry_requirements",
    "restrictions": "restrictions",
    "registration restrictions": "restrictions",
    "course restrictions": "restrictions",
    "limited to": "restrictions",
    "class standing": "restrictions",
    "confirmation": "confirmation",
    "course confirmation": "confirmation",
    "confirmed by": "confirmation",
    "last revised": "confirmation",
    # content & learning
    "course description": "course_description",
    "catalog description": "course_description",
    "description": "course_description",
    "about this course": "course_description",
    "course overview": "course_description",
    "overview": "course_description",
    "course content": "course_content",
    "course topics": "course_content",
    "topics covered": "course_content",
    "subject matter": "course_content",
    "course outline": "course_content",
    "outline": "course_content",
    "topics": "course_content",
    "syllabus topics": "course_content",
    "learning objectives": "learning_outcomes",
    "learning outcomes": "learning_outcomes",
    "course objectives": "learning_outcomes",
    "student learning outcomes": "learning_outcomes",
    "objectives": "learning_outcomes",
    "course learning outcomes": "learning_outcomes",
    "outcomes": "learning_outcomes",
    "course goals": "learning_outcomes",
    "goals": "learning_outcomes",
    "form of teaching": "teaching_form",
    "teaching form": "teaching_form",
    "method of teaching": "teaching_form",
    "teaching methods": "teaching_form",
    "instructional method": "teaching_form",
    "modes of instruction": "teaching_form",
    "format of instruction": "teaching_form",
    "delivery": "teaching_form",
    "language of instruction": "language_of_instruction",
    "language of teaching": "language_of_instruction",
    "teaching language": "language_of_instruction",
    # materials & assessment
    "textbook": "materials",
    "textbooks": "materials",
    "required textbook": "materials",
    "required textbooks": "materials",
    "required text": "materials",
    "required texts": "materials",
    "required reading": "materials",
    "required readings": "materials",
    "required books": "materials",
    "books required": "materials",
    "course materials": "materials",
    "materials required": "materials",
    "text book": "materials",
    "textbook information": "materials",
    "required textbook information": "materials",
    "books and materials": "materials",
    "required course materials": "materials",
    "materials and supplies": "materials",
    "software requirements": "materials",
    "weekly readings": "weekly_readings",
    "weekly reading": "weekly_readings",
    "reading schedule": "weekly_readings",
    "assigned readings": "weekly_readings",
    "reading list": "weekly_readings",
    "weekly reading schedule": "weekly_readings",
    "assignments": "assignments",
    "assignment": "assignments",
    "course assignments": "assignments",
    "homework": "assignments",
    "projects": "assignments",
    "term project": "assignments",
    "course project": "assignments",
    "assessment": "assessment",
    "methods of assessment": "assessment",
    "assessment methods": "assessment",
    "evaluation methods": "assessment",
    "grading": "grading_policy",
    "grading policy": "grading_policy",
    "grade policy": "grading_policy",
    "grading criteria": "grading_policy",
    "how your grade is calculated": "grading_policy",
    "assessment criteria": "grading_policy",
    "grades": "grades",
    "grade scale": "grades",
    "grading scale": "grades",
    "course evaluation": "course_evaluation",
    "evaluation of the course": "course_evaluation",
    "student evaluation of the course": "course_evaluation",
    "course feedback": "course_evaluation",
    # schedule
    "course schedule": "course_schedule",
    "class schedule": "course_schedule",
    "weekly schedule": "course_schedule",
    "schedule of classes": "course_schedule",
    "lecture schedule": "course_schedule",
    "course calendar": "course_schedule",
    "schedule": "course_schedule",
    "calendar": "course_schedule",
    "meeting time": "meeting_time",
    "meeting times": "meeting_time",
    "class times": "meeting_time",
    "class meeting times": "meeting_time",
    "class time": "meeting_time",
    "meeting days": "meeting_days",
    "meeting day": "meeting_days",
    "class days": "meeting_days",
    "class meeting days": "meeting_days",
    "days": "meeting_days",
    "time and location": "meeting_time",
    "times and locations": "meeting_time",
    "class location": "meeting_time",
    "class times and locations": "meeting_time",
    "when and where": "meeting_time",
    "meeting place": "meeting_time",
    "exam schedule": "exam_schedule",
    "exam dates": "exam_schedule",
    "examination schedule": "exam_schedule",
    "test schedule": "exam_schedule",
    "test dates": "exam_schedule",
    "midterm schedule": "exam_schedule",
    "final exam schedule": "exam_schedule",
    "exam dates and times": "exam_schedule",
    "grading schedule": "exam_schedule",
    "exams": "exam_schedule",
    "important dates": "session_dates",
    "key dates": "session_dates",
    "academic calendar": "session_dates",
    "term dates": "session_dates",
    "schedule of important dates": "session_dates",
    "course start date": "session_dates",
    "last day of class": "session_dates",
    "dates": "session_dates",
    "term": "term",
    "semester": "term",
    "academic term": "term",
    "term and date": "term",
    "add drop deadline": "drop_add_deadline",
    "add/drop": "drop_add_deadline",
    "add/drop deadline": "drop_add_deadline",
    "add/drop period": "drop_add_deadline",
    "add/drop information": "drop_add_deadline",
    "withdrawal deadline": "drop_add_deadline",
    "drop deadline": "drop_add_deadline",
    "course withdrawal": "drop_add_deadline",
    "deadlines": "drop_add_deadline",
    # policies
    "attendance": "attendance_policy",
    "attendance policy": "attendance_policy",
    "attendance requirements": "attendance_policy",
    "attendance and participation": "attendance_policy",
    "class attendance": "attendance_policy",
    "attendance expectations": "attendance_policy",
    "participation": "attendance_policy",
    "participation policy": "attendance_policy",
    "class participation": "attendance_policy",
    "academic integrity": "academic_integrity",
    "academic integrity policy": "academic_integrity",
    "academic honesty": "academic_integrity",
    "integrity policy": "academic_integrity",
    "honor code": "academic_integrity",
    "honour code": "academic_integrity",
    "plagiarism": "academic_integrity",
    "academic misconduct": "academic_integrity",
    "student conduct": "academic_integrity",
    "code of conduct": "academic_integrity",
    "course policies": "course_policies",
    "class policies": "course_policies",
    "course policy": "course_policies",
    "general policies": "course_policies",
    "course rules": "course_policies",
    "class rules": "course_policies",
    "policies and procedures": "course_policies",
    "policies": "course_policies",
    "disability services": "disability_services",
    "services for students with disabilities": "disability_services",
    "disability accommodations": "disability_services",
    "students with disabilities": "disability_services",
    "accessibility services": "disability_services",
    "accommodations": "disability_services",
    "special services": "disability_services",
    "accessibility": "disability_services",
    "additional information": "additional_information",
    "other information": "additional_information",
    "extra information": "additional_information",
    "more information": "additional_information",
    "important information": "additional_information",
    "notes": "additional_information",
    "note": "additional_information",
    "miscellaneous": "additional_information",
}

#: Headings that only mark a boundary; their content stays unlabelled.
_BOUNDARY_ONLY_HEADINGS = {
    "introduction", "purpose", "background", "course introduction", "course information",
    "course information and policies", "important information", "notice", "notes to students",
    "change of schedule", "late work", "late policy", "make up work", "makeup work",
    "extra credit", "grading and assessment", "grading and evaluation", "evaluation",
    "assessment and grading", "assignments and grading", "requirements", "course requirements",
    "office location", "location", "building", "room", "classroom", "meetings", "meeting",
    "schedule of topics", "weekly topic", "schedule of lectures", "lecture topics",
    "course format", "format", "credit", "grading scheme", "marking", "marking scheme",
    "exam", "final exam", "midterm", "midterm exam", "final examination", "quiz",
    "project", "paper", "term paper", "research project", "presentation",
    "reading", "readings", "supplementary reading", "supplementary readings",
    "resources", "course resources", "software", "technology requirements",
    "communication", "communications", "announcements", "email policy", "email usage",
    "cell phone", "cellphone policy", "recording policy", "course recording",
    "student participation", "student responsibilities", "responsibilities",
    "learning goals", "expected learning outcomes", "course summary", "summary",
    "the course", "about", "at a glance", "quick facts", "course highlights",
}

_HEADING_NUMBER = re.compile(
    r"^\s*(?:[0-9]{1,2}[.)]|[ivxlcdm]{1,7}[.)]|[a-z][.)]|[•\-\*▪◦]+)\s+", re.I)
_HEADING_TRAILER = re.compile(r"[:\s\-–—]+$")
_HEADING_MAX_CHARS = 70

MISSING_VALUES = {"", "not available", "n/a", "none", "null", "nan", "-", "tba"}

_WINDOW_CHARS = 420
_MAX_WINDOWS_PER_VALUE = 3
_MAX_SECTION_CHARS = 2400

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

_HTML_TAG = re.compile(r"<[^>]{0,4000}>")
_HTML_ENTITY = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,10});")
_LATIN_RATIO = 0.5


def clean_html(text: str) -> tuple[str, list[int]]:
    """Strip HTML tags and entities while keeping an original->clean offset map.

    The ingested syllabi are raw HTML, but the section entity offsets are
    relative to that raw text, so the offsets have to survive the cleanup.
    """
    out: list[str] = []
    mapping: list[int] = []
    position = 0
    for match in _HTML_TAG.finditer(text):
        for char in text[position:match.start()]:
            out.append(char)
            mapping.append(len(out) - 1)
        out.append(" ")
        mapping.extend([len(out) - 1] * (match.end() - match.start()))
        position = match.end()
    for char in text[position:]:
        out.append(char)
        mapping.append(len(out) - 1)
    tagged = "".join(out)

    final: list[str] = []
    final_map: list[int] = []
    position = 0
    for match in _HTML_ENTITY.finditer(tagged):
        for char in tagged[position:match.start()]:
            final.append(char)
            final_map.append(len(final) - 1)
        replacement = html.unescape(match.group(0))
        if replacement == match.group(0):
            replacement = " "
        replacement = replacement if replacement else " "
        final.append(replacement[0])
        for _ in range(len(replacement) - 1):
            final.append("")
        index = len(final) - 1
        final_map.extend([index] * len(match.group(0)))
        position = match.end()
    for char in tagged[position:]:
        final.append(char)
        final_map.append(len(final) - 1)
    return "".join(final), final_map


def _mostly_latin(text: str) -> bool:
    """True when the text is mostly Latin-script, so the corpus stays one language."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if c.isascii())
    return latin / len(letters) >= _LATIN_RATIO


def _normalize_heading(line: str) -> str:
    """Strip numbering, bullets and trailing punctuation from a heading line."""
    text = line.strip()
    text = _HEADING_NUMBER.sub("", text)
    text = _HEADING_TRAILER.sub("", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def detect_document_sections(text: str) -> list[dict[str, Any]]:
    """Find real section headings in a cleaned syllabus by line structure.

    A heading is a line that, after removing numbering and trailing punctuation,
    is exactly a known syllabus section name. Sections are only trusted when the
    whole line is the heading, which removes the false positives that come from
    matching a phrase inside a sentence. A heading may also carry an inline
    value (``Office Hours: Mon 2-4pm``), which is kept as the section body.
    """
    sections: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = offset
        offset += len(line)
        if not stripped or len(stripped) > _HEADING_MAX_CHARS:
            continue
        normalized = _normalize_heading(stripped)
        if not normalized:
            continue
        label = SYLLABUS_SECTION_MAP.get(normalized)
        inline = ""
        if label is None:
            # "Office Hours: Mon 2-4pm" -- heading and value on one line.
            head, sep, tail = stripped.partition(":")
            if sep and len(head) <= _HEADING_MAX_CHARS:
                head_normalized = _normalize_heading(head)
                label = SYLLABUS_SECTION_MAP.get(head_normalized)
                if label is not None:
                    inline = tail.strip()
                    normalized = head_normalized
        if label is None and normalized not in _BOUNDARY_ONLY_HEADINGS:
            continue
        sections.append({
            "start": start,
            "end": start + len(line),
            "text": stripped,
            "label": label,
            "inline": inline,
        })
    return sections


def normalize_value(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_missing(text: str) -> bool:
    return normalize_value(text) in MISSING_VALUES


def window_text(text: str, window_chars: int = _WINDOW_CHARS,
                max_windows: int = _MAX_WINDOWS_PER_VALUE) -> list[str]:
    """Cut one natural value into a few sentence-aligned windows."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > window_chars:
            windows.append(current)
            current = sentence
            if len(windows) == max_windows:
                return windows
        else:
            current = f"{current} {sentence}".strip()
    if current and len(windows) < max_windows:
        windows.append(current)
    return windows[:max_windows]


def _record(text: str, label: str, group_id: str, source: str, source_field: str,
            provenance: str, source_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "text": text,
        "label": label,
        "group_id": group_id,
        "source": source,
        "source_field": source_field,
        "provenance": provenance,
        "source_id": source_id,
        "text_sha256": document_hash(text),
    }
    if meta:
        record["meta"] = meta
    return record


# ---------------------------------------------------------------------------
# Source 1: USC course catalog cells
# ---------------------------------------------------------------------------

def load_usc_field_examples(path: str | Path = "data/raw/usc_course_data.csv") -> list[dict[str, Any]]:
    """Natural catalog cell values mapped to harmonised field types."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            class_id = (row.get("Class ID") or "").strip()
            match = re.match(r"([A-Za-z]+)\s*[- ]?\s*\d", class_id)
            subject = (match.group(1).upper() if match
                       else (row.get("Program Code") or "UNKNOWN").strip().upper())
            source_id = f"usc-2024-spring:{index}:{class_id}"
            for column, label in USC_COLUMN_MAP.items():
                text = (row.get(column) or "").strip()
                if _is_missing(text):
                    continue
                if column == "Class Name":
                    text = re.sub(r"^\s*[^:]{2,16}:\s*", "", text).strip()
                if column == "Prerequisites":
                    text = re.sub(r"^\s*Prerequisite:\s*", "", text, flags=re.I).strip()
                if column == "Restrictions":
                    text = re.sub(r"^\s*Restriction:\s*", "", text, flags=re.I).strip()
                if not text:
                    continue
                rows.append(_record(text, label, f"usc-subject:{subject}",
                                    "USC/USC-Course-Catalog", column,
                                    "2024 Spring term dataset; original CSV cell value", source_id))
    return rows


# ---------------------------------------------------------------------------
# Source 2: GU syllabus field passages
# ---------------------------------------------------------------------------

def _gu_course_key(course_code: str, confirmation: str) -> str:
    """A stable course identity so one course never spans two splits."""
    name = re.sub(r"[^A-Za-z0-9 ]+", " ", confirmation or "")[:60]
    return f"{course_code}|{normalize_value(name)}"


#: The GU corpus repeats a machine-made prefix and the field name in every value.
_GU_PREFIX = re.compile(r"^\s*passage:\s*course code:\s*[A-Za-z0-9]+\s*,\s*", re.I)
_GU_FIELD_PHRASE = {
    "confirmation": "confirmation",
    "position in the educational system": "position in the educational system",
    "entry requirements": "entry requirements",
    "learning outcomes": "learning outcomes",
    "course content": "course content",
    "form of teaching": "form of teaching",
    "assessment": "assessment",
    "grades": "grades",
    "course evaluation": "course evaluation",
    "additional information": "additional information",
}


def _strip_gu_prefix(value: str) -> str:
    return _GU_PREFIX.sub("", value).strip()


def _strip_gu_field_phrase(value: str, column: str) -> str:
    phrase = _GU_FIELD_PHRASE.get(column.casefold())
    if not phrase:
        return value
    return re.sub(rf"^\s*{re.escape(phrase)}\b[\s:,]*", "", value, flags=re.I).strip()


def load_gu_field_examples(path: str | Path = "data/raw/gu_course_syllabus.parquet"
                           ) -> list[dict[str, Any]]:
    """Natural syllabus field passages mapped to harmonised field types.

    Every GU value is stored as ``passage: course code: <code>, <field name> <body>``.
    The machine prefix is removed. Each value yields both a *body* example with
    the field-name phrase removed -- so the model cannot win by string-matching a
    heading -- and an *as written* example that keeps it, which matches how a
    real document reads.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for index, record in frame.iterrows():
        course_code = str(record.get("course_code") or "").strip()
        confirmation = str(record.get("Confirmation") or "")
        group = f"gu-course:{_gu_course_key(course_code, confirmation)}"
        source_id = f"gu:{course_code}:{index}"
        for column, label in GU_COLUMN_MAP.items():
            if column not in frame.columns:
                continue
            value = record.get(column)
            if value is None or (isinstance(value, float) and value != value):
                continue
            raw = str(value).strip()
            if not raw or _is_missing(raw):
                continue
            written = _strip_gu_prefix(raw)
            body = _strip_gu_field_phrase(written, column)
            for window in window_text(body):
                rows.append(_record(window, label, group, "anordkvist/gu-course-syllabus",
                                    f"{column} (body)", "GU syllabus field body without its heading cue",
                                    source_id))
            if written and normalize_value(written) != normalize_value(body):
                for window in window_text(written, max_windows=1):
                    rows.append(_record(window, label, group, "anordkvist/gu-course-syllabus",
                                        f"{column} (as written)", "GU syllabus field passage as written",
                                        source_id))
    return rows


# ---------------------------------------------------------------------------
# Source 3: real multi-institution syllabi
# ---------------------------------------------------------------------------

def _syllabus_paths(root: str | Path) -> tuple[Path, Path, Path]:
    base = Path(root) / "syllabus-cleaned"
    return (base / "ingested" / "syllabi_with_text.jsonl",
            base / "labeled" / "syllabus_entities.jsonl",
            base / "labeled" / "golden_facts.jsonl")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _gen() -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    return _gen()


def load_syllabus_field_examples(root: str | Path = "data/raw/hf") -> list[dict[str, Any]]:
    """Real syllabus section bodies, headings and structured facts.

    Section boundaries come from line-anchored headings in the document itself,
    so a field's body never bleeds into the next section. The raw HTML is
    stripped first. Nothing is rewritten: the heading and the text under it are
    exactly what the institution published.
    """
    ingested_path, entities_path, facts_path = _syllabus_paths(root)
    ingested = {row["id"]: row for row in _read_jsonl(ingested_path)}
    rows: list[dict[str, Any]] = []

    for doc in _read_jsonl(entities_path):
        meta = ingested.get(doc.get("id"))
        if not meta:
            continue
        raw_text = meta.get("text") or ""
        text, _offsets = clean_html(raw_text)
        school = meta.get("school") or "unknown"
        document_id = doc.get("id")
        group = f"syllabus-doc:{document_id}"
        source = "Dev-the-dev91/syllabus-cleaned"
        source_id = f"syllabus:{document_id}"
        document_meta = {"school": school, "document_id": document_id}

        sections = detect_document_sections(text)
        for index, section in enumerate(sections):
            label = section["label"]
            if label is None:
                continue
            rows.append(_record(section["text"], label, group, source, "section heading",
                                "real section heading in a real syllabus", source_id,
                                meta=document_meta))
            body_start = section["end"]
            body_end = sections[index + 1]["start"] if index + 1 < len(sections) else len(text)
            body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()[:_MAX_SECTION_CHARS]
            inline = section.get("inline") or ""
            if inline and len(inline) >= 4 and any(c.isalpha() for c in inline):
                body = f"{inline} {body}".strip()
            if len(body) < 40 or not _mostly_latin(body):
                continue
            for window in window_text(body, window_chars=_WINDOW_CHARS, max_windows=2):
                rows.append(_record(window, label, group, source, "section body",
                                    "text under a real section heading in a real syllabus",
                                    source_id, meta=document_meta))

    for entry in _read_jsonl(facts_path):
        facts = entry.get("facts") or {}
        document_id = facts.get("document_id")
        meta = ingested.get(document_id) or {}
        school = meta.get("school") or "unknown"
        group = f"syllabus-doc:{document_id}"
        source = "Dev-the-dev91/syllabus-cleaned"
        source_id = f"syllabus:{document_id}"
        simple = {
            "course_code": "course_code",
            "course_title": "course_title",
            "instructor": "instructor",
            "term": "term",
        }
        for key, label in simple.items():
            value = facts.get(key)
            if not value or not str(value).strip():
                continue
            rows.append(_record(re.sub(r"\s+", " ", str(value)).strip(), label, group, source,
                                f"document fact: {key}",
                                "structured fact extracted from a real syllabus", source_id,
                                meta={"school": school, "document_id": document_id}))
        for category in facts.get("grading") or []:
            name = str((category or {}).get("category") or "").strip()
            if name:
                rows.append(_record(name, "grading_policy", group, source, "document fact: grading",
                                    "grading category from a real syllabus", source_id,
                                    meta={"school": school, "document_id": document_id}))
            weight = (category or {}).get("weight_pct")
            if weight is not None:
                rows.append(_record(f"{float(weight):g}%", "grading_policy", group, source,
                                    "document fact: grading weight",
                                    "grading weight from a real syllabus", source_id,
                                    meta={"school": school, "document_id": document_id}))
        for task in facts.get("all_tasks") or []:
            title = str((task or {}).get("title") or "").strip()
            if title:
                rows.append(_record(title, "assignments", group, source, "document fact: task",
                                    "assessment task title from a real syllabus", source_id,
                                    meta={"school": school, "document_id": document_id}))
            due = (task or {}).get("due_date")
            if due:
                rows.append(_record(str(due), "due_date", group, source, "document fact: task due date",
                                    "task due date from a real syllabus", source_id,
                                    meta={"school": school, "document_id": document_id}))
        pattern = str(facts.get("class_meeting_pattern") or "").strip()
        if pattern:
            rows.append(_record(pattern, "meeting_time", group, source,
                                "document fact: class meeting pattern",
                                "class meeting pattern from a real syllabus", source_id,
                                meta={"school": school, "document_id": document_id}))
            room = re.split(r"\s+in\s+", pattern, maxsplit=1)
            if len(room) == 2 and room[1].strip():
                rows.append(_record(room[1].strip(), "class_meeting_room", group, source,
                                    "document fact: meeting room",
                                    "class meeting room from a real syllabus", source_id,
                                    meta={"school": school, "document_id": document_id}))
        weeks = facts.get("term_weeks")
        if weeks is not None:
            rows.append(_record(f"{int(weeks)} weeks", "session_dates", group, source,
                                "document fact: term length",
                                "term length from a real syllabus", source_id,
                                meta={"school": school, "document_id": document_id}))
    return rows


# ---------------------------------------------------------------------------
# Source 4: real text that is not an academic field value
# ---------------------------------------------------------------------------

def load_other_examples(path: str | Path = "data/raw/cu_docbench.json",
                        limit: int = 4000) -> list[dict[str, Any]]:
    """Real document blocks that are not academic field values.

    Used only for the ``other`` class so the classifier can decline instead of
    always naming the closest field.
    """
    if not Path(path).exists():
        return []
    documents = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.get("metadata") or {}
        if str(metadata.get("document_type") or "").lower() == "syllabus":
            continue
        name = str(metadata.get("file_name") or metadata.get("document") or "unknown")
        for page in (document.get("ground_truth") or {}).get("pages") or []:
            for block in page.get("blocks") or []:
                text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
                if len(text) < 15 or len(text) > 1200:
                    continue
                if not _mostly_latin(text):
                    continue
                rows.append(_record(text, "other", f"cudocbench-doc:{name}",
                                    "iristun/CU-DocBench", f"block:{block.get('type')}",
                                    "real document block that is not an academic field value",
                                    f"cudocbench:{name}:{block.get('id')}"))
    if len(rows) > limit:
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def exclude_intrinsically_ambiguous(rows: list[dict[str, Any]]
                                    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Drop values whose exact normalised text carries conflicting field labels.

    A text-only model cannot recover the source field from an identical string,
    so these rows are unusable as supervised examples and are reported rather
    than silently kept.
    """
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_text[normalize_value(row["text"])].add(row["label"])
    ambiguous = {text: sorted(labels) for text, labels in labels_by_text.items() if len(labels) > 1}
    clean = [row for row in rows if normalize_value(row["text"]) not in ambiguous]
    return clean, ambiguous


def build_field_corpus(usc_path: str | Path = "data/raw/usc_course_data.csv",
                       gu_path: str | Path = "data/raw/gu_course_syllabus.parquet",
                       hf_root: str | Path = "data/raw/hf",
                       cu_path: str | Path = "data/raw/cu_docbench.json",
                       cap_per_label: int | None = None,
                       seed: int = 20261019) -> dict[str, Any]:
    """Merge every real source into one labelled corpus."""
    import random

    from .field_sources import (load_int_business_syllabus_examples, load_mit_field_examples,
                                load_syllabus_span_examples, load_umich_field_examples,
                                load_uw_field_examples)

    rows: list[dict[str, Any]] = []
    rows.extend(load_usc_field_examples(usc_path))
    rows.extend(load_gu_field_examples(gu_path))
    rows.extend(load_syllabus_field_examples(hf_root))
    rows.extend(load_other_examples(cu_path))
    rows.extend(load_uw_field_examples())
    rows.extend(load_mit_field_examples())
    rows.extend(load_syllabus_span_examples())
    rows.extend(load_umich_field_examples())
    rows.extend(load_int_business_syllabus_examples())

    total_before = len(rows)
    rows, ambiguous = exclude_intrinsically_ambiguous(rows)
    after_exclusion = len(rows)
    for row in rows:
        row["source_key"] = row["source"].split("/")[0]

    capped = False
    if cap_per_label:
        rng = random.Random(seed)
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        trimmed: list[dict[str, Any]] = []
        for label, items in by_label.items():
            if len(items) > cap_per_label:
                capped = True
                rng.shuffle(items)
                items = items[:cap_per_label]
            trimmed.extend(items)
        rows = trimmed

    return {
        "rows": rows,
        "stats": {
            "rows_before_ambiguity_exclusion": total_before,
            "rows_after_ambiguity_exclusion": after_exclusion,
            "rows_excluded_as_ambiguous": total_before - after_exclusion,
            "rows": len(rows),
            "ambiguous_text_values": len(ambiguous),
            "capped_per_label": cap_per_label,
            "was_capped": capped,
            "by_source": dict(Counter(r["source"] for r in rows)),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "groups": len({r["group_id"] for r in rows}),
        },
        "ambiguous": ambiguous,
    }


def split_field_groups(rows: list[dict[str, Any]], seed: int = 20261019,
                        fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
                        group_prefix: str | None = None
                        ) -> dict[str, list[dict[str, Any]]]:
    """Hold out whole groups (subject, course, document) so nothing leaks."""
    import random

    if abs(sum(fractions) - 1) > 1e-8 or any(f <= 0 for f in fractions):
        raise ValueError("split fractions must be positive and sum to one")
    groups = sorted({row["group_id"] for row in rows})
    if group_prefix:
        groups = [g for g in groups if g.startswith(group_prefix)]
    if len(groups) < 3:
        raise ValueError("at least three distinct groups are required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    first = round(len(groups) * fractions[0])
    second = round(len(groups) * (fractions[0] + fractions[1]))
    assignment = {g: "train" for g in groups[:first]}
    assignment.update({g: "validation" for g in groups[first:second]})
    assignment.update({g: "test" for g in groups[second:]})
    return {name: [row for row in rows if assignment.get(row["group_id"]) == name]
            for name in ("train", "validation", "test")}
