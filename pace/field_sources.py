"""Real labelled field examples from additional public academic datasets.

Three more real sources join the corpus, all of them genuine institutional text:

``twangodev/uwcourses`` (UW-Madison course guide, meetings and sections)
    8,930 catalogued courses and ~700,000 real class meetings with the
    building, room, meeting time, meeting type and instructor that the
    registrar publishes. Only the non-generated columns are used; every
    ``llm_*`` column in that dataset is ignored.

``chaseharmon/MIT_Course_Catalog``
    5,794 MIT catalog rows: class number, title, description, term, level,
    units, prerequisites and laboratory component.

``syllabus-finetune`` (span-annotated real syllabi)
    Real university syllabi with annotated entity spans: instructor, email,
    due date, course date, assignment, reading, grading weight, concept.

Rendering rules: timestamps are printed the way a schedule prints them
(``2:00PM - 2:50PM``), booleans are dropped, and nothing is paraphrased.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .field_corpus import SYLLABUS_SECTION_MAP, _record, normalize_value, window_text

UW_ROOT = "data/raw/hf/uwcourses/public"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(part for part in (_clean(item) for item in value) if part)
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return _clean(value.tolist())
        except (TypeError, ValueError):
            pass
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _format_time(value: Any) -> str:
    if value is None or value != value:
        return ""
    stamp = value
    hour = int(stamp.hour)
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display}:{stamp.minute:02d}{suffix}"


def _format_date(value: Any) -> str:
    if value is None or value != value:
        return ""
    if hasattr(value, "strftime"):
        return f"{value.strftime('%B')} {value.day}, {value.year}"
    return str(value)


# ---------------------------------------------------------------------------
# UW-Madison course guide
# ---------------------------------------------------------------------------


def _is_clean_span(text: str, minimum: int = 3) -> bool:
    """Reject annotation fragments such as URL paths or markup leftovers."""
    if len(text) < minimum or len(text) > 400:
        return False
    if any(marker in text for marker in ("<", ">", "http", "|", "{", "}")):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return len(letters) / len(text) >= 0.6


def load_uw_field_examples(root: str | Path = UW_ROOT) -> list[dict[str, Any]]:
    """Real UW-Madison catalog fields and ~700k real class meetings."""
    import pandas as pd

    base = Path(root)
    courses_path = base / "courses_current.parquet"
    if not courses_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    source = "twangodev/uwcourses"

    courses = pd.read_parquet(courses_path)
    courses = courses[courses["run_id"] == courses["run_id"].max()]
    for record in courses.to_dict("records"):
        course_id = _clean(record.get("course_id"))
        if not course_id:
            continue
        group = f"uw-course:{course_id}"
        source_id = f"uw-course:{course_id}"
        for column, label in (("course_id", "course_code"),
                               ("title", "course_title"),
                               ("description", "course_description"),
                               ("requirements_text", "prerequisites")):
            text = _clean(record.get(column))
            if not text:
                continue
            for window in window_text(text, max_windows=2):
                rows.append(_record(window, label, group, source, column,
                                    "UW-Madison course guide field", source_id))
        credits = record.get("credits_min")
        if credits is not None and credits == credits:
            text = f"{float(credits):g} credits"
            if record.get("credits_max") not in (None,) and \
                    record.get("credits_max") == record.get("credits_max") and \
                    float(record["credits_max"]) != float(credits):
                text = f"{float(credits):g}-{float(record['credits_max']):g} credits"
            rows.append(_record(text, "credits", group, source, "credits",
                                "UW-Madison course guide field", source_id))
        subjects = _clean(record.get("subjects"))
        if subjects:
            rows.append(_record(subjects, "program", group, source, "subjects",
                                "UW-Madison course guide field", source_id))
        outcomes_raw = _clean(record.get("official_learning_outcomes_json"))
        if outcomes_raw:
            try:
                outcomes = json.loads(outcomes_raw)
            except (TypeError, ValueError):
                outcomes = []
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                if _clean(outcome.get("source")) != "catalog":
                    continue  # only catalog-sourced outcomes, never generated ones
                text = _clean(outcome.get("text"))
                if text:
                    rows.append(_record(text, "learning_outcomes", group, source,
                                        "official learning outcomes",
                                        "official catalog learning outcome", source_id))
        designations = record.get("designations")
        if designations is not None:
            for designation in designations:
                text = _clean(designation)
                if not text:
                    continue
                label = "course_level" if text.lower().startswith("level - ") \
                    else "additional_information"
                rows.append(_record(text, label, group, source, "designation",
                                    "UW-Madison course guide field", source_id))

    # --- meetings: the real schedule -------------------------------------
    meetings_path = base / "meetings_current.parquet"
    if meetings_path.exists():
        meetings = pd.read_parquet(meetings_path)
        meetings = meetings[meetings["run_id"] == meetings["run_id"].max()]
        course_id_by_uid = dict(zip(courses["course_uid"], courses["course_id"]))
        for record in meetings.to_dict("records"):
            course_id = course_id_by_uid.get(record.get("course_uid"))
            if not course_id:
                continue
            group = f"uw-course:{course_id}"
            source_id = f"uw-meeting:{record.get('meeting_id')}"
            starts = _format_time(record.get("starts_at"))
            ends = _format_time(record.get("ends_at"))
            if starts and ends:
                rows.append(_record(f"{starts} - {ends}", "meeting_time", group, source,
                                    "meeting time", "real class meeting time", source_id))
            building = _clean(record.get("building"))
            room = _clean(record.get("room"))
            if building and room:
                rows.append(_record(f"{building} {room}", "class_meeting_room", group, source,
                                    "meeting location", "real class meeting location", source_id))
            elif building:
                rows.append(_record(building, "class_meeting_room", group, source,
                                    "meeting location", "real class meeting location", source_id))
            meeting_type = _clean(record.get("meeting_type"))
            if meeting_type:
                rows.append(_record(meeting_type, "class_type", group, source,
                                    "meeting type", "real class meeting type", source_id))
            name = _clean(record.get("name"))
            if name:
                rows.append(_record(name, "class_section", group, source,
                                    "meeting name", "real section name", source_id))
            instructors = _clean(record.get("instructor_names"))
            if instructors:
                rows.append(_record(instructors, "instructor", group, source,
                                    "meeting instructor", "real instructor name", source_id))

    # --- sections: modality, capacity, term dates -------------------------
    sections_path = base / "sections_current.parquet"
    offering_sections_path = base / "offering_sections.parquet"
    offerings_path = base / "offerings_current.parquet"
    if sections_path.exists() and offering_sections_path.exists() and offerings_path.exists():
        sections = pd.read_parquet(sections_path)
        sections = sections[sections["run_id"] == sections["run_id"].max()]
        offering_sections = pd.read_parquet(offering_sections_path)
        offering_sections = offering_sections[
            offering_sections["run_id"] == offering_sections["run_id"].max()]
        offerings = pd.read_parquet(offerings_path)
        offerings = offerings[offerings["run_id"] == offerings["run_id"].max()]
        course_id_by_offering = dict(zip(offerings["offering_id"], offerings["course_id"]))
        course_id_by_section: dict[str, str] = {}
        for offering_id, section_uid in zip(offering_sections["offering_id"],
                                            offering_sections["section_uid"]):
            course_id = course_id_by_offering.get(offering_id)
            if course_id:
                course_id_by_section[section_uid] = course_id
        for record in sections.to_dict("records"):
            course_id = course_id_by_section.get(record.get("section_uid"))
            if not course_id:
                continue
            group = f"uw-course:{course_id}"
            source_id = f"uw-section:{record.get('section_uid')}"
            for column, label in (("section_number", "class_section"),
                                   ("section_type", "class_type"),
                                   ("instruction_mode", "course_modality")):
                text = _clean(record.get(column))
                if text:
                    rows.append(_record(text, label, group, source, column,
                                        "real section attribute", source_id))
            capacity = record.get("capacity")
            if capacity is not None and capacity == capacity and int(capacity) > 0:
                rows.append(_record(f"{int(capacity)} students", "capacity", group, source,
                                    "capacity", "real section capacity", source_id))
            start = _format_date(record.get("start_date"))
            end = _format_date(record.get("end_date"))
            if start and end:
                rows.append(_record(f"{start} - {end}", "session_dates", group, source,
                                    "section start and end date", "real section date range", source_id))

    # --- offerings: term the course is usually offered in ------------------
    offerings_path = base / "offerings_current.parquet"
    if offerings_path.exists():
        offerings = pd.read_parquet(offerings_path)
        offerings = offerings[offerings["run_id"] == offerings["run_id"].max()]
        for record in offerings.to_dict("records"):
            course_id = _clean(record.get("course_id"))
            term = _clean(record.get("typically_offered"))
            if not course_id or not term:
                continue
            rows.append(_record(term, "term", f"uw-course:{course_id}", source,
                                "typically offered", "real offering term",
                                f"uw-offering:{record.get('offering_id')}"))

    # --- instructor directory: public institutional email ------------------
    instructors_path = base / "instructors.parquet"
    if instructors_path.exists():
        instructors = pd.read_parquet(instructors_path)
        seen: set[tuple[str, str]] = set()
        for record in instructors.to_dict("records"):
            name = _clean(record.get("name"))
            email = _clean(record.get("email"))
            if not email or "@" not in email:
                continue
            key = (name, email)
            if key in seen:
                continue
            seen.add(key)
            group = f"uw-instructor:{email}"
            rows.append(_record(email, "contact_information", group, source, "email",
                                "public institutional instructor email",
                                f"uw-instructor:{email}"))
    return rows


# ---------------------------------------------------------------------------
# MIT course catalog
# ---------------------------------------------------------------------------


def load_mit_field_examples(path: str | Path = "data/raw/hf/MIT_Course_Catalog/train.csv"
                            ) -> list[dict[str, Any]]:
    """Real MIT catalog fields."""
    import pandas as pd

    frame_path = Path(path)
    if not frame_path.exists():
        return []
    frame = pd.read_csv(frame_path)
    rows: list[dict[str, Any]] = []
    source = "chaseharmon/MIT_Course_Catalog"
    for record in frame.to_dict("records"):
        class_number = _clean(record.get("Class Number"))
        if not class_number:
            continue
        group = f"mit-course:{class_number}"
        source_id = f"mit:{class_number}"
        for column, label in (("Class Number", "course_code"),
                              ("Name", "course_title"),
                              ("Description", "course_description"),
                              ("Term", "term"),
                              ("Level", "course_level"),
                              ("Units", "credits"),
                              ("Prerequisites", "prerequisites"),
                              ("Equivalents", "additional_information")):
            text = _clean(record.get(column))
            if not text or text.lower() == "nan":
                continue
            for window in window_text(text, max_windows=2):
                rows.append(_record(window, label, group, source, column,
                                    "MIT catalog field", source_id))
        lab = record.get("Lab")
        partial = record.get("Partial Lab")
        if lab is True or partial is True:
            component = "Partial Lab" if partial is True else "Lab"
            rows.append(_record(component, "class_type", group, source, "Lab/Partial Lab",
                                "MIT catalog component", source_id))
    return rows


# ---------------------------------------------------------------------------
# Span-annotated real syllabi
# ---------------------------------------------------------------------------

#: Annotated entity label -> harmonised field type.
SPAN_LABEL_MAP: dict[str, str] = {
    "INSTRUCTOR": "instructor",
    "EMAIL": "contact_information",
    "DUE_DATE": "due_date",
    "COURSE_DATE": "course_schedule",
    "ASSIGNMENT": "assignments",
    "READING": "weekly_readings",
    "GRADING_WEIGHT": "grading_policy",
    "PERCENT": "grading_policy",
    "CONCEPT": "course_content",
    "COURSE": "course_code",
}


# ---------------------------------------------------------------------------
# University of Michigan course pages
# ---------------------------------------------------------------------------

_UMICH_DAY = r"(?:Mo|Tu|We|Th|Fr|Sa|Su)"
_UMICH_TIME = r"\d{1,2}:\d{2}(?:AM|PM)"
_UMICH_SECTION = re.compile(
    r"Section\s+(\d+)\s*\(([^)]*)\)\s+(\d+)\s+(\w+)\s+(\d+|N/A)\s+(\d+|N/A)\s+"
    r"(.{0,80}?)\s+(In Person|Online|Hybrid|In-Person)\s+"
    r"((?:" + _UMICH_DAY + r"\s*){1,6})\s+(" + _UMICH_TIME + r"\s*-\s*" + _UMICH_TIME + r")\s+"
    r"([0-9]{2,5}\s+[A-Z]{1,4}(?:\s+[0-9]{2,5}\s+[A-Z]{1,4})*)")
_UMICH_CREDITS = re.compile(r"Credits:\s*([0-9.]+)")
_UMICH_PREREQ = re.compile(r"(?:Advisory\s+)?Prerequisites?:\s*([^\n]*?)(?:\.\s|\n|Median|$)")
_UMICH_MEDIAN = re.compile(r"Median Grade:\s*([A-Z][+-]?)")
_UMICH_CODE = re.compile(r"Description of course\s+([A-Z&]{2,10}\s*\d{3}[A-Za-z]*)")
_UMICH_PROFILE = re.compile(r"Course Profile\s+([A-Z&]{2,10}\s*\d{3}[A-Za-z]*)\s+([^\n]+)")


def load_umich_field_examples(path: str | Path = "data/raw/hf/umich-course-descriptions/train.jsonl"
                              ) -> list[dict[str, Any]]:
    """Real UMich course pages: catalog fields and the published section table."""
    source_path = Path(path)
    if not source_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    source = "alecwatts/umich-course-descriptions"
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = re.sub(r"[ \t]+", " ", str(record.get("text") or ""))
            url = str((record.get("metadata") or {}).get("url") or "")
            match = _UMICH_CODE.search(text)
            course_code = match.group(1).strip() if match else ""
            group = f"umich-course:{course_code or record.get('id')}"
            source_id = f"umich:{record.get('id')}"

            profile = _UMICH_PROFILE.search(text)
            if profile:
                rows.append(_record(profile.group(1).strip(), "course_code", group, source,
                                    "course code", "real course page field", source_id))
                rows.append(_record(profile.group(2).strip(), "course_title", group, source,
                                    "course title", "real course page field", source_id))
            credits = _UMICH_CREDITS.search(text)
            if credits:
                rows.append(_record(credits.group(1), "credits", group, source, "credits",
                                    "real course page field", source_id))
            median = _UMICH_MEDIAN.search(text)
            if median:
                rows.append(_record(median.group(1), "grades", group, source, "median grade",
                                    "real course page field", source_id))
            prereq = _UMICH_PREREQ.search(text)
            if prereq and prereq.group(1).strip():
                rows.append(_record(prereq.group(1).strip(), "prerequisites", group, source,
                                    "prerequisites", "real course page field", source_id))
            description = re.split(r"Course Profile|Course Sections", text, maxsplit=1)
            if len(description) > 1:
                body = description[1]
                body = re.sub(r"^.*?\n\s*", "", body, count=1)
                if len(body) > 60 and "Scroll down" not in body[:60]:
                    rows.append(_record(body.strip()[:600], "course_description", group, source,
                                        "course description", "real course page text", source_id))

            for section in _UMICH_SECTION.finditer(text):
                number, kind, _class_number, _status, _open, cap, instructor, mode, days, times, room = \
                    section.groups()
                rows.append(_record(f"Section {number}", "class_section", group, source,
                                    "section", "real published section listing", source_id))
                if kind.strip():
                    rows.append(_record(kind.strip(), "class_type", group, source,
                                        "section type", "real published section listing", source_id))
                if cap and cap != "N/A":
                    rows.append(_record(f"{cap} seats", "capacity", group, source,
                                        "enrollment cap", "real published section listing", source_id))
                instructor = instructor.strip(" ,")
                if instructor and len(instructor) <= 60:
                    rows.append(_record(instructor, "instructor", group, source,
                                        "instructor", "real published section listing", source_id))
                rows.append(_record(mode, "course_modality", group, source,
                                    "instruction mode", "real published section listing", source_id))
                rows.append(_record(" ".join(days.split()), "meeting_days", group, source,
                                    "days", "real published section listing", source_id))
                rows.append(_record(times, "meeting_time", group, source,
                                    "time", "real published section listing", source_id))
                rows.append(_record(" ".join(room.split()), "class_meeting_room", group, source,
                                    "location", "real published section listing", source_id))
    return rows


# ---------------------------------------------------------------------------
# University of Florida business syllabus pages
# ---------------------------------------------------------------------------


def load_int_business_syllabus_examples(
        path: str | Path = "data/raw/hf/INT-Business-Syllabus/documents.jsonl") -> list[dict[str, Any]]:
    """Real UF business syllabus pages, split by their own section headings."""
    from .field_corpus import detect_document_sections

    source_path = Path(path)
    if not source_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    source = "INT-Business-Syllabus"
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = re.sub(r"[ \t]+", " ", str(record.get("text") or ""))
            document_id = str(record.get("id") or "unknown")
            group = f"uf-syllabus-doc:{document_id}"
            source_id = f"uf-syllabus:{document_id}"
            sections = detect_document_sections(text)
            for index, section in enumerate(sections):
                label = section["label"]
                if label is None:
                    continue
                rows.append(_record(section["text"], label, group, source, "section heading",
                                    "real section heading in a real syllabus", source_id))
                body_start = section["end"]
                body_end = sections[index + 1]["start"] if index + 1 < len(sections) else len(text)
                body = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
                inline = section.get("inline") or ""
                if inline:
                    body = f"{inline} {body}".strip()
                if len(body) < 40:
                    continue
                for window in window_text(body, max_windows=2):
                    rows.append(_record(window, label, group, source, "section body",
                                        "text under a real section heading", source_id))
    return rows


def load_syllabus_span_examples(path: str | Path = "data/raw/hf/syllabus-finetune/train.jsonl"
                                ) -> list[dict[str, Any]]:
    """Real span-annotated syllabus entities."""
    source_path = Path(path)
    if not source_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    source = "syllabus-finetune"
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            if not messages:
                continue
            try:
                payload = json.loads(messages[-1]["content"])
            except (TypeError, ValueError):
                continue
            document_id = payload.get("document_id") or "unknown"
            group = f"syllabus-span-doc:{document_id}"
            source_id = f"syllabus-span:{document_id}"
            for entity in payload.get("entities") or []:
                label = SPAN_LABEL_MAP.get(entity.get("label") or "")
                if label is None:
                    continue
                text = re.sub(r"\s+", " ", str(entity.get("text") or "")).strip()
                if not _is_clean_span(text):
                    continue
                rows.append(_record(text, label, group, source,
                                    f"annotated span: {entity.get('label')}",
                                    "entity span annotated in a real syllabus", source_id))
            for name in payload.get("section_names") or []:
                label = SYLLABUS_SECTION_MAP.get(normalize_value(str(name)))
                if label is None:
                    continue
                text = re.sub(r"\s+", " ", str(name)).strip()
                if _is_clean_span(text):
                    rows.append(_record(text, label, group, source, "section name",
                                        "section name in a real syllabus", source_id))
    return rows
