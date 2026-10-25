"""Seeded synthetic academic-world generator with template-held-out text documents.

This generates auditable synthetic text examples only. It does not claim rendered PDF/image
realism or real OCR performance; OCR-like text corruption is an augmentation experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .synthetic_transcript import TRANSCRIPT_FAMILIES, render_transcript_document

SCHEMA_VERSION = "pace-academic-ir/0.2"
DOCUMENT_TYPES = (
    "syllabus", "schedule", "gradebook", "transcript", "report_card",
    "course_catalog", "requirements", "assignment_sheet", "course_selection", "other",
)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
YEAR_STRINGS = ["2026", "2027", "2028", "2029", "2030"]


def academic_date(rng: random.Random) -> str:
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.choice(YEAR_STRINGS)}"

COURSES = [
    ("BIO 210", "Cell Biology", "Biology 210"),
    ("CHEM 118", "General Chemistry II", "Chemistry II"),
    ("CSCI 240", "Data Structures", "Computer Science 240"),
    ("MATH 221", "Calculus II", "Integral Calculus"),
    ("HIST 130", "World History", "Global History"),
    ("ENGL 204", "Writing and Rhetoric", "Academic Writing"),
    ("PHYS 122", "Physics: Waves and Optics", "Physics II"),
    ("PSYC 101", "Introduction to Psychology", "Psychology 101"),
    ("ART 105", "Foundations of Design", "Design Studio"),
    ("ECON 202", "Microeconomic Analysis", "Microeconomics"),
    ("STAT 205", "Applied Statistics", "Statistics for Applications"),
    ("FREN 110", "Elementary French", "French I"),
]

ALIASES = {code: sorted({code, title, alias, code.replace(" ", "-"), code.split()[0] + code.split()[1]})
           for code, title, alias in COURSES}
SCHOOL_ADJECTIVES=["North","Cedar","Lakeview","Westfield","Pinecrest","East Harbor","Silver","Maple","Redwood","Summit","Clearwater","Brookfield","Meadow","Granite","Oak","Willow","Canyon","River","Stone","Bright"]
SCHOOL_NOUNS=["Ridge","Valley","Heights","Grove","Point","Hills","Crossing","Harbor","Meadows","Park","Springs","Terrace","View","Gardens","Creek","Falls","Field","Landing","Woods","Center"]
SCHOOL_KINDS=["High School","University","College"]
SCHOOLS=[(f"{a} {n} {k}",f"{a} {n}") for a in SCHOOL_ADJECTIVES for n in SCHOOL_NOUNS[:3] for k in SCHOOL_KINDS]
TEACHER_TITLES=["Dr.","Ms.","Mr.","Prof.","Mrs."]
FIRST_NAMES=["Avery","Riley","Jordan","Casey","Taylor","Quinn","Dakota","Cameron","Jamie","Skylar","Robin","Alex","Morgan","Reese","Parker","Finley","Arden","Sage","Emerson","Rowan","Noah","Maya","Amir","Zoe"]
LAST_NAMES=["Morgan","Chen","Patel","Brooks","Singh","Rivera","Nguyen","Lee","Wilson","Ahmed","Green","Kim","Diaz","Martin","Clark","Wright","Scott","Bennett","Lopez","Price","Shah","Cruz","Baker","Reed"]
ROOM_PREFIXES=["Science","Hall","Room","North Lab","West Wing","Studio","Building C","Auditorium","East Lab","Seminar"]
ROOMS=[f"{prefix} {number}" for prefix in ROOM_PREFIXES for number in range(1,31)]
TERMS = ["Fall 2026", "Autumn 2026", "Semester 1, 2026-27", "Spring 2027", "Winter 2027", "Term 2 2027"]
DAYS = [["Mon", "Wed"], ["Tue", "Thu"], ["Mon", "Wed", "Fri"], ["Tue", "Thu", "Fri"], ["Friday"]]
TIMES = [("8:30am", "9:45am"), ("9:00am", "10:15am"), ("10:30am", "11:45am"), ("1:00pm", "2:15pm"), ("2:00pm", "3:20pm")]

CATEGORY_VARIANTS = {
    "tests": ["Tests", "Exams", "Unit Assessments", "Major Tests", "Summative Assessments"],
    "labs": ["Labs", "Laboratory Work", "Practical Assessments", "Lab Reports"],
    "homework": ["Homework", "Assignments", "Practice Work", "Problem Sets"],
    "final": ["Final Grade", "Semester Final", "End-of-Course Exam", "Comprehensive Final"],
}

# The names of dated pieces of work. They are deliberately disjoint from the grade
# category vocabulary: a category carries a weight, an assessment carries a date.
ASSESSMENT_NAME_VARIANTS = (
    ["Unit 1 Exam", "Unit 1 Examination", "Midterm Assessment", "First Major Test"],
    ["Final Examination", "End-of-Term Assessment", "Final Exam", "Course Final"],
)
DELIVERABLE_NAME_VARIANTS = ["Lab Report 2", "Research Project 1", "Problem Set 3", "Design Brief"]

IRRELEVANT_TEXT = [
    "Late work may lose 10% per day after the deadline.",
    "Please silence phones during class.",
    "Office hours are Tuesday from 2:00 to 3:00 in {room}.",
    "Course materials were revised on December 12, 2026.",
    "Bring the assigned textbook to each meeting.",
    "The school office is located at 14 College Avenue.",
    "A 15% late penalty applies after the grace period.",
    "Student services open at 8:00am on weekdays.",
    "The department website was updated on October 20, 2026.",
]

DOCUMENT_TEMPLATES = {
    "syllabus": ["syllabus_heading_first", "syllabus_narrative", "syllabus_compact", "syllabus_table_like"],
    "schedule": ["schedule_rows", "schedule_compact", "schedule_narrative", "schedule_heading"],
    "gradebook": ["gradebook_table", "gradebook_compact", "gradebook_narrative", "gradebook_heading"],
    "transcript": list(TRANSCRIPT_FAMILIES),
    "report_card": ["report_card_rows", "report_card_compact", "report_card_narrative", "report_card_heading"],
    "course_catalog": ["catalog_entry", "catalog_narrative", "catalog_compact", "catalog_heading"],
    "requirements": ["requirement_list", "requirement_compact", "requirement_narrative", "requirement_heading"],
    "assignment_sheet": ["assignment_notice", "assignment_compact", "assignment_narrative", "assignment_heading"],
    "course_selection": ["selection_form", "selection_compact", "selection_narrative", "selection_heading"],
    "other": ["other_notice", "other_compact", "other_narrative", "other_heading"],
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _label_doc_type(text: str) -> str:
    t = text.casefold()
    checks = [
        ("gradebook", ("gradebook", "grade categories", "grade weights", "current grades")),
        ("transcript", ("transcript", "academic history", "term gpa", "cumulative gpa")),
        ("report_card", ("report card", "progress report", "term grades", "semester marks")),
        ("requirements", ("graduation requirement", "degree requirement", "credits required", "requirement audit")),
        ("assignment_sheet", ("assignment sheet", "due date", "submit by", "project instructions")),
        ("course_selection", ("course selection", "enrollment choices", "select courses", "registration options")),
        ("schedule", ("class schedule", "weekly timetable", "meets on", "period schedule")),
        ("course_catalog", ("course catalog", "catalogue entry", "course description", "catalog listing")),
        ("syllabus", ("syllabus", "course outline", "grading policy", "learning outcomes")),
    ]
    for label, cues in checks:
        if any(cue in t for cue in cues):
            return label
    return "other"


def generate_academic_world(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    school, short_name = rng.choice(SCHOOLS)
    term = rng.choice(TERMS)
    selected = rng.sample(COURSES, k=rng.randint(2, 5))
    courses = []
    for idx, (code, title, alias) in enumerate(selected):
        start, end = rng.choice(TIMES)
        cuts = sorted(rng.sample(range(1, 100), 3))
        parts = [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], 100 - cuts[2]]
        rng.shuffle(parts)
        categories = {
            rng.choice(CATEGORY_VARIANTS["tests"]): parts[0],
            rng.choice(CATEGORY_VARIANTS["labs"]): parts[1],
            rng.choice(CATEGORY_VARIANTS["homework"]): parts[2],
            rng.choice(CATEGORY_VARIANTS["final"]): parts[3],
        }
        courses.append({
            "course_code": code,
            "canonical_title": title,
            "aliases": sorted(set([title, alias] + ALIASES[code])),
            "teacher": f"{rng.choice(TEACHER_TITLES)} {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "room": rng.choice(ROOMS),
            "term": term,
            "credits": rng.choice([0.5, 1.0, 2.0, 3.0, 4.0]),
            "schedule": {"days": rng.choice(DAYS), "start": start, "end": end},
            "categories": categories,
            "assessments": [
                {"name": rng.choice(ASSESSMENT_NAME_VARIANTS[0]), "date": academic_date(rng)},
                {"name": rng.choice(ASSESSMENT_NAME_VARIANTS[1]), "date": academic_date(rng)},
            ],
            "deliverables": [
                {"name": rng.choice(DELIVERABLE_NAME_VARIANTS), "due": academic_date(rng)},
            ],
            "prerequisites": [selected[idx - 1][0]] if idx > 0 and rng.random() < 0.80 else [],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "school": {"name": school, "short_name": short_name, "school_id": short_name.casefold().replace(' ', '-'), "academic_year": "2026-2027"},
        "term": term,
        "courses": courses,
    }


def _render_body(world: dict[str, Any], doc_type: str, rng: random.Random) -> tuple[list[str], dict[str, Any]]:
    course = rng.choice(world["courses"])
    name = rng.choice(course["aliases"])
    code = course["course_code"]
    teacher = course["teacher"]
    room = course["room"]
    credits = course["credits"]
    days = ", ".join(course["schedule"]["days"])
    start, end = course["schedule"]["start"], course["schedule"]["end"]
    title = f"{name} ({code})"
    irrelevant = rng.choice(IRRELEVANT_TEXT).format(room=room)
    assessments = course["assessments"]
    deliverable = course["deliverables"][0]

    if doc_type == "syllabus":
        title_line = rng.choice([f"{title}", f"{world['school']['name']} — {title}", f"{title} | {world['term']}"])
        rows = [title_line, f"{rng.choice(['Instructor', 'Course lead', 'Teacher'])}: {teacher}",
                rng.choice(["Evaluation Methods", "Course Policies and Assessment", "How Your Grade Is Calculated"])]
        if course['prerequisites']:
            rows.append(f"{rng.choice(['Prerequisite:', 'Required before enrollment:', 'Must complete'])} {course['prerequisites'][0]}")
        for cat, weight in course["categories"].items():
            rows.append(f"{cat}{rng.choice([' — ', ': ', ' ........ ', '    '])}{weight}%")
        rows += ["Important Dates", f"{assessments[0]['name']} — {assessments[0]['date']}",
                 f"{assessments[1]['name']} — {assessments[1]['date']}", irrelevant]
        if course["prerequisites"]:
            rows.append(f"{rng.choice(['Prerequisite:', 'Required prior course:', 'Must complete'])} {course['prerequisites'][0]}")
        # A course code that follows a prerequisite line is a plain course code, not the
        # prerequisite itself. Without this contrast the only course code that ever
        # appears after a "Prerequisite:" cue is the prerequisite, so the tagger learns
        # the cue instead of the local relation.
        other_courses = [entry for entry in world["courses"]
                         if entry["course_code"] != code
                         and entry["course_code"] not in course["prerequisites"]]
        if other_courses:
            other_code = rng.choice(other_courses)["course_code"]
            rows.append(f"{rng.choice(['Equivalent course', 'Also listed as', 'Cross-listed with'])}: {other_code}")
        # The letter grades a course can award are a scale, not marks awarded to a
        # student. Leaving them unlabelled teaches the tagger that a bare letter in a
        # list is not a mark; the literal pattern extractor proposes them instead.
        rows.append(rng.choice([
            "Grade scale: A, B, C, D, F",
            "Grades: A, B, C, D, F",
            "Final grades are A, B, C, D, F",
            "Grading scale: A through F",
        ]))
    elif doc_type == "schedule":
        rows = [rng.choice(["Weekly Plan", "Meeting Information", "Section Details"]),
                f"{rng.choice(['Course', 'Class', 'Section'])}: {title}",
                f"{rng.choice(['Meets on', 'Days / time', 'Schedule'])}: {days} {start}–{end}",
                # A room number followed by a full stop is still one room value.
                f"{rng.choice(['Room', 'Location'])}: {room}{rng.choice(['', '.'])}",
                f"{rng.choice(['Teacher', 'Instructor'])}: {teacher}",
                irrelevant]
    elif doc_type == "gradebook":
        rows = [rng.choice(["Current Standing", "Class Summary", "Assessment Record"]), title,
                f"{rng.choice(CATEGORY_VARIANTS['tests'])}: {rng.randint(25,55)}%",
                f"{rng.choice(CATEGORY_VARIANTS['homework'])}: {rng.randint(10,30)}%",
                f"{rng.choice(['Room', 'Location'])}: {room}{rng.choice(['', '.'])}", irrelevant]
    elif doc_type == "transcript":
        rows = [rng.choice(["Student Record", "Completed Coursework", "Term Summary"]),
                f"{world['term']} | {code} | {name} | A- | {credits:.1f} credits",
                f"Cumulative GPA: 3.62 | Units earned: {credits:.1f}"]
    elif doc_type == "report_card":
        rows = [rng.choice(["Student Progress", "Term Summary", "Learning Update"]),
                f"{code} {name}: A-", f"Term grade: 91% | Attendance: 96%", irrelevant]
    elif doc_type == "course_catalog":
        rows = [rng.choice(["Course Information", "Program Offering", "Course Details"]), title,
                rng.choice([f"An introduction to {name.lower()}.", f"Students examine key concepts in {name.lower()}."]),
                f"Credit: {credits:.1f} units"]
        if course['prerequisites']:
            rows.append(f"Prerequisite: {course['prerequisites'][0]}")
    elif doc_type == "requirements":
        rows = [rng.choice(["Program Progress", "Credit Summary", "Completion Status"]),
                f"Required credits: {rng.choice([18,20,24,30,36])}.0", f"Earned credits: {rng.choice([6,12,18,21])}.5", f"Science: {rng.choice([2,3,4])}.0 credits", f"Mathematics: {rng.choice([3,4,5])}.0 credits"]
    elif doc_type == "assignment_sheet":
        rows = [rng.choice(["Independent Work", "Task Brief", "Submission Details"]),
                f"{name}: {deliverable['name']}", f"Submit by: {deliverable['due']}", "Late work may lose 10% per day."]
    elif doc_type == "course_selection":
        rows = [rng.choice(["Next Term Options", "Planning Choices", "Registration Form"]),
                f"Select: {title}", f"Credits awarded: {credits:.1f}"]
        if course['prerequisites']:
            rows.append(f"Required before enrollment: {course['prerequisites'][0]}")
    else:
        rows = [rng.choice(["Campus Notice", "Library Information", "Student Services Bulletin"]),
                "Library hours: Monday-Friday, 8:00am-6:00pm", "Bring your student ID to the front desk."]

    gold = {
        "course_code": code, "current_course_code": code, "course_title": course["canonical_title"],
        "course_aliases": course["aliases"], "teacher": teacher,
        "room": room, "credits": credits, "prerequisites": course["prerequisites"],
        "grade_categories": course["categories"], "assessments": course["assessments"],
        "deliverables": course["deliverables"],
        # The literal code spellings that may appear in this document. A course code is
        # a value from the world, not a shape that happens to look like one: prose
        # such as "located at 14 College Avenue" is not a course code.
        "course_code_forms": sorted({form for entry in world["courses"]
                                     for form in course_code_forms(entry["course_code"])}),
        "decoy_text": irrelevant,
    }
    return rows, gold


def _find_all_casefold(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    return [(match.start(), match.end()) for match in re.finditer(re.escape(value), text, flags=re.I)]


def _find_all_boundary(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    return [(match.start(), match.end()) for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", text, flags=re.I)]


def course_code_forms(code: str) -> list[str]:
    """The literal spellings a course code may take in a document."""
    return [code, code.replace(" ", "-"), code.replace(" ", "")]


def _first_boundary(text: str, values: list[str]) -> tuple[int, int] | None:
    """The earliest boundary match among several candidate values, if any."""
    found = [span for value in values for span in _find_all_boundary(text, value)]
    return min(found, key=lambda span: (span[0], span[1])) if found else None


def extract_gold_entities(text: str, gold: dict[str, Any], doc_type: str) -> list[dict[str, Any]]:
    """Create character-span labels directly from the generated canonical world state."""
    candidates: list[dict[str, Any]] = []

    # Codes and their lexical course-name aliases are source-grounded canonical entities.
    code_forms = gold.get("course_code_forms", [])
    for form in code_forms:
        for start, end in _find_all_boundary(text, form):
            candidates.append({"type": "COURSE_CODE", "start": start, "end": end, "text": text[start:end]})
    for alias in sorted(set([gold["course_title"], *gold.get("course_aliases", ALIASES.get(gold["course_code"], []))]), key=len, reverse=True):
        for start,end in _find_all_boundary(text,alias):
            candidates.append({"type":"COURSE","start":start,"end":end,"text":text[start:end]})
    for kind,key in (("TEACHER","teacher"),("ROOM","room")):
        for start,end in _find_all_boundary(text,gold.get(key,"")):
            candidates.append({"type":kind,"start":start,"end":end,"text":text[start:end]})
    # Track room mentions in irrelevant prose, but do not mislabel them as the course room.
    if gold.get("room"):
        for start,end in _find_all_boundary(text,gold["room"]):
            if not any(c['type']=='ROOM' and c['start']==start and c['end']==end for c in candidates):
                candidates.append({"type":"ROOM","start":start,"end":end,"text":text[start:end]})

    # A dated piece of work is an assessment: its name comes from the world state, so
    # every name the document prints is covered, not only the ones a regex lists.
    for entry in list(gold.get("assessments", [])) + list(gold.get("deliverables", [])):
        for start, end in _find_all_boundary(text, entry.get("name", "")):
            candidates.append({"type": "ASSESSMENT", "start": start, "end": end, "text": text[start:end]})

    # Context-gated numeric entities avoid labeling policy penalties as grade weights.
    lines = []
    cursor = 0
    for line in text.splitlines():
        start = text.find(line, cursor)
        if start < 0:
            continue
        cursor = start + len(line)
        lines.append((line, start))
    weight_categories = sorted({variant for variants in CATEGORY_VARIANTS.values() for variant in variants}, key=len, reverse=True)
    # A time of day and a calendar date are those things wherever they appear. The
    # tagger is not told to look only on a line that names an assessment.
    for line, line_start in lines:
        for match in re.finditer(r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", line, flags=re.I):
            candidates.append({"type": "TIME", "start": line_start + match.start(), "end": line_start + match.end(), "text": match.group(0)})
        for match in re.finditer(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b", line, flags=re.I):
            candidates.append({"type": "DATE", "start": line_start + match.start(), "end": line_start + match.end(), "text": match.group(0)})
    for line, line_start in lines:
        low = line.casefold()
        if "%" in line:
            category_match = next((re.search(re.escape(cat), line, flags=re.I) for cat in weight_categories if re.search(re.escape(cat), line, flags=re.I)), None)
            percent_match = re.search(r"\b\d{1,3}\s*%", line)
            if category_match and percent_match:
                candidates.append({"type": "GRADE_CATEGORY", "start": line_start + category_match.start(), "end": line_start + category_match.end(), "text": category_match.group(0)})
                candidates.append({"type": "WEIGHT", "start": line_start + percent_match.start(), "end": line_start + percent_match.end(), "text": percent_match.group(0)})

        credit_cue_pattern=re.compile(r"\b(?:required credits|earned credits|units earned|credits awarded|credits?|units|science|mathematics)\s*:",flags=re.I)
        for credit_cue in credit_cue_pattern.finditer(line):
            # Parse the value after each explicit credit-related cue. A leading GPA/year
            # on the same row must never be captured as units.
            tail=line[credit_cue.end():].split("|",1)[0]
            number=re.search(r"\b\d+(?:\.\d+)?\b",tail)
            if number:
                number_start=credit_cue.end()+number.start()
                number_end=credit_cue.end()+number.end()
                value_start=line_start+number_start;value_end=line_start+number_end
                if not any(c['type']=='COURSE_CODE' and value_start < c['end'] and c['start'] < value_end for c in candidates):
                    candidates.append({"type":"CREDIT","start":value_start,"end":value_end,"text":line[number_start:number_end]})


        if any(cue in low for cue in ("prerequisite", "required before enrollment", "required prior course", "must complete")):
            cue_match=re.search(r"(?:prerequisite\s*:|required before enrollment\s*:|required prior course\s*:|must complete\s+)",line,flags=re.I)
            search_line=line[cue_match.end():] if cue_match else line
            prereq=_first_boundary(search_line, code_forms)
            if prereq:
                prereq_start=(cue_match.end() if cue_match else 0)+prereq[0]
                prereq_end=(cue_match.end() if cue_match else 0)+prereq[1]
                value_start=line_start+prereq_start;value_end=line_start+prereq_end
                candidates.append({"type":"PREREQUISITE","start":value_start,"end":value_end,"text":line[prereq_start:prereq_end]})
                for code in [c for c in candidates if c['type']=='COURSE_CODE' and c['start']==value_start and c['end']==value_end]:
                    candidates.remove(code)
            # The cue's line may also name the primary course. Only the post-cue code is prerequisite.
            continue

    # Remove regex-shaped false course codes that point at prose/percentages, and remove
    # non-code course aliases whose only match is a source text malformed as a code token.
    candidates = [c for c in candidates if not (c["type"] == "COURSE_CODE" and re.fullmatch(r"lose\s+10", c["text"], flags=re.I))]
    # Drop aliases embedded in a code-shaped token; the exact code is a separate entity.
    candidates = [c for c in candidates if not (c["type"] == "COURSE" and re.fullmatch(r"[A-Z]{2,6}[- ]?\d{2,4}[A-Z]?", c["text"], flags=re.I))]
    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["type"]))
    accepted = []
    priority = {"PREREQUISITE": 0, "COURSE_CODE": 1}
    for item in candidates:
        if item["start"] < 0 or item["end"] > len(text) or item["start"] >= item["end"]:
            continue
        if text[item["start"]:item["end"]] != item["text"]:
            raise ValueError("Entity span does not match source evidence")
        conflicts = [other for other in accepted if item["start"] < other["end"] and other["start"] < item["end"]]
        if conflicts:
            # This flat BIO schema cannot encode overlapping roles. Preserve the specific
            # prerequisite relation role over a generic course-code mention at the same span.
            same_span = [other for other in conflicts if (other["start"], other["end"]) == (item["start"], item["end"])]
            if same_span and priority.get(item["type"], 2) < priority.get(same_span[0]["type"], 2):
                accepted = [other for other in accepted if other not in same_span]
            elif item["type"] == "COURSE_CODE" and all(item["start"] <= other["start"] and item["end"] >= other["end"] for other in conflicts):
                accepted = [other for other in accepted if other not in conflicts]
            else:
                continue
        accepted.append(item)
    return sorted(accepted, key=lambda item: (item["start"], item["end"], item["type"]))


def render_course_document(world: dict[str, Any], seed: int, doc_type: str | None = None):
    rng = random.Random(seed ^ 0x5A17)
    if doc_type is None:
        doc_type = rng.choice(DOCUMENT_TYPES)
    if doc_type not in DOCUMENT_TEMPLATES:
        raise ValueError(f"Unsupported document type: {doc_type}")
    template = rng.choice(DOCUMENT_TEMPLATES[doc_type])
    if doc_type == "transcript":
        # Transcripts are secondary-school records: render them with the transcript
        # generator and a school identity drawn from the shared school pool so the
        # school hold-out in split_by_template still applies.
        high_schools = [entry for entry in SCHOOLS if entry[0].endswith("High School")]
        school, short_name = high_schools[rng.randrange(len(high_schools))]
        document = render_transcript_document(
            world, seed, template, school, short_name.casefold().replace(" ", "-"), _digest,
        )
        document["schema_version"] = SCHEMA_VERSION
        document["world_seed"] = world["seed"]
        document["ocr_text"] = apply_ocr_noise(document["text"], seed * 97, 0.02)
        return document
    lines, gold = _render_body(world, doc_type, rng)
    header = rng.choice([world["school"]["name"], world["school"]["short_name"], f"{world['term']} • {world['school']['short_name']}"])
    style = rng.choice(["plain", "caps", "numbered", "bulleted"])
    if style == "caps":
        lines = [line.upper() if rng.random() < 0.3 else line for line in lines]
    elif style == "numbered":
        lines = [f"{i + 1}. {line}" for i, line in enumerate(lines)]
    elif style == "bulleted":
        lines = [f"• {line}" if i > 0 else line for i, line in enumerate(lines)]

    if template.endswith("compact"):
        text = " | ".join([header, *lines])
    elif template.endswith("narrative"):
        text = f"{header}\n\n" + "\n".join(lines)
    elif template.endswith("heading"):
        text = f"{header}\n{'-' * min(len(header), 50)}\n" + "\n\n".join(lines)
    elif template.endswith("table_like") or template.endswith("rows") or template.endswith("form") or template.endswith("entry") or template.endswith("list") or template.endswith("notice"):
        text = f"{header}\n" + "\n".join(lines)
    else:
        text = f"{header}\n\n" + "\n".join(lines)

    blocks = []
    for idx, line in enumerate(lines):
        # Labels are controlled by the underlying canonical state / generated section type.
        normalized = line.lstrip("0123456789. •-*\t").strip()
        low = normalized.casefold()
        if idx == 0:
            label = "course_overview" if doc_type in {"syllabus", "course_catalog", "course_selection"} else doc_type
        elif re.search(r"\b(?:prerequisite|required before enrollment|must complete)\b", low):
            label = "prerequisites"
        elif re.search(r"\b(?:instructor|teacher|course lead)\b", low):
            label = "contact"
        elif re.search(r"\b(?:meets on|days / time|schedule|room|location)\b", low):
            label = "schedule"
        elif "%" in low or re.search(r"\b(?:tests|exams|homework|labs|grading|assessment)\b", low):
            label = "grading"
        elif re.search(r"\b(?:credit|units|credits)\b", low):
            label = "credits"
        elif re.search(r"\b(?:due|exam|assessment|submit by)\b", low):
            label = "assignments"
        else:
            label = "unknown"
        blocks.append({"text": line, "section_type": label, "evidence_sha256": _digest(line)})

    # Keep raw source text unchanged; OCR corruption is stored as a separate augmented input.
    # Character-span labels refer to exact canonical clean text, never to a mutated string.
    clean_text = text
    corrupted_text = apply_ocr_noise(clean_text, seed * 97, 0.02)
    entities = extract_gold_entities(clean_text, gold, doc_type)
    return {
        "document_id": f"synthetic-{seed}-{_digest(text)[:12]}",
        "schema_version": SCHEMA_VERSION,
        "document_type": doc_type,
        "template_family": template,
        "seed": seed,
        "world_seed": world["seed"],
        "world_school_id": world["school"]["school_id"],
        "text": clean_text,
        "ocr_text": corrupted_text,
        "gold": gold,
        "entities": entities,
        "blocks": blocks,
    }


def apply_ocr_noise(text: str, seed: int, corruption_rate: float = 0.02,
                    protect: list[tuple[int, int]] | None = None) -> str:
    """Apply mild OCR-like text corruption; this is not a substitute for image OCR.

    `protect` lists character spans whose characters are never changed. Corruption
    inside a value changes how the value is tokenized, so a measurement that wants
    to isolate the effect of noise in the surrounding text can protect the values.
    """
    if not 0 <= corruption_rate <= 0.25:
        raise ValueError("corruption_rate must be in [0, 0.25]")
    rng = random.Random(seed)
    swaps = {"o": "0", "O": "0", "i": "1", "l": "1", "S": "5", "B": "8", "m": "rn", "c": "e"}
    keep = bytearray(len(text))
    for start, end in protect or ():
        for index in range(max(0, start), min(len(text), end)):
            keep[index] = 1
    out = []
    for index, char in enumerate(text):
        # Keep character offsets stable so canonical spans remain aligned. `rn` replaces
        # one source character with two characters, so exclude it from span-labeled text.
        replacement = swaps.get(char)
        if replacement is not None and len(replacement) == 1 and not keep[index] and rng.random() < corruption_rate:
            out.append(replacement)
        else:
            out.append(char)
    return "".join(out)


def generate_dataset(count: int, seed: int, corruption_rate: float = 0.0) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    docs = []
    for i in range(count):
        doc_seed = seed + i
        world = generate_academic_world(doc_seed)
        doc_type = DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)]
        doc = render_course_document(world, doc_seed * 17, doc_type=doc_type)
        doc["ocr_text"] = apply_ocr_noise(doc["text"], doc_seed * 97, corruption_rate)
        docs.append(doc)
    return docs


def split_by_template(documents: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Hold out school identities and layout families with label coverage in each split."""
    rng=random.Random(seed)
    schools=sorted({d['world_school_id'] for d in documents})
    if len(schools)<6:raise ValueError('Need at least six distinct synthetic school identities')
    rng.shuffle(schools)
    ntrain=max(3,round(len(schools)*.70));nval=max(1,round(len(schools)*.15))
    families_by_type={}
    for d in documents:families_by_type.setdefault(d['document_type'],set()).add(d['template_family'])
    family_assignments={name:{} for name in ('train','validation','test')}
    # A template is only a leak if the same family appears in multiple splits for the same class.
    for doc_type,families in sorted(families_by_type.items()):
        families=sorted(families)
        if len(families)<3:raise ValueError(f"Document type {doc_type!r} needs at least three template families")
        rng.shuffle(families)
        family_assignments['train'][doc_type]=set(families[:2])
        family_assignments['validation'][doc_type]={families[2]}
        family_assignments['test'][doc_type]=set(families[3:])
    # Preserve both group boundaries. If a school/template intersection is empty,
    # fail with an actionable message rather than leaking a school across splits.
    school_groups={'train':set(schools[:ntrain]),'validation':set(schools[ntrain:ntrain+nval]),'test':set(schools[ntrain+nval:])}
    splits={name:[] for name in ('train','validation','test')}
    for name in ('train','validation','test'):
        for doc_type in sorted(families_by_type):
            family_set=family_assignments[name][doc_type]
            candidates=[d for d in documents if d['document_type']==doc_type and d['template_family'] in family_set]
            eligible=[d for d in candidates if d['world_school_id'] in school_groups[name]]
            if not eligible:
                raise ValueError(f"No school/template examples for type {doc_type!r} in {name}; increase --count")
            splits[name].extend(eligible)

    school_sets={name:{d['world_school_id'] for d in rows} for name,rows in splits.items()}
    if any(school_sets[a] & school_sets[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise RuntimeError("Synthetic school identity leaked across evaluation splits")
    expected=set(families_by_type)
    for name,rows in splits.items():
        found={r['document_type'] for r in rows}
        if found!=expected:raise ValueError(f"School/template split {name} misses classes: {sorted(expected-found)}; increase count")
    # Store audit facts for reproducibility; caller persists exact IDs below.
    for name,rows in splits.items():
        for row in rows:row['evaluation_split']=name
    return splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate seeded synthetic academic text documents.")
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", default="data/synthetic/academic_documents.jsonl")
    parser.add_argument("--ocr-noise", type=float, default=0.02)
    parser.add_argument("--template-split", action="store_true")
    args = parser.parse_args(argv)
    docs = generate_dataset(args.count, args.seed, args.ocr_noise)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs), encoding="utf-8")
    split_summary = None
    if args.template_split:
        splits = split_by_template(docs, args.seed)
        split_summary = {}
        for name, rows in splits.items():
            target = out.with_name(f"{out.stem}_{name}{out.suffix}")
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            split_summary[name] = {"rows": len(rows), "template_families": sorted({r["template_family"] for r in rows})}
    print(json.dumps({"count": len(docs), "seed": args.seed, "output": str(out), "template_split": split_summary}, indent=2))


if __name__ == "__main__":
    main()
