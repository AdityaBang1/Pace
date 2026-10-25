"""Turn a syllabus or course document into a structured record.

The field classifier answers "what kind of text is this?"; this module answers
"what does it say?". Values are pulled by format-specific rules (a credit count,
a grading weight, a course code, a meeting slot) and the classifier supplies the
field identity, while the confidence attached to a value is computed from the
evidence for that value (where it sits, whether it is labelled, whether it has
the shape the field requires and the body is about the field at all). The    section softmax is not reused as a value probability: it is most certain exactly
where a correct heading is followed by the wrong body.

Every value carries an evidence span — the exact characters it was read from —
so a reader can check it, and so evaluation can tell a real extraction from a
plausible one. Values a rule reads and does not keep go to the loss ledger
(``pace.field_ledger``) rather than disappearing, so the record can be audited
for what it dropped as well as for what it emitted.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .field_corpus import SYLLABUS_SECTION_MAP, clean_html, detect_document_sections
from .field_ledger import EVIDENCE_CHARS, Ledger

# A breakdown is consistent when its parts account for the whole grade. One
# band for the per-value flag, the confidence bump and the record's flag: the
# weights are integers read off the page, so ±5 points is the widest band that
# still means "these are the parts" rather than "some of the parts".
GRADING_TOTAL = 100
GRADING_SUM_TOLERANCE = 5


def grading_sum_consistent(total: float) -> bool:
    return abs(total - GRADING_TOTAL) <= GRADING_SUM_TOLERANCE

# A grading line names a category and a percentage, in either order.
# A category may carry a number ("Paper 1") and a note before its weight
# ("Paper 1 (5-6 pages) 20%").
# The gap between the category and its percentage may be a colon, a dash, or the
# dot leaders a printed grade table uses ("Essays ...... 30%"). Without the dot
# run, the commonest layout in a photographed syllabus read as zero categories.
_WEIGHT_FIRST = re.compile(
    r"^\s*[-•*]?\s*(?P<cat>[A-Za-z][A-Za-z0-9 &/+'\-]{1,40}?)"
    r"(?:\s*\([^)%]{0,40}\))?[\s.:\-–]*\(?(?P<w>\d{1,3})\s*%")
_PERCENT_FIRST = re.compile(
    r"^\s*[-•*]?\s*(?P<w>\d{1,3})\s*%\s*[:\-–]?\s*(?P<cat>[A-Za-z][A-Za-z &/+'\-]{1,40})$")
_PERCENT = re.compile(r"(?P<w>\d{1,3}(?:\.\d{1,2})?)\s*%")
_BARE_PERCENT = re.compile(r"^\d{1,3}(?:\.\d{1,2})?\s*%$")
_PAIR_BREAK = re.compile(r"[•·,;|+]|\band\b|&")
_LEADING_NOISE = re.compile(r"^[\s\d.)\-(•·]+")

_CREDITS = re.compile(
    r"\b(?P<n>\d{1,2}(?:\.\d{1,2})?)\s*(?:semester\s+|academic\s+)?"
    r"(?:units?|credits?|credit\s+hours?|semester\s+hours?)\b", re.I)
_CREDITS_INLINE = re.compile(
    r"\b(?:credits?|units?)\s*[:\-]\s*(?P<n>\d{1,2}(?:\.\d{1,2})?)\b", re.I)
_COURSE_CODE = re.compile(
    r"\b(?P<dept>[A-Z]{2,6})[\s\-]?(?P<num>\d{1,4}[A-Z]?(?:-\d{1,3})?)\b")
_LABELLED_CODE = re.compile(
    r"\b(?:course\s+(?:code|number|no\.?)|catalog\s+(?:number|no\.?)|course\s*#)\s*[:#]",
    re.I)

# "Teacher:" is how most secondary-school syllabi label the instructor; the
# field reader only knew the university wording and read no instructor at all on
# a real high-school syllabus.
_INSTRUCTOR = re.compile(r"^\s*(?:instructors?|teachers?|taught by|professor)\s*[:\-]\s*(?P<v>.+)$", re.I)
_ROOM = re.compile(r"\b(?:room|rm\.?|hall|building|bldg\.?)\s*[#:]?\s*(?P<v>[A-Z0-9][A-Z0-9\-]{1,10})\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# Two-letter codes written apart ("TU TH", "Mo We Fr", "TU,TH") are what
# registrar exports print, and the compact forms below never matched them, so a
# real syllabus's meeting line produced no meeting at all.
_DAY_PAIR = r"(?:MO|TU|WE|TH|FR|SA|SU)(?:[ ,/&]+(?:MO|TU|WE|TH|FR|SA|SU))+"
_DAY_TOKEN = re.compile(
    r"(?<![A-Za-z])(?P<days>" + _DAY_PAIR + r"|MTWThF|MTWTh|MWThF|MWTh|MWF|MThF|TuTh|TTh|TR|MW|WF|MTW|WTh|M/W/F|"
    r"M/W|T/R)(?![A-Za-z])", re.I)
_DAY_MAP = {
    "M": "MO", "T": "TU", "W": "WE", "R": "TH", "F": "FR",
}
_NAV_WORDS = re.compile(
    r"\b(menu|login|log in|sign in|navigation|breadcrumb|skip to|copyright|©|privacy|"
    r"search|home page|user account|toggle|cookie|javascript|news center|campus calendar|"
    r"offices and departments|visitor|accreditation|admissions|sitemap|directory|"
    r"georgia tech resources|all rights reserved)\b", re.I)
_WEIGHT_STOP = {"the", "of", "at", "least", "more", "than", "those", "who", "will", "result",
               "accumulation", "accumulate", "possible", "total", "points", "percent",
               "grade", "grades", "grading", "earn", "earned", "must", "should", "this",
               "week", "each", "and", "or", "for"}
_GENERIC_TITLES = {"introduction", "syllabus", "catalog description", "course description",
                   "overview", "contents", "resources", "policies", "welcome", "description",
                   "general information", "course information", "class information"}
# A heading that labels a block of a document, not the document's subject.
# A heading that labels a block of a document, not the document's subject. The
# leading words only make it a heading when a block noun follows them: "General
# Chemistry: Macroscopic Investigations and Reaction Principles" starts with
# "General" and is a course name, while "General Information" is a heading.
_GENERIC_HEADING = re.compile(
    r"^(?:general|catalog|course|class|key|basic|important|other|additional)\s+"
    r"(?:information|details|description|overview|contents|requirements|policies|"
    r"policy|schedule|materials|objectives|expectations|contact|staff|grading|"
    r"resources|readings|calendar|hours|outline|topics)\s*$"
    r"|\b(?:information|details|description|requirements|expectations|objectives|"
    r"contact|policy|policies|schedule|materials|grading|assistants?|staff|readings|"
    r"resources|calendar|hours|content)\s*$", re.I)
# A line that names the document rather than the course: a file name, a page
# marker, or a heading built out of "Syllabus".
_TITLE_DOCUMENT = re.compile(
    r"\bsyllabus\b|\bcourse\s+outline\b"
    r"|\.(?:pdf|docx?|html?|txt|rtf)\b"
    r"|_"
    r"|\bpage\s+\d+(?:\s*(?:of|/)\s*\d+)?\b"
    r"|^\s*\d{1,3}\s*/\s*\d{1,3}\s*$"
    r"|[|\\]", re.I)
# "Collaboration, Not Competition" is a rhetorical heading, not a course name.
# "Data, Models, and Decisions" is a real course name, so only the contrastive
# form is rejected.
_TITLE_SLOGAN = re.compile(r",\s*(?:not|but|yet)\b", re.I)
# A line whose part before the colon is an attribute name ("Skills and Experience
# Flag: Quantitative Reasoning", "Course Format: Online") is metadata about the
# course, not what the course is called. A title may carry a subtitle ("General
# Chemistry: Macroscopic Investigations..."); an attribute label may not.
_TITLE_ATTRIBUTE = re.compile(r"^([^:]{3,60}):\s*\S")
_ATTRIBUTE_LABELS = {
    "flag", "flags", "format", "formats", "credit", "credits", "prerequisite",
    "prerequisites", "requirement", "requirements", "policy", "policies",
    "information", "info", "hours", "time", "times", "location", "instructor",
    "instructors", "author", "authors", "contact", "website", "url", "link",
    "email", "office", "description",
}
# "Course Website: Canvas" is a labelled link, not a name; a title that merely
# contains a colon ("General Chemistry: Macroscopic Investigations...") is not.
_TITLE_WEIGHT = re.compile(r"\d{1,3}(?:\.\d{1,2})?\s*%\s*$")
_TITLE_LINK = re.compile(
    r"^(?:course\s+)?(?:website|web\s*site|home\s*page|homepage|url|link|canvas|"
    r"moodle|blackboard|email|e-mail)\s*:", re.I)

# The threshold the command ships with. It is an operating point, not a rule, and it
# is chosen from measurement: on both splits no threshold certifies the 0.95 precision
# target, and on the hand-labelled test documents every positive threshold makes
# precision *worse* (0.750 at 0.0, 0.529 at 0.99), because the per-value percentage is
# a section-type softmax that is confidently wrong about where a value came from.
# Raising it would discard true values while the false ones survive, so it stays at
# zero until the confidence itself is recalibrated. See
# artifacts/field-extract/EXTRACTION_EVAL.md.
DEFAULT_MIN_CONFIDENCE = 0.0


def _looks_like_nav(text: str) -> bool:
    if _NAV_WORDS.search(text):
        return True
    letters = sum(ch.isalnum() or ch.isspace() for ch in text)
    return bool(text) and letters / len(text) < 0.6


def _days_from_token(token: str) -> list[str]:
    """Expand a compact weekday token (MWF, TuTh, TR, M/W/F) to day codes."""
    separated = [part for part in re.split(r"[ ,/&]+", token.strip()) if part]
    codes = {"MO", "TU", "WE", "TH", "FR", "SA", "SU"}
    if len(separated) > 1 and all(part.upper() in codes for part in separated):
        return list(dict.fromkeys(part.upper() for part in separated))
    cleaned = token.replace("/", "")
    if cleaned.lower() in {"tuth", "tth", "tr"}:
        return ["TU", "TH"]
    days: list[str] = []
    index = 0
    while index < len(cleaned):
        if cleaned[index:index + 2].lower() == "th":
            days.append("TH")
            index += 2
            continue
        code = _DAY_MAP.get(cleaned[index].upper())
        if code:
            days.append(code)
        index += 1
    return list(dict.fromkeys(days))

# Words that look like course codes but are not.
_CODE_STOP = {"ROOM", "PAGE", "FALL", "SPRING", "SUMMER", "WINTER", "V", "WWW", "HTTP",
              "HTTPS", "PDF", "HTML", "TBA", "TBD", "N", "A", "ISBN", "ISSN", "DOI",
              "TEL", "FAX", "PHONE", "EMAIL", "ZIP", "SKYPE", "OFFICE", "UNIT"}
_CATEGORY_STOP = {"total", "points", "grade", "grades", "grading", "percent", "percentage",
                  "weight", "weighting", "score", "score total"}

# Values read out of a section rather than a single line.
_SECTION_VALUES = {
    "office_hours": "office_hours",
    "attendance_policy": "attendance_policy",
    "academic_integrity": "academic_integrity",
    "disability_services": "disability_services",
    "materials": "materials",
}


def _lines(text: str):
    """Yield (line, start, end) with true offsets into ``text``."""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        yield stripped, offset, offset + len(stripped)
        offset += len(line)


def _clean_candidate(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;-\u2013")


def _norm_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_BAD_TAIL = {"is", "are", "be", "was", "were", "counts", "count", "worth", "equals",
             "and", "or", "the", "a", "an", "of", "on", "in", "to", "for", "with",
             "this", "that", "will", "should", "must", "as", "if", "when", "after",
             "before", "up", "at", "by", "from"}
# "20% of your grade" is a ceiling or a share, not a category weight.
_OF_GRADE = re.compile(r"\bof\s+(?:your|the)\s+(?:course\s+|final\s+|overall\s+|semester\s+)?"
                       r"grade\s*$", re.I)
_FUNCTION_WORD = _BAD_TAIL | {"based", "following", "computed", "using", "your", "my",
                             "grades", "grade", "from", "by", "at"}


_CATEGORY_LEAD_STOP = {"with", "and", "or", "for", "of", "to", "in", "on", "at", "by",
                       "from", "that", "this", "these", "those", "your", "our", "their",
                       "is", "are", "was", "were", "will", "can", "should", "must"}


def _category_ok(category: str) -> bool:
    text = _clean_candidate(category).lower()
    words = text.split()
    if not (3 <= len(text) <= 40 and sum(ch.isalpha() for ch in text) >= 3):
        return False
    if text in _CATEGORY_STOP or len(words) > 4 or not words:
        return False
    # "with your third absence" is a clause, not a category name.
    if words[0] in _CATEGORY_LEAD_STOP:
        return False
    if words[-1] in _BAD_TAIL:
        return False
    return not any(word in _WEIGHT_STOP for word in words)


_QTY_PREFIX = re.compile(r"^(?:the|a|an|each|every|one|two|three|four|five|both|all|"
                         r"your|my|our)\s+", re.I)


def _strip_category_prefix(text: str) -> str:
    """"The Final: 25%" and "Each of two tests 25%" name a category too."""
    value = _clean_candidate(text)
    lower = value.lower()
    if lower.startswith("of "):
        tail = value[3:].strip()
    elif " of " in lower:
        tail = value[lower.rfind(" of ") + 4:].strip()
    else:
        tail = ""
    if tail and _category_ok(tail):
        value = tail
    for _ in range(3):
        stripped = _QTY_PREFIX.sub("", value).strip()
        if stripped == value:
            break
        value = stripped
    return value


def _trailing_category(text: str) -> str:
    """The noun phrase at the end of a clause: "... based on homework (20%)"."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9&/+'\-]*", text)
    if not words:
        return ""
    picked = [words[-1]]
    for word in reversed(words[:-1]):
        if word.lower() in _FUNCTION_WORD or len(picked) >= 3:
            break
        picked.insert(0, word)
    return " ".join(picked)


def _probability(engine, text: str, field: str) -> float:
    """Calibrated probability that ``text`` belongs to ``field``."""
    if not text.strip():
        return 0.0
    return float(engine.classify(text, top_k=1)["probabilities"].get(field, 0.0))


# --- confidence from the evidence, not from the section-type softmax ---------
#
# The classifier's percentage answers "what kind of section is this text?" and it
# is confidently wrong when a correct heading is followed by the wrong body. The
# number attached to a *value* therefore comes from checks on that value's own
# evidence: where it sits, whether it carries an explicit label, whether it has
# the shape the field requires, and whether the body is about the field at all.
# The model still types sections; it no longer supplies the reported probability.

_POLICY_CUES = {
    "office_hours": re.compile(
        r"(office hour|appointment|\bby e-?mail\b|\bschedul\w*\b|"
        r"\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*\b|"
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b)", re.I),
    "attendance_policy": re.compile(r"(attend|absence|absent|make-?up)", re.I),
    "academic_integrity": re.compile(
        r"(academic|plagiar|integrity|dishonest|honou?r|submitted work|your own work|"
        r"collaborat|cit(?:e|ation|ing)|\bsources?\b)", re.I),
    "disability_services": re.compile(r"(disabilit|accommodat|accessib)", re.I),
    "materials": re.compile(r"(textbook|required|isbn|materials?|readings?|\bbook\b)", re.I),
}
# A heading that names when the class meets is not a heading for when the
# instructor is available; a section the classifier mislabels as office hours
# with one of these headings is rejected on the heading alone.
_SCHEDULE_HEADING = re.compile(
    r"^\s*(?:(?:class|lecture|laborator(?:y|ies)|lab|seminar|recitation|meeting)\s+"
    r"(?:times?|schedules?|meetings?|hours?)|class\s+schedule)\b", re.I)
# What makes a body an office-hours value: it says when, where or how a student
# can reach the instructor. A body that opens with course prose or a contact
# block is the next section's text, however many office-hours words it uses
# later; the opening sentence is where the value states itself.
_AVAILABILITY = re.compile(
    r"(?:\bappointment\b|\boffice hours?\b|\bstudent hours?\b|\bconsultation hours?\b|"
    r"\b(?:drop|walk)[- ]?in\b|\bopen hours?\b|\b(?:stop|come) by\b|"
    r"\bby e-?mail\b|\bschedul\w*)", re.I)
_PLACEHOLDER = re.compile(
    r"^[\s•·\-–—:.]*(?:tbd|tba|n/?a|none|to be announced|to be determined)\b", re.I)
_POINTER = re.compile(
    r"(posted|canvas|e-?learning|website|web site|online|\bsee\b|e-?mail|schedul\w*|contact|"
    r"appointment)", re.I)
_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)
_LOCATION_CUE = re.compile(
    r"(\boffice\b|\broom\b|\brm\.?\b|\bbldg\b|\bbuilding\b|\bhall\b|\blab\b|"
    r"\bcenter\b|\buniversity\b)", re.I)
_SESSION_CODE = re.compile(r"^(?:SA|FA|SP|SU|WI|WS|SS|S)[A-Z]?\d{2}$", re.I)
# A line that opens another labelled field ("Course Dates: May 16").
_NEXT_SECTION_LABEL = re.compile(r"^\s*(?P<label>[A-Z][A-Za-z &/.'\-]{2,40}?)\s*:\s*\S")
# Labels that start a time/place block rather than the field being read.
_FOREIGN_LABELS = {"class", "classes", "lecture", "lectures", "lecture times",
                   "lecture place", "course dates", "meeting times", "meeting place",
                   "screening", "seminar", "recitation", "lab", "laboratory"}
# A flattened table glues the next field onto the same line; cut there.
_VALUE_CUT = re.compile(
    r"\s+(?=(?:course dates|lecture times?|lecture place|class (?:time|room)|"
    r"meeting (?:times?|place)|screening|course (?:code|number)|semester|term)\s*:)", re.I)


def _policy_body(text: str, sections: list[dict[str, Any]],
                 section: dict[str, Any]) -> tuple[str, bool]:
    """The section's value, preferring the inline text on the heading line.

    ``Office Hours: M 7-9pm`` carries its value on the heading line, so reading
    the text *after* the line returns the next section instead of the value.
    """
    inline = _norm_space(section.get("inline") or "")
    if inline:
        return inline, True
    return _section_body(text, sections, section), False


_ABBREVIATION = re.compile(r"(?:\b(?:dr|mr|mrs|ms|prof|professor|st|vs)\.)$", re.I)


def _opening_sentence(text: str) -> str:
    """The first sentence, without breaking at 'Dr.' or 'Prof.'."""
    for match in re.finditer(r"[.!?](?=\s)", text):
        if _ABBREVIATION.search(text[:match.end()]):
            continue
        return text[:match.end()]
    return text


def _names_availability(text: str) -> bool:
    """Does the opening sentence say when or how the instructor is available?"""
    opening = _opening_sentence(text)[:160]
    return bool(_AVAILABILITY.search(opening) or _CLOCK.search(opening)
                or _DAY_TOKEN.search(opening) or _DAY_WORD.search(opening)
                or _DAY_PHRASE.search(opening))


def _policy_decision(field: str, body: str, inline: bool,
                     heading: str = "") -> dict[str, Any]:
    """The field's value, or the reason this body is not one.

    Returning the reason rather than ``None`` is what lets the record show a
    rejected candidate instead of silently dropping it.
    """
    text = _VALUE_CUT.split(_norm_space(body), maxsplit=1)[0].strip(" .;")
    if not text:
        return {"value": None, "confidence": 0.0, "reason": "empty body"}
    if _PLACEHOLDER.match(text):
        return {"value": None, "confidence": 0.0,
                "reason": "placeholder ('TBD'/'TBA'/'None')"}
    if _looks_like_nav(text):
        return {"value": None, "confidence": 0.0, "reason": "navigation text"}
    if field == "office_hours" and _SCHEDULE_HEADING.match(_norm_space(heading)):
        return {"value": None, "confidence": 0.0,
                "reason": "a class-time heading, not office hours"}
    # The body may be the *next* section's text when the document puts the value
    # on the heading line and the heading was not recognised as carrying one.
    lead = _NEXT_SECTION_LABEL.match(text)
    if lead:
        label = re.sub(r"\s+", " ", lead.group("label")).strip().casefold()
        other = SYLLABUS_SECTION_MAP.get(label)
        if (other is not None and other != field) or label in _FOREIGN_LABELS:
            return {"value": None, "confidence": 0.0,
                    "reason": f"body opens another field ('{lead.group('label')}')"}
    if field == "office_hours" and not _names_availability(text):
        return {"value": None, "confidence": 0.0,
                "reason": "body opens without naming when the instructor is available"}
    if not _POLICY_CUES.get(field, re.compile(r".")).search(text):
        return {"value": None, "confidence": 0.0,
                "reason": "body is not about the field"}
    has_day = bool(_CLOCK.search(text)) or bool(_DAY_TOKEN.search(text)) or \
        bool(re.search(r"\b(?:mon|tue|wed|thu|fri)[a-z]*\b", text, re.I))
    # A short body is a fragment unless it names a time ("Tue 2-3pm in WEL 5.239").
    if not inline and len(text) < 40 and not _CLOCK.search(text):
        return {"value": None, "confidence": 0.0, "reason": "body too short to be a value"}
    if field == "office_hours":
        if _CLOCK.search(text):
            confidence = 0.95 if inline else 0.9
        elif has_day:
            confidence = 0.85
        elif _POINTER.search(text):
            confidence = 0.72
        else:
            confidence = 0.6
    else:
        confidence = 0.85 if inline else 0.8
    return {"value": text[:300], "confidence": round(confidence, 4), "reason": None}


def _policy_value(field: str, body: str, inline: bool,
                  heading: str = "") -> dict[str, Any] | None:
    """A policy value, or ``None`` when the body is not a value for that field."""
    decision = _policy_decision(field, body, inline, heading)
    if decision["value"] is None:
        return None
    return {"value": decision["value"], "confidence": decision["confidence"]}


def _value(value: Any, field: str, confidence: float, text: str,
           start: int, end: int, rule: str, **extra) -> dict[str, Any]:
    return {
        "value": value,
        "field": field,
        "confidence": round(confidence, 4),
        "rule": rule,
        "evidence": {"text": text, "start": start, "end": end},
        **extra,
    }


def _sentence_pairs(line: str, inside_parens: bool = False) -> list[tuple[str, int]]:
    """Every (category, weight) on a line written as a list rather than a table.

    Covers both orders — ``Homework 20%, Quizzes 10%`` and ``15% Homework,
    10% Quizzes`` — and the bulleted form ``• WebAssign 15% (quiz)``. A weight
    inside parentheses is a component of the category beside it, not a category
    of its own, unless ``inside_parens`` asks for those components.
    """
    percents = list(_PERCENT.finditer(line))
    if len(percents) < 2:
        return []
    pairs: list[tuple[str, int]] = []
    for match in percents:
        if _in_range(line, match.start("w")):
            continue   # neither end of a "0-1%" range is a category weight
        if not inside_parens and _component_paren(line, match.start("w")):
            continue   # a nested weight is a component, not a category
        before = line[:match.start()]
        if inside_parens:
            # Read the component from inside its own group, not the whole line.
            before = re.split(r"[(\n]", before)[-1]
        if _OF_GRADE.search(before.strip()):
            continue
        left = _PAIR_BREAK.split(before)[-1]
        # "Exams (3 x 20%)" names the category, not the parenthetical.
        left = re.sub(r"\([^)]*$", " ", left)
        left = _LEADING_NOISE.sub("", left).strip(" :-\u2013(")
        right = _PAIR_BREAK.split(line[match.end():])[0]
        if inside_parens:
            right = re.split(r"[)\n]", right)[0]
        right = re.sub(r"\([^)]*\)", " ", right).strip(" :-\u2013.")
        weight = int(float(match.group("w")))
        candidates = [left, right]
        if not left.strip() and right:
            # "20% Exam 1 composed of a written test (15%) ...": the category
            # names the whole, and the parenthetical numbers are its parts.
            candidates.insert(0, re.split(
                r"\b(?:composed|consisting|based|made\s+up|broken|includes?)\b",
                right, maxsplit=1)[0])
        if before.rstrip().endswith("(") or len(_clean_candidate(left).split()) > 4:
            candidates.insert(0, _trailing_category(left))
        for candidate in candidates:
            candidate = _clean_candidate(candidate)
            if len(candidate.split()) <= 4 and _category_ok(candidate):
                pairs.append((candidate, weight))
                break
    return pairs


def _subset_summing_to_100(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The breakdown that adds to 100, when duplicates inflate the raw list.

    A syllabus often prints its weights twice (a table, then a list), so the
    raw sum is 200. The real breakdown is the largest subset that sums to
    about 100; prose about grade cut-offs has no such subset and is rejected.
    """
    total = sum(item["weight"] for item in items)
    if grading_sum_consistent(total):
        return items
    best: list[dict[str, Any]] = []
    best_key: tuple[int, int] | None = None
    count = len(items)
    if count > 12:
        return best
    for mask in range(1, 1 << count):
        subset = [items[index] for index in range(count) if mask & (1 << index)]
        if len(subset) < 2:
            continue
        weight = sum(item["weight"] for item in subset)
        if not grading_sum_consistent(weight):
            continue
        key = (len(subset), -abs(weight - 100))
        if best_key is None or key > best_key:
            best_key, best = key, subset
    if not best and total < 95:
        # A partial breakdown (the rest is in prose elsewhere) is still what the
        # document says; `grading_total_consistent` reports that it does not add up.
        return items
    return best


def _subweight_line(line: str) -> bool:
    """True when a parenthetical carries two weights: a nested breakdown.

    ``Exams 1, 2 (25% lowest, 35% best)`` states one category's internal
    split, not two top-level categories. Reading each number as a category
    invents rows the document does not have.
    """
    return any(len(_PERCENT.findall(group)) >= 2
               for group in re.findall(r"\([^()]*\)", line))


def extract_grading(text: str, engine, min_weight: int = 1,
                    ledger: Ledger | None = None) -> list[dict[str, Any]]:
    """Grading categories and weights, with a sum check.

    Three shapes are read: a label on its own line with the number after it
    (the flattened form of a two-column table), a single category and weight on
    one line, and a list inside a sentence — both orders, parenthesised or not.
    A nested breakdown inside parentheses and a ``0-1%`` range are not
    categories, so neither becomes a row.

    Every candidate that was read and then dropped (a component of a parent
    weight, a duplicate of a category printed twice, a row outside the subset
    that sums to 100) is noted on ``ledger`` when one is given, so the rejected
    values stay visible instead of disappearing from the record.
    """
    def note(item: dict[str, Any], reason: str) -> None:
        if ledger is None:
            return
        ledger.note("grading", f"{item['category']} {item['weight']}%", reason,
                    (item.get("evidence") or {}).get("text", ""))
    found: list[dict[str, Any]] = []
    # (category, weight) -> the parent category that already counts it.
    components: dict[tuple[str, int], str] = {}
    previous: tuple[str, int, int] | None = None
    for line, start, end in _lines(text):
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 300:
            previous = (stripped, start, end)
            continue
        # A table flattened to text puts the category and "20%" on separate lines.
        if _BARE_PERCENT.fullmatch(stripped) and previous:
            category = _clean_candidate(previous[0].rstrip(":"))
            if _category_ok(category) and len(category.split()) <= 4:
                weight = int(float(_PERCENT.search(stripped).group("w")))
                if min_weight <= weight <= 100:
                    found.append({
                        "category": category,
                        "weight": weight,
                        "confidence": 0.82,
                        "evidence": {"text": f"{previous[0][:80]} {stripped}",
                                     "start": previous[1], "end": end},
                    })
        if "%" not in stripped:
            previous = (stripped, start, end)
            continue
        pairs: list[tuple[str, int]] = []
        top: tuple[str, int] | None = None
        match = _WEIGHT_FIRST.match(stripped) or _PERCENT_FIRST.match(stripped)
        if match and not _in_range(stripped, match.start("w")):
            top = (_strip_category_prefix(match.group("cat")), int(match.group("w")))
            pairs.append(top)
        # "WebAssign 15% (Lecture-quizzes 5% + Homework 10%)": keep the category,
        # drop the components. A line with only parenthetical weights has no
        # top-level pair to keep and stays out of the breakdown.
        if top is not None:
            for category, weight in _parenthetical_components(stripped):
                if (category.lower(), weight) != (top[0].lower(), top[1]):
                    components[(category.lower(), weight)] = top[0]
        # A group that carries two weights is a breakdown of the category beside
        # it; reading each number as a category invents rows the document lacks.
        sentence = [] if _subweight_line(stripped) else _sentence_pairs(stripped)
        # "20% Exam 1 composed of a written test (15%) and an oral test (5%)":
        # the parenthetical numbers are components that add up to the category.
        if top is not None and sentence and sum(w for _, w in sentence) == top[1]:
            sentence = []
        pairs.extend(sentence)
        if not pairs:
            # A line states a weight that never became a row: either every
            # number sat in a range or a nested group, or the text beside the
            # weight is too long to be a category name. The second is a real
            # loss and is logged; the first was never a row.
            unparsed = None if _subweight_line(stripped) else _unparsed_weight(stripped)
            if unparsed:
                note({"category": unparsed[0], "weight": unparsed[1],
                      "evidence": {"text": stripped[:160]}},
                     "a weight with no category name beside it")
            previous = (stripped, start, end)
            continue
        confidence = 0.9
        if _SENTENCE_VERB.search(stripped) or "based on" in stripped.lower():
            confidence = 0.8       # weights stated inside running prose
        if len(pairs) >= 6:
            confidence = min(confidence, 0.8)
        for category, weight in pairs:
            if not _category_ok(category):
                note({"category": category, "weight": weight,
                      "evidence": {"text": stripped[:160]}},
                     "the text beside the weight is not a category name")
                continue
            if not min_weight <= weight <= 100:
                note({"category": category, "weight": weight,
                      "evidence": {"text": stripped[:160]}},
                     f"a weight outside {min_weight}-100%")
                continue
            # The document also lists a nested component as its own line. It is
            # already part of the parent weight, so counting it again inflates
            # the breakdown (or, when the numbers happen to add up, double-counts).
            if (category.lower(), weight) in components:
                note({"category": category, "weight": weight,
                      "evidence": {"text": stripped[:160]}},
                     f"component of '{components[(category.lower(), weight)]}'")
                continue
            found.append({
                "category": category,
                "weight": weight,
                "confidence": round(confidence, 4),
                "evidence": {"text": stripped[:160], "start": start, "end": end},
            })
        previous = (stripped, start, end)
    # The same breakdown is often printed twice (table + list); count it once.
    deduped: dict[str, dict[str, Any]] = {}
    for item in found:
        key = (item["category"].lower(), item["weight"])
        if key in components:
            note(item, f"component of '{components[key]}'")
            continue   # a nested component the parent weight already counts
        category = item["category"].lower()
        if category in deduped:
            note(item, f"duplicate of '{deduped[category]['category']}'")
            continue
        deduped[category] = item
    found = list(deduped.values())
    if len(found) < 2:
        for item in found:
            note(item, "the only category found; a breakdown needs at least two")
        return []
    kept = _subset_summing_to_100(found)
    kept_ids = {id(item) for item in kept}
    for item in found:
        if id(item) not in kept_ids:
            note(item, "not in the subset of rows that sums to 100")
    found = kept
    if not found:
        return []
    total = sum(item["weight"] for item in found)
    consistent = grading_sum_consistent(total)
    if consistent:
        # The parts were read off the page and they account for the whole grade;
        # that agreement is evidence about the individual values.
        for item in found:
            item["confidence"] = round(min(0.97, item["confidence"] + 0.05), 4)
    for item in found:
        item["weights_sum"] = total
        item["weights_sum_consistent"] = consistent
    return found


_CREDIT_CAVEAT = re.compile(
    r"\b(prereq\w*|requirement|minimum|up to|at least|max(?:imum)?|elective|repeat)\b", re.I)


def extract_credits(text: str, engine,
                    ledger: Ledger | None = None) -> dict[str, Any] | None:
    """The credit count, preferring a clean statement over a requirements clause."""
    candidates: list[dict[str, Any]] = []
    for line, start, end in _lines(text):
        for pattern in (_CREDITS, _CREDITS_INLINE):
            match = pattern.search(line)
            if not match:
                continue
            number = float(match.group("n"))
            if not 0 < number <= 30:
                continue
            stripped = line.strip()
            if _CREDIT_CAVEAT.search(stripped):
                confidence = 0.7      # "Prereq: 6 credits of history" is not this course
            elif len(stripped) > 90:
                confidence = 0.85     # inside running prose, not a stated credit line
            else:
                confidence = 0.95
            candidates.append(_value(number, "credits", confidence, match.group(0), start, end,
                                     "credit_pattern"))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["confidence"])
    if ledger is not None:
        for item in candidates:
            if item is best:
                continue
            ledger.note("credits", item["value"], "lower confidence than the chosen line",
                        item["evidence"]["text"])
    return best


def _unparsed_weight(line: str) -> tuple[str, int] | None:
    """The first weight whose text could have named a category.

    Only a label-shaped prefix counts: two to ten words of letters, spaces and
    ``&``/``/``/``'``/``-`` with no other punctuation. That keeps grade scales
    (``90.0% = A-``), CSS (``margin: 0 10%``) and running prose out of the
    ledger, while a real category that is simply too long for the reader —
    the case this exists for — is still a visible loss.
    """
    for match in _PERCENT.finditer(line):
        if _in_range(line, match.start("w")):
            continue
        prefix = _clean_candidate(line[:match.start()])
        if not 2 <= len(prefix.split()) <= 10:
            return None
        if not re.fullmatch(r"[A-Za-z][A-Za-z &/'\-]*", prefix):
            return None
        return prefix[:60], int(float(match.group("w")))
    return None


def _in_range(line: str, position: int) -> bool:
    """True when the number at ``position`` is either end of an ``X-Y%`` range."""
    if re.search(r"\d\s*[-–]\s*$", line[max(0, position - 6):position]):
        return True
    return bool(re.match(r"\d*\s*[-–]\s*\d+\s*%", line[position:position + 10]))


def _in_parens(line: str, position: int) -> bool:
    """True when the character at ``position`` sits inside parentheses."""
    return line.count("(", 0, position) > line.count(")", 0, position)


def _parenthetical_components(line: str) -> list[tuple[str, int]]:
    """Weights stated in parentheses as parts of the category beside them."""
    parts: list[tuple[str, int]] = []
    for match in _PERCENT.finditer(line):
        if not _component_paren(line, match.start("w")):
            continue
        opened = line.rfind("(", 0, match.start())
        category = _clean_candidate(_PAIR_BREAK.split(line[opened + 1:match.start()])[-1])
        if category and _category_ok(category):
            parts.append((category, int(float(match.group("w")))))
    return parts


def _component_paren(line: str, position: int) -> bool:
    """True when the parenthesised weight at ``position`` is a component.

    ``homework (20%)`` states the category's weight; ``WebAssign 15%
    (quizzes 5% + homework 10%)`` and ``20% Exam 1 (15%) and (5%)`` state
    parts. A group with two weights, or one that follows a weight already
    stated on the line, is a part.
    """
    if not _in_parens(line, position):
        return False
    opened = line.rfind("(", 0, position)
    closed = line.find(")", position)
    group = line[opened:closed if closed != -1 else len(line)]
    if len(_PERCENT.findall(group)) >= 2:
        return True
    return any(match.end() <= opened and not _in_parens(line, match.start("w"))
               for match in _PERCENT.finditer(line))


def extract_course_code(text: str, engine, window: int = 3000,
                        ledger: Ledger | None = None) -> dict[str, Any] | None:
    """The course code, not the office, room or session code that shares its shape."""
    lines = list(_lines(text))
    head = _norm_space(" ".join(line for line, _, _ in lines[:6]))
    candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    def reject(value: str, reason: str, evidence: str) -> None:
        # No cap: a code-shaped token that was rejected is a loss, and a cap
        # would move it out of the ledger without saying so.
        rejects.append({"value": value, "reason": reason, "evidence": evidence})

    for line, start, end in lines:
        stripped = line.strip()
        if not stripped:
            continue
        labelled = bool(_LABELLED_CODE.search(stripped))
        if start > window and not labelled:
            continue
        # "MWF 1351" and "MTWR 00" are schedules, not catalogue numbers.
        if _DAY_TOKEN.match(stripped) or re.match(
                r"^(?:MWF|MTWTh|MTThF|MWTh|TTh|TuTh|TR|MW|WF)\b", stripped):
            continue
        for match in _COURSE_CODE.finditer(stripped):
            code = match.group(0).strip()
            if match.group("dept") in _CODE_STOP:
                reject(code, "department is a stop word", stripped)
                continue
            number = match.group("num")
            if re.fullmatch(r"(19|20)\d\d", number):
                reject(code, "a year, not a course number", stripped)
                continue
            if _SESSION_CODE.match(code.replace(" ", "")):
                reject(code, "session code", stripped)
                continue
            # "WEL 5.132" is a room; "Office: MAT 328" is the instructor's office.
            tail = stripped[match.end():match.end() + 4]
            if tail.startswith(".") and tail[1:2].isdigit():
                reject(code, "room number", stripped)
                continue
            before = stripped[max(0, match.start() - 14):match.start()].lower()
            if re.search(r"(room|rm\.?|office|bldg|building|hall|lab|in)\s*$", before):
                reject(code, "office or room number", stripped)
                continue
            if _LOCATION_CUE.search(stripped) and not labelled:
                reject(code, "line is about a place, not the course", stripped)
                continue
            if labelled:
                confidence = 0.97
            elif _norm_space(code) in head:
                confidence = 0.93
            elif start < 400:
                confidence = 0.85
            else:
                confidence = 0.5
            candidates.append(_value(code, "course_code", confidence, code, start, end,
                                     "code_pattern"))
    if not candidates:
        if ledger is not None:
            ledger.extend("course_code", rejects)
        return None
    best = max(candidates, key=lambda item: item["confidence"])
    if ledger is not None:
        for item in candidates:
            if item is best:
                continue
            ledger.note("course_code", item["value"], "lower confidence than the chosen line",
                        item["evidence"]["text"])
        ledger.extend("course_code", rejects)
    return best


def extract_instructor(text: str, engine, sections: list[dict[str, Any]],
                       ledger: Ledger | None = None) -> dict[str, Any] | None:
    chosen: dict[str, Any] | None = None
    for line, start, end in _lines(text):
        match = _INSTRUCTOR.match(line)
        if not match:
            continue
        value = _clean_candidate(match.group("v"))
        if len(value) < 3:
            continue
        confidence = 0.95
        if re.search(r"\d|@", value):
            confidence = 0.7    # an email or an office number is not the teacher's name
        candidate = _value(value[:120], "instructor", confidence, value, start, end,
                           "instructor_label")
        if chosen is None:
            chosen = candidate
        elif ledger is not None:
            ledger.note("instructor", value, "later labelled line ignored", line.strip())
    if chosen is not None:
        return chosen

    for section in sections:
        if section.get("label") != "instructor":
            continue
        body = _section_body(text, sections, section)
        if not body or _looks_like_nav(body):
            continue
        # The body often glues the name to an office, an email or a policy
        # paragraph; only the name part is a value.
        head = _clean_candidate(re.split(r"[,\n]", body, maxsplit=1)[0])[:80]
        if not re.match(r"^(?:(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+)?"
                        r"[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3}$", head):
            if ledger is not None and 3 <= len(head) <= 80:
                ledger.note("instructor", head, "section body is not a person's name",
                            section["text"])
            continue
        confidence = 0.8
        if re.search(r"\d|@", head):
            confidence = 0.6
        return _value(head, "instructor", confidence, head,
                      section["start"], section["end"], "instructor_section")
    return None


_SENTENCE_VERB = re.compile(
    r"\b(?:is|are|was|were|will|introduces|covers|examines|explores|focuses|provides|"
    r"includes|serves|aims|offers|requires|designed|intended)\b", re.I)
_TITLE_TAIL = {"of", "and", "the", "for", "to", "in", "with", "a", "an", "or", "on",
               "by", "at", "from", "as", "that", "which"}
_FILE_SHAPE = re.compile(r"\.(?:pdf|docx?|html?|txt|rtf)\b|_|\bdownload\b|\barchive\b", re.I)
# A room, a department or a campus name is a place, not a course name.
_TITLE_PLACE = re.compile(
    r"\b(room|hall|building|bldg|auditorium|department of|university|college of|"
    r"campus|center|centre|library|institute)\b", re.I)
_TITLE_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
# "January 9-13 (Week 1)" heads a row of a schedule, and "Barbour & Wright,
# Chapter 1" names a reading. Both carry a number and name no course.
_TITLE_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\bweek\s*\d+\b|\(\s*week\b", re.I)
_TITLE_READING = re.compile(
    r"\b(?:chapters?|ch\.|sections?|pp?\.|pages?|vol\.?|readings?)\s*\d+", re.I)
# A contact block ("email", "office hours", a phone) is not a course name, and
# neither is a meeting slot: "MWF 1-2 pm, O. Hallatschek" is when and where the
# class meets plus who teaches it, not what the course is called.
_TITLE_CONTACT = re.compile(
    r"\b(?:e-?mail|office hours|phone|tel|fax|appointment)\b"
    r"|\b(?:MTWThF|MTWTh|MWThF|MWTh|MWF|MThF|TuTh|TTh|TR|MW|WF|MTW|WTh)\b"
    r"[^\n]{0,20}\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*[-–—]\s*\d"
    r"|\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\s*(?:&|and|/)?\s*"
    r"(?:mon|tue|wed|thu|fri|sat|sun)?[a-z]*\.?,?\s*\d{1,2}(?::\d{2})?\s*[-–—]",
    re.I)
# A rejected line can still carry the course's own name: what remains after the
# document/task words are stripped. "Fall 2003 Syllabus" leaves nothing;
# "Physics 7A Syllabus, Fall 2018" leaves "Physics 7A".
_LABEL_WORDS = re.compile(
    r"\b(?:course|class|courses|classes|section|sections|lecture|lectures|seminar|"
    r"syllabus|syllabi|outline|policy|policies|information|info|page|of|the|for|and|"
    r"spring|summer|fall|winter|autumn|term|terms|schedule|calendar|meeting|meetings)\b",
    re.I)
TITLE_DOCUMENT_LABEL = "a document label, not a course name"
TITLE_SESSION_LINE = "a session or term line"
# A label that carries a code is the page saying it has a course, so it can vouch
# for a candidate elsewhere in the header block.
_STRONG_LABEL_REASONS = {TITLE_DOCUMENT_LABEL, TITLE_SESSION_LINE}

# The title field's own decision to emit nothing. The evaluation folds losses by
# string, so this vocabulary lives in one place: a reason typed at a call site
# would silently become a new cause in the published table.
TITLE_ABSTENTION = {
    "refused": "no course name: every title-shaped line names something other than the course",
    "unreadable": "no course name: a line shaped like a title could not be read cleanly",
    "empty": "no course name: nothing on the page is shaped like a title",
    "floor": "no course name: the best line is not a printed title",
}
TITLE_BELOW_FLOOR = "below the plausibility floor"
TITLE_WEAKER = "not emitted: weaker than the document's own title line"


def _name_core(line: str) -> str:
    """What is left of a document or session line once its task words are gone."""
    core = re.sub(r"\b(?:19|20)\d{2}\b", " ", line)
    core = _LABEL_WORDS.sub(" ", core)
    return re.sub(r"[\s\-–—|/\\_,:;.()\[\]]+", " ", core).strip()


def _names_a_course(line: str) -> bool:
    """Does a rejected label line name anything course-shaped at all?"""
    core = _name_core(line)
    parts = core.split()
    words = [part for part in parts if part.isalpha() and len(part) >= 2]
    if len(parts) < 2 or not words:
        return False
    return not _TITLE_ADMIN.search(core)


# A catalogue number printed in front of the name ("21-120 : Differential and
# Integral Calculus") or a term/section clause printed after it ("Interpretation
# and Argument, Fall 2005, Section AA") is metadata, not part of the title. The
# name inside the line is what the page calls the course.
_TERM_CLAUSE = re.compile(
    r"\b(?:spring|summer|fall|winter|autumn|semester|term|quarter)\b"
    r"|\bsection\s*[A-Z0-9]+\b|\b(?:19|20)\d{2}\b", re.I)


def _without_metadata(line: str) -> tuple[str, bool]:
    """The name a line carries, with its printed catalogue/term/session metadata
    stripped off, and whether a numeric catalogue code came off the front.
    Empty when too little is left to be a name."""
    name = line.strip()
    prefix = _numeric_prefix(name)
    in_front = prefix is not None
    if prefix:
        name = prefix[1]
    parts = [part.strip() for part in name.split(",")]
    while len(parts) > 1 and _TERM_CLAUSE.search(parts[-1]):
        parts.pop()
    name = ", ".join(parts).strip()
    if len(name.split()) < 2:
        return "", False
    return name, in_front


# "Does this line carry a course code?" is asked four times in this field: for the
# score, for the plausibility floor, for the promotion gate and for the document
# label's own code. It is one question, so it has one owner here. A code is a
# capitalised department token followed by a catalogue number ("Chemistry 1A",
# "Hist. 151"), a line of initials ("M 408C", "P.S. 101"), or a hyphenated
# numeric code ("21-120", "76-101"). An enumerated list marker ("1.") is none of
# these, and a version ("V 1.4") is not either.
# A month is not a department: "May 16" heads a date, not a course.
_CODE_DEPARTMENT = r"(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?)[A-Z][A-Za-z]{1,9}\.?"
_CODE_INITIALS = r"[A-Z]{1,6}(?:\.[A-Z]){0,3}\.?"
_CODE_NUMBER = r"(?!(?:19|20)\d{2}\b)\d{1,5}[A-Za-z]?(?!\.\d)"
_CODE_PATTERNS = (
    re.compile(r"\b" + _CODE_DEPARTMENT + r"[\s\-]?" + _CODE_NUMBER + r"\b"),
    re.compile(r"\b" + _CODE_INITIALS + r"\s" + _CODE_NUMBER + r"\b"),
    re.compile(r"\b(?!(?:19|20)\d{2}\b)\d{2,3}-\d{1,4}\b"),
)
_NUMERIC_CODE = re.compile(r"^(?P<code>\d{2,3}-\d{1,4})(?=\s*[:.\-–—]\s+)")


def catalogue_code(line: str) -> tuple[int, int] | None:
    """Where the line's own catalogue code sits, if it has one."""
    found: tuple[int, int] | None = None
    for pattern in _CODE_PATTERNS:
        match = pattern.search(line)
        if match and (found is None or match.start() < found[0]):
            found = (match.start(), match.end())
    return found


def code_names_a_course(line: str) -> bool:
    """A catalogue code *and* a name beside it: "Chemistry 1A: General
    Chemistry" yes, "HIS 315k External" no, "1. Read the syllabus" no.

    The code has to lead the line (within six tokens), because a number anywhere
    in a registrar row is not the course's own code, and at least two name words
    have to sit beside it: a code with a mode of delivery attached ("External")
    identifies a section, not the course. A colon after the code introduces the
    name, so one word there is a name; a code at the end of the line ("General
    Chemistry 103") counts when a name precedes it.
    """
    head = " ".join(line.split()[:6])
    span = catalogue_code(head)
    if span is None:
        return False
    tail = line[span[1]:]
    after = [word for word in tail.split() if any(ch.isalpha() for ch in word)]
    before = [word for word in line[:span[0]].split() if any(ch.isalpha() for ch in word)]
    if len(after) >= 2 or (len(after) == 1 and tail.lstrip().startswith(":")):
        return True
    return not after and bool(before)


def _numeric_prefix(line: str) -> tuple[str, str] | None:
    """A purely numeric code printed in front of the name ("21-120 : ...") and
    the name that follows it. An enumerated list marker ("1. ...") is not a
    code, so it is never stripped or counted as evidence."""
    match = _NUMERIC_CODE.match(line.strip())
    if not match:
        return None
    rest = re.sub(r"^\s*[:.\-–—]\s+", "", line.strip()[match.end():]).strip()
    return (match.group("code"), rest) if rest else None


def _label_shaped(line: str) -> bool:
    """`label: value` with a short label is metadata, even in a header block."""
    match = _TITLE_ATTRIBUTE.match(line)
    return bool(match) and len(match.group(1).split()) <= 4


# A line the reader could actually see on the page, as opposed to scraped script,
# CSS or markup that a rejection can quote when the page is a raw HTML dump.
_SCRIPT_NOISE = re.compile(
    r"[(){}=;<>]|\bfunction\b|\bvar\b|\bw\[l\]|!:|\bcolor\s*:|\bnavbar\b"
    r"|\bwindow\b|\bdocument\.|[\w]+_[a-z]", re.I)
# What makes a line a piece of page code rather than printed text. Narrower than
# the evidence filter above: parentheses are ordinary in a heading, a brace or an
# assignment is not.
_SCRIPT_LINE = re.compile(r"[{};=]|\bfunction\b|\bwindow\b|\bvar\b|\bw\[l\]|!important")


def _printed(value: Any) -> bool:
    """A heading a person reading the page would see, not a line of code."""
    text = str(value or "").strip()
    if not 2 <= len(text) <= EVIDENCE_CHARS or _SCRIPT_NOISE.search(text):
        return False
    return sum(ch.isalnum() or ch.isspace() for ch in text) / len(text) >= 0.85


def _first_printed_line(text: str) -> str:
    """The first line a reader would see, when no rejected line is printable."""
    for line, _, _ in _lines(text):
        if _printed(line):
            return line.strip()
        if line.strip() and len(line.strip()) > 90:
            break
    return ""


def _best_refusal(rejected: list[dict[str, Any]], strong: list[tuple[str, int]],
                  named: list[str]) -> str:
    """The line a reader should look at when the title field comes back empty.

    Preference is for a printed line, and among those for one that carries the
    page's catalogue code: that is the line that named the document, which is the
    decision the empty value is reporting.
    """
    for value, _ in strong:
        if _printed(value):
            return value
    for item in rejected:
        if _printed(item.get("value")):
            return str(item["value"])
    for value in named:
        if _printed(value):
            return value
    return ""


# A person's name (the instructor's line, a staff list) is not a course name. The
# honourific, the credential or a middle initial is what distinguishes it from a
# two-word course name like "Organic Chemistry".
_TITLE_PERSON = re.compile(
    r"^(?:dr|prof|professor)\.?\s+[A-Z]"
    r"|^[A-Z][\w'\-]+\s+(?:(?:[A-Z]\.|[A-Z][\w'\-]+)\s+){0,2}[A-Z][\w'\-]+"
    r"\s*,?\s*(?:ph\.?d|m\.?s|m\.?a|m\.?ed|ed\.?d|b\.?s|b\.?a)\.?$", re.I)
_TITLE_PERSON_INITIAL = re.compile(r"^(?:Dr\.?\s+)?[A-Z]\.\s+[A-Z][\w'\-]+")
# "Paul Light (Brookings Institution)" is a speaker with their affiliation, and
# the affiliation is what makes it a person rather than a course name.
_TITLE_AFFILIATION = re.compile(
    r"^[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){1,2}\s*\(\s*[^)]*"
    r"\b(?:University|College|Institute|Institution|School|Department|Center|Centre|"
    r"Foundation|Group|Inc|LLC|Corp)\b[^)]*\)$")


# "Lecture 3, Fall 2011" names a session, not a course.
_TITLE_ADMIN = re.compile(
    r"^(?:lecture|section|recitation|lab(?:oratory)?|seminar|class|term|unit|week|module)\b"
    r"|\b(?:fall|spring|summer|winter|autumn)\s+(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\s*$", re.I)


def title_plausible(text: str) -> bool:
    """A course name, not a file name and not a sentence that starts capitalised."""
    words = text.split()
    if not 2 <= len(words) <= 10:
        return False
    # "Final Exam 15%" is a grading row. Course names are not weights, and a
    # weight line that outscored the real title produced a course called
    # "Final Exam 15%" on a live syllabus.
    if _TITLE_WEIGHT.search(text):
        return False
    if words[-1].lower() in _TITLE_TAIL:
        return False
    if _FILE_SHAPE.search(text):
        return False
    # "CH301 - McCord - Syllabus - Spring 2011" is a file name pasted as a heading.
    if text.count(" - ") >= 2:
        return False
    if _SENTENCE_VERB.search(text):
        return False
    return True


def extract_course_name(text: str, engine, sections: list[dict[str, Any]],
                        ledger: Ledger | None = None) -> dict[str, Any] | None:
    """The course name: repeated early text, not a generic heading.

    A page repeats its course name in the title, the breadcrumb and the header,
    while a section heading like "Catalog Description" appears once. Repetition
    plus a length and navigation filter separates the two.
    """
    candidates: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    # Lines rejected as the document's own label that still name something
    # (`named`) or carry a catalogue code (`strong`). A strong line means the page
    # did name a course, just not in a title slot: if no surviving candidate is
    # corroborated, the field abstains instead of promoting whatever came next.
    named: list[str] = []
    strong: list[tuple[str, int]] = []
    # Lines dropped without a refusal reason: they are not evidence that the page
    # has no course name, only that this reader could not read them as one.
    unread: list[str] = []

    def reject(value: str, reason: str, at: int | None = None) -> None:
        if ledger is not None:
            rejected.append({"value": value, "reason": reason,
                             "evidence": "text in the first 800 characters"})
        if reason == TITLE_DOCUMENT_LABEL and _names_a_course(value):
            named.append(value)
        # A line vouches for the page's own code only if it also *names* a course.
        # "Class meets Monday 8:00-8:50 in Room 118" carries a code-shaped number
        # and was being read as proof that a name was printed, which then blocked
        # the real title ("AP Chemistry") for being weaker than that line.
        if (reason in _STRONG_LABEL_REASONS and catalogue_code(_name_core(value))
                and _names_a_course(value)
                and not _TITLE_CLOCK.search(value) and not _TITLE_DATE.search(value)
                and not _TITLE_PLACE.search(value)):
            strong.append((value, at if at is not None else -1))

    def abstain(kind: str, evidence: str) -> None:
        """The field's own decision: it emits nothing, and this is why."""
        if ledger is None:
            return
        ledger.extend("course_title", rejected)
        ledger.note("course_title", None, TITLE_ABSTENTION[kind],
                    evidence[:EVIDENCE_CHARS])

    # A raw page often glues its heading to a script block on one line. Read as
    # one line the heading is unreadable, and the field would report the page as
    # having no title when it prints one. The printed runs are what a reader sees,
    # so they are what the reader reads; the original text is still what grounds
    # the value, so nothing can be invented by the split.
    view = "\n".join(
        "\n".join(part for part in re.split(r"\s{2,}", line) if part.strip())
        if _SCRIPT_LINE.search(line) else line
        for line in text.splitlines())

    # A document names itself at the top; searching the whole page lets a
    # description sentence outscore the title. The header block is the few
    # printed lines around the line that carries the document's own code.
    non_empty = [start for line, start, _ in _lines(view[:800]) if _clean_candidate(line)]
    order = {start: index for index, start in enumerate(non_empty)}

    def in_header(start: int) -> bool:
        here = order.get(start)
        if here is None or not strong:
            return False
        return any(at in order and abs(here - order[at]) <= 3 for _, at in strong)

    for line, start, end in _lines(view[:800]):
        candidate = _clean_candidate(line)
        # "21-120 : Differential and Integral Calculus" and "Interpretation and
        # Argument, Fall 2005, Section AA" both print the name with metadata
        # attached. The evidence stays the whole printed line.
        # The number itself is the line's evidence that it names a course, so it is
        # kept as a flag even though it is not part of the emitted name. A list
        # marker is not a number in this sense and is never stripped or flagged.
        without_metadata, code_in_front = _without_metadata(candidate)
        if without_metadata and without_metadata != candidate:
            candidate = without_metadata
        # Ligatures ("O\ufb03ce") hide the label shape these filters look for.
        probe = _fold(candidate)
        if not (4 <= len(candidate) <= 90):
            continue
        if candidate.lower().strip(":") in _GENERIC_TITLES:
            reject(candidate, "generic section heading")
            continue
        if _TITLE_DOCUMENT.search(candidate):
            reject(candidate, TITLE_DOCUMENT_LABEL, start)
            continue
        attribute = _TITLE_ATTRIBUTE.match(probe)
        if attribute and attribute.group(1).split()[-1].lower() in _ATTRIBUTE_LABELS:
            reject(candidate, "a labelled attribute, not a course name")
            continue
        span = catalogue_code(candidate)
        if span is not None and span == (0, len(candidate)):
            reject(candidate, "a course code, not a name")
            continue
        if _TITLE_ADMIN.search(probe):
            reject(candidate, TITLE_SESSION_LINE, start)
            continue
        if _GENERIC_HEADING.search(probe):
            reject(candidate, "a section heading")
            continue
        if _TITLE_PLACE.search(probe):
            reject(candidate, "a place name")
            continue
        if _TITLE_CLOCK.search(probe):
            reject(candidate, "a clock time")
            continue
        if _TITLE_DATE.search(probe):
            reject(candidate, "a date or week line, not a course name")
            continue
        if _TITLE_READING.search(probe):
            reject(candidate, "a reading or page reference, not a course name")
            continue
        if _TITLE_CONTACT.search(probe):
            reject(candidate, "a contact line")
            continue
        if (_TITLE_PERSON.search(probe) or _TITLE_PERSON_INITIAL.search(probe)
                or _TITLE_AFFILIATION.match(candidate)):
            reject(candidate, "a person's name, not a course name")
            continue
        # A title starts with a capital; description fragments usually do not.
        if not probe[:1].isupper() or probe.endswith((",", ";")):
            unread.append(candidate)
            continue
        if not re.search(r"[A-Za-z]{3,}", probe) or _looks_like_nav(probe):
            unread.append(candidate)
            continue
        if re.search(r"(https?://|www\.|\.edu\b|@)", probe, re.I):
            unread.append(candidate)
            continue
        # "Course Website: ..." is a labelled link, not a name.
        if _TITLE_LINK.match(probe):
            reject(candidate, "a labelled link, not a course name")
            continue
        # Digits belong in a course name ("Chemistry 1A"); punctuation does not.
        if sum(ch.isalnum() or ch.isspace() for ch in candidate) / len(candidate) < 0.85:
            unread.append(candidate)
            continue
        if not title_plausible(probe):
            unread.append(candidate)
            continue
        # "Collaboration, Not Competition" survives the shape filters but is a
        # heading; prose with the same conjunction is caught by the sentence rule
        # above, so this only sees title-shaped lines.
        if _TITLE_SLOGAN.search(probe):
            reject(candidate, "a heading, not a course name")
            continue
        entry = candidates.setdefault(candidate, {"count": 0, "start": start, "end": end,
                                                 "text": line.strip(),
                                                 "code_in_front": False})
        entry["count"] += 1
        entry["code_in_front"] = entry["code_in_front"] or code_in_front
    if not candidates:
        # Nothing survived. An empty title is a decision, not a silence, and it is
        # only the right decision when every title-shaped line on the page was
        # refused as *not a course name*. When nothing was shaped like a title at
        # all the field says that instead, so a reader can tell "this page has no
        # course name" from "a name was spelled in a way this reader cannot read".
        if ledger is not None:
            refused = bool(strong or named
                           or [item for item in rejected if _printed(item["value"])])
            unreadable = [line for line in unread if _printed(line)]
            if refused:
                kind, evidence = "refused", _best_refusal(rejected, strong, named)
            elif unreadable:
                # Saying "nothing here is shaped like a title" would be false when a
                # line nearly was: the reader has to see that line too.
                kind, evidence = "unreadable", unreadable[0]
            else:
                kind, evidence = "empty", ""
            abstain(kind, evidence or _first_printed_line(text))
        return None
    scored = []
    for candidate, entry in candidates.items():
        # A course name usually carries a catalogue code ("Chemistry 1A"); a
        # section heading never does.
        has_code = ((code_names_a_course(candidate) and not candidate.isupper())
                    or entry["code_in_front"])
        score_now = _probability(engine, candidate, "course_title")
        position_bonus = 1.0 + 0.25 * max(0.0, 1.0 - entry["start"] / 4000)
        score = (score_now + (2.0 if has_code else 0.0)) * \
                (1 + 0.2 * (entry["count"] - 1)) * position_bonus
        scored.append((score, score_now, entry["count"], candidate, entry, has_code))
    scored.sort(key=lambda item: item[0], reverse=True)

    def corroborated(item) -> bool:
        """A line with more evidence than winning by default: a catalogue code, a
        repeat, or a name-length line in the header block that carries the page's
        own code. Menu entries ("Exam Answers") are two words and do not qualify."""
        _, _, times, text, where, _ = item
        name_shaped = not _label_shaped(text)
        # A line that carries its own catalogue code ("Chemistry 1A: General
        # Chemistry") vouches for itself even when the colon makes it look like a
        # label: the code is the course, and the text after it is the name.
        carries_code = ((code_names_a_course(text) and not text.isupper())
                        or where.get("code_in_front"))
        return bool(carries_code or times >= 2
                    or (name_shaped and in_header(where["start"])
                        and len(text.split()) >= 3))

    # The page's own label carried a course code, so a name was printed. If the
    # top-scoring line is not in that header block and is not otherwise corroborated,
    # the header block's own line wins; if there is none, the field abstains rather
    # than promoting whatever came next. A wrong title is worse than none.
    promoted = False
    if strong and not corroborated(scored[0]):
        preferred = [item for item in scored if corroborated(item)]
        if not preferred:
            if ledger is not None:
                _, _, _, text, where, _ = scored[0]
                abstain("refused", _best_refusal(rejected, strong, named) or where["text"]
                        or _first_printed_line(text))
                ledger.note("course_title", text, TITLE_WEAKER,
                            where["text"][:EVIDENCE_CHARS])
            return None
        scored = preferred + [item for item in scored if item not in preferred]
        promoted = True
    _, model_score, count, candidate, entry, has_code = scored[0]
    # A line chosen from the header block because the page's own code vouches for it
    # is not a weak lone candidate: the plausibility floor is for lines that won by
    # default, and this one did not.
    after_label = promoted and not _label_shaped(candidate) and len(candidate.split()) >= 3
    # The floor is for lines that won by default. A code with a name attached is
    # checkable evidence, and so is a line the header block's own code vouches for;
    # a bare code with a delivery mode ("HIS 315k External") is not.
    if (not code_names_a_course(candidate) and not entry["code_in_front"]
            and model_score < 0.25 and not after_label):
        if ledger is not None:
            # The best line still did not clear the plausibility floor; it is a
            # loss like any other and needs its reason, and the field needs its
            # own: the reader has to see that a title-shaped line was found and
            # deliberately not emitted, not that the page was quietly empty.
            ledger.note("course_title", candidate, TITLE_BELOW_FLOOR,
                        entry["text"][:120])
            abstain("floor", entry["text"])
        return None
    if ledger is not None:
        ledger.extend("course_title", rejected)
        for score, _, _, alt, _, _ in scored[1:]:
            ledger.note("course_title", alt, "ranked below the chosen line",
                        f"score {score:.2f}")
    # The confidence describes the evidence for *this* value: a code and a
    # repeated header are checkable facts about the line, not the classifier's
    # own confidence in a label it gives every short heading.
    repeated = count >= 2
    if has_code and repeated:
        confidence = 0.92
    elif has_code:
        confidence = 0.86
    elif repeated:
        confidence = 0.82
    elif after_label:
        confidence = 0.8
    else:
        confidence = 0.7
    # Grounding: the value returned is text the document itself contains, so it can
    # be found again by a reader, rather than assembled from a file name or a guess.
    if _norm_space(candidate) not in _norm_space(text):
        return None
    if has_code:
        rule = "code_bearing_title"
    elif after_label:
        rule = "title_after_label"
    else:
        rule = "repeated_title_line"
    return _value(candidate, "course_title", confidence, candidate, entry["start"],
                  entry["end"], rule, occurrences=entry["count"], grounded=True)


_MEETING_LINE = re.compile(
    r"(?<![\d.])(?P<start>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?)\s*(?:-{1,3}|–|—|to)\s*"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?)(?![\d.])", re.I)
_PLACE = re.compile(r"\bin\s+(?P<v>[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*)*\s+(?:Hall|Building|Center|Room|Auditorium))")
# Real timetable rooms rarely say "Room": they are a building code and a number
# (MM 103), a building name and a number (Hewlett 201), or a number and a name
# (1351 Chemistry). The keyword-only pattern above misses all three.
_ROOM_CODE = re.compile(r"\b(?P<v>[A-Z]{2,6}[- ]?\d{1,3}(?:\.\d{1,3})?[A-Z]?)\b")
_ROOM_NAMED = re.compile(
    r"\b(?P<v>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:Room\s+)?\d{1,4}[A-Z]?)\b")
_ROOM_NUMBER_NAME = re.compile(r"\b(?P<v>\d{2,4}[A-Z]?\s+[A-Z][a-z]{3,})\b")
_ROOM_STOP = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|"
    r"december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|fall|spring|summer|winter|"
    r"week|weeks|semester|quarter|credit|credits|unit|units|page|pages|section|sections|"
    r"percent|am|pm|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lecture|lectures|"
    r"lab|labs|recitation|recitations|class|classes|discussion|tutorial|exam|exams|quiz|"
    r"quizzes|problem|problems|homework|chapter|part|module|modules|assignment|assignments|"
    r"lecture\s+\d+|section\s+\d+)\b", re.I)
_ROOM_GLUED = re.compile(r"\b(?P<v>\d{3,4}[A-Z][a-z]{2,})\b")
_DAY_CONCAT = re.compile(r"(?<![A-Za-z])(?P<run>(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun){2,5})(?![A-Za-z])")
_DAY_LETTERS = re.compile(r"(?<![A-Za-z])(?P<run>[MTWRFS](?:\s*[,/]\s*[MTWRFS]){2,})(?![A-Za-z])")
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
              "\ufb05": "ft", "\u00a0": " ", "\u2009": " ", "\u202f": " "}
_SINGLE_DAY = {"M": "MO", "T": "TU", "W": "WE", "R": "TH", "F": "FR", "S": "SA"}
_DAY_WORD = re.compile(r"\b(?:Mon(?:day)?s?|Tues?(?:day)?s?|Wed(?:nesday)?s?|"
                       r"Thu(?:rs?(?:day)?)?s?|Fri(?:day)?s?|Sat(?:urday)?s?|Sun(?:day)?s?)\b",
                       re.I)
_DAY_NAME = (r"(?:Mon(?:day)?|Tues?(?:day)?|Wed(?:nesday)?|Thu(?:rs?(?:day)?)?|Fri(?:day)?|"
             r"Sat(?:urday)?|Sun(?:day)?)s?")
_DAY_PHRASE = re.compile(
    r"(?<![A-Za-z])" + _DAY_NAME +
    r"(?:\s*(?:,|and|&|through|thru|to|-|–|/)\s*" + _DAY_NAME + r")+(?![A-Za-z])", re.I)
_DAY_ORDER = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
_DAY_WORDS = {"mon": "MO", "tue": "TU", "wed": "WE", "thu": "TH", "fri": "FR",
              "sat": "SA", "sun": "SU"}
_OFFICE_HOURS_LINE = re.compile(r"office\s*hours?|\boh\s*:", re.I)
# An exam sits in its own field: its date and clock time are not class meetings.
_EXAM_LINE = re.compile(r"\b(?:final|midterm|make-?up)\s+exam|\bexam(?:ination)?s?\s*\d*\s*:"
                        r"|\bfinal\s+exam", re.I)


def _fold(text: str) -> str:
    """Fold typographic ligatures so "O\ufb03ce Hours" is still read as office hours."""
    for bad, good in _LIGATURES.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def _concat_days(line: str) -> list[str]:
    match = _DAY_CONCAT.search(line)
    if not match:
        return []
    run = match.group("run")
    codes: list[str] = []
    for index in range(0, len(run), 3):
        code = _DAY_WORDS.get(run[index:index + 3].lower())
        if code and code not in codes:
            codes.append(code)
    return [code for code in _DAY_ORDER if code in codes]


def _letter_days(line: str) -> list[str]:
    match = _DAY_LETTERS.search(line)
    if not match:
        return []
    codes = [_SINGLE_DAY[letter.upper()] for letter in re.findall(r"[MTWRFS]", match.group("run"))]
    return [code for code in _DAY_ORDER if code in set(codes)]


def _spelled_days(line: str) -> list[str]:
    """Days from a written run such as "Monday, Wednesday, Friday" or "TueThu"."""
    phrase = _DAY_PHRASE.search(line)
    if not phrase:
        return []
    text = phrase.group(0)
    words = [match.group(0).lower()[:3] for match in _DAY_WORD.finditer(text)]
    codes: list[str] = []
    for word in words:
        code = _DAY_WORDS.get(word)
        if code and code not in codes:
            codes.append(code)
    if len(codes) == 2 and re.search(r"\bthrough\b|\bthru\b", text, re.I):
        first, last = sorted(codes, key=_DAY_ORDER.index)
        codes = _DAY_ORDER[_DAY_ORDER.index(first): _DAY_ORDER.index(last) + 1]
    return [code for code in _DAY_ORDER if code in codes]


def _room_from(line: str, after: int = 0) -> str | None:
    """The room a schedule line names, in any of the usual shapes."""
    for pattern in (_PLACE, _ROOM):
        match = pattern.search(line)
        if match:
            return match.group("v").strip()
    tail = line[after:] if after else line
    for pattern in (_ROOM_CODE, _ROOM_NAMED, _ROOM_NUMBER_NAME, _ROOM_GLUED):
        for match in pattern.finditer(tail):
            value = match.group("v").strip()
            if _ROOM_STOP.search(value):
                continue
            last = value.split()[-1]
            # A four-digit number is a year in prose, but a room number in a timetable.
            if re.fullmatch(r"\d{4}", last) and 1600 <= int(last) <= 2100:
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}", value):
                continue
            # "MWF 12" is a day token that swallowed a clock hour, not a room.
            letters = re.match(r"[A-Za-z]+", value)
            if letters and (_DAY_TOKEN.fullmatch(letters.group(0))
                            or _DAY_CONCAT.fullmatch(letters.group(0))):
                continue
            return value
    return None


def _clock(value: str) -> str | None:
    """Normalise a clock time; a bare 1-6 is read as afternoon, as timetables do."""
    match = re.match(r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]\.?m\.?)?", value.strip(), re.I)
    if not match:
        return None
    hour = int(match.group("h"))
    minute = int(match.group("m") or 0)
    meridiem = (match.group("ap") or "").lower().replace(".", "")
    if meridiem.startswith("p") and hour < 12:
        hour += 12
    elif not meridiem and 1 <= hour <= 6:
        hour += 12
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _days_in(line: str) -> list[str]:
    """Weekdays in a schedule line, however the document writes them."""
    token = _DAY_TOKEN.search(line)
    if token:
        return _days_from_token(token.group("days"))
    for reader in (_concat_days, _spelled_days, _letter_days):
        days = reader(line)
        if days:
            return days
    return []


def _day_groups(line: str) -> int:
    return len(_DAY_PHRASE.findall(line)) + len(_DAY_CONCAT.findall(line)) \
        + len(_DAY_TOKEN.findall(line)) + len(_DAY_LETTERS.findall(line))


def _meeting_text(item: dict[str, Any]) -> str:
    """One line for a meeting, used when the value is logged or filtered."""
    parts = [*item["days"], item.get("start_time") or "", item.get("end_time") or "",
             item.get("room") or ""]
    return " ".join(part for part in parts if part)


def extract_meetings(text: str, engine,
                     ledger: Ledger | None = None) -> list[dict[str, Any]]:
    """Meeting days, time and room from schedule lines.

    Read line by line rather than through the schedule reader's slot
    association: that reader clears weekdays when one day group owns several
    time slots, which is right for a calendar but loses the value here. A line
    such as ``MWF 9:10-10:00 or 11:10-12:00 in Pimentel Hall`` yields one
    meeting per slot, all sharing the days, marked as alternatives.

    A line that reads like a schedule and produces no meeting (an exam or
    office-hours line, several day groups, a weekday the group does not cover,
    a range that is a date, a repeat of a slot already emitted) is noted on
    ``ledger`` with the reason, so a dropped slot stays visible.
    """
    extracted: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def drop(value: Any, reason: str, line: str) -> None:
        if ledger is not None:
            ledger.note("meetings", str(value)[:120], reason, line)

    for line, start, end in _lines(text):
        probe = _fold(line)
        if _OFFICE_HOURS_LINE.search(probe):
            drop(line.strip(), "an office-hours line, not a class meeting", line)
            continue
        if _EXAM_LINE.search(probe):
            drop(line.strip(), "an exam line, not a class meeting", line)
            continue
        # Two separate day groups on one line describe opening hours, not a class.
        if _day_groups(probe) > 1:
            drop(line.strip(), "several day groups on one line", line)
            continue
        days = _days_in(probe)
        if not days:
            continue
        # A weekday the resolved group does not account for means the line lists
        # several schedules ("Monday-Thursday 2-7pm and Friday 2-5pm").
        listed = {_DAY_WORDS[word.group(0).lower()[:3]]
                  for word in _DAY_WORD.finditer(probe)
                  if word.group(0).lower()[:3] in _DAY_WORDS}
        if listed and not listed <= set(days):
            drop(" ".join(days), "a weekday outside the day group", line)
            continue
        token = _DAY_TOKEN.search(probe)
        ranges = list(_MEETING_LINE.finditer(probe))
        room = _room_from(probe, ranges[-1].end() if ranges else 0)
        if not ranges:
            # "TTh in WEL 2.224" names a place and days but no clock time.
            if not room:
                drop(" ".join(days), "days with no clock time or room", line)
                continue
            key = (tuple(days), None, None, room)
            if key not in seen:
                seen.add(key)
                extracted.append({
                    "days": days, "start_time": None, "end_time": None, "room": room,
                    "alternatives": False,
                    "confidence": 0.82,
                    "evidence": {"text": line.strip()[:160], "start": start, "end": end},
                })
            else:
                drop(f"{' '.join(days)} {room}", "duplicate of a meeting already emitted",
                     line)
            continue
        # The slot's own shape is the evidence: days, a clock range and often a room.
        confidence = 0.95 if room else 0.9
        for match in ranges:
            raw = match.group(0)
            # "2.10-3" is a date, not a meeting: a real slot names a minute or a meridiem,
            # or is written in the bare-hour form ("MWF 11-12") next to a day token.
            bare_hours = bool(token) and all(
                re.fullmatch(r"\d{1,2}\s*(?:[ap]\.?m\.?)?", part.strip(), re.I)
                for part in (match.group("start"), match.group("end")))
            if ":" not in raw and not re.search(r"[ap]\.?m\.?", raw, re.I) and not bare_hours:
                drop(raw, "a range that is a date, not a class time", line)
                continue
            start_time, end_time = _clock(match.group("start")), _clock(match.group("end"))
            if not start_time or not end_time:
                drop(raw, "a clock time that could not be parsed", line)
                continue
            key = (tuple(days), start_time, end_time, room)
            if key in seen:
                drop(f"{' '.join(days)} {start_time}-{end_time} {room or ''}".strip(),
                     "duplicate of a meeting already emitted", line)
                continue
            seen.add(key)
            extracted.append({
                "days": days,
                "start_time": start_time,
                "end_time": end_time,
                "room": room,
                "alternatives": len(ranges) > 1,
                "confidence": round(confidence, 4),
                "evidence": {"text": line.strip()[:160], "start": start, "end": end},
            })
    return extracted


_HEADING_FILLER = {"and", "of", "for", "the", "to", "with", "in", "on", "a", "an", "&"}


def _heading_like(line: str) -> bool:
    """A short capitalised line that names a section rather than saying something."""
    if not (2 <= len(line) <= 48) or line.endswith("."):
        return False
    if any(char.isdigit() for char in line) or ":" in line:
        return False
    words = line.split()
    if not 1 <= len(words) <= 5:
        return False
    return all(word[0].isupper() or word.lower() in _HEADING_FILLER for word in words)


def _section_body(text: str, sections: list[dict[str, Any]], section: dict[str, Any],
                  limit: int = 400) -> str:
    """The section's own text, stopped by the next heading even if it was not typed.

    Section detection misses headings it does not recognise ("Required Materials"
    on a syllabus that has no such field type), and without this cut the previous
    section swallows the next one's prose.
    """
    index = sections.index(section)
    end = len(text)
    for later in sections[index + 1:]:
        if later["start"] > section["end"]:
            end = later["start"]
            break
    lines = text[section["end"]:end].splitlines()
    kept: list[str] = []
    seen = 0
    for position, line in enumerate(lines):
        stripped = line.strip()
        # A labelled line after the value ("Course Dates: May 16") opens the next
        # field; without the cut the section swallows it and the value comes back
        # as the following section's text.
        if stripped and any(item.strip() for item in kept) \
                and _NEXT_SECTION_LABEL.match(stripped):
            break
        if (seen >= 30 and stripped and _heading_like(stripped)
                and kept and not kept[-1].strip()):
            break
        kept.append(line)
        seen += len(stripped)
    return re.sub(r"\s+", " ", "\n".join(kept)).strip()[:limit]


def type_sections(text: str, engine, context_engine=None, threshold: float = 0.8,
                  limit: int = 60) -> list[dict[str, Any]]:
    """Classify each detected section, heading first, body as fallback."""
    typed: list[dict[str, Any]] = []
    sections = detect_document_sections(text)[:limit]
    for section in sections:
        heading = section["text"]
        if context_engine is not None:
            from .field_model import classify_section
            body = _section_body(text, sections, section, limit=300)
            decision = classify_section(heading, body, engine, context_engine,
                                        threshold=threshold)
            label, confidence = decision["label"], decision["confidence"]
            source = decision["source"]
        else:
            result = engine.classify(heading, top_k=1)
            label, confidence, source = result["label"], result["confidence"], "heading"
        typed.append({**section, "label": label, "confidence": round(confidence, 4),
                      "source": source})
    return typed


def extract_record(text: str, engine, context_engine=None, threshold: float = 0.8,
                   min_confidence: float = 0.0) -> dict[str, Any]:
    """The structured record: what the document says, with evidence and confidence."""
    cleaned, _ = clean_html(text)
    sections = type_sections(cleaned, engine, context_engine, threshold)
    fields: dict[str, Any] = {}
    # Everything read and then not emitted: a lower-ranked candidate, a
    # component, a duplicate, a section that was overwritten, or a value below
    # the threshold. Nothing that looked like a value leaves without a trace.
    suppressed = Ledger()

    def emit(name: str, value: dict[str, Any] | None,
             text: Any = None) -> dict[str, Any] | None:
        """The value when it clears the threshold; otherwise it is logged once."""
        if value is None or value["confidence"] >= min_confidence:
            return value
        suppressed.note(name, value["value"] if text is None else text,
                        "below min_confidence", value["evidence"]["text"])
        return None

    for name, getter in (
        ("course_title", lambda: extract_course_name(cleaned, engine, sections,
                                                     ledger=suppressed)),
        ("course_code", lambda: extract_course_code(cleaned, engine, ledger=suppressed)),
        ("credits", lambda: extract_credits(cleaned, engine, ledger=suppressed)),
        ("instructor", lambda: extract_instructor(cleaned, engine, sections,
                                                  ledger=suppressed)),
    ):
        kept = emit(name, getter())
        if kept is not None:
            fields[name] = kept

    grading: list[dict[str, Any]] = []
    for item in extract_grading(cleaned, engine, ledger=suppressed):
        if emit("grading", item, f"{item['category']} {item['weight']}%") is not None:
            grading.append(item)
    meetings: list[dict[str, Any]] = []
    for item in extract_meetings(cleaned, engine, ledger=suppressed):
        if emit("meetings", item, _meeting_text(item)) is not None:
            meetings.append(item)

    policies: dict[str, Any] = {}
    by_field: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        target = _SECTION_VALUES.get(section.get("label"))
        if not target:
            continue
        body, inline = _policy_body(cleaned, sections, section)
        decision = _policy_decision(target, body, inline, section["text"])
        by_field.setdefault(target, []).append({
            "value": decision["value"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "section": section["text"][:120],
            "evidence": {"text": section["text"], "start": section["start"],
                         "end": section["end"]},
        })
    def status_of(entry: dict[str, Any],
                  chosen: dict[str, Any] | None) -> tuple[str, str | None]:
        """How one section ended up, for both the record's candidates and its losses."""
        if entry is chosen:
            return "chosen", None
        if entry["value"] is None:
            return "rejected", entry["reason"]
        if entry["confidence"] < min_confidence:
            return "below min_confidence", "below min_confidence"
        return "overwritten", "a later section supplied the value"

    for target, entries in by_field.items():
        accepted = [entry for entry in entries
                    if entry["value"] is not None and entry["confidence"] >= min_confidence]
        chosen = accepted[-1] if accepted else None
        for entry in entries:
            status, reason = status_of(entry, chosen)
            if status != "chosen":
                suppressed.note(target, entry["value"], reason, entry["section"])
        if chosen is None:
            continue
        policies[target] = {
            "value": chosen["value"],
            "field": target,
            "confidence": chosen["confidence"],
            "evidence": chosen["evidence"],
            "candidates": [
                {"value": entry["value"], "confidence": entry["confidence"],
                 "section": entry["section"],
                 "status": status_of(entry, chosen)[0],
                 "reason": entry["reason"]}
                for entry in entries
            ],
        }

    record = {
        "course": {key: fields[key] for key in ("course_title", "course_code", "credits",
                                                "instructor") if key in fields},
        "grading": grading,
        "meetings": meetings,
        "policies": policies,
        "suppressed": suppressed.losses(),
        "sections_typed": len(sections),
        "min_confidence": min_confidence,
    }
    if grading:
        total = grading[0]["weights_sum"]
        record["grading_total_percent"] = total
        record["grading_total_consistent"] = grading_sum_consistent(total)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract a structured record (course, meetings, grading, policies) from text")
    parser.add_argument("path", help="UTF-8 text file, or - for stdin")
    parser.add_argument("--checkpoint", default="artifacts/field-classifier-minilm")
    parser.add_argument("--context-checkpoint", default="artifacts/field-classifier-context",
                        help="Heading+value model; omit to type headings only")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Heading confidence below which the value model takes over")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Filter on the model's field probability; see "
                             "artifacts/field-extract/EXTRACTION_EVAL.md for how the default "
                             "was chosen and what it certifies")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    import sys

    import torch

    from .field_model import FieldEngine

    text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FieldEngine(args.checkpoint, device=device)
    context = None
    if args.context_checkpoint and Path(args.context_checkpoint).exists():
        context = FieldEngine(args.context_checkpoint, device=device)
    record = extract_record(text, engine, context, threshold=args.threshold,
                            min_confidence=args.min_confidence)
    record["source"] = args.path
    record["checkpoint"] = str(args.checkpoint)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
