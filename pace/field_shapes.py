"""Mechanical value shapes that should never reach the model.

The field classifier is a semantic model: it decides whether a line is a grading
policy or a course description. It is not the right tool for a bare ``Room 303``
or ``Wed 2-4pm``, where the shape alone is the answer — and measurement showed it
getting both wrong at 95–99% confidence when no heading was present:

    Room 303                      -> course_code  95.08
    Dr. Johnson                   -> course_code  99.39
    Wed 2-4pm                     -> credits      99.32

A fine-tune on generated shapes fixed those nine probes (all at 99.9%) but cost
six macro-F1 points on new syllabus documents, because averaging sharp pattern
knowledge into the semantic weights pushes out something else. The better split is
the one this module implements: resolve the shapes deterministically, and keep the
model for what actually needs reading.

Rules here fire only on shapes where there is nothing left to interpret, and they
return the reason so a caller can audit the decision. Anything ambiguous — a bare
number, ``Room capacity 30``, ``Office hours by appointment`` — is left to the
model. This is deliberately narrow; widening it is how rule layers start
overriding the model on text they do not understand.
"""
from __future__ import annotations

import re

#: (label, compiled pattern). Order matters: the first match wins.
_DAYS = r"(?:M|T|W|Th|F|Sa|Su|Mon|Tue|Wed|Thu|Fri|Sat|Sun|MWF|TTh|TR|MW|M-F|daily)"
_TIME = r"\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.)?"
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")

SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "3 credits", "4.0 units", "1 credit hour" -- a number and a unit word only.
    ("credits", re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:credits?|units?|credit hours?)\s*$", re.I)),
    # "Room 303", "Rm. 12", "Sexton 204", "Dwinelle Hall, Room 210", "Office 118".
    ("class_meeting_room",
     re.compile(r"^\s*(?:room|rm\.?|office)\s*#?\s*[A-Za-z]?-?\d{1,4}[A-Za-z]?\s*$", re.I)),
    ("class_meeting_room",
     re.compile(r"^\s*[A-Z][A-Za-z'.\-]+(?:\s+Hall)?,?\s+(?:Room|Rm\.?)\s*#?\s*\d{1,4}\s*$")),
    # "Wed 2-4pm", "MWF 10:00-10:50", "TTh 1:00 PM - 2:30 PM", "Period 4".
    ("meeting_time",
     re.compile(rf"^\s*{_DAYS}\s*,?\s*{_TIME}\s*(?:-|–|to)\s*{_TIME}\s*$", re.I)),
    ("meeting_time", re.compile(r"^\s*(?:period|p)\s*\.?\s*[1-8]\s*$", re.I)),
    # "Dr. Johnson", "Prof. Allison Johnson", "Johnson, Allison".
    ("instructor",
     re.compile(r"^\s*(?:Dr|Prof|Professor|Mr|Mrs|Ms|Mx)\.?\s+[A-Z][A-Za-z'\-]+"
                r"(?:\s+[A-Z][A-Za-z'\-]+)*\s*$")),
    ("instructor",
     re.compile(r"^\s*[A-Z][A-Za-z'\-]+,\s*[A-Z][A-Za-z'\-]+(?:\s+[A-Z]\.)?\s*$")),
    # "Final Examination December 16", "Midterm: Oct 21". A bare "<Word> 204"
    # such as "Sexton 204" is deliberately not matched: "Biology 101" has the
    # same shape and is a course, not a room.
    ("exam_schedule",
     re.compile(rf"^\s*(?:final|midterm|unit|comprehensive|make-?up|entrance)\s*"
                rf"(?:exam(?:ination)?|test)?\s*[:,–-]?\s*(?:{_MONTHS})\.?\s+\d{{1,2}}\s*$",
                re.I)),
    # "Fall 2026", "Spring 2027".
    ("term", re.compile(r"^\s*(?:fall|spring|summer|winter|autumn|sp)\s*\d{4}\s*$", re.I)),
)

#: Short label names used by the application record.
TASKS = {
    "credits": "credits",
    "class_meeting_room": "meeting",
    "meeting_time": "meeting",
    "instructor": "instructor",
    "exam_schedule": "exam_schedule",
    "term": "term",
}


def resolve(text: str) -> dict[str, object] | None:
    """The field a bare value's shape decides, or ``None`` to let the model decide."""
    candidate = " ".join(str(text or "").split())
    if not candidate or len(candidate) > 60:
        return None
    for label, pattern in SHAPES:
        if pattern.match(candidate):
            return {"label": label, "task": TASKS.get(label, "other"),
                    "source": "shape", "reason": f"matches the {label} value shape"}
    return None
