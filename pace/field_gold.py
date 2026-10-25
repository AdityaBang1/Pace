"""Value-level gold for the fields the extractor promises, and its reliability.

The published section-span gold answers "did a value come from the right
section". That is not the same question as "is the value true", and it says
nothing at all about a field the document simply does not contain: a course
document that never prints credits produced no gold row, so an invented credit
count was never counted as an error.

This module carries a second, smaller gold set that was read field by field by a
person: for every document it lists the values the document actually states, and
marks the named fields the document does *not* state. An emitted value on an
absent field is then a false positive — which is exactly the failure the mission
cares about, and the one span gold is structurally unable to see.

On top of that gold it computes the two tables a confidence claim needs:

* **Bands** — observed precision in each confidence band, with a Clopper-Pearson
  lower bound, so "0.99 confidence" can be compared with what actually happened.
* **Sweep** — what the command would emit at each candidate default
  ``min_confidence``: coverage, precision, and the precision certified at 95%.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .field_calibration import error_upper_bound

FIELDS = ("course_title", "course_code", "credits", "instructor", "meetings", "grading",
          "office_hours")

# What decides a verdict, in words, next to the matcher that implements it. The
# held-out artifact publishes one of these per emitted value, so a reader can see
# the rule rather than re-deriving it from the report.
MATCH_RULES = {
    "course_title": "substring match after normalisation, either string containing the other",
    "course_code": "alphanumeric compaction, either string contained in the other",
    "credits": "numeric equality with the printed credit count",
    "instructor": "the gold surname occurs in the emitted name",
    "meetings": "identical day set and clock times, plus the printed room when gold has one",
    "grading": "equal integer weight and category-token overlap",
    "office_hours": "every content token of the gold entry occurs in the emitted text",
}
ABSENT_RULE = "the gold marks this field absent, so any emitted value is a false positive"
UNJUDGED_RULE = "no gold item to judge against: the document neither states the field nor marks it absent"

# The claim the mission asks for, in two strengths.
TARGET_PRECISION = 0.95
FLOOR_PRECISION = 0.90
CONFIDENCE = 0.95
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995)

DAY_ALIASES = {
    "m": "MO", "mo": "MO", "mon": "MO", "monday": "MO",
    "tu": "TU", "tue": "TU", "tues": "TU", "tuesday": "TU",
    "w": "WE", "we": "WE", "wed": "WE", "wednesday": "WE",
    "th": "TH", "thu": "TH", "thur": "TH", "thurs": "TH", "thursday": "TH",
    "f": "FR", "fr": "FR", "fri": "FR", "friday": "FR",
    "sa": "SA", "sat": "SA", "saturday": "SA",
    "su": "SU", "sun": "SU", "sunday": "SU",
}

_STOPWORDS = {"the", "and", "of", "in", "for", "to", "a", "an", "on", "with", "by", "at",
              "course", "class", "student", "students", "your", "you", "is", "are", "will",
              "be", "this", "that", "from", "or", "as"}


def normalise(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def tokens(text: Any) -> set[str]:
    """Content words, with a crude plural fold so "Essay"/"Essays" agree."""
    words = {word for word in normalise(text).split()
             if len(word) > 2 and word not in _STOPWORDS and not word.isdigit()}
    return {word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words}


def compact(text: Any) -> str:
    """Alphanumerics only: ``AMH 4373`` and ``AMH4373`` are the same code.

    An absent value compacts to the empty string, not to ``"none"``: the meeting
    matcher treats an empty gold room as "this document prints no room", and
    ``str(None)`` would silently turn that into a room the extractor has to find.
    """
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_gold(path: str | Path) -> dict[str, dict[str, Any]]:
    """Gold documents, keyed by id. The one loader for the gold schema.

    Validation lives here, not in each caller, so a malformed payload fails the
    same way everywhere: a missing document list, a document without an id, or a
    duplicate id is an error rather than a silently short gold set.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"{path}: gold payload has no 'documents' list")
    by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        document_id = document.get("id")
        if not document_id:
            raise ValueError(f"{path}: a gold document has no id")
        if document_id in by_id:
            raise ValueError(f"{path}: duplicate gold document id {document_id}")
        by_id[document_id] = document
    return by_id


def load_gold_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _day_codes(days: Iterable[Any]) -> frozenset[str]:
    codes = set()
    for day in days or []:
        text = str(day).strip().lower()
        if text in DAY_ALIASES:
            codes.add(DAY_ALIASES[text])
        elif len(text) == 2 and text.upper() in {"MO", "TU", "WE", "TH", "FR", "SA", "SU"}:
            codes.add(text.upper())
        elif len(text) >= 3:
            codes.add(DAY_ALIASES.get(text[:3], text.upper()))
    return frozenset(codes)


def _clock(value: Any) -> str | None:
    if value is None:
        return None
    match = re.match(r"^\s*(\d{1,2}):(\d{2})", str(value))
    if not match:
        return str(value).strip() or None
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def _meeting_match(gold: dict[str, Any], value: dict[str, Any]) -> bool:
    if _day_codes(gold.get("days")) != _day_codes(value.get("days")):
        return False
    for key in ("start", "end"):
        mine = _clock(gold.get(key) or gold.get(f"{key}_time"))
        theirs = _clock(value.get(key) or value.get(f"{key}_time"))
        if (mine is None) != (theirs is None):
            return False
        if mine is not None and mine != theirs:
            return False
    gold_room = compact(gold.get("room"))
    if not gold_room:
        return True
    room = compact(value.get("room"))
    return bool(room) and (gold_room in room or room in gold_room)


def _grading_match(gold: dict[str, Any], value: dict[str, Any]) -> bool:
    try:
        if int(gold["weight"]) != int(value.get("weight")):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    left, right = tokens(gold["category"]), tokens(value.get("category") or value.get("text"))
    if not left or not right:
        return False
    if left <= right or right <= left:
        return True
    return len(left & right) >= max(1, min(len(left), len(right)) - 1) and \
        len(left & right) / len(left | right) >= 0.5


def _text_match(gold: str, value: Any) -> bool:
    left, right = normalise(gold), normalise(value)
    if not left or not right:
        return False
    return left in right or right in left


def match_entry(field: str, gold: list[Any], value: dict[str, Any]) -> Any | None:
    """The gold entry an emitted value agrees with, or None.

    The only place the matching rules live. ``value_matches`` is the boolean
    view of this; the held-out artifact uses the entry itself so every verdict
    names what it was judged against.
    """
    if field == "course_title":
        return next((item for item in gold if _text_match(item, value.get("text"))), None)
    if field == "course_code":
        mine = compact(value.get("text"))
        return next((item for item in gold
                     if compact(item) and (compact(item) in mine or mine in compact(item))), None)
    if field == "credits":
        try:
            number = float(re.search(r"\d+(?:\.\d+)?", str(value.get("text"))).group(0))
        except (AttributeError, TypeError, ValueError):
            return None
        return next((item for item in gold
                     if str(item).strip() and abs(number - float(item)) < 1e-6), None)
    if field == "instructor":
        emitted = normalise(value.get("text"))
        for item in gold:
            parts = normalise(item).split()
            surname = parts[-1] if parts else ""
            if surname and surname in emitted:
                return item
        return None
    if field == "meetings":
        return next((item for item in gold if _meeting_match(item, value)), None)
    if field == "grading":
        return next((item for item in gold if _grading_match(item, value)), None)
    if field == "office_hours":
        emitted = tokens(value.get("text"))
        for item in gold:
            wanted = tokens(item)
            if wanted and wanted <= emitted:
                return item
        return None
    return None


def value_matches(field: str, gold: list[Any], value: dict[str, Any]) -> bool:
    """Does one emitted value agree with one of these gold entries?"""
    return match_entry(field, gold, value) is not None


def judge(value: dict[str, Any], gold: dict[str, Any]) -> tuple[bool, bool]:
    """(judged, matched) for one emitted value against one document's gold."""
    field = value["field"]
    absent = set(gold.get("absent") or [])
    if field in absent:
        return True, False
    if field not in gold or not gold[field]:
        return False, False
    entries = gold[field]
    if field in {"meetings", "grading"}:
        return True, any(value_matches(field, [item], value) for item in entries)
    return True, value_matches(field, entries, value)


def score_document(values: list[dict[str, Any]], gold: dict[str, Any]) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    for value in values:
        was_judged, matched = judge(value, gold)
        judged.append({**value, "judged": was_judged, "matched": matched})
    return judged


def recall(gold: dict[str, Any], values: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """Gold items with at least one emitted value that agrees with them."""
    hits: dict[str, list[int]] = {}
    for field in FIELDS:
        entries = gold.get(field) or []
        if field in set(gold.get("absent") or []) or not entries:
            continue
        found = 0
        for item in entries:
            if field in {"meetings", "grading"}:
                hit = any(value_matches(field, [item], value) for value in values)
            else:
                hit = any(value_matches(field, entries, value) for value in values)
            found += int(hit)
        hits[field] = [found, len(entries)]
    return {field: (hit, total) for field, (hit, total) in hits.items()}


def reliability(values: list[dict[str, Any]],
                bands: tuple[tuple[float, float], ...] = ((0.0, 0.7), (0.7, 0.9),
                                                          (0.9, 0.99), (0.99, 1.01)),
                ) -> list[dict[str, Any]]:
    """Observed precision per confidence band, with a one-sided lower bound."""
    judged = [value for value in values if value["judged"]]
    rows: list[dict[str, Any]] = []
    for low, high in bands:
        inside = [value for value in judged if low <= value["confidence"] < high]
        correct = sum(1 for value in inside if value["matched"])
        errors = len(inside) - correct
        bound = error_upper_bound(errors, len(inside), CONFIDENCE)
        rows.append({
            "low": low, "high": min(high, 1.0), "values": len(inside), "correct": correct,
            "precision": correct / len(inside) if inside else None,
            "precision_lower_95": None if bound is None else 1 - bound,
        })
    return rows


def sweep(values: list[dict[str, Any]],
          thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """What the command emits at each candidate default threshold.

    Coverage is over the values gold can speak to, not over every value the rule
    engine emitted: an emitted value on a field this gold never labelled is not
    something a threshold can be credited with keeping or dropping.
    """
    judged = [value for value in values if value["judged"]]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [value for value in judged if value["confidence"] >= threshold]
        correct = sum(1 for value in accepted if value["matched"])
        bound = error_upper_bound(len(accepted) - correct, len(accepted), CONFIDENCE)
        rows.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "judged_values": len(judged),
            "coverage": len(accepted) / len(judged) if judged else 0.0,
            "precision": correct / len(accepted) if accepted else None,
            "precision_lower_95": None if bound is None else 1 - bound,
            "errors": len(accepted) - correct,
        })
    return rows


def operating_point(rows: list[dict[str, Any]], target: float = TARGET_PRECISION
                    ) -> dict[str, Any]:
    """Lowest threshold whose *certified* precision clears the target.

    Certified, not observed: the point of the table is to stop quoting a
    precision that a handful of documents cannot support.
    """
    for row in rows:
        bound = row["precision_lower_95"]
        if bound is not None and bound >= target:
            return {"threshold": row["threshold"], "target": target, "certified": True,
                    "precision": row["precision"], "precision_lower_95": bound,
                    "coverage": row["coverage"]}
    best = max((row for row in rows if row["precision_lower_95"] is not None),
               key=lambda row: row["precision_lower_95"], default=None)
    return {
        "threshold": None, "target": target, "certified": False,
        "note": f"no threshold certifies precision {target:.2f} at "
                f"{CONFIDENCE:.0%} confidence on this sample",
        "best": None if best is None else {
            "threshold": best["threshold"], "precision": best["precision"],
            "precision_lower_95": best["precision_lower_95"], "coverage": best["coverage"],
        },
    }


def by_field(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for value in values:
        row = stats.setdefault(value["field"], {"emitted": 0, "judged": 0, "matched": 0})
        row["emitted"] += 1
        if value["judged"]:
            row["judged"] += 1
            row["matched"] += int(value["matched"])
    for row in stats.values():
        row["precision"] = row["matched"] / row["judged"] if row["judged"] else None
    return dict(sorted(stats.items()))
