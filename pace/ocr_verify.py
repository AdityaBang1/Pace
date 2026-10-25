"""Check every recognised cell against the pixels it came from, and against the page.

A whole-page OCR pass is a compromise: the detector decides what a cell is, then
one recogniser run reads a line. On a small or blurry page a single digit can flip
(``90`` -> ``06``). Flagging the page as "OCR, not exact" is honest but not enough
when the number itself is wrong.

So each value the parser emitted from an OCR page is put through three questions:

1. **Its own crop.** The box that produced the characters is cropped out and the
   recogniser reads it alone, twice (as-is and contrast-stretched). A cell read on
   its own has no neighbouring line to compete with and no detector to lose it.
2. **Its own rules.** A mark is a whole number in 0..100, a weight and a credit are
   positive numbers, a local course code is digits then a letter. A reading that
   cannot be that field is not that field, whatever the model said.
3. **The page's arithmetic.** Row credits must add up to the printed total. When
   they do not, no credit on that page is trusted, because the page does not say
   which row is wrong.

The outcome for each cell is one of ``confirmed``, ``repaired`` (the focused read
disagreed and was stable, and it passes the rules) or ``unread`` (the readings
disagree with each other or with the page's own arithmetic). An ``unread`` cell is
emitted as ``null`` with the reason and every reading attached — never replaced by
a guess. Repairs are listed too, so a person can see the page pass said ``06`` and
the focused read said ``90``.

Two things make that record trustworthy rather than merely detailed:

* **One write path.** A cell's status and its value are written together by
  ``_write_cell`` from the same ``CellCheck``, and ``cell_checks`` carries the
  emitted value next to the status. When the page's arithmetic later refuses a
  credit, it goes through the same path (``refuse_cell``), so a status can never
  say ``confirmed`` above a ``null``.
* **Derived tallies.** ``cells_checked`` is the sum of the outcomes at page and
  document level — never a separate counter to keep in step — so
  ``cells_checked == confirmed + repaired + unconfirmed + unread`` holds by
  construction, including after a credit is distrusted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from .ingest import Document, PageText

CODE_PATTERN = re.compile(r"^\d{3,6}[A-Za-z]{1,2}$")
NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d{1,6})?$")

CELL_FIELDS = ("course_code", "mark", "weight", "credit")
NUMERIC_FIELDS = frozenset({"mark", "weight", "credit"})

# The outcomes partition the cells that were looked at. `cells_checked` is the sum
# of these, at page and document level, so the arithmetic cannot drift apart.
CELL_OUTCOMES = ("confirmed", "repaired", "unconfirmed", "unread")

# The one table that decides whether a status emits a value. `_write_cell` reads it
# for both the status and the value it writes, so the two cannot disagree.
EMITS_VALUE = {"confirmed": True, "repaired": True, "unconfirmed": True,
               "unread": False}

# Why a value the cell checker never looked at is still in the record. A reader who
# sees a title or a GPA must be able to see, on that value, that nobody checked it.
TITLE_REASON = ("a title is long, wraps, and is not a number: it is read by the page "
                "pass only and never re-read from its own crop")
RECORD_TEXT_REASON = ("this value is transcribed from the page and never cell-checked; "
                      "only course code, mark, weight and credit are")
STANDING_REASON = ("a footer figure is read once from the page and never re-read from "
                   "its own crop, so it is not part of the per-cell guarantee")
HEADER_REASON = ("a header value is read once from the page and never re-read from "
                 "its own crop, so it is not part of the per-cell guarantee")
ASSESSMENT_REASON = ("assessment values are parsed from page text and never "
                     "cell-checked, so they are not part of the guarantee")
ATTENDANCE_REASON = ("attendance counts are parsed from a table that is never "
                     "cell-checked, so they are not part of the guarantee")

# A cell crop is padded so the recogniser sees the whole glyph and a little paper,
# then doubled: the recogniser trains on line images about 32 pixels tall, and a
# mark on a small page is barely 20.
CROP_PAD_X = 5
CROP_PAD_Y = 4
CROP_SCALE = 2

# Weights and credits are per-course numbers. Anything above this is a misread of
# something else (a code, a mark) rather than a course weight.
MAX_WEIGHT = 10
MAX_CREDIT = 10

# A focused reading may replace the page's reading only when the recogniser itself
# considered it acceptable (its default `text_score`). Below that, a disagreeing
# crop is not evidence enough to overwrite anything.
MIN_REPAIR_SCORE = 0.5

# A digit in this recogniser's line images is about 0.55 of the line height wide.
# A cell box much wider than the digits read from it lost characters somewhere —
# the 600-pixel rung read a two-digit mark as "4" at 0.35 confidence, and the box
# was still two digits wide.
DIGIT_WIDTH_RATIO = 0.55
MISSING_CHARACTER = 0.75


@dataclass
class CellCheck:
    """One cell's page reading, focused re-reads and the decision between them."""

    field: str
    page_reading: str
    status: str
    readings: list[tuple[str, float]] = field(default_factory=list)
    reason: str | None = None
    # The value this cell emits. `None` means the record carries `null`, and only
    # a status whose EMITS_VALUE is False may leave it that way.
    emitted: Any = None
    box: list[float] | None = None

    @property
    def score(self) -> float | None:
        return max((score for _, score in self.readings), default=None)

    def brief(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "status": self.status,
            # The value sits beside the status so a reader cannot see one without
            # the other: a `confirmed` cell above a null is a visible absurdity.
            "value": self.emitted,
            "page_reading": self.page_reading,
            "focused_readings": [text for text, _ in self.readings],
        }
        if self.score is not None:
            view["score"] = round(self.score, 3)
        if self.reason:
            view["reason"] = self.reason
        return view


def _number(text: Any) -> Decimal | None:
    """The numeric value of a reading, or None when it is not a number at all."""
    candidate = str(text).strip().replace(",", ".")
    if not NUMBER_PATTERN.match(candidate):
        return None
    try:
        return Decimal(candidate)
    except InvalidOperation:  # pragma: no cover - the pattern already guards this
        return None


def _code(text: Any) -> str | None:
    candidate = str(text).strip().replace(" ", "")
    # The letter suffix is a local-code marker, not a distinction: 220725Y and
    # 220725y are the same code and must not be "repaired" into each other.
    return candidate.lower() if CODE_PATTERN.match(candidate) else None


def _comparable(field_name: str, text: Any) -> Any:
    if field_name in NUMERIC_FIELDS:
        return _number(text)
    return _code(text)


def _typed(field_name: str, text: Any) -> Any:
    """The JSON value a reading of this field is emitted as."""
    if field_name in {"weight", "credit"}:
        return float(_number(text))
    return str(text).strip()


def _rule_reason(field_name: str, text: str) -> str | None:
    """Why a reading cannot be the field it claims to be, or None if it can."""
    if field_name == "course_code":
        if _code(text) is None:
            return "a local course code is three to six digits followed by a letter"
        return None
    number = _number(text)
    if number is None:
        return f"a {field_name} must be a number, and this reading is not one"
    if field_name == "mark":
        if number != number.to_integral_value():
            return "a mark in this layout is a whole percentage"
        if not Decimal(0) <= number <= Decimal(100):
            return "a mark must be between 0 and 100"
        return None
    if number <= 0:
        return f"a {field_name} must be greater than zero"
    ceiling = MAX_WEIGHT if field_name == "weight" else MAX_CREDIT
    if number > ceiling:
        return f"a {field_name} above {ceiling} is not a course {field_name}"
    return None


def _box_for(page: PageText, start: int, end: int, trim: bool = True
             ) -> dict[str, Any] | None:
    """The part of a recognised box that holds exactly the wanted characters.

    A detector often merges a course code with its title into one box: the page
    text still separates them, but one crop would contain both. The box is trimmed
    by the value's share of the box's characters, so the crop is the cell, not the
    row.
    """
    local_start = start - page.start
    local_end = end - page.start
    covering = [box for box in page.boxes
                if box["start"] <= local_start and box["end"] >= local_end]
    if not covering:
        return None
    # A span can touch two boxes when a value is glued to its neighbour; the one
    # that holds the whole span wins, and the tightest box after that.
    exact = [box for box in covering
             if box["start"] == local_start and box["end"] == local_end]
    if exact:
        return exact[0]
    box = min(covering, key=lambda item: (item["box"][2] - item["box"][0])
              * (item["box"][3] - item["box"][1]))
    text = box["text"]
    if not text:
        return None
    if not trim:
        return box
    first = (local_start - box["start"]) / len(text)
    last = (local_end - box["start"]) / len(text)
    if last <= first:
        return None
    x0, y0, x1, y1 = box["box"]
    width = x1 - x0
    return {"start": local_start, "end": local_end, "text": text, "score": box.get("score"),
            "box": [x0 + width * first, y0, x0 + width * last, y1]}


def _truncated(field_name: str, text: str, box: list[float] | None) -> tuple[bool, float]:
    """Does the cell's printed width fit more characters than were read?"""
    if not box:
        return False, 0.0
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    expected = width / (height * DIGIT_WIDTH_RATIO)
    digits = sum(1 for character in str(text) if character.isdigit())
    return (expected - digits) >= MISSING_CHARACTER, expected


def _first_reading(field_name: str, text: str) -> str | None:
    """The cell's value at the start of a crop reading, if the crop is a bit wide."""
    candidate = text.strip()
    if field_name == "course_code":
        # A code is printed in front of its title, and the detector often keeps them
        # in one box, so the reading is "220725y Accel Adv" — the code is its head.
        match = re.match(r"(\d{3,6}[A-Za-z]{1,2})", candidate)
        return match.group(1) if match else candidate
    match = re.match(r"(\d+(?:\.\d{1,6})?)", candidate)
    return match.group(1) if match else candidate


def _crop_variants(image: Any, box: list[float]) -> list[Any]:
    """The padded, doubled crop of a box, plain and with its contrast stretched."""
    from PIL import ImageOps

    left = max(0, int(box[0]) - CROP_PAD_X)
    top = max(0, int(box[1]) - CROP_PAD_Y)
    right = min(image.width, int(box[2]) + CROP_PAD_X)
    bottom = min(image.height, int(box[3]) + CROP_PAD_Y)
    if right <= left or bottom <= top:
        return []
    crop = image.crop((left, top, right, bottom))
    crop = crop.resize((crop.width * CROP_SCALE, crop.height * CROP_SCALE))
    return [crop, ImageOps.autocontrast(crop)]


def _read_cells(backend: Any, images: list[Any]) -> list[tuple[str, float] | None]:
    """Recognise many crops in one pass, so the work stays cheap per page."""
    if not images:
        return []
    batch = getattr(backend, "read_cells", None)
    if callable(batch):
        results = batch(images)
        if isinstance(results, list):
            return [result if result is not None else None for result in results]
    return [backend.read_cell(image) if hasattr(backend, "read_cell") else None
            for image in images]


def _decide(check: CellCheck, field_name: str) -> CellCheck:
    page_value = _comparable(field_name, check.page_reading)
    # A crop can include a sliver of its neighbour; the cell's value is the run at
    # the start of the reading, not the whole string.
    extracted = [(_first_reading(field_name, text), score)
                 for text, score in check.readings]
    readings = [(text, score) for text, score in extracted
                if _comparable(field_name, text) is not None]
    if not readings:
        if field_name == "course_code" and _code(check.page_reading) is not None:
            check.status = "unconfirmed"
            check.reason = ("the focused re-read of this code returned nothing that "
                            "fits; the page code matches the printed pattern")
            return check
        check.status = "unread"
        check.reason = ("the focused re-read of this cell returned nothing readable "
                        "that fits the field")
        return check
    values = {_comparable(field_name, text) for text, _ in readings}
    if values == {page_value}:
        if field_name in NUMERIC_FIELDS:
            truncated, expected = _truncated(field_name, check.page_reading, check.box)
            if truncated:
                check.status = "unread"
                check.reason = (f"the cell reads {check.page_reading!r}, but its printed "
                                f"width fits about {expected:.1f} characters; characters "
                                "are missing from the reading")
                return check
        check.status = "confirmed"
        return check
    if len(values) == 1 and len(readings) >= 2:
        candidate = readings[0][0]
        reason = _rule_reason(field_name, candidate)
        if field_name == "course_code":
            # Codes are identifiers, not numbers: two letters of a code look alike
            # and a crop is weak evidence against a code that already matches the
            # printed pattern. Keep the page's code, and show the disagreement.
            check.status = "unconfirmed"
            check.reason = (f"the focused re-read said {candidate!r}; the page code "
                            "matches the printed pattern but was not confirmed")
            return check
        if reason is not None:
            check.status = "unread"
            check.reason = f"the focused re-read said {candidate!r}, but {reason}"
            return check
        if max(score for _, score in readings) < MIN_REPAIR_SCORE:
            check.status = "unread"
            check.reason = (f"the focused re-read said {candidate!r} but scored below "
                            f"{MIN_REPAIR_SCORE}; not enough evidence to replace "
                            f"the page's {check.page_reading!r}")
            return check
        check.status = "repaired"
        check.emitted = _typed(field_name, candidate)
        check.reason = ("the whole-page pass and the focused re-read disagreed; the "
                        "focused read is stable across two renderings and passes the "
                        f"field's rules (page said {check.page_reading!r})")
        return check
    check.status = "unread"
    check.reason = ("the focused re-read is not stable: "
                    + ", ".join(repr(text) for text, _ in readings))
    return check


def _write_cell(record: dict[str, Any], check: CellCheck) -> None:
    """Write a cell's value, its status and its companion entry in one place.

    This is the only function that puts a value on a record, so the status shown
    for a cell and the value beside it come from the same decision. The companion
    lists are rewritten, never appended to: when a status changes after the fact —
    a credit distrusted by the page's arithmetic — the stale repair or refusal is
    dropped instead of sitting next to the new one.
    """
    if check.status not in EMITS_VALUE:
        raise ValueError(f"unknown cell status {check.status!r}")
    emits = EMITS_VALUE[check.status]
    if check.status == "repaired" and check.emitted is None:
        raise ValueError("a repaired cell must carry the value it was repaired to")
    if not emits and check.emitted is not None:
        raise ValueError(f"an {check.status} cell for {check.field!r} cannot carry a value")
    if emits and check.emitted is None:
        # A confirmed or unconfirmed cell keeps the value the parser emitted; the
        # focused read only had to agree with it (or be too weak to replace it).
        check.emitted = record.get(check.field)
    # No status may be recorded without the value that matches it: checked statuses
    # must emit something, and `unread` must emit nothing. This is the writer's
    # invariant, not a promise the caller is trusted to keep.
    if emits and check.emitted is None:
        raise ValueError(f"a {check.status} cell for {check.field!r} has no value to emit")
    record[check.field] = check.emitted
    if check.field == "mark":
        record["mark_percent"] = (int(_number(check.emitted))
                                  if check.emitted is not None else None)
    record.setdefault("cell_checks", {})[check.field] = check.brief()
    if check.status == "repaired":
        record.setdefault("repairs", {})[check.field] = {
            "page_reading": check.page_reading,
            "focused_reading": check.emitted,
            "score": None if check.score is None else round(check.score, 3),
            "reason": check.reason,
        }
    else:
        (record.get("repairs") or {}).pop(check.field, None)
    if check.status == "unread":
        record.setdefault("unread", {})[check.field] = {
            "reason": check.reason,
            "page_reading": check.page_reading,
            "focused_readings": [text for text, _ in check.readings],
            "evidence": (record.get("evidence") or {}).get(check.field),
        }
    else:
        (record.get("unread") or {}).pop(check.field, None)


def refuse_cell(record: dict[str, Any], field_name: str, reason: str,
                *, page_reading: str | None = None) -> None:
    """Refuse a cell from outside the focused pass, through the same write path.

    The page's own arithmetic can distrust a credit no crop disagreed with. That is
    still one outcome for one cell, so it is written as an `unread` cell — status,
    null value and reason together — rather than as a null bolted onto a record
    whose status still says `confirmed`.
    """
    previous = (record.get("cell_checks") or {}).get(field_name) or {}
    reading = page_reading or previous.get("page_reading")
    if reading is None:
        reading = ((record.get("evidence") or {}).get(field_name) or {}).get("text")
    score = previous.get("score")
    readings = [(text, float(score or 0.0))
                for text in previous.get("focused_readings") or []]
    check = CellCheck(field=field_name, page_reading=reading, status="unread",
                      readings=readings, reason=reason)
    _write_cell(record, check)


def verify_document(parsed: dict[str, Any], document: Document) -> dict[str, Any]:
    """Re-read and check every cell parsed from an OCR page; return the tally."""
    backend = getattr(document, "backend", None)
    stats: dict[str, Any] = {
        "backend": getattr(backend, "name", None),
        "pages": [],
    }
    ocr_pages = [page for page in document.pages if page.method == "ocr"]
    if not ocr_pages:
        stats["status"] = "not_needed"
        return stats
    if backend is None or not any(page.boxes for page in ocr_pages):
        stats["status"] = "unavailable"
        stats["reason"] = ("this OCR backend does not report where each cell was, so "
                           "no cell could be cropped and checked")
        return stats

    records_by_page: dict[int, list[dict[str, Any]]] = {}
    for record in parsed.get("course_records") or []:
        records_by_page.setdefault(record.get("page"), []).append(record)

    for page in ocr_pages:
        records = records_by_page.get(page.number) or []
        if not records or not page.boxes:
            continue
        # One rendering per page; it is never held beyond this loop.
        image = document.page_image(page)
        # Gather every crop for the page first: one batched recognition call costs
        # far less than one call per cell.
        jobs: list[tuple[dict[str, Any], str, str, list[float], CellCheck]] = []
        images: list[Any] = []
        for record in records:
            for field_name in CELL_FIELDS:
                value = record.get(field_name)
                evidence = (record.get("evidence") or {}).get(field_name)
                if value is None or not evidence:
                    continue
                box = _box_for(page, evidence["start"], evidence["end"],
                               trim=field_name != "course_code")
                if box is None:
                    continue
                variants = _crop_variants(image, box["box"])
                if not variants:
                    continue
                check = CellCheck(field=field_name, page_reading=evidence["text"],
                                  status="pending", box=box["box"])
                jobs.append((record, field_name, evidence["text"], box["box"], check))
                images.extend(variants)

        results = _read_cells(backend, images)
        for index, (record, field_name, page_reading, box, check) in enumerate(jobs):
            readings = [reading for reading in results[index * 2:index * 2 + 2]
                        if reading is not None]
            check.readings = readings
            _decide(check, field_name)
            _write_cell(record, check)

    # No tally is accumulated while cells are decided: it is derived once from the
    # settled records, so it cannot count a cell twice or miss a late refusal.
    stats["status"] = "checked"
    _retally(parsed, stats)
    return stats


def _retally(parsed: dict[str, Any], stats: dict[str, Any]) -> None:
    """Derive every tally from the records, so page and document agree.

    Counting outcomes as they are written leaves two counters free to drift: the
    focused pass and the page's arithmetic both change what a cell is. Recounting
    from the settled records makes `cells_checked` the sum of the outcomes at both
    levels, by construction.
    """
    pages: dict[Any, dict[str, Any]] = {}
    for record in parsed.get("course_records") or []:
        page_number = record.get("page")
        bucket = pages.setdefault(page_number, {
            "page": page_number, **{name: 0 for name in CELL_OUTCOMES}})
        for view in (record.get("cell_checks") or {}).values():
            status = view.get("status")
            if status not in EMITS_VALUE:
                raise ValueError(f"record carries an unknown cell status {status!r}")
            bucket[status] += 1
    ordered = [pages[number] for number in
               sorted(pages, key=lambda number: (number is None, number))]
    for bucket in ordered:
        bucket["cells_checked"] = sum(bucket[name] for name in CELL_OUTCOMES)
    stats["pages"] = ordered
    for name in CELL_OUTCOMES:
        stats[name] = sum(bucket[name] for bucket in ordered)
    stats["cells_checked"] = sum(bucket["cells_checked"] for bucket in ordered)


def refresh_verification(parsed: dict[str, Any],
                         stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-derive the verification tallies after values have changed.

    Refusing a credit happens after the focused pass, so the tallies it wrote are
    no longer the truth. They are recomputed from the records rather than patched,
    which is what keeps `unread` from exceeding `cells_checked`.
    """
    stats = stats if stats is not None else (parsed.get("verification") or {})
    if stats.get("status") == "checked":
        _retally(parsed, stats)
    return stats


def refresh_checks(parsed: dict[str, Any]) -> None:
    """Recompute the document-level checks after repairs and unread cells.

    The parser's checks describe the values it emitted. Verification can change
    those values, so the arithmetic has to be redone; a stale "credits add up" flag
    next to a repaired credit would be a lie.
    """
    records = parsed.get("course_records") or []
    checks = parsed.setdefault("checks", {})
    credits = [record["credit"] for record in records if record.get("credit") is not None]
    weights = [record["weight"] for record in records if record.get("weight") is not None]
    checks["computed_total_credits"] = round(sum(credits), 4)
    checks["computed_total_weight"] = round(sum(weights), 4)
    checks["records_without_weight"] = len(records) - len(weights)
    checks["marks_numeric"] = all(record.get("mark") is not None for record in records)
    checks["credit_sum_complete"] = len(credits) == len(records)
    reported = checks.get("reported_total_credits")
    if reported is not None and checks["credit_sum_complete"]:
        difference = round(checks["computed_total_credits"] - reported, 4)
        checks["credit_sum_difference"] = difference
        checks["credit_sum_matches_reported_total"] = abs(difference) < 0.005
    else:
        checks["credit_sum_matches_reported_total"] = None
    unread = _count_status(records, "unread")
    repaired = _count_status(records, "repaired")
    checks["unread_cells"] = unread
    checks["repaired_cells"] = repaired
    checks["unconfirmed_cells"] = _count_status(records, "unconfirmed")


def _count_status(records: list[dict[str, Any]], status: str) -> int:
    """How many cells carry a status, read from the cells themselves."""
    return sum(1 for record in records
               for view in (record.get("cell_checks") or {}).values()
               if view.get("status") == status)


def distrust_unbalanced_credits(parsed: dict[str, Any]) -> list[str]:
    """When the printed total and the rows still disagree, trust none of the rows.

    The page states one total and prints one credit per course. If they disagree
    after every credit was re-read, the page does not say which row is wrong — so
    keeping four of five credits would be a guess dressed as data.
    """
    checks = parsed.get("checks") or {}
    if checks.get("credit_sum_matches_reported_total") is not False:
        return []
    if not checks.get("credit_sum_complete"):
        return []
    reported = checks.get("reported_total_credits")
    computed = checks.get("computed_total_credits")
    reason = (f"the row credits add up to {computed}, but the page prints {reported}; "
              "no row can be identified as the wrong one, so no credit is trusted")
    touched = 0
    for record in parsed.get("course_records") or []:
        if record.get("credit") is None:
            continue
        # The same write path the focused pass uses: status and value together.
        refuse_cell(record, "credit", reason)
        touched += 1
    checks["credit_sum_matches_reported_total"] = None
    checks["credit_sum_complete"] = False
    checks["distrusted_credit_rows"] = touched
    checks["unread_cells"] = _count_status(parsed.get("course_records") or [],
                                           "unread")
    return [reason] if touched else []


# Keys that are not values: provenance, geometry and the bookkeeping of checking.
# The list is deliberately an exclusion list — a newly added value field is marked
# unverified by default, because nothing checked it.
_RECORD_METADATA = frozenset({
    "kind", "page", "start", "end", "line_start", "line_end", "line_index",
    "column", "title_wrapped", "instructor_wrapped", "section_confidence", "evidence",
    "evidence_method", "evidence_exact", "cell_checks", "repairs", "unread",
    "unverified", "requires_review", "mark_percent",
}) | frozenset(CELL_FIELDS)
_STANDING_METADATA = frozenset({"heading", "source", "unverified"})
_HEADER_METADATA = frozenset({"unverified"})


def mark_unverified_values(parsed: dict[str, Any]) -> dict[str, int]:
    """Mark, value by value, everything the cell checker never looked at.

    Only course code, mark, weight and credit are re-read from their own crops.
    Titles, term text, the footer GPA and total credits, and assessment values are
    transcriptions of the page that nothing checked. A document-level list of
    field names is not enough: a reader who takes one title out of the JSON has no
    way to know it was never verified. So the mark is written on the record that
    carries the value — ``"unverified": {"title": "..."}`` — and the verification
    block counts them, so the boundary is visible from either direction.
    """
    marks: dict[str, int] = {}

    def mark(container: dict[str, Any], field_name: str, reason: str) -> None:
        container.setdefault("unverified", {})[field_name] = reason
        marks[field_name] = marks.get(field_name, 0) + 1

    for record in parsed.get("course_records") or []:
        # The dict grows while it is read: take a snapshot of the keys first.
        for field_name, value in list(record.items()):
            if field_name in _RECORD_METADATA or value is None:
                continue
            mark(record, field_name,
                 TITLE_REASON if field_name == "title" else RECORD_TEXT_REASON)
    header = parsed.get("header") or {}
    for field_name, value in list(header.items()):
        if field_name in _HEADER_METADATA or not isinstance(value, dict):
            continue
        present = value.get("value", value.get("text"))
        if present is None:
            continue
        mark(header, field_name, HEADER_REASON)
    standing = parsed.get("academic_standing") or {}
    for field_name, value in list(standing.items()):
        if field_name in _STANDING_METADATA:
            continue
        present = value.get("value") if isinstance(value, dict) else value
        if present is None:
            continue
        mark(standing, field_name, STANDING_REASON)
    for record in parsed.get("assessment_records") or []:
        for field_name, value in list(record.items()):
            if field_name in _RECORD_METADATA or value is None:
                continue
            mark(record, field_name, ASSESSMENT_REASON)
    for record in parsed.get("attendance_records") or []:
        for field_name, value in list(record.items()):
            if field_name in _RECORD_METADATA or value is None:
                continue
            mark(record, field_name, ATTENDANCE_REASON)

    stats = parsed.get("verification")
    if stats is not None:
        # Only a field whose cells were actually re-read may be called verified.
        # With no box geometry nothing was checked, and in a layout that prints no
        # credit there is no credit cell to check — so the list is read off the
        # cells that carry a status rather than assumed from the field names.
        checked = {name for record in parsed.get("course_records") or []
                   for name in (record.get("cell_checks") or {})}
        stats["verified_fields"] = [name for name in CELL_FIELDS if name in checked]
        stats["unverified_values"] = sum(marks.values())
        stats["unverified_fields"] = sorted(marks)
    return marks


ROW_START_PATTERN = re.compile(r"^\s*(\d{3,6}[A-Za-z]{1,2})\s+\S")


def unparsed_row_lines(parsed: dict[str, Any], document: Document) -> list[str]:
    """OCR lines that start like a course row but produced no record.

    A row disappears when a misread cell stops the row from being a row at all
    (``t6`` where ``94`` was printed). Silence there would look like a page with
    seven courses, so every such line is named in the warnings.
    """
    reasons: list[str] = []
    # A row read by another table reader is covered: an attendance row is a course
    # row the page printed, not a course row that was lost.
    rows = (list(parsed.get("course_records") or [])
            + list(parsed.get("attendance_records") or []))
    for page in document.pages:
        if page.method != "ocr":
            continue
        covered = [(record["start"], record["end"]) for record in rows
                   if record.get("page") == page.number]
        offset = page.start
        for line in page.text.splitlines():
            start, end = offset, offset + len(line)
            offset = end + 1
            if not ROW_START_PATTERN.match(line):
                continue
            if any(not (end <= span_start or start >= span_end)
                   for span_start, span_end in covered):
                continue
            reasons.append(
                f"page {page.number}: {line.strip()[:70]!r} starts like a course row, "
                "but no complete row could be read from it"
            )
    return reasons


def verification_summary(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """The compact verification block the CLI prints under ``--summary``."""
    stats = parsed.get("verification")
    if not stats:
        return None
    view = {"status": stats.get("status"), "backend": stats.get("backend")}
    for key in ("cells_checked", "confirmed", "repaired", "unconfirmed", "unread",
                "distrusted_credit_rows", "verified_fields", "unverified_values"):
        if key in stats:
            view[key] = stats[key]
    if stats.get("reason"):
        view["reason"] = stats["reason"]
    return view
