"""Coverage for cell-level verification: confirm, repair, and refuse.

The real recogniser is used where the point is the recogniser (the control page and
the user's 256x202 capture). The unread path is exercised with a backend that
answers nonsense, so the test pins the decision, not the model.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pace.ingest import ocr_backend_name, parse_document, read_document
from pace.ocr_verify import CELL_OUTCOMES, distrust_unbalanced_credits
from pace.transcript_extract import parse_transcript

pytest.importorskip("PIL.Image")

FIXTURE = Path("data/samples/transcript_term_marks.txt")
GRADE_FIXTURE = Path("data/samples/grade_report.txt")
GRADE_PAGE = Path("data/samples/grade_report.png")
CONTROL = Path("data/samples/transcript_term_marks.png")
CAPTURE = Path("data/samples/transcript_term_marks_256x202.png")
PDF = Path("data/samples/transcript_term_marks.pdf")

BACKEND = ocr_backend_name()
real_ocr = pytest.mark.skipif(BACKEND is None, reason="no OCR backend installed on this machine")


def _values(parsed: dict) -> list[tuple]:
    return [(record["course_code"], record["mark"], record["weight"], record["credit"])
            for record in parsed["course_records"]]


@pytest.fixture(scope="module")
def fixture_values() -> list[tuple]:
    return _values(parse_transcript(FIXTURE.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def control_document():
    return read_document(CONTROL)


@pytest.fixture(scope="module")
def control(control_document):
    return parse_document(control_document)


@pytest.fixture(scope="module")
def capture():
    return parse_document(read_document(CAPTURE))


@pytest.fixture(scope="module")
def grade_page():
    if not GRADE_PAGE.exists() or not GRADE_FIXTURE.exists():
        pytest.skip("grade report page is unavailable")
    return parse_document(read_document(GRADE_PAGE))


@real_ocr
def test_a_grade_report_page_read_from_pixels_keeps_its_codes_and_grades(grade_page):
    """A grade report has no date, credit or weight to anchor on, only its code."""
    fixture = parse_transcript(GRADE_FIXTURE.read_text(encoding="utf-8"))

    assert grade_page["layout"] == "course_task_grade"
    assert [(record["course_code"], record["section_number"])
            for record in grade_page["course_records"]] == [
        (record["course_code"], record["section_number"])
        for record in fixture["course_records"]]
    # Every grade survives the page pass: a misread mark is repaired from its own
    # crop, and a mark that cannot be confirmed is null instead of wrong.
    assert [record["mark"] for record in grade_page["course_records"]] == [
        record["mark"] for record in fixture["course_records"]]
    assert len(grade_page["assessment_records"]) == len(fixture["assessment_records"])
    stats = grade_page["verification"]
    assert stats["status"] == "checked"
    assert stats["unread"] == 0
    # This page prints no credit at all, so no credit cell is called verified.
    assert stats["verified_fields"] == ["course_code", "mark"]
    assert grade_page["checks"]["credits_printed"] is False


# --------------------------------------------------------------------------- #
# The control page: every cell should confirm, nothing should be "repaired"
# --------------------------------------------------------------------------- #


@real_ocr
def test_every_cell_on_the_control_page_is_confirmed(control, fixture_values):
    stats = control["verification"]

    assert stats["status"] == "checked"
    assert stats["cells_checked"] == 32
    assert stats["confirmed"] == 32
    assert stats["repaired"] == 0
    assert stats["unread"] == 0
    assert _values(control) == fixture_values


@real_ocr
def test_a_confirmed_cell_says_what_the_page_and_the_crop_read(control):
    record = control["course_records"][2]
    check = record["cell_checks"]["mark"]

    assert check["status"] == "confirmed"
    assert check["page_reading"] == check["focused_readings"][0] == "90"
    assert "reason" not in check
    assert "repairs" not in record


# --------------------------------------------------------------------------- #
# The user's 256x202 capture: the two misread marks are repaired, not passed on
# --------------------------------------------------------------------------- #


@real_ocr
def test_the_low_resolution_capture_matches_every_fixture_value(capture, fixture_values):
    assert _values(capture) == fixture_values
    stats = capture["verification"]
    assert stats["cells_checked"] == 32
    assert stats["unread"] == 0
    assert stats["repaired"] == 2


@real_ocr
def test_a_misread_mark_is_repaired_from_its_crop_and_the_repair_is_visible(capture):
    repairs = {record["course_code"]: record.get("repairs", {})
               for record in capture["course_records"]}
    first = repairs["42075y"]["mark"]
    second = repairs["700605y"]["mark"]

    assert (first["page_reading"], first["focused_reading"]) == ("06", "90")
    assert (second["page_reading"], second["focused_reading"]) == ("86", "98")
    assert "disagreed" in first["reason"]
    assert capture["checks"]["repaired_cells"] == 2


# --------------------------------------------------------------------------- #
# Refusing: a cell that cannot be read is emitted as null with the reason
# --------------------------------------------------------------------------- #


class _NonsenseCells:
    """A backend whose focused reads are gibberish: nothing can be confirmed."""

    name = "nonsense"

    def __init__(self) -> None:
        self.batches = 0
        self.images = 0

    def read(self, image):  # pragma: no cover - the page was read before this
        raise AssertionError("the document was already read")

    def read_cells(self, images):
        self.batches += 1
        self.images += len(images)
        return [("zz", 0.9)] * len(images)


@real_ocr
def test_unreadable_cells_become_null_with_a_reason(control_document):
    backend = _NonsenseCells()
    document = dataclasses.replace(control_document, backend=backend)
    parsed = parse_document(document)

    assert backend.batches == 1, "all cell crops on a page must be one batch"
    assert backend.images == 64
    for record in parsed["course_records"]:
        assert record["mark"] is None and record["mark_percent"] is None
        assert "mark" in record["unread"]
        assert "nothing readable" in record["unread"]["mark"]["reason"]
    # Codes keep their page value (they match the printed pattern); every number
    # is refused, because nothing could confirm it.
    assert parsed["checks"]["unread_cells"] == 8 * 3
    assert parsed["checks"]["unconfirmed_cells"] == 8
    assert parsed["checks"]["marks_numeric"] is False


@real_ocr
def test_a_single_batch_per_page_keeps_cell_work_cheap(control_document):
    class _Counting:
        name = "counting"

        def __init__(self, inner):
            self.inner = inner
            self.batches = 0
            self.cells = 0

        def read(self, image):  # pragma: no cover
            return self.inner.read(image)

        def read_cells(self, images):
            self.batches += 1
            self.cells += len(images)
            return self.inner.read_cells(images)

    inner = control_document.backend
    counting = _Counting(inner)
    document = dataclasses.replace(control_document, backend=counting)
    parsed = parse_document(document)

    assert counting.batches == 1
    assert counting.cells == 64
    assert parsed["verification"]["confirmed"] == 32


# --------------------------------------------------------------------------- #
# One record, one truth: a status and its value cannot disagree, and the tallies
# are the sum of the outcomes rather than counters kept in step by hand
# --------------------------------------------------------------------------- #


def assert_reconciled(parsed: dict) -> None:
    """The invariant a reader is entitled to, checked on a whole parsed document."""
    stats = parsed["verification"]
    records = parsed["course_records"]
    statuses = [view["status"] for record in records
                for view in (record.get("cell_checks") or {}).values()]

    # checked equals confirmed + repaired + unconfirmed + unread, at page and document.
    assert stats["cells_checked"] == len(statuses)
    assert stats["cells_checked"] == sum(stats[name] for name in CELL_OUTCOMES)
    assert sum(page["cells_checked"] for page in stats["pages"]) == stats["cells_checked"]
    for name in CELL_OUTCOMES:
        assert sum(page[name] for page in stats["pages"]) == stats[name]
    # The companion lists are exactly the cells they claim to be about.
    assert stats["unread"] == sum(len(record.get("unread") or {}) for record in records)
    assert stats["repaired"] == sum(len(record.get("repairs") or {}) for record in records)
    assert stats["unread"] == statuses.count("unread")
    # No status may claim a value that is null, and no null may sit under a status
    # that says the cell was read.
    for record in records:
        checks = record.get("cell_checks") or {}
        for field, view in checks.items():
            if view["status"] == "unread":
                assert record[field] is None, (field, view)
                assert view["value"] is None, (field, view)
                assert field in (record.get("unread") or {}), (field, view)
            else:
                assert record[field] is not None, (field, view)
                assert view["value"] == record[field], (field, view)
                assert field not in (record.get("unread") or {}), (field, view)
        for field in (record.get("unread") or {}):
            assert checks.get(field, {}).get("status") == "unread", field
        for field in (record.get("repairs") or {}):
            assert checks.get(field, {}).get("status") == "repaired", field


@real_ocr
def test_a_page_whose_credits_are_distrusted_reconciles_its_tally(tmp_path):
    """The 768-pixel rung the audit reproduced: seven credits refused, one repair."""
    parsed = parse_document(read_document(_downscale(tmp_path, 768)))

    assert_reconciled(parsed)
    stats = parsed["verification"]
    assert stats["distrusted_credit_rows"] == 7
    assert stats["cells_checked"] == 28
    refused = [record for record in parsed["course_records"]
               if record["credit"] is None]
    assert len(refused) == 7
    for record in refused:
        check = record["cell_checks"]["credit"]
        assert check["status"] == "unread"
        assert check["value"] is None
        assert "add up to" in check["reason"]
        assert "add up to" in record["unread"]["credit"]["reason"]
    # The page tally is the document tally, not a second opinion: it said `unread: 0`
    # above seven null credits before this was derived from the records.
    assert stats["pages"][0]["unread"] == stats["unread"] == 7
    assert stats["pages"][0]["cells_checked"] == stats["cells_checked"]
    # The document-level checks block agrees with the verification block.
    assert parsed["checks"]["unread_cells"] == stats["unread"]
    assert parsed["checks"]["repaired_cells"] == stats["repaired"]


@real_ocr
def test_a_repaired_cell_and_a_refused_cell_on_one_page_still_reconcile(tmp_path):
    """600 pixels: one mark repaired from its crop, two marks refused, credits distrusted."""
    parsed = parse_document(read_document(_downscale(tmp_path, 600)))

    assert_reconciled(parsed)
    stats = parsed["verification"]
    repaired = [record for record in parsed["course_records"]
                if "mark" in (record.get("repairs") or {})]
    refused = [record for record in parsed["course_records"]
               if (record.get("unread") or {}).get("mark")]
    assert repaired and refused
    # One page carries both, and the arithmetic still closes.
    assert {record["page"] for record in repaired + refused} == {1}
    assert stats["cells_checked"] == 24
    assert stats["cells_checked"] == sum(stats[name] for name in CELL_OUTCOMES)
    for record in refused:
        assert record["mark"] is None and record["mark_percent"] is None
        check = record["cell_checks"]["mark"]
        assert check["value"] is None and check["reason"]
        # The refusal names why it could not be read, and keeps the readings.
        assert check["page_reading"] and check["focused_readings"] is not None
    for record in repaired:
        assert record["cell_checks"]["mark"]["status"] == "repaired"
        assert record["cell_checks"]["mark"]["value"] == record["mark"]


@real_ocr
def test_the_control_page_reconciles_too(control):
    assert_reconciled(control)
    assert control["verification"]["cells_checked"] == 32


# --------------------------------------------------------------------------- #
# The honestedge of the record: a value nothing checked says so, on the value
# --------------------------------------------------------------------------- #


@real_ocr
def test_a_title_and_a_footer_figure_are_marked_unverified_where_they_sit(control):
    title = control["course_records"][0]
    standing = control["academic_standing"]

    assert "title" in title["unverified"]
    assert "read by the page pass only" in title["unverified"]["title"]
    assert {"gpa", "total_credits"} <= set(standing["unverified"])
    # The verified fields are not marked, and the document says which they are.
    for field in ("course_code", "mark", "weight", "credit"):
        assert field not in title["unverified"]
    stats = control["verification"]
    assert stats["verified_fields"] == ["course_code", "mark", "weight", "credit"]
    marked = [name for record in control["course_records"]
              for name in (record.get("unverified") or {})]
    header = control["header"].get("unverified") or {}
    assert stats["unverified_values"] == len(marked) + len(standing["unverified"]) + len(header)
    assert stats["unverified_fields"] == sorted(
        set(marked) | set(standing["unverified"]) | set(header))


def test_a_header_value_is_marked_unverified_too():
    """A name and a school printed at the top are readings nothing re-read."""
    from pace.ocr_verify import mark_unverified_values

    parsed = {
        "header": {"document_title": {"text": "Official Transcript", "start": 0,
                                      "end": 18}},
        "course_records": [], "assessment_records": [],
        "academic_standing": {"gpa": None, "total_credits": None},
        "verification": {"status": "checked"},
    }
    marks = mark_unverified_values(parsed)

    assert marks == {"document_title": 1}
    assert "never re-read" in parsed["header"]["unverified"]["document_title"]
    assert parsed["verification"]["unverified_values"] == 1


def test_a_backend_without_boxes_calls_no_field_verified():
    """`unavailable` must not leave four field names looking checked."""
    from pace.ocr_verify import mark_unverified_values

    parsed = {
        "header": {}, "course_records": [], "assessment_records": [],
        "academic_standing": {"gpa": None, "total_credits": None},
        "verification": {"status": "unavailable",
                         "reason": "no box geometry"},
    }
    mark_unverified_values(parsed)

    assert parsed["verification"]["verified_fields"] == []


@real_ocr
def test_the_compact_summary_marks_the_figures_it_never_checked(control):
    from pace.transcript_extract import summarize

    summary = summarize(control)
    assert set(summary["unverified"]) == {"gpa", "reported_total_credits"}
    assert "never re-read" in summary["unverified"]["gpa"]
    # A bare parse runs no verification stage at all, so it has nothing to mark:
    # the marks come from the document path, not from `parse_transcript`.
    assert "unverified" not in summarize(
        parse_transcript(FIXTURE.read_text(encoding="utf-8")))


def _marked(parsed: dict) -> set[tuple[str, str]]:
    """The (container, field) pairs a record says nothing ever checked."""
    pairs = {("record", field) for record in parsed["course_records"]
             for field in (record.get("unverified") or {})}
    for container in ("header", "academic_standing"):
        pairs |= {(container, field)
                  for field in ((parsed.get(container) or {}).get("unverified") or {})}
    pairs |= {("assessment", field)
              for record in parsed.get("assessment_records") or []
              for field in (record.get("unverified") or {})}
    return pairs


# What an OCR record for this page admits it never checked.
UNCHECKED = {("record", "title"), ("record", "term"),
             ("academic_standing", "gpa"), ("academic_standing", "total_credits")}


@pytest.mark.skipif(not PDF.exists(), reason="the rendered sample PDF is unavailable")
def test_an_exact_record_admits_the_values_it_never_checked():
    """`not_needed` says no crop stage ran; it does not say every value was checked.

    No OCR backend is needed to see this: the marks and the provenance are properties
    of the record, so this half must run everywhere the fixture does.
    """
    from pace.transcript_extract import summarize

    exact = parse_document(read_document(FIXTURE))

    assert _marked(exact) == UNCHECKED
    assert exact["verification"]["unverified_fields"] == [
        "gpa", "term", "title", "total_credits"]
    assert exact["verification"]["unverified_values"] == 18
    # The compact view flattens the same two figures, so it carries their marks too.
    assert set(summarize(exact)["unverified"]) == {"gpa", "reported_total_credits"}
    # Provenance still says the characters are the author's, and that no crop stage ran.
    assert exact["source"]["kind"] == "text"
    assert exact["source"]["exact_source_text"] is True
    assert all(record["evidence_exact"] for record in exact["course_records"])
    assert exact["verification"]["status"] == "not_needed"
    assert "cells_checked" not in exact["verification"]
    assert exact["verification"]["verified_fields"] == []
    # The same admission on the rendered PDF, whose page says how it was read.
    pdf = parse_document(read_document(PDF))
    assert _marked(pdf) == UNCHECKED
    assert pdf["source"]["pages"][0]["method"] == "text_layer"
    assert pdf["verification"]["status"] == "not_needed"
    # Adding the marks did not touch a value: the rows are the parser's rows.
    raw = parse_transcript(FIXTURE.read_text(encoding="utf-8"))
    assert _values(exact) == _values(raw)
    assert [(record["title"], record["term"]) for record in exact["course_records"]] == [
        (record["title"], record["term"]) for record in raw["course_records"]]


@real_ocr
@pytest.mark.skipif(not PDF.exists(), reason="the rendered sample PDF is unavailable")
def test_an_exact_record_and_an_ocr_record_admit_the_same_unchecked_values(control):
    """The boundary is a property of the record, not of how the characters arrived.

    Comparing the two paths needs a recogniser; the exact-path half above does not.
    """
    exact = parse_document(read_document(FIXTURE))

    assert _marked(exact) == _marked(control) == UNCHECKED
    assert exact["verification"]["unverified_fields"] == (
        control["verification"]["unverified_fields"])
    assert exact["verification"]["unverified_values"] == 18
    assert control["verification"]["unverified_values"] == 18


@pytest.mark.skipif(not PDF.exists(), reason="the rendered sample PDF is unavailable")
def test_a_text_layer_page_is_not_verified_against_itself():
    parsed = parse_document(read_document(PDF))

    assert parsed["verification"]["status"] == "not_needed"
    assert all("cell_checks" not in record for record in parsed["course_records"])


def test_the_writer_refuses_a_status_without_its_matching_value():
    """The invariant belongs to the writer, not to the caller's good behaviour."""
    from pace.ocr_verify import CellCheck, _write_cell

    record = {"credit": None, "evidence": {}}
    for status in ("confirmed", "unconfirmed"):
        with pytest.raises(ValueError, match="has no value to emit"):
            _write_cell(record, CellCheck(field="credit", page_reading="1",
                                          status=status))
    with pytest.raises(ValueError, match="cannot carry a value"):
        _write_cell(record, CellCheck(field="credit", page_reading="1",
                                      status="unread", emitted=1.0))
    # Nothing was written before the contradiction was refused.
    assert record == {"credit": None, "evidence": {}}


def test_credit_sum_disagreement_nulls_every_credit():
    parsed = {
        "checks": {"credit_sum_matches_reported_total": False,
                   "credit_sum_complete": True,
                   "computed_total_credits": 5.0,
                   "reported_total_credits": 7.5},
        "course_records": [
            {"credit": 1.0, "evidence": {"credit": {"text": "1"}},
             "cell_checks": {"credit": {"focused_readings": ["1"]}}},
            {"credit": 4.0, "evidence": {"credit": {"text": "4"}},
             "cell_checks": {"credit": {"focused_readings": ["4"]}}},
        ],
    }
    reasons = distrust_unbalanced_credits(parsed)

    assert [record["credit"] for record in parsed["course_records"]] == [None, None]
    assert "add up to 5.0" in reasons[0]
    assert parsed["checks"]["credit_sum_complete"] is False
    # A refused credit is an unread cell, not a null stapled under `confirmed`.
    for record in parsed["course_records"]:
        check = record["cell_checks"]["credit"]
        assert check["status"] == "unread"
        assert check["value"] is None
        # The evidence survives the refusal: the page reading and every focused
        # reading are still there to be looked at.
        assert check["page_reading"] in {"1", "4"}
        assert check["focused_readings"] == [check["page_reading"]]
        assert record["unread"]["credit"]["page_reading"] == check["page_reading"]
        assert "add up to" in record["unread"]["credit"]["reason"]


# --------------------------------------------------------------------------- #
# The ladder: where a smaller page stops reading, it stops by refusing
# --------------------------------------------------------------------------- #


def _downscale(tmp_path: Path, width: int) -> Path:
    from PIL import Image

    control = Image.open(CONTROL).convert("L")
    height = round(control.height * width / control.width)
    path = tmp_path / f"page_{width}.png"
    control.resize((width, height), Image.LANCZOS).save(path)
    return path


@real_ocr
def test_a_truncated_mark_is_unread_not_confirmed(tmp_path):
    parsed = parse_document(read_document(_downscale(tmp_path, 600)))
    record = next(item for item in parsed["course_records"]
                  if item["course_code"] == "200715b")

    assert record["mark"] is None
    assert "width fits about" in record["unread"]["mark"]["reason"]
    # Nothing wrong is left behind on this rung.
    assert all(value != "4" for value in (record["mark"],))


# --------------------------------------------------------------------------- #
# The artifact makes the same admission: the guarantee counts every measurement,
# and the values nothing checked are listed rather than omitted
# --------------------------------------------------------------------------- #

ARTIFACTS = Path("artifacts/ingest")


@pytest.mark.skipif(not (ARTIFACTS / "photo_ladder.json").exists(),
                    reason="the photo ladder has not been measured here")
def test_the_photo_ladder_guarantee_counts_the_user_capture():
    import json

    payload = json.loads((ARTIFACTS / "photo_ladder.json").read_text(encoding="utf-8"))
    guarantee = payload["guarantee"]
    capture = payload["user_capture"]
    counted = payload["ladder"] + ([capture] if capture is not None else [])

    assert guarantee["wrong_values"] == sum(
        entry["wrong_total"] for entry in counted) == 0
    if capture is not None:
        assert guarantee["includes_user_capture"] is True
        assert guarantee["measurements"] == len(payload["ladder"]) + 1
    assert guarantee["unverified"]["checked"] == 0
    assert set(guarantee["unverified"]["fields"]) >= {"title", "gpa", "total_credits"}
    for entry in counted:
        assert entry["cells_refused"] == entry["cells"]["unread"]
        assert entry["wrong_total"] == len(entry["wrong_values"])
        assert entry["cells"]["cells_checked"] == (
            entry["cells"]["confirmed"] + entry["cells"]["repaired"]
            + entry["cells"]["unconfirmed"] + entry["cells"]["unread"])
        assert entry["unverified_values"] == sum(
            detail["values"] for detail in entry["unverified"].values())
        # The photo artifact now publishes the same page rows as the resolution
        # ladder: the reconciliation is visible, not only enforced in code.
        if entry["rows_recovered"]:
            assert entry["pages"], entry["file"]
        for page in entry["pages"]:
            assert page["cells_checked"] == sum(page[name] for name in CELL_OUTCOMES)
        assert sum(page["cells_checked"] for page in entry["pages"]) == (
            entry["cells"]["cells_checked"])
        assert sum(page["unread"] for page in entry["pages"]) == (
            entry["cells"]["unread"])
    # The rung the audit reproduced, as it stands in the artifact: seven credits
    # distrusted, one mark repaired, and the arithmetic closing on the page.
    sensor = next(entry for entry in payload["ladder"]
                  if entry["condition"] == {"width": 768})
    assert sensor["cells"] == {"cells_checked": 28, "confirmed": 20, "repaired": 1,
                               "unconfirmed": 0, "unread": 7}
    assert sensor["cells_refused"] == 7


@pytest.mark.skipif(not (ARTIFACTS / "resolution_ladder.json").exists(),
                    reason="the resolution ladder has not been measured here")
def test_the_resolution_ladder_pages_reconcile_and_its_guarantee_counts_capture():
    import json

    payload = json.loads((ARTIFACTS / "resolution_ladder.json").read_text(encoding="utf-8"))
    counted = payload["ladder"] + ([payload["user_capture"]]
                                   if payload["user_capture"] else [])
    assert payload["guarantee"]["wrong_values"] == sum(
        entry["wrong_total"] for entry in counted) == 0
    for entry in counted:
        stats = entry["verification"]
        assert stats["cells_checked"] == sum(
            stats[name] for name in CELL_OUTCOMES)
        for page in entry["pages"]:
            assert page["cells_checked"] == sum(page[name] for name in CELL_OUTCOMES)
        assert sum(page["cells_checked"] for page in entry["pages"]) == (
            stats["cells_checked"])
        assert sum(page["unread"] for page in entry["pages"]) == stats["unread"]


@real_ocr
def test_a_row_that_could_not_be_read_is_named_in_the_warnings(tmp_path):
    parsed = parse_document(read_document(_downscale(tmp_path, 800)))
    warnings = parsed["source"]["warnings"]

    assert parsed["checks"]["unparsed_row_lines"] >= 1
    assert any("starts like a course row" in warning for warning in warnings)
    for record in parsed["course_records"]:
        assert record["mark"] != "t6"
