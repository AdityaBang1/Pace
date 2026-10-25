"""The spotter's decoder: peaks in, records out, refusals visible.

These tests never train. The decoder tests need no torch; the public-entry tests
run the network in evaluation mode, three of them on the real checkpoint and the
real fixture pages. They pin the properties the experiment rests on: no template is
supplied to the decoder, a tilted row is still one row, a column layout with no
printed commas still splits into its four fields, a character the model is unsure
about makes its field unread rather than wrong, and `detect_and_read` is callable
the way a caller calls it. They also pin the agreement mechanism: rows are paired
only where the pairing is mutual and unique, the second reading is the documented
3 px / 2 px / 1.015 warp of the page rather than the page again, and a network that
cannot be widened to float64 raises rather than being read un-widened.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import pace.page_spotter as spotter_module
from pace.page_spotter import (CHARSET, FIELDS, PAGE_HEIGHT, PAGE_WIDTH,
                               SECOND_VIEW, STRIDE, PageSpotter, _merge_views,
                               agreeing_readings, decode_page, detect_and_read)


def _stamp(presence, scores, character: str, x: float, y: float,
           confidence: float) -> None:
    cell_x, cell_y = int(round(x / STRIDE)), int(round(y / STRIDE))
    # A peak has to be the strict maximum of its neighbourhood, so the ring
    # around each character is high but lower than its centre.
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            presence[cell_y + offset_y, cell_x + offset_x] = (
                confidence if (offset_y, offset_x) == (0, 0) else 0.5)
    scores[cell_y, cell_x, CHARSET.index(character)] = 1.0


def _canvas():
    height, width = 320, 576
    return (np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width, len(CHARSET)), dtype=np.float32))


def _page(row_text: str, *, spacing: float = 9.0, baseline: float = 40.0,
          tilt: float = 0.0, confidence: float = 0.99):
    """A presence map with one printed row and its one-hot character scores."""
    presence, scores = _canvas()
    for index, character in enumerate(row_text):
        x = 20.0 + index * spacing
        _stamp(presence, scores, character, x, baseline + tilt * x, confidence)
    return presence, scores


def _row(items, *, baseline: float = 40.0, spacing: float = 9.0,
         confidence: float = 0.99):
    """One printed row drawn in columns: (text, left) pairs, left to right."""
    presence, scores = _canvas()
    for text, left in items:
        for index, character in enumerate(text):
            if character == " ":
                continue
            _stamp(presence, scores, character, left + index * spacing,
                   baseline, confidence)
    return presence, scores


def test_one_printed_row_decodes_to_its_four_fields_with_no_template():
    presence, scores = _page("220725Y,94,1.0000,1")

    records = decode_page(presence, scores)

    assert len(records) == 1
    record = records[0]
    assert record["course_code"] == "220725Y"
    assert record["mark"] == "94"
    assert record["weight"] == 1.0
    assert record["credit"] == 1.0


def test_a_lower_case_course_code_is_representable():
    presence, scores = _page("220725y,94,1.0000,1")

    record = decode_page(presence, scores)[0]

    assert record["course_code"] == "220725y"


def test_a_column_row_without_commas_splits_into_its_four_fields():
    # The real page prints the code beside the title, then mark, weight and
    # credit right-aligned in their own columns. There is no comma to split on.
    presence, scores = _row([("220725y", 20.0), ("Accel Adv", 120.0),
                             ("94", 282.0), ("1.0000", 326.0), ("1", 430.0)])

    record = decode_page(presence, scores)[0]

    assert record["course_code"] == "220725y"
    assert record["mark"] == "94"
    assert record["weight"] == 1.0
    assert record["credit"] == 1.0


def test_a_weight_printed_without_its_point_is_refused_not_read_as_10000():
    presence, scores = _page("220725Y,94,10000,1")

    record = decode_page(presence, scores)[0]

    assert record["weight"] is None
    assert record["mark"] == "94"


def test_a_number_with_a_point_and_no_trailing_digit_is_refused():
    presence, scores = _page("220725Y,94,1.0000,0.")

    record = decode_page(presence, scores)[0]

    assert record["credit"] is None
    assert record["weight"] == 1.0


def test_a_duplicated_final_peak_refuses_the_code():
    # A second presence maximum four pixels after the last code character is one
    # glyph detected twice, not a two-letter suffix: the code must be refused
    # rather than emitted as a different, legal-looking code.
    presence, scores = _row([("220725Y", 20.0), ("Accel Adv", 120.0),
                             ("94", 282.0), ("1.0000", 326.0), ("1", 430.0)])
    _stamp(presence, scores, "Y", 78.0, 40.0, 0.9)

    record = decode_page(presence, scores)[0]

    assert record["course_code"] is None
    assert record["unread_cells"]["course_code"] is True
    assert record["mark"] == "94"


def test_a_repeated_digit_inside_a_code_is_not_a_duplicate():
    # Two printed zeros sit one advance apart and are two characters; the
    # duplicate rule only looks at the closing letter, so the code is read.
    presence, scores = _row([("700605y", 20.0), ("Accel Adv", 140.0),
                             ("94", 400.0), ("1.0000", 444.0), ("1", 520.0)])

    record = decode_page(presence, scores)[0]

    assert record["course_code"] == "700605y"


def _reading(code: str = "220725y", mark: str = "94", weight: str = "1.0000",
             credit: str = "1", y: float = 40.0):
    return {"field_text": {"course_code": code, "mark": mark, "weight": weight,
                           "credit": credit}, "row_text": "", "row_y": y}


def test_agreement_keeps_fields_both_readings_share():
    merged = agreeing_readings([_reading()], [_reading(y=41.0)])

    assert merged[0]["course_code"] == "220725y"
    assert merged[0]["mark"] == "94"
    assert merged[0]["weight"] == 1.0
    assert merged[0]["credit"] == 1.0


def test_a_disagreement_refuses_one_field_and_keeps_the_rest():
    merged = agreeing_readings([_reading()], [_reading(mark="95")])

    assert merged[0]["mark"] is None
    assert merged[0]["unread_cells"]["mark"] is True
    assert merged[0]["course_code"] == "220725y"


def test_a_character_difference_in_the_code_refuses_the_code():
    merged = agreeing_readings([_reading()], [_reading(code="220726y")])

    assert merged[0]["course_code"] is None
    assert merged[0]["mark"] == "94"


def test_a_row_missing_from_the_second_reading_is_refused_whole():
    merged = agreeing_readings([_reading()], [_reading(y=200.0)])

    assert all(merged[0][field] is None
               for field in ("course_code", "mark", "weight", "credit"))


def test_an_ambiguous_row_is_refused_rather_than_paired_with_a_neighbour():
    """Two second-view rows inside the tolerance: no pairing can be trusted.

    A nearest-row search pairs the 1.2 px one and emits the code; either candidate
    could be the row itself, so the confirmation would be another printed row's.
    """
    first = [_reading(y=100.0)]
    second = [_reading(y=98.8), _reading(y=101.2)]

    merged, unpaired = _merge_views(first, second)

    assert merged[0]["paired"] is False
    assert all(merged[0][field] is None for field in FIELDS)
    assert [row["view"] for row in unpaired] == ["first", "second", "second"]


def test_two_rows_near_one_partner_are_both_refused():
    first = [_reading(y=100.0), _reading(y=101.0, code="914929p")]

    merged, unpaired = _merge_views(first, [_reading(y=100.5)])

    assert [row["paired"] for row in merged] == [False, False]
    assert [row["view"] for row in unpaired] == ["first", "first", "second"]


def test_a_single_view_pairs_every_row_with_itself():
    records = [_reading(y=100.0), _reading(y=101.0, code="914929p")]

    merged, unpaired = _merge_views(records, records)

    assert [row["paired"] for row in merged] == [True, True]
    assert [row["course_code"] for row in merged] == ["220725y", "914929p"]
    assert unpaired == []


def test_a_tilted_row_is_still_one_row():
    presence, scores = _page("220725Y,94,1.0000,1", tilt=0.08)

    records = decode_page(presence, scores)

    assert len(records) == 1
    assert records[0]["course_code"] == "220725Y"


def test_an_uncertain_character_makes_its_field_unread():
    presence, scores = _page("220725Y,94,1.0000,1")
    # The last code character is detected but the model is not confident of it.
    cell_x, cell_y = int(round((20.0 + 6 * 9.0) / STRIDE)), int(round(40.0 / STRIDE))
    presence[cell_y, cell_x] = 0.3

    record = decode_page(presence, scores)[0]

    assert record["course_code"] is None
    assert record["unread_cells"]["course_code"] is True
    # The rest of the row is unaffected: one weak character is one refused field.
    assert record["mark"] == "94"
    assert record["weight"] == 1.0


def test_a_row_that_does_not_split_into_four_fields_is_unread():
    presence, scores = _page("220725Y,94,1.0000")

    record = decode_page(presence, scores)[0]

    assert all(record[field] is None
               for field in ("course_code", "mark", "weight", "credit"))
    assert record["row_text"] == "220725Y,94,1.0000"


def _replayed_logits(text: str, y: float = 100.0):
    """Full-page logits that read exactly ``text`` on one printed row."""
    torch = pytest.importorskip("torch")
    rows, columns = PAGE_HEIGHT // STRIDE, PAGE_WIDTH // STRIDE
    logits = torch.full((1 + len(CHARSET), rows, columns), -12.0,
                        dtype=torch.float16)
    cell_y = int(round(y / STRIDE))
    for index, character in enumerate(text):
        cell_x = int(round((40.0 + index * 9.0) / STRIDE))
        logits[0, cell_y, cell_x] = 12.0
        logits[1 + CHARSET.index(character), cell_y, cell_x] = 12.0
    return logits


def _in_second_view(y: float) -> float:
    """Where a row at page ``y`` is printed in the second view's frame."""
    centre = (PAGE_HEIGHT - 1) / 2.0
    return centre + (y - centre) * SECOND_VIEW["scale"] + SECOND_VIEW["dy"]


class _Replay:
    """A network that replays one prepared reading per call."""

    def __init__(self, readings: list[list[tuple[str, float]]]):
        torch = pytest.importorskip("torch")
        self.readings = [list(reading) for reading in readings]
        self.batches = []
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.parameter

    def __call__(self, batch):
        # Keep the pixels: a stub that ignored them could not tell one view from
        # the other, which is exactly what the second reading must be checked on.
        self.batches.append(batch.detach().clone())
        return pytest.importorskip("torch").stack(
            [_replayed_logits(text, y) for text, y in self.readings.pop(0)])


def test_reading_copy_widens_the_network_and_leaves_the_original_alone():
    torch = pytest.importorskip("torch")
    from pace.page_spotter import reading_copy

    spotter = PageSpotter(len(CHARSET))

    widened = reading_copy(spotter)

    assert next(widened.parameters()).dtype == torch.float64
    assert next(spotter.parameters()).dtype == torch.float32


def test_a_network_that_cannot_widen_is_not_used_unwidened():
    from pace.page_spotter import reading_copy

    class _Stubborn:
        def double(self):
            raise AttributeError("widen")

    with pytest.raises(AttributeError):
        reading_copy(_Stubborn())


def test_a_stub_with_no_widening_is_used_as_it_is():
    from pace.page_spotter import reading_copy

    class _Stub:
        pass

    stub = _Stub()

    assert reading_copy(stub) is stub


def _real_chain_page(name: str):
    """The checkpoint and one fixture page, or a skip when either is missing."""
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from PIL import Image
    from scripts.train_page_spotter import fit_page

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts" / "page-spotter" / "model.pt"
    picture = root / "data" / "samples" / name
    if not checkpoint.exists() or not picture.exists():
        pytest.skip("the checkpoint or the fixture page is missing")
    spotter = PageSpotter(len(CHARSET))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    spotter.load_state_dict(state)
    spotter.eval()
    page, _ = fit_page(Image.open(picture).convert("L"), [])
    return spotter, page, root


def test_the_real_page_reads_its_known_rows_through_the_second_view():
    pytest.importorskip("torch")

    spotter, page, _ = _real_chain_page("transcript_term_marks.png")

    read = detect_and_read(spotter, [page])[0]

    rows = read["rows"]
    assert len(rows) == 15
    assert all(row["paired"] for row in rows)
    assert read["unpaired"] == []
    assert [tuple(row[field] for field in FIELDS) for row in rows
            if all(row[field] is not None for field in FIELDS)] == [
        ("220725y", "94", 1.0, 1.0),
        ("200715b", "94", 0.5, 0.5),
        ("42075y", "90", 1.0, 1.0),
        ("32115y", "93", 1.0, 1.0),
        ("700605y", "98", 1.0, 1.0),
        ("708325y", "100", 1.0, 1.0),
        ("12135y", "91", 1.0, 1.0),
        ("50715y", "97", 1.0, 1.0),
    ]


def test_the_capture_page_refuses_because_the_views_cannot_be_paired(monkeypatch):
    """The known refusals of the 256x202 capture, and the shift that causes them.

    57 rows are decoded, none is a course row and 21 are seen by one reading only:
    9 in the framed view, 12 in the shifted one. Replacing the second view with the
    unshifted page changes the pairing (34 unpaired), which is what makes this test
    fail if `second_view` stops shifting the page.
    """
    pytest.importorskip("torch")

    from scripts._ladder_common import gold_rows
    from scripts.train_page_spotter import summarize

    spotter, page, root = _real_chain_page("transcript_term_marks_256x202.png")

    read = detect_and_read(spotter, [page])[0]

    rows = read["rows"]
    views = [row["view"] for row in read["unpaired"]]
    assert len(rows) == 57
    assert sum(1 for row in rows if row["paired"]) == 48
    assert (views.count("first"), views.count("second")) == (9, 12)

    gold = gold_rows(root / "data" / "samples" / "transcript_term_marks.txt")
    report = summarize([{"gold": gold, "document": "capture"}], [rows])
    assert (report["rows_read_correctly"], report["rows_partly_read"],
            report["rows_refused"]) == (0, 0, 8)
    assert all(report["fields"][field]["emitted"] == 0 for field in FIELDS)

    monkeypatch.setattr(spotter_module, "second_view",
                        lambda picture: np.asarray(picture.convert("L")))
    unshifted = detect_and_read(spotter, [page])[0]

    assert len(unshifted["rows"]) == 57
    assert len(unshifted["unpaired"]) == 34  # 17 + 17, not 21: the shift pairs rows


def test_cpu_and_cuda_read_the_real_page_identically():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from pathlib import Path
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "artifacts" / "page-spotter" / "model.pt"
    page = root / "data" / "samples" / "transcript_term_marks.png"
    if not checkpoint.exists() or not page.exists():
        pytest.skip("the checkpoint or the fixture page is missing")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    cpu = PageSpotter(len(CHARSET))
    cpu.load_state_dict(state)
    cpu.eval()
    gpu = PageSpotter(len(CHARSET))
    gpu.load_state_dict(state)
    gpu.to("cuda")
    gpu.eval()
    image = Image.open(page).convert("L")

    def fields(read):
        return [(row["row_text"], row["course_code"], row["mark"], row["weight"],
                 row["credit"]) for row in read["rows"]]

    assert fields(detect_and_read(cpu, [image])[0]) == fields(
        detect_and_read(gpu, [image])[0])


def test_a_single_view_reads_once_and_pairs_each_row_with_itself():
    pytest.importorskip("torch")
    from PIL import Image

    network = _Replay([[("220725Y,94,1.0000,1", 100.0)]])

    read = detect_and_read(network, [Image.new("L", (320, 400), 255)], views=1)[0]

    assert read["rows"][0]["course_code"] == "220725Y"
    assert read["rows"][0]["paired"] is True
    assert read["unpaired"] == []
    assert network.readings == []  # exactly one forward, no second view


def _inked_page():
    """A page with one block of printed ink, for checking what a view sees."""
    from PIL import Image

    canvas = np.full((400, 320), 255, dtype=np.uint8)
    canvas[100:112, 60:120] = 0
    return Image.fromarray(canvas, mode="L")


def _documented_second_view(image, *, dx: float = 3.0, dy: float = 2.0,
                            scale: float = 1.015) -> np.ndarray:
    """The documented view: 3 px right, 2 px down, enlarged 1.5% about centre."""
    from PIL import Image

    picture = image.convert("L")
    centre_x, centre_y = (picture.width - 1) / 2.0, (picture.height - 1) / 2.0
    coefficients = (1.0 / scale, 0.0, centre_x - (dx + centre_x) / scale,
                    0.0, 1.0 / scale, centre_y - (dy + centre_y) / scale)
    moved = picture.transform(picture.size, Image.AFFINE, coefficients,
                              resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(moved, dtype=np.uint8)


def test_the_second_reading_is_the_documented_warp_of_the_page():
    torch = pytest.importorskip("torch")
    from pace.page_spotter import page_tensor

    page = _inked_page()
    network = _Replay([[("220725Y,94,1.0000,1", 100.0)],
                       [("220725Y,94,1.0000,1", _in_second_view(100.0))]])

    read = detect_and_read(network, [page])[0]

    assert read["rows"][0]["mark"] == "94"
    assert len(network.batches) == 2, "one forward per view"
    first, second = network.batches
    assert torch.equal(first, page_tensor(page).double())
    assert not torch.equal(second, first), "the second view is not the page again"
    assert torch.equal(second, page_tensor(_documented_second_view(page)).double())

    # The documented geometry, checked on the ink rather than on the coefficients.
    assert SECOND_VIEW == {"dx": 3.0, "dy": 2.0, "scale": 1.015}
    assert not np.array_equal(spotter_module.second_view(page), np.asarray(page))
    ink = np.argwhere(np.asarray(page) < 128)
    moved = np.argwhere(spotter_module.second_view(page) < 128)
    centre = (np.array(page.size[::-1], dtype=float) - 1) / 2.0
    expected = (SECOND_VIEW["scale"] * (ink.mean(axis=0) - centre) + centre
                + np.array([SECOND_VIEW["dy"], SECOND_VIEW["dx"]]))
    assert np.allclose(moved.mean(axis=0), expected, atol=1.0)


def test_the_wrong_value_audit_trail_is_not_truncated():
    from scripts.train_page_spotter import summarize

    gold = [{"course_code": f"10000{i}", "mark": "90", "weight": 1.0, "credit": 1.0}
            for i in range(45)]
    records = [{"course_code": f"90000{i}", "mark": "90", "weight": 1.0, "credit": 1.0}
               for i in range(45)]

    report = summarize([{"gold": gold, "document": 0}], [records])

    assert report["wrong_total"] == 45
    assert len(report["wrong_values"]) == 45


def test_the_evaluation_counts_rows_only_one_reading_saw():
    from scripts.train_page_spotter import _count_unpaired

    report: dict = {}
    _count_unpaired(report, [{"unpaired": [{"view": "first"},
                                            {"view": "second"},
                                            {"view": "second"}]}])

    assert report["unpaired_rows"] == 3
    assert report["unpaired_first_view"] == 1
    assert report["unpaired_second_view"] == 2


def test_a_rescore_refuses_to_invent_a_training_record(tmp_path):
    from scripts.train_page_spotter import _stored_record

    assert _stored_record(tmp_path, report_only=True) == {}
    with pytest.raises(SystemExit, match="no metrics.json"):
        _stored_record(tmp_path, report_only=False)

    record = {"training": {"epochs": 12, "wall_seconds": 932.2}}
    (tmp_path / "metrics.json").write_text(json.dumps(record), encoding="utf-8")

    assert _stored_record(tmp_path, report_only=False) == record


def test_detect_and_read_drives_a_spotter_from_page_images():
    pytest.importorskip("torch")
    from PIL import Image

    spotter = PageSpotter(len(CHARSET)).eval()

    reads = detect_and_read(spotter, [Image.new("L", (320, 400), 255)])

    assert len(reads) == 1
    assert set(reads[0]) == {"rows", "unpaired"}
    for row in reads[0]["rows"]:
        assert set(FIELDS) <= set(row)
        if any(row[field] is None for field in FIELDS):
            assert set(row["unread_cells"]) == set(FIELDS)


def test_detect_and_read_emits_only_where_the_second_view_agrees():
    pytest.importorskip("torch")
    from PIL import Image

    network = _Replay([[("220725Y,94,1.0000,1", 100.0)],
                       [("220725Y,94,1.0000,5", _in_second_view(100.0))]])

    read = detect_and_read(network, [Image.new("L", (320, 400), 255)])[0]

    assert len(read["rows"]) == 1
    row = read["rows"][0]
    assert row["course_code"] == "220725Y"
    assert row["mark"] == "94"
    assert row["weight"] == 1.0
    assert row["credit"] is None
    assert row["unread_cells"]["credit"] is True
    assert row["paired"] is True
    assert read["unpaired"] == []
    # Both views were read: the second call consumed the second reading.
    assert network.readings == []


def test_detect_and_read_hands_back_rows_only_one_reading_saw():
    pytest.importorskip("torch")
    from PIL import Image

    blank = Image.new("L", (320, 400), 255)
    network = _Replay([
        [("", 0.0), ("220725Y,94,1.0000,1", 100.0)],
        [("220725Y,94,1.0000,1", _in_second_view(100.0)), ("", 0.0)],
    ])

    first, second = detect_and_read(network, [blank, blank], batch_size=2)

    assert first["rows"] == []
    assert [row["view"] for row in first["unpaired"]] == ["second"]
    assert second["rows"][0]["paired"] is False
    assert all(second["rows"][0][field] is None for field in FIELDS)
    assert [row["view"] for row in second["unpaired"]] == ["first"]
    assert second["unpaired"][0]["row_text"] == "220725Y,94,1.0000,1"
