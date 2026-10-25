"""Coverage for reading a PDF or an image: provenance, the OCR path, and refusal."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pace import ingest
from pace.ingest import (
    Document,
    IngestError,
    OcrUnavailable,
    PageText,
    boxes_to_text,
    ocr_backend_name,
    parse_document,
    prepare_for_ocr,
    read_document,
    repair_ocr_spacing,
)
from pace.transcript_extract import parse_transcript

FIXTURE = Path("data/samples/transcript_term_marks.txt")

pytest.importorskip("pypdf")
reportlab = pytest.importorskip("reportlab")


def _document(path: Path, **kwargs) -> Document:
    return read_document(path, **kwargs)


def _row_view(parsed: dict) -> list[tuple]:
    return [(record["course_code"], record["title"], record["mark"],
             record["weight"], record["credit"])
            for record in parsed["course_records"]]


@pytest.fixture(scope="module")
def fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory, fixture_text) -> Path:
    from pace.transcript_render import render_transcript_pdf

    return render_transcript_pdf(fixture_text, tmp_path_factory.mktemp("pdf") / "page.pdf")


@pytest.fixture(scope="module")
def png_path(tmp_path_factory, pdf_path) -> Path:
    pypdfium2 = pytest.importorskip("pypdfium2")
    assert pypdfium2 is not None
    from pace.transcript_render import render_transcript_png

    return render_transcript_png(pdf_path, tmp_path_factory.mktemp("png") / "page.png")


class _EchoBackend:
    """A recogniser stand-in: proves the wiring without pretending to read pixels."""

    name = "echo"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def read(self, image):
        from pace.ingest import OcrReading

        self.calls += 1
        return OcrReading(self.text, {"backend": self.name, "boxes": 1, "lines": 1})


# --------------------------------------------------------------------------- #
# The text layer: the author's own characters
# --------------------------------------------------------------------------- #


def test_a_pdf_text_layer_reads_the_same_rows_as_the_text_fixture(pdf_path, fixture_text):
    document = _document(pdf_path, backend=_EchoBackend("should never be called"))
    parsed = parse_document(document)

    assert document.kind == "pdf"
    assert [page.method for page in document.pages] == ["text_layer"]
    assert document.exact is True
    assert parsed["source"]["exact_source_text"] is True
    assert _row_view(parsed) == _row_view(parse_transcript(fixture_text))
    assert parsed["checks"]["computed_total_credits"] == 7.5
    assert parsed["checks"]["reported_total_credits"] == 7.5
    assert parsed["checks"]["credit_sum_matches_reported_total"] is True


def test_text_layer_values_are_exact_slices_of_the_extracted_page_text(pdf_path):
    document = _document(pdf_path)
    parsed = parse_document(document)

    assert all(record["evidence_exact"] is True for record in parsed["course_records"])
    assert all(record["evidence_method"] == "text_layer" for record in parsed["course_records"])
    for record in parsed["course_records"]:
        # The code's evidence span is inside the page's own characters, so it can
        # be quoted back to a reader rather than described.
        span = record["evidence"]["course_code"]
        assert document.text[span["start"]:span["end"]] == record["course_code"]


def test_a_text_file_is_a_page_of_its_own_kind(fixture_text):
    document = _document(FIXTURE)
    parsed = parse_document(document)

    assert document.kind == "text"
    assert document.pages[0].method == "text_file"
    assert parsed["source"]["exact_source_text"] is True
    assert len(parsed["course_records"]) == 8


def test_a_pdf_page_without_a_text_layer_is_read_from_pixels(tmp_path):
    from reportlab.pdfgen import canvas

    blank = tmp_path / "scan.pdf"
    page = canvas.Canvas(str(blank))
    page.showPage()  # a page with no text at all, like a scanned sheet
    page.save()

    backend = _EchoBackend("220725y Accel Adv    94 1.0000 1\nCredit: 1.000 GPA: 4.0")
    document = _document(blank, backend=backend)
    assert [page.method for page in document.pages] == ["ocr"]
    assert backend.calls == 1


# --------------------------------------------------------------------------- #
# The OCR path: a reading of pixels, never the author's text
# --------------------------------------------------------------------------- #


def test_an_image_is_read_from_pixels_and_flagged_as_not_exact(png_path, fixture_text):
    parsed = parse_document(_document(png_path, backend=_EchoBackend(fixture_text)))

    assert parsed["source"]["methods"] == {"ocr": 1}
    assert parsed["source"]["exact_source_text"] is False
    assert "reading of rendered pixels" in parsed["source"]["note"]
    assert [record["evidence_method"] for record in parsed["course_records"]] == ["ocr"] * 8
    assert all(record["evidence_exact"] is False for record in parsed["course_records"])


def test_an_image_is_upscaled_before_recognition(png_path):
    document = _document(png_path, backend=_EchoBackend("text"))
    detail = document.pages[0].detail
    assert detail["upscale"] > 1.0
    assert detail["backend"] == "echo"


def test_ocr_always_reads_a_text_layer_pdf_from_pixels_too(pdf_path):
    backend = _EchoBackend("220725y Accel Adv    94 1.0000 1\nCredit: 1.000 GPA: 4.0")
    document = _document(pdf_path, ocr="always", backend=backend)
    assert [page.method for page in document.pages] == ["ocr"]
    assert backend.calls == 1


def test_a_sparse_text_layer_page_is_recognised_instead_of_trusted(tmp_path):
    from reportlab.pdfgen import canvas

    thin = tmp_path / "thin.pdf"
    page = canvas.Canvas(str(thin))
    page.setFont("Helvetica", 9)
    page.drawString(40, 700, "Page 1 of 3")  # a stamp, not the document
    page.showPage()
    page.save()

    backend = _EchoBackend("220725y Accel Adv    94 1.0000 1\nCredit: 1.000 GPA: 4.0")
    document = _document(thin, backend=backend)
    assert document.pages[0].method == "ocr"
    assert backend.calls == 1


# --------------------------------------------------------------------------- #
# Refusals: never empty text that looks like a blank page
# --------------------------------------------------------------------------- #


def test_a_missing_backend_error_says_the_two_commands_that_fix_it(monkeypatch, png_path):
    def missing():
        raise OcrUnavailable(ingest.OCR_INSTALL_HINT)

    monkeypatch.setattr(ingest, "ocr_backend", missing)
    with pytest.raises(IngestError) as caught:
        read_document(png_path)
    message = str(caught.value)
    assert "rapidocr-onnxruntime" in message
    assert "tesseract" in message


def test_an_image_with_ocr_never_is_refused_rather_than_read_as_nothing(png_path):
    with pytest.raises(IngestError, match="no text layer"):
        read_document(png_path, ocr="never")


def test_a_png_with_a_text_suffix_is_not_mistaken_for_text(tmp_path):
    fake = tmp_path / "screenshot.txt"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    # The bytes, not the suffix, decide what a file is; this is a broken image, and
    # the error says so instead of complaining about text encoding.
    with pytest.raises(IngestError, match="as an image"):
        read_document(fake)


# --------------------------------------------------------------------------- #
# Small pieces that make the OCR text readable
# --------------------------------------------------------------------------- #


def test_columns_survive_the_journey_through_boxes():
    boxes = [
        ([[10, 10], [90, 10], [90, 30], [10, 30]], "220725y", "0.9"),
        ([[100, 10], [200, 10], [200, 30], [100, 30]], "Accel Adv", "0.9"),
        ([[300, 10], [330, 10], [330, 30], [300, 30]], "94", "0.9"),
        ([[360, 10], [420, 10], [420, 30], [360, 30]], "1.0000", "0.9"),
        ([[460, 10], [480, 10], [480, 30], [460, 30]], "1", "0.9"),
    ]
    text, detail = boxes_to_text(boxes)
    assert text.startswith("220725y Accel Adv")
    # Each column gap became two or more spaces, which is what a cell is.
    assert re.search(r"\s{2,}94\s+1\.0000\s+1$", text)
    assert detail["boxes"] == 5
    assert detail["min_confidence"] == 0.9


def _one_page(text: str, method: str) -> Document:
    return Document(path="x", kind="image",
                    pages=(PageText(number=1, text=text, method=method, start=0),))


def test_a_read_page_whose_rows_disagree_with_the_footer_is_flagged():
    text = ("2025-2026 Grade 09 Term 2\nCourse  Mark  Weight  Credit\n"
            "220725y Accel Adv    94  1.0000  1\nCredit: 9.000  GPA: 4.0\n")
    parsed = parse_document(_one_page(text, "ocr"))

    assert parsed["checks"]["credit_sum_matches_reported_total"] is False
    assert any("do not add up" in warning for warning in parsed["source"]["warnings"])


def test_an_exact_page_is_not_flagged_for_the_same_arithmetic():
    text = ("2025-2026 Grade 09 Term 2\nCourse  Mark  Weight  Credit\n"
            "220725y Accel Adv    94  1.0000  1\nCredit: 9.000  GPA: 4.0\n")
    parsed = parse_document(_one_page(text, "text_layer"))

    assert "warnings" not in parsed["source"]


def test_a_space_lost_inside_a_term_banner_is_repaired_and_reported():
    text, repairs = repair_ocr_spacing("2025-2026Grade09Term2")
    assert text == "2025-2026 Grade 09 Term 2"
    assert repairs and "term banner spacing" in repairs[0]


def test_a_banner_that_lost_only_some_spaces_is_repaired_too():
    text, repairs = repair_ocr_spacing("2025-2026 Grade09 Term2")
    assert text == "2025-2026 Grade 09 Term 2"
    assert repairs


def test_an_already_spaced_banner_is_not_reported_as_a_repair():
    line = "2025-2026 Grade 09 Term 2"
    text, repairs = repair_ocr_spacing(line)
    assert text == line
    assert repairs == []


def test_repairs_leave_a_course_code_alone():
    line = "220725y  Accel Adv    94 1.0000 1"
    text, repairs = repair_ocr_spacing(line)
    assert text == line
    assert repairs == []


def test_a_tall_image_is_not_scaled_down():
    from PIL import Image

    image = Image.new("L", (2400, 3200))
    prepared, factor = prepare_for_ocr(image)
    assert prepared is image and factor == 1.0


def test_a_low_resolution_capture_says_so_in_the_summary(tmp_path):
    document = Document(
        path="x", kind="image",
        pages=(PageText(number=1, text=FIXTURE.read_text(encoding="utf-8"),
                        method="ocr", start=0),),
        warnings=("this image is only 256x202 pixels",),
    )
    parsed = parse_document(document)
    from pace.ingest import summarize_source

    compact = summarize_source(parsed)
    assert compact["warnings"] == ["this image is only 256x202 pixels"]


def test_page_offsets_line_up_with_the_joined_text():
    pages = (
        PageText(number=1, text="abc", method="text_layer", start=0),
        PageText(number=2, text="def", method="ocr", start=5),
    )
    document = Document(path="x", kind="pdf", pages=pages)
    assert document.text == "abc\n\ndef"
    assert document.page_for(0).number == 1
    assert document.page_for(5).number == 2
    assert document.page_for(4).number == 1


# --------------------------------------------------------------------------- #
# Careless inputs: a user should get a reason, not a traceback
# --------------------------------------------------------------------------- #


def test_a_directory_is_named_as_a_directory_not_a_permission_problem(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    with pytest.raises(IngestError, match="is a directory"):
        read_document(folder)


def test_a_corrupt_pdf_is_a_clean_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(IngestError, match="as a PDF"):
        read_document(path)


def test_a_truncated_image_is_a_clean_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 40)
    with pytest.raises(IngestError, match="as an image"):
        read_document(path)


def test_a_blank_image_does_not_tell_the_user_to_install_an_ocr_backend(tmp_path):
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.new("RGB", (20, 10), "white").save(path)
    with pytest.raises(IngestError, match="no readable text") as caught:
        read_document(path)
    assert "install" not in str(caught.value)


def test_a_screenshot_without_an_extension_is_read_as_an_image(tmp_path):
    import shutil

    path = tmp_path / "screenshot"
    shutil.copyfile("data/samples/transcript_term_marks.png", path)
    document = read_document(path)
    parsed = parse_document(document)

    assert document.kind == "image"
    assert document.pages[0].method == "ocr"
    assert len(parsed["course_records"]) == 8


def test_a_windows_1252_file_is_read_with_a_visible_warning(tmp_path):
    path = tmp_path / "saved-by-word.txt"
    path.write_bytes(("Caf\u00e9 High School\n" + FIXTURE.read_text(encoding="utf-8"))
                     .encode("cp1252"))
    document = read_document(path)
    parsed = parse_document(document)

    assert document.pages[0].detail["encoding"] == "cp1252"
    assert any("Windows-1252" in warning for warning in document.warnings)
    assert len(parsed["course_records"]) == 8


def test_stdin_is_text_input(monkeypatch):
    import io
    import sys

    payload = FIXTURE.read_bytes()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))
    document = read_document("-")

    assert document.path == "<stdin>"
    assert document.kind == "text"
    assert len(parse_document(document)["course_records"]) == 8


def test_a_multi_frame_image_says_only_the_first_frame_is_read(tmp_path):
    from PIL import Image

    path = tmp_path / "pages.tiff"
    with Image.open("data/samples/transcript_term_marks.png") as handle:
        page = handle.convert("L")
    page.save(path, save_all=True, append_images=[page])

    document = read_document(path)
    assert any("frames; only the first frame" in warning for warning in document.warnings)


def test_ocr_always_on_text_says_it_does_not_apply():
    document = read_document(FIXTURE, ocr="always")
    assert any("--ocr does not apply" in warning for warning in document.warnings)


def test_each_term_block_is_read_not_only_the_first(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "two-pages.pdf"
    page = canvas.Canvas(str(path))
    for _ in range(2):
        y = 740
        for line in FIXTURE.read_text(encoding="utf-8").splitlines():
            page.drawString(40, y, line.rstrip())
            y -= 12
        page.showPage()
    page.save()

    parsed = parse_document(read_document(path))
    per_page = [record["page"] for record in parsed["course_records"]]

    assert len(parsed["course_records"]) == 16
    assert per_page.count(1) == 8 and per_page.count(2) == 8
    assert all(record["term"] == "2025-2026 Grade 09 Term 2"
               for record in parsed["course_records"])


# --------------------------------------------------------------------------- #
# The real recogniser, when this machine has one
# --------------------------------------------------------------------------- #

BACKEND = ocr_backend_name()


@pytest.mark.skipif(BACKEND is None, reason="no OCR backend installed on this machine")
def test_a_photo_turned_by_the_camera_is_straightened_before_it_is_read(png_path, tmp_path):
    """A tilted capture loses the table; the page is turned back before recognition."""
    from PIL import Image

    with Image.open(png_path) as handle:
        page = handle.convert("L")
    margin = 60
    canvas = Image.new("L", (page.width + 2 * margin, page.height + 2 * margin), 210)
    canvas.paste(page, (margin, margin))
    photo = canvas.rotate(4, resample=Image.BICUBIC, expand=True, fillcolor=210)
    source = tmp_path / "hand-held.png"
    photo.save(source)

    document = read_document(source)
    parsed = parse_document(document)
    angle = document.pages[0].detail.get("deskew_degrees")

    assert angle is not None and -5.0 <= angle <= -3.0
    assert len(parsed["course_records"]) == 8
    assert parsed["checks"]["computed_total_credits"] == 7.5
    assert parsed["checks"]["credit_sum_matches_reported_total"] is True
    # The focused cell re-reads use the corrected geometry; if they used the
    # tilted original every box would land in the wrong place.
    assert parsed["verification"]["unread"] == 0


@pytest.mark.skipif(BACKEND is None, reason="no OCR backend installed on this machine")
def test_the_real_backend_reads_the_rendered_page_row_for_row(png_path, fixture_text):
    """The end-to-end OCR result, not a stub. Skipped when no backend is installed."""
    parsed = parse_document(_document(png_path))

    assert parsed["source"]["pages"][0]["method"] == "ocr"
    assert parsed["source"]["pages"][0]["detail"]["backend"] == BACKEND
    assert _row_view(parsed) == _row_view(parse_transcript(fixture_text))
    assert parsed["checks"]["computed_total_credits"] == 7.5
    assert parsed["checks"]["credit_sum_matches_reported_total"] is True
