"""Print a parsed transcript back onto a page, as a PDF or a PNG.

This exists so the ingest path can be checked the way a reader would check it: the
eight rows of a fixture are printed, then read back by `pace parse-transcript`, and
the two answers are compared. The renderer is driven by the parser's own output, so
the page and the fixture cannot drift apart without a test noticing.
"""
from __future__ import annotations

from pathlib import Path

from .transcript_extract import parse_transcript

MARKS_X = 352
WEIGHT_X = 424
CREDIT_X = 486
CODE_X = 40
TITLE_X = 92
TOP = 720
LINE = 16


def _printed_title_lines(text: str, record: dict) -> list[str]:
    """The title exactly as the fixture prints it, one entry per printed line."""
    return [text[span["start"]:span["end"]].strip()
            for span in record["evidence"].get("title_lines") or []]


def render_transcript_pdf(text: str, pdf_path: str | Path) -> Path:
    """Draw a MARK / WEIGHT / CREDIT transcript page from the parsed rows."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    parsed = parse_transcript(text)
    term = (parsed.get("terms") or [{}])[0].get("value") or ""
    standing = parsed["academic_standing"]

    page = canvas.Canvas(str(pdf_path), pagesize=letter)
    page.setFont("Helvetica-Bold", 11)
    page.drawString(CODE_X, TOP, term)

    page.setFont("Helvetica", 10)
    y = TOP - LINE * 1.6
    page.drawString(CODE_X, y, "Course")
    for x, label in ((MARKS_X, "Mark"), (WEIGHT_X, "Weight"), (CREDIT_X, "Credit")):
        page.drawRightString(x, y, label)

    y -= LINE
    page.setFont("Helvetica", 9)
    for record in parsed["course_records"]:
        lines = _printed_title_lines(text, record)
        if not lines:
            continue
        page.drawString(CODE_X, y, record["course_code"])
        page.drawString(TITLE_X, y, lines[0])
        weight = (record["evidence"].get("weight") or {}).get("text") or ""
        credit = (record["evidence"].get("credit") or {}).get("text") or ""
        page.drawRightString(MARKS_X, y, record["mark"])
        page.drawRightString(WEIGHT_X, y, weight)
        page.drawRightString(CREDIT_X, y, credit)
        for extra in lines[1:]:
            y -= LINE
            page.drawString(TITLE_X, y, extra)
        y -= LINE

    y -= LINE * 0.4
    credits = standing["total_credits"]["text"] if standing["total_credits"] else ""
    gpa = standing["gpa"]["text"] if standing["gpa"] else ""
    page.setFont("Helvetica-Bold", 9)
    page.drawString(240, y, f"Credit: {credits}   GPA: {gpa}")
    page.showPage()
    page.save()
    return Path(pdf_path)


def render_transcript_png(pdf_path: str | Path, png_path: str | Path,
                          scale: float = 2.0) -> Path:
    """Rasterise the first page, so the PNG really is the same document.

    ``scale=2.0`` gives about 160 dpi: text a person can read, and small enough
    that the recogniser's own working size is exercised on a realistic screenshot.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        bitmap = document[0].render(scale=scale)
        bitmap.to_pil().convert("L").save(png_path, format="PNG", optimize=True)
    finally:
        document.close()
    return Path(png_path)
