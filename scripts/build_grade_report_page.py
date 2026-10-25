"""Draw the grade-report fixture onto a page, and that page as a PNG.

    python scripts/build_grade_report_page.py

`transcript_render.py` prints a MARK / WEIGHT / CREDIT page from the parser's own
output, so it cannot draw a layout the parser has no rows for. This script draws the
fixture's printed lines verbatim instead, one monospace line per printed line, which
keeps the page and the fixture from drifting apart: what is checked is exactly what
the file says, wrapped lines and lost labels included.

The page exists so `pace parse-transcript` can be pointed at pixels and the answer
compared with the same file read as text.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.transcript_render import render_transcript_png  # noqa: E402

FIXTURE = ROOT / "data" / "samples" / "grade_report.txt"
PDF = ROOT / "data" / "samples" / "grade_report.pdf"
PNG = ROOT / "data" / "samples" / "grade_report.png"

LEFT = 36
TOP = 740
LINE = 11.5
FONT_SIZE = 8


def render(path: str | Path = PDF) -> Path:
    """Draw every printed line of the fixture at its own monospace position."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    page = canvas.Canvas(str(path), pagesize=letter)
    page.setFont("Courier", FONT_SIZE)
    y = TOP
    for line in lines:
        if line.strip():
            page.drawString(LEFT, y, line)
        y -= LINE
    page.showPage()
    page.save()
    return Path(path)


if __name__ == "__main__":
    print(render())
    print(render_transcript_png(PDF, PNG))
