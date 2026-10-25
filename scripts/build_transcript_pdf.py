"""Render the sample transcript fixture as a PDF, and that PDF as a PNG.

    python scripts/build_transcript_pdf.py

Both artifacts live beside the fixture so `pace parse-transcript` can be pointed at
them directly; `tests/test_ingest.py` rebuilds them in a temporary directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.transcript_render import render_transcript_pdf, render_transcript_png  # noqa: E402

FIXTURE = ROOT / "data" / "samples" / "transcript_term_marks.txt"
PDF = ROOT / "data" / "samples" / "transcript_term_marks.pdf"
PNG = ROOT / "data" / "samples" / "transcript_term_marks.png"


if __name__ == "__main__":
    text = FIXTURE.read_text(encoding="utf-8")
    print(render_transcript_pdf(text, PDF))
    print(render_transcript_png(PDF, PNG))
