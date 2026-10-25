"""Record where the sample transcript PDF prints each text run.

A PDF text layer does not arrive as lines: it arrives as the runs the author drew,
each with its own position. PDFKit hands the app one of those per line, which is
why a table read over a text layer arrived as `Course`, then `Mark`, then eight
course codes each on its own line. The app rebuilds the printed lines from the
boxes, and this fixture is the geometry that rebuild is tested against: the real
positions of the real sample, not a hand-drawn approximation.

    python scripts/build_pdf_runs.py

Writes `data/samples/transcript_term_marks_runs.json`, normalised to the page the
way the app's block geometry is.
"""
from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "samples" / "transcript_term_marks.pdf"
OUT = ROOT / "data" / "samples" / "transcript_term_marks_runs.json"

# Helvetica's average advance, in ems. Only used to estimate a run's width, which
# PDFKit reports exactly; the estimate decides how many spaces go between columns.
AVERAGE_ADVANCE = 0.5


def runs_of(page) -> list[dict]:
    runs: list[dict] = []

    def visit(text: str, cm, tm, font_dict, font_size) -> None:
        if not text or not text.strip():
            return
        size = font_size if isinstance(font_size, (int, float)) and font_size > 0 else 10.0
        runs.append({"x": float(tm[4]), "baseline": float(tm[5]), "size": float(size), "text": text.strip()})

    page.extract_text(visitor_text=visit)
    return runs


def main() -> None:
    page = PdfReader(str(PDF)).pages[0]
    box = page.mediabox
    width = float(box.width)
    height = float(box.height)

    raw = runs_of(page)
    blocks = []
    for index, run in enumerate(raw):
        # PDFKit reports a run's own width. The estimate is the run's advance; it is
        # cut short where the next run on the same baseline starts, so a column gap
        # cannot be swallowed by an over-wide box.
        advance = AVERAGE_ADVANCE * run["size"] * len(run["text"])
        same_line = [
            other["x"]
            for other in raw
            if other is not run and abs(other["baseline"] - run["baseline"]) < 0.5 and other["x"] > run["x"]
        ]
        if same_line:
            advance = min(advance, min(same_line) - run["x"] - 2)
        advance = max(advance, 1.0)
        blocks.append(
            {
                "id": f"p0-run-{index}",
                "text": run["text"],
                "page": 0,
                "x": round(run["x"] / width, 6),
                "y": round(1 - (run["baseline"] + run["size"]) / height, 6),
                "width": round(advance / width, 6),
                "height": round(run["size"] / height, 6),
                "confidence": 1.0,
            }
        )

    OUT.write_text(json.dumps(blocks, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(blocks)} runs")


if __name__ == "__main__":
    main()
