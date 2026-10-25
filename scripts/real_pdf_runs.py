"""Record the runs of a PDF as the app's block geometry, for offline diagnosis.

    python scripts/real_pdf_runs.py data/samples/real_transcript.pdf out.json

A text layer arrives as the runs the author drew, which is what PDFKit reports
one per "line". Positions are the text matrix combined with the CTM, because a
generator may put the position in either one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader

AVERAGE_ADVANCE = 0.5


def runs_of(page) -> list[dict]:
    runs: list[dict] = []

    def visit(text: str, cm, tm, font_dict, font_size) -> None:
        if not text or not text.strip():
            return
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        size = font_size if isinstance(font_size, (int, float)) and font_size > 0 else 10.0
        runs.append({"x": x, "y": y, "size": float(size), "text": text.strip()})

    page.extract_text(visitor_text=visit)
    return runs


def main() -> None:
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    reader = PdfReader(str(source))
    blocks: list[dict] = []
    for page_index, page in enumerate(reader.pages):
        box = page.mediabox
        width = float(box.width)
        height = float(box.height)
        runs = runs_of(page)
        for index, run in enumerate(runs):
            advance = AVERAGE_ADVANCE * run["size"] * len(run["text"])
            same_line = [
                other["x"]
                for other in runs
                if other is not run and abs(other["y"] - run["y"]) < 0.6 and other["x"] > run["x"]
            ]
            if same_line:
                advance = min(advance, min(same_line) - run["x"] - 1)
            blocks.append(
                {
                    "id": f"p{page_index}-run-{index}",
                    "text": run["text"],
                    "page": page_index,
                    "x": round(run["x"] / width, 6),
                    "y": round(1 - (run["y"] + run["size"]) / height, 6),
                    "width": round(max(advance, 1.0) / width, 6),
                    "height": round(run["size"] / height, 6),
                    "confidence": 1.0,
                }
            )
    target.write_text(json.dumps(blocks, indent=1) + "\n", encoding="utf-8")
    print(f"{source} -> {target}: {len(blocks)} runs, {len(reader.pages)} page(s)")


if __name__ == "__main__":
    main()
