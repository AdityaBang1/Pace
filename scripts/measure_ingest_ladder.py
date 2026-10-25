"""Measure how small this transcript can get before OCR stops reading it exactly.

    python scripts/measure_ingest_ladder.py

The control is the rendered sample page (1224x1584). It is downscaled rung by
rung, and at every rung the CLI path is run in-process: read the image, parse it,
verify each cell, compare against the eight rows the fixture states. The user's
actual 256x202 capture is measured as its own row.

What is counted, per rung: rows found, and for each of mark/code/weight/credit
correct, unread (the verifier refused to emit a value) and wrong (a value that
differs from the fixture). Zero wrong is the bar; unread is honest.

Writes artifacts/ingest/resolution_ladder.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.ingest import parse_document, read_document  # noqa: E402
from scripts._ladder_common import (  # noqa: E402
    FIELDS, field_tally, gold_rows, guarantee_block, unverified_counts)

CONTROL = ROOT / "data" / "samples" / "transcript_term_marks.png"
CAPTURE = ROOT / "data" / "samples" / "transcript_term_marks_256x202.png"
FIXTURE = ROOT / "data" / "samples" / "transcript_term_marks.txt"
OUTPUT = ROOT / "artifacts" / "ingest" / "resolution_ladder.json"

WIDTHS = (1224, 1100, 1024, 900, 800, 700, 600, 512, 448, 400,
          352, 320, 288, 256, 224, 192, 160, 128)


def _measure(path: Path, gold: list[dict]) -> dict:
    started = time.perf_counter()
    document = read_document(path)
    parsed = parse_document(document)
    elapsed = round((time.perf_counter() - started) * 1000)
    rows = parsed["course_records"]
    summary = field_tally(rows, gold)
    verification = parsed.get("verification") or {}
    source = parsed["source"]
    return {
        "file": path.name,
        "width": source["pages"][0]["detail"].get("source_size", [None])[0]
        if source["kind"] == "image" else None,
        "rows": len(rows),
        "elapsed_ms": elapsed,
        "verification": {key: verification.get(key) for key in
                         ("status", "cells_checked", "confirmed", "repaired",
                          "unconfirmed", "unread", "verified_fields",
                          "unverified_values")},
        "pages": verification.get("pages") or [],
        "unverified": unverified_counts(parsed),
        "fields": summary["fields"],
        "wrong_total": summary["wrong_total"],
        "unread_total": sum(summary["fields"][field]["unread"] for field in FIELDS),
        "exact_rows": summary["exact_rows"],
        "source": {"kind": source["kind"], "exact_source_text": source["exact_source_text"]},
    }


def main() -> None:
    from PIL import Image

    gold = gold_rows(FIXTURE)
    ladder = []
    control = Image.open(CONTROL).convert("L")
    with tempfile.TemporaryDirectory(prefix="pace-ladder-") as folder:
        for width in WIDTHS:
            height = max(1, round(control.height * width / control.width))
            path = Path(folder) / f"page_{width}x{height}.png"
            control.resize((width, height), Image.LANCZOS).save(path)
            entry = _measure(path, gold)
            entry["width"], entry["height"] = width, height
            ladder.append(entry)
            print(f"{width:5d}x{height:<5d} rows={entry['rows']} "
                  f"wrong={entry['wrong_total']:2d} unread={entry['unread_total']:2d} "
                  f"repaired={entry['verification'].get('repaired')} "
                  f"exact={entry['exact_rows']} {entry['elapsed_ms']}ms")

    capture = _measure(CAPTURE, gold) if CAPTURE.exists() else None
    if capture is not None:
        print(f"user capture {capture['width']}px rows={capture['rows']} "
              f"wrong={capture['wrong_total']} unread={capture['unread_total']} "
              f"repaired={capture['verification'].get('repaired')}")

    try:
        import rapidocr_onnxruntime
        backend_version = getattr(rapidocr_onnxruntime, "__version__", "unknown")
    except Exception:  # pragma: no cover - only when the backend is absent
        backend_version = None

    exact = [entry["width"] for entry in ladder if entry["exact_rows"]]
    wrong = [entry["width"] for entry in ladder if entry["wrong_total"]]
    counted = ladder + ([capture] if capture is not None else [])
    # The safe boundary is the smallest width that is exact with every larger rung
    # also exact; a lucky rung below a lossy one is not a guarantee.
    contiguous = []
    for entry in ladder:
        if not entry["exact_rows"]:
            break
        contiguous.append(entry["width"])
    payload = {
        "schema_version": 1,
        "control": str(CONTROL.relative_to(ROOT)),
        "capture": str(CAPTURE.relative_to(ROOT)) if CAPTURE.exists() else None,
        "backend": {"name": "rapidocr", "version": backend_version},
        "gold_rows": len(gold),
        "exact_widths": exact,
        "exact_down_to_width": min(contiguous) if contiguous else None,
        "wrong_widths": wrong,
        "smallest_exact_width": min(exact) if exact else None,
        "largest_wrong_width": max(wrong) if wrong else None,
        # One implementation of the sum, shared with the photo ladder.
        "guarantee": guarantee_block(counted, rungs=len(ladder)),
        "ladder": ladder,
        "user_capture": capture,
        "notes": [
            "Each rung is the control page downscaled with LANCZOS to that width.",
            "wrong counts values that differ from the fixture; unread counts values "
            "the verifier refused to emit. The bar is zero wrong.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")
    print(f"exact at: {exact}")
    print(f"wrong at: {wrong}")


if __name__ == "__main__":
    main()
