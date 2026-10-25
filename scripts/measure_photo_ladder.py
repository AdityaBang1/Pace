"""Measure the transcript page the way a phone camera delivers it.

    python scripts/measure_photo_ladder.py

The control is the rendered sample page (1224x1584). Each rung degrades it the
way a hand-held capture does — a few degrees of rotation, keystone perspective,
an illumination gradient, a dark exposure, defocus, JPEG compression, a smaller
sensor — alone and in combination. Every rung is then read through the real
in-process CLI path: `read_document` (OCR, page text, boxes), `parse_document`
(rows, cell verification) and a comparison against the eight rows the fixture
states.

Counted per rung: rows recovered, cells confirmed/repaired/unread by the focused
re-read, values that differ from the fixture, and wall time. The bar is zero
wrong values at every rung: a value is either right or visibly unread with a
reason. Anything else is a defect to fix, not a number to publish.

Writes artifacts/ingest/photo_ladder.json.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
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
OUTPUT = ROOT / "artifacts" / "ingest" / "photo_ladder.json"


def _perspective_coefficients(destination, source) -> tuple[float, ...]:
    """PIL coefficients mapping destination corners onto the page's own corners."""
    import numpy as np

    matrix = []
    target = []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        matrix.append([0, 0, 0, u, v, 1, -y * u, -y * v])
        target.extend((x, y))
    solution = np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(target, dtype=float))
    return tuple(float(value) for value in solution)


def _on_canvas(image, margin: int, background: int):
    """Put the page on a plain background, as a camera sees it on a desk."""
    from PIL import Image

    canvas = Image.new("L", (image.width + 2 * margin, image.height + 2 * margin),
                       background)
    canvas.paste(image, (margin, margin))
    return canvas


def _degrade(image, *, rotation: float = 0.0, perspective: float = 0.0,
             gradient: float = 1.0, exposure: float = 1.0, blur: float = 0.0,
             motion: int = 0, jpeg: int | None = None, width: int | None = None):
    """Apply the degradations a hand-held capture applies, in capture order.

    The page sits on a plain background with a margin, so perspective moves its
    corners inside the frame instead of cropping the printed edge — the first
    version of this harness clipped the first characters and measured a crop, not
    a photograph.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    background = 210
    margin = round(image.width * 0.06)
    result = _on_canvas(image, margin, background)
    if perspective:
        # The page is warped into a trapezoid inside the photo; every printed
        # character stays on the page, so nothing is cropped away.
        inset = perspective * image.width
        left, top = margin, margin
        right, bottom = margin + image.width, margin + image.height
        destination = ((left + inset, top), (right - inset, top),
                       (right, bottom), (left, bottom))
        source = ((left, top), (right, top), (right, bottom), (left, bottom))
        coefs = _perspective_coefficients(destination, source)
        result = result.transform(result.size, Image.PERSPECTIVE, coefs,
                                  resample=Image.BICUBIC, fillcolor=background)
    if rotation:
        result = result.rotate(rotation, resample=Image.BICUBIC, expand=True,
                               fillcolor=background)
    if gradient != 1.0 or exposure != 1.0:
        pixels = np.asarray(result, dtype=np.float32)
        # Light falls off across the sheet: full brightness at the top-left, and
        # `gradient` of it at the bottom-right. Exposure then scales the whole page.
        ramp = np.linspace(1.0, gradient, result.width, dtype=np.float32)[None, :]
        ramp = ramp * np.linspace(1.0, 1.0, result.height, dtype=np.float32)[:, None]
        pixels = pixels * ramp * exposure
        result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
    if blur:
        result = result.filter(ImageFilter.GaussianBlur(blur))
    if motion:
        # A 5-pixel horizontal smear: the hand moved while the shutter was open.
        kernel = [1 / motion if row == 2 else 0.0
                  for row in range(5) for _column in range(5)]
        result = result.filter(ImageFilter.Kernel((5, 5), kernel))
    if jpeg is not None:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=jpeg)
        buffer.seek(0)
        with Image.open(buffer) as handle:
            result = handle.convert("L")
    if width and result.width != width:
        height = max(1, round(result.height * width / result.width))
        result = result.resize((width, height), Image.LANCZOS)
    return result


def _cli(path: Path) -> tuple[dict, int]:
    """Run the real `pace parse-transcript` command and read its JSON."""
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-m", "pace.cli", "parse-transcript", str(path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = round((time.perf_counter() - started) * 1000)
    if process.returncode != 0:
        raise SystemExit(f"parse-transcript failed for {path.name}: "
                         f"{process.stderr.strip()[:500]}")
    return json.loads(process.stdout), elapsed


def _measure(path: Path, gold: list[dict]) -> dict:
    parsed, elapsed = _cli(path)
    rows = parsed["course_records"]
    summary = field_tally(rows, gold)
    verification = parsed.get("verification") or {}
    page = (parsed["source"].get("pages") or [{}])[0]
    return {
        "file": path.name,
        "deskew_degrees": page.get("detail", {}).get("deskew_degrees"),
        "rows_recovered": len(rows),
        "rows_expected": len(gold),
        "wall_ms": elapsed,
        "cells": {key: verification.get(key) for key in
                  ("cells_checked", "confirmed", "repaired", "unconfirmed", "unread")},
        # The page-level tallies, so the artifact shows the same reconciliation the
        # resolution ladder does: checked equals the sum of the four outcomes.
        "pages": verification.get("pages") or [],
        "cells_repaired": sum(len(record.get("repairs") or {}) for record in rows),
        # A value the verifier refused (a null with a reason) is not a wrong
        # value; this counts the cells whose status is `unread`, including the
        # credits the page's arithmetic refused.
        "cells_refused": verification.get("unread", 0),
        "cells_missing": summary["missing_total"],
        # Every value outside the verified fields is named here, per field, with
        # the number of such values this rung emitted.
        "unverified": unverified_counts(parsed),
        "unverified_values": verification.get("unverified_values", 0),
        "fields": summary["fields"],
        "wrong_values": summary["wrong_values"],
        "wrong_total": summary["wrong_total"],
        "exact_rows": summary["exact_rows"],
    }


CONDITIONS = [
    ("clean_control", {}),
    ("rotate_2deg", {"rotation": 2}),
    ("rotate_4deg", {"rotation": 4}),
    ("rotate_6deg", {"rotation": 6}),
    ("keystone_5pct", {"perspective": 0.05}),
    ("keystone_9pct", {"perspective": 0.09}),
    ("gradient_0.55", {"gradient": 0.55}),
    ("exposure_0.45", {"exposure": 0.45}),
    ("exposure_0.30", {"exposure": 0.30}),
    ("blur_0.8", {"blur": 0.8}),
    ("blur_1.5", {"blur": 1.5}),
    ("motion_blur_5px", {"motion": 5}),
    ("jpeg_45", {"jpeg": 45}),
    ("jpeg_30", {"jpeg": 30}),
    ("sensor_768", {"width": 768}),
    ("phone_1024", {"rotation": 4, "perspective": 0.05, "gradient": 0.6,
                    "blur": 0.8, "jpeg": 50, "width": 1024}),
    ("phone_768", {"rotation": 3, "perspective": 0.04, "gradient": 0.7,
                   "blur": 0.6, "jpeg": 45, "width": 768}),
    ("phone_512", {"rotation": 3, "perspective": 0.04, "gradient": 0.7,
                   "blur": 0.6, "jpeg": 45, "width": 512}),
]


def _timed(path: Path, repeats: int = 3) -> dict:
    """Warm per-page cost of reading and verifying the same image, in milliseconds."""
    reads, verifies, totals = [], [], []
    for _ in range(repeats):
        started = time.perf_counter()
        document = read_document(path)
        read_ms = (time.perf_counter() - started) * 1000
        parsed = parse_document(document)
        total_ms = (time.perf_counter() - started) * 1000
        reads.append(read_ms)
        totals.append(total_ms)
        verification_ms = (parsed.get("verification") or {}).get("elapsed_ms")
        if verification_ms is not None:
            verifies.append(float(verification_ms))
    middle = lambda values: round(sorted(values)[len(values) // 2], 1) if values else None
    return {"read_ms": middle(reads), "verify_ms": middle(verifies),
            "total_ms": middle(totals), "repeats": repeats}


def _on_device(paths: dict[str, Path]) -> dict:
    """What the OCR path costs on a phone: model bytes on disk and ms per page."""
    import rapidocr_onnxruntime

    package = Path(rapidocr_onnxruntime.__file__).parent
    models = [{"file": str(file.relative_to(package)).replace("\\", "/"),
               "bytes": file.stat().st_size}
              for file in sorted(package.rglob("*.onnx"))]
    return {
        "backend": "rapidocr",
        "model_files": models,
        "model_bytes": sum(model["bytes"] for model in models),
        "pages": {name: _timed(path) for name, path in paths.items()},
        "note": ("Wall time includes deskew estimation, upscaling, recognition, the "
                 "focused cell re-reads and parsing; the recogniser is warm after the "
                 "first page. Measured on this CPU, not on a phone."),
    }


def main() -> None:
    from PIL import Image

    gold = gold_rows(FIXTURE)
    control = Image.open(CONTROL).convert("L")
    ladder = []
    with tempfile.TemporaryDirectory(prefix="pace-photo-") as folder:
        # Warm the recogniser before the first rung, so the ladder measures the
        # path and not one-time model loading; the cold cost is recorded below.
        warm_path = Path(folder) / "warm.png"
        control.save(warm_path)
        started = time.perf_counter()
        warm = parse_document(read_document(warm_path))
        cold_ms = round((time.perf_counter() - started) * 1000)
        paths = {}
        for name, settings in CONDITIONS:
            image = _degrade(control, **settings)
            path = Path(folder) / f"{name}.png"
            image.save(path)
            paths[name] = path
            entry = _measure(path, gold)
            entry["condition"] = settings
            entry["input_size"] = [image.width, image.height]
            ladder.append(entry)
            print(f"{name:16s} {image.width}x{image.height:<5d} "
                  f"rows={entry['rows_recovered']} wrong={len(entry['wrong_values']):2d} "
                  f"refused={entry['cells_refused']:2d} missing={entry['cells_missing']:2d} "
                  f"repaired={entry['cells_repaired']} "
                  f"deskew={entry['deskew_degrees']} {entry['wall_ms']}ms")

        capture = None
        if CAPTURE.exists():
            capture = _measure(CAPTURE, gold)
            capture["condition"] = {"source": "user_capture"}
            capture["input_size"] = list(Image.open(CAPTURE).size)

        on_device = _on_device({"clean_control": paths["clean_control"],
                                "phone_768": paths["phone_768"]})
        from pace.deskew import estimate_skew

        with Image.open(CONTROL) as handle:
            page = handle.convert("L")
        deskew_times = []
        for _ in range(3):
            started = time.perf_counter()
            estimate_skew(page)
            deskew_times.append((time.perf_counter() - started) * 1000)
        on_device["deskew_ms"] = round(sorted(deskew_times)[1], 1)
        on_device["cold_first_page_ms"] = cold_ms
        on_device["cold_first_page"] = {
            "rows_recovered": len(warm["course_records"]),
            "model_loading_included": True,
        }

    try:
        import rapidocr_onnxruntime
        backend_version = getattr(rapidocr_onnxruntime, "__version__", "unknown")
    except Exception:  # pragma: no cover - only when the backend is absent
        backend_version = None

    wrong = [entry["condition"] for entry in ladder if entry["wrong_values"]]
    counted = ladder + ([capture] if capture is not None else [])
    payload = {
        "schema_version": 1,
        "control": str(CONTROL.relative_to(ROOT)),
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "backend": {"name": "rapidocr", "version": backend_version},
        "gold_rows": len(gold),
        "verified_fields": list(FIELDS),
        # One implementation of the sum, shared with the resolution ladder.
        "guarantee": guarantee_block(counted, rungs=len(ladder)),
        "wrong_conditions": wrong + (["user_capture"] if capture
                                     and capture["wrong_values"] else []),
        "ladder": ladder,
        "user_capture": capture,
        "on_device": on_device,
        "notes": [
            "Each rung applies the named degradations to the control render in capture "
            "order: perspective, rotation, illumination/exposure, defocus, JPEG, resize; "
            "the page is placed on a plain background with a margin so perspective "
            "moves its corners instead of cropping the printed edge.",
            "wrong counts values that differ from the fixture; refused counts values the "
            "verifier or the credit arithmetic replaced with null and a reason; missing "
            "counts values in rows that were not recovered. The bar is zero wrong.",
            "Only course_code, mark, weight and credit are verified. Titles, term text, "
            "GPA, printed total credits and assessment values are marked `unverified` "
            "per value in each record, so no emitted value reads as checked unless it "
            "was checked.",
            "Deskew is applied before recognition and its angle is in deskew_degrees; "
            "the same angle is replayed when a cell is cropped for its focused re-read.",
            "Each rung is a fresh `pace parse-transcript` process, so its wall_ms includes "
            "interpreter and model startup; warm per-page cost is in on_device.",
            "Beyond the reader: a 5-pixel motion smear recovers 4 of 8 rows; JPEG q45 "
            "loses credit cells on 3 rows, q30 on 1; the 768-pixel sensor loses 1 row to "
            "missing cells; the combined phone captures at 768-1024 px lose cells to "
            "defocus + JPEG + downscaling and recover 4-5 rows; the 512-pixel combination "
            "recovers no row because the recogniser no longer detects the mark and credit "
            "columns, so the table layout is not recognised. Missing cells are null or "
            "absent, not wrong.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")
    print(f"wrong at: {wrong}")


if __name__ == "__main__":
    main()
