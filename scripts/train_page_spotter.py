"""Train and evaluate the page spotter on the transcript family this project reads.

    python scripts/train_page_spotter.py

The previous version of this script invented its own page family: landscape
sheets, comma-separated fields, upper-case six-digit codes. The model learned that
renderer and then read nothing on a real transcript page — the type arrived at
about 7 px and the real codes were not even representable.

This version draws the family the shipped reader actually parses, using the
geometry of the project's own renderer (``pace/transcript_render.py``):

* portrait US letter at two pixels per printer's point, so 9 pt type is 18 px —
  the same scale as the rendered fixture page and its raster;
* the renderer's column positions and line pitch, with per-document jitter around
  them: course code, a wrapped course title, then right-aligned mark, weight and
  credit;
* local course codes such as ``220725y`` and ``42075y`` — digits and a lower-case
  suffix — and the printed number formats (``1.0000`` weight, ``1`` credit).

A document is drawn whole and pages are split by document, never by page: the
held-out split contains no page of any training document. Every printed glyph is
a presence target, so the model must find all the ink; only the four measured
fields carry a class target, so the title column and headings are located but not
transcribed.

Writes ``artifacts/page-spotter/model.pt`` and ``metrics.json``.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pace.page_spotter import (CHARSET, DEFAULT_MIN_CONFIDENCE,  # noqa: E402
                               DEFAULT_PEAK_THRESHOLD, FIELDS, PAGE_HEIGHT,
                               PAGE_WIDTH, SECOND_VIEW, PageSpotter,
                               character_loss, detect_and_read, presence_loss,
                               targets_from_characters)
from scripts._ladder_common import gold_rows, same_value  # noqa: E402

# Two pixels per printer's point: the raster this project's renderer produces at
# scale 2.0, and the scale at which a letter page's 9 pt type is 18 px tall.
SCALE = 2
PAPER = 255
INK = 15
PAGE = (612 * SCALE, 792 * SCALE)

FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
WEIGHTS = (0.5, 1.0, 0.25, 0.75, 1.5)
TITLES = (
    "Accel Adv", "Alg:Concept&Connect", "B:Concept&Connect", "AP Human Geography",
    "Biology I Honors", "Chemistry", "Geometry", "World History", "Physics",
    "Pre-Calc", "Digital Design", "Marketing", "Algebra II", "English IV",
    "US Government", "Economics", "Spanish II", "Intro to Software Tech",
    "Literature & Composition I Hnr", "Intro to Business and Tech",
    "Advanced Placement Human Geography: Concepts and Connections",
    "Honors Integrated Mathematics III: Algebra, Geometry and Statistics",
)

_FONTS: dict[tuple[str, int], Any] = {}


@dataclass(frozen=True)
class Style:
    """One document's geometry, drawn once and shared by all of its pages."""

    body: int
    header: int
    banner: int
    line: int
    top: int
    code_x: int
    title_x: int
    marks_x: int
    weight_x: int
    credit_x: int


@dataclass(frozen=True)
class Document:
    style: Style
    rows_per_page: int
    records: list[dict[str, Any]]
    titles: list[str]


def _font(path: str, size: int):
    from PIL import ImageFont

    key = (path, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(path, size)
    return _FONTS[key]


def _draw(draw, text: str, x: float, top: float, font, size: int, *,
          labelled: bool, out: list[tuple]) -> float:
    """Draw one left-starting run and record each glyph's centre in page pixels."""
    cursor = float(x)
    for character in text:
        advance = font.getlength(character)
        if character != " ":
            draw.text((cursor, top), character, font=font, fill=INK)
            out.append((character, cursor + advance / 2, top + size * 0.48, labelled))
        cursor += advance
    return cursor


def _draw_right(draw, text: str, right: float, top: float, font, size: int, *,
                labelled: bool, out: list[tuple]) -> None:
    _draw(draw, text, right - font.getlength(text), top, font, size,
          labelled=labelled, out=out)


def _wrap(text: str, font, width: float) -> list[str]:
    """Break a course title at the column edge, the way the page wraps it."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and font.getlength(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + [current] if current else lines


def _style(rng: random.Random) -> Style:
    body = rng.choice((18, 18, 20))
    code_x = rng.randint(74, 96)
    marks_x = rng.randint(686, 730)
    weight_x = marks_x + rng.randint(132, 158)
    credit_x = weight_x + rng.randint(112, 140)
    return Style(body=body, header=body + 2, banner=body + 4,
                 line=rng.choice((30, 32, 32, 34)), top=rng.randint(118, 158),
                 code_x=code_x, title_x=code_x + rng.randint(96, 116),
                 marks_x=marks_x, weight_x=weight_x, credit_x=credit_x)


def _record(rng: random.Random) -> dict[str, Any]:
    """One printed course row, in the formats the real page uses."""
    code = f"{rng.randint(10000, 999999)}{rng.choice('abcdefghijklmnopqrstuvwxyz')}"
    weight = rng.choice(WEIGHTS)
    return {"course_code": code, "mark": str(rng.randint(55, 100)),
            "weight": weight, "credit": weight,
            "weight_text": f"{weight:.4f}",
            "credit_text": f"{weight:g}"}


def _title(rng: random.Random) -> str:
    title = rng.choice(TITLES)
    if rng.random() < 0.25:
        title = f"{rng.choice(TITLES)} {title}"
    return title


def _document(rng: random.Random) -> Document:
    count = rng.randint(6, 13)
    return Document(style=_style(rng), rows_per_page=rng.randint(6, 9),
                    records=[_record(rng) for _ in range(count)],
                    titles=[_title(rng) for _ in range(count)])


def render_document(rng: random.Random) -> list[tuple[Any, list[tuple], list[dict]]]:
    """Draw every page of one document; return image, glyphs and that page's gold."""
    from PIL import Image, ImageDraw

    document = _document(rng)
    style = document.style
    body = _font(FONT, style.body)
    header = _font(FONT, style.header)
    banner = _font(FONT_BOLD, style.banner)
    bold = _font(FONT_BOLD, style.body)
    year = rng.choice((2024, 2025, 2026))
    banner_text = f"{year}-{year + 1} Grade {rng.randint(6, 12):02d}   Term {rng.randint(1, 4)}"
    pages = []
    for page_index, start in enumerate(range(0, len(document.records),
                                             document.rows_per_page)):
        image = Image.new("L", PAGE, PAPER)
        draw = ImageDraw.Draw(image)
        glyphs: list[tuple] = []
        header_top = style.top + int(style.banner * 2.8)
        _draw(draw, banner_text, style.code_x, style.top, banner, style.banner,
              labelled=False, out=glyphs)
        _draw(draw, "Course", style.code_x, header_top, header, style.header,
              labelled=False, out=glyphs)
        for label, right in (("Mark", style.marks_x), ("Weight", style.weight_x),
                             ("Credit", style.credit_x)):
            _draw_right(draw, label, right, header_top, header, style.header,
                        labelled=False, out=glyphs)

        y = header_top + style.line
        rows: list[dict] = []
        for record, title in zip(document.records[start:start + document.rows_per_page],
                                 document.titles[start:start + document.rows_per_page]):
            _draw(draw, record["course_code"], style.code_x, y, body, style.body,
                  labelled=True, out=glyphs)
            # Leave a real gutter before the mark column: a title that ran right
            # up to the number would merge with it in any gap-based reader.
            lines = _wrap(title, body, style.marks_x - style.title_x - 60)
            for line_index, line in enumerate(lines):
                _draw(draw, line, style.title_x, y + line_index * style.line, body,
                      style.body, labelled=False, out=glyphs)
            _draw_right(draw, record["mark"], style.marks_x, y, body, style.body,
                        labelled=True, out=glyphs)
            _draw_right(draw, record["weight_text"], style.weight_x, y, body,
                        style.body, labelled=True, out=glyphs)
            _draw_right(draw, record["credit_text"], style.credit_x, y, body,
                        style.body, labelled=True, out=glyphs)
            rows.append(record)
            y += style.line * len(lines)

        footer = ("Credit: 7.500   GPA: 4.13333" if page_index else
                  "Credit: 3.750   GPA: 4.08123")
        _draw(draw, footer, style.code_x + 400, y + style.line * 0.7, bold,
              style.body, labelled=False, out=glyphs)
        pages.append((image, glyphs, rows))
    return pages


def fit_page(image, characters):
    """Scale the page into the network's frame and return the scaled centres."""
    from PIL import Image

    scale = min(PAGE_HEIGHT / image.height, PAGE_WIDTH / image.width)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.BILINEAR)
    frame = Image.new("L", (PAGE_WIDTH, PAGE_HEIGHT), PAPER)
    offset_x = (PAGE_WIDTH - size[0]) // 2
    offset_y = (PAGE_HEIGHT - size[1]) // 2
    frame.paste(resized, (offset_x, offset_y))
    centres = []
    for entry in characters:
        character, x, y = entry[0], entry[1], entry[2]
        scaled = (character, x * scale + offset_x, y * scale + offset_y)
        centres.append(scaled + (entry[3],) if len(entry) > 3 else scaled)
    return frame, centres


def frame_page(image, characters, rng: random.Random):
    """Put the sheet in a capture: pad, crop inside the padding, tilt slightly."""
    import math

    from PIL import Image

    pad = 26
    framed = Image.new("L", (image.width + 2 * pad, image.height + 2 * pad), PAPER)
    framed.paste(image, (pad, pad))
    left, top, right, bottom = (rng.uniform(0, pad) for _ in range(4))
    framed = framed.crop((round(left), round(top),
                          round(framed.width - right), round(framed.height - bottom)))
    characters = [(entry[0], entry[1] - left + pad, entry[2] - top + pad)
                  + tuple(entry[3:]) for entry in characters]

    angle = rng.uniform(-0.4, 0.4)
    if angle:
        centre = (framed.width / 2, framed.height / 2)
        framed = framed.rotate(angle, resample=Image.BICUBIC, fillcolor=PAPER,
                               center=centre)
        theta = math.radians(angle)
        cos, sin = math.cos(theta), math.sin(theta)
        characters = [
            (entry[0],
             centre[0] + (entry[1] - centre[0]) * cos + (entry[2] - centre[1]) * sin,
             centre[1] - (entry[1] - centre[0]) * sin + (entry[2] - centre[1]) * cos)
            + tuple(entry[3:])
            for entry in characters
        ]
    return fit_page(framed, characters)


def degrade(image, rng: random.Random, *, heldout: bool = False):
    """Illumination, blur and JPEG, applied after the page is framed."""
    from PIL import Image, ImageFilter

    pixels = np.asarray(image, dtype=np.float32)
    gradient = rng.uniform(0.75, 1.0) if heldout else rng.uniform(0.85, 1.0)
    exposure = rng.uniform(0.8, 1.0) if heldout else rng.uniform(0.9, 1.0)
    ramp = np.linspace(1.0, gradient, image.width, dtype=np.float32)[None, :]
    image = Image.fromarray(np.clip(pixels * ramp * exposure, 0, 255).astype(np.uint8),
                            mode="L")
    blur = rng.uniform(0.0, 1.1) if heldout else rng.uniform(0.0, 0.7)
    if blur > 0.05:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    quality = rng.randint(55, 85) if heldout else rng.randint(65, 92)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("L")


def make_pages(documents: int, seed: int, *, heldout: bool = False
               ) -> list[dict[str, Any]]:
    """Pages from whole documents, each with its gold rows and glyph centres."""
    rng = random.Random(seed)
    pages: list[dict[str, Any]] = []
    for document in range(documents):
        for image, glyphs, rows in render_document(rng):
            page, centres = frame_page(image, glyphs, rng)
            if heldout or rng.random() >= 0.25:
                page = degrade(page, rng, heldout=heldout)
            pages.append({"image": np.asarray(page, dtype=np.uint8),
                          "characters": centres, "gold": rows,
                          "document": document})
    return pages


def page_batch(pages: list[dict[str, Any]], indices) -> Any:
    """Normalised grey pages as the network's input, computed per batch."""
    import torch

    stack = np.stack([pages[int(index)]["image"] for index in indices]
                     ).astype(np.float32) / 255.0
    return torch.from_numpy((stack - 0.5) / 0.5)[:, None]


def read_pages_agreeing(pages: list[dict[str, Any]], model, *,
                        views: int = 2,
                        batch_size: int = 2,
                        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
                        min_confidence: float = DEFAULT_MIN_CONFIDENCE
                        ) -> list[dict[str, Any]]:
    """Every page through the module's public entry; rows and unpaired rows."""
    return detect_and_read(model, [page["image"] for page in pages],
                           views=views,
                           batch_size=batch_size,
                           peak_threshold=peak_threshold,
                           min_confidence=min_confidence)


def _count_unpaired(report: dict[str, Any],
                    reads: list[dict[str, Any]]) -> None:
    """Add the rows one reading saw and the other did not to a report."""
    report["unpaired_rows"] = sum(len(read["unpaired"]) for read in reads)
    report["unpaired_first_view"] = sum(
        1 for read in reads for row in read["unpaired"] if row["view"] == "first")
    report["unpaired_second_view"] = (report["unpaired_rows"]
                                      - report["unpaired_first_view"])


def summarize(pages: list[dict[str, Any]], decoded: list[list[dict[str, Any]]]
              ) -> dict[str, Any]:
    """Per-field emitted / refused / wrong, counted against each page's gold.

    Rows are matched by course code, so a misread code does not shift the rows
    after it into a pile of wrong values. A code printed as a different string is
    counted wrong even though no gold row can be matched to it.
    """
    counts = {field: {"emitted": 0, "refused": 0, "wrong": 0} for field in FIELDS}
    rows_expected = rows_decoded = rows_read = rows_refused = rows_partial = 0
    wrong_values: list[dict[str, Any]] = []
    for page, records in zip(pages, decoded):
        gold = page["gold"]
        rows_expected += len(gold)
        rows_decoded += len(records)
        by_code: dict[str, dict[str, Any]] = {}
        for record in records:
            code = record.get("course_code")
            if code:
                by_code.setdefault(code.strip().lower(), record)
        expected_codes = {str(row["course_code"]).strip().lower() for row in gold}
        for expected in gold:
            row = by_code.get(str(expected["course_code"]).strip().lower())
            correct = emitted = 0
            for field in FIELDS:
                value = None if row is None else row.get(field)
                if value is None:
                    counts[field]["refused"] += 1
                    continue
                emitted += 1
                counts[field]["emitted"] += 1
                if same_value(field, value, expected[field]):
                    correct += 1
                else:
                    counts[field]["wrong"] += 1
                    wrong_values.append({"field": field, "expected": expected[field],
                                         "got": value})
            if correct == len(FIELDS):
                rows_read += 1
            elif emitted:
                rows_partial += 1
            else:
                rows_refused += 1
        for code, row in by_code.items():
            if code not in expected_codes:
                counts["course_code"]["emitted"] += 1
                counts["course_code"]["wrong"] += 1
                wrong_values.append({"field": "course_code", "expected": None,
                                     "got": row.get("course_code")})
    return {
        "documents": len({page["document"] for page in pages}),
        "pages": len(pages),
        "rows_expected": rows_expected,
        "rows_decoded": rows_decoded,
        "rows_read_correctly": rows_read,
        "rows_partly_read": rows_partial,
        "rows_refused": rows_refused,
        "fields": counts,
        "wrong_total": len(wrong_values),
        "wrong_values": wrong_values,
    }


def _real_page_images() -> list[tuple[str, Any]]:
    """The real pages already in this tree: the render, its raster and a capture."""
    from PIL import Image, ImageFilter
    import pypdfium2 as pdfium

    samples = ROOT / "data" / "samples"
    images: list[tuple[str, Any]] = []
    document = pdfium.PdfDocument(str(samples / "transcript_term_marks.pdf"))
    try:
        images.append(("pdf page, 2x raster",
                       document[0].render(scale=2.0).to_pil().convert("L")))
    finally:
        document.close()
    images.append(("rendered PNG",
                   Image.open(samples / "transcript_term_marks.png").convert("L")))
    images.append(("256x202 capture",
                   Image.open(samples / "transcript_term_marks_256x202.png").convert("L")))
    degraded = Image.open(samples / "transcript_term_marks.png").convert("L").rotate(
        0.5, resample=Image.BICUBIC, fillcolor=PAPER)
    degraded = degraded.filter(ImageFilter.GaussianBlur(0.6))
    buffer = io.BytesIO()
    degraded.save(buffer, format="JPEG", quality=60)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        images.append(("degraded PNG (0.5 deg, blur 0.6, JPEG 60)", handle.convert("L")))
    return images


def measure_real_pages(model, *,
                       peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
                       min_confidence: float = DEFAULT_MIN_CONFIDENCE
                       ) -> dict[str, Any]:
    """Read the real fixture page with the model as trained; report per page."""
    fixture = ROOT / "data" / "samples" / "transcript_term_marks.txt"
    gold = gold_rows(fixture)
    pages = []
    names = []
    for name, image in _real_page_images():
        page, _ = fit_page(image, [])
        names.append(name)
        pages.append({"image": np.asarray(page, dtype=np.uint8), "gold": gold,
                      "document": name})
    reads = read_pages_agreeing(pages, model,
                                peak_threshold=peak_threshold,
                                min_confidence=min_confidence)
    per_page = {}
    for name, page, read in zip(names, pages, reads):
        report = summarize([page], [read["rows"]])
        _count_unpaired(report, [read])
        per_page[name] = report
    return {"gold_rows": len(gold), "pages": per_page}


def _stored_record(output: Path, *, report_only: bool) -> dict[str, Any]:
    """The record a re-score extends, or nothing when it only prints.

    A re-score that writes must extend an existing record: the training block,
    the seeds and the training-split counts come from it. Deriving them from this
    run's arguments instead would describe a training that never happened.
    """
    saved = output / "metrics.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    if report_only:
        return {}
    raise SystemExit(
        f"no metrics.json in {output}: a re-score cannot write a training record. "
        "Use --report-only to score without writing, or --output with the "
        "directory that holds the checkpoint's record.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-documents", type=int, default=600)
    parser.add_argument("--heldout-documents", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--heldout-seed", type=int, default=880123)
    parser.add_argument("--peak-threshold", type=float, default=DEFAULT_PEAK_THRESHOLD)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "page-spotter"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="re-score the saved checkpoint instead of training")
    parser.add_argument("--report-only", action="store_true",
                        help="re-score and print; leave model.pt and metrics.json "
                             "untouched (implies --eval-only)")
    args = parser.parse_args()
    if args.report_only:
        # Printing a report never trains; the flag says re-score.
        args.eval_only = True

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    previous: dict[str, Any] = {}
    if args.eval_only:
        previous = _stored_record(Path(args.output), report_only=args.report_only)

    if args.eval_only:
        train_pages: list[dict[str, Any]] = []
    else:
        print(f"rendering {args.train_documents} training documents ...", flush=True)
        train_pages = make_pages(args.train_documents, args.seed)
    print(f"rendering {args.heldout_documents} held-out documents ...", flush=True)
    heldout = make_pages(args.heldout_documents, args.heldout_seed, heldout=True)
    print(f"{len(train_pages)} train pages, {len(heldout)} held-out pages in "
          f"{time.perf_counter() - started:.1f}s", flush=True)

    spotter = PageSpotter(len(CHARSET))
    model = spotter.net.to(device)
    if args.eval_only:
        saved = torch.load(Path(args.output) / "model.pt", map_location=device,
                           weights_only=False)
        spotter.load_state_dict(saved["state_dict"])
    model.eval()

    steps = (math.ceil(len(train_pages) / args.batch_size) * args.epochs
             if train_pages else 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                   total_steps=max(1, steps))

    step = 0
    for epoch in range(1, args.epochs + 1):
        if not train_pages:
            break
        model.train()
        epoch_loss = batches = 0
        permutation = torch.randperm(len(train_pages))
        for start in range(0, len(permutation), args.batch_size):
            selection = permutation[start:start + args.batch_size]
            batch = page_batch(train_pages, selection).to(device)
            targets = [targets_from_characters(train_pages[int(index)]["characters"])
                       for index in selection]
            presence = torch.stack([target[0] for target in targets]).to(device)
            classes = torch.stack([target[1] for target in targets]).to(device)
            logits = model(batch)
            loss = presence_loss(logits[:, 0], presence) + character_loss(
                logits[:, 1:], classes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step < steps:  # OneCycle cannot be stepped past its total
                scheduler.step()
            step += 1
            epoch_loss += float(loss.detach())
            batches += 1
            if step >= steps:
                break
        print(f"epoch {epoch:2d}/{args.epochs}  loss {epoch_loss / max(1, batches):.4f}  "
              f"{time.perf_counter() - started:.1f}s", flush=True)
        if step >= steps:
            break

    model.eval()
    reads = read_pages_agreeing(heldout, model, batch_size=2,
                                peak_threshold=args.peak_threshold,
                                min_confidence=args.min_confidence)
    heldout_report = summarize(heldout, [read["rows"] for read in reads])
    _count_unpaired(heldout_report, reads)
    single = read_pages_agreeing(heldout, model, views=1, batch_size=2,
                                 peak_threshold=args.peak_threshold,
                                 min_confidence=args.min_confidence)
    single_report = summarize(heldout, [read["rows"] for read in single])
    _count_unpaired(single_report, single)
    print(json.dumps({key: heldout_report[key] for key in
                      ("rows_expected", "rows_decoded", "rows_read_correctly",
                       "unpaired_rows", "fields", "wrong_total")}, indent=2))

    real = measure_real_pages(model, peak_threshold=args.peak_threshold,
                              min_confidence=args.min_confidence)
    for name, report in real["pages"].items():
        print(f"{name}: {report['rows_decoded']} rows, "
              f"{report['rows_read_correctly']}/{report['rows_expected']} read, "
              f"{report['unpaired_rows']} unpaired, {report['fields']}")

    elapsed = round(time.perf_counter() - started, 1)
    if args.report_only:
        print("report only: model.pt and metrics.json untouched", flush=True)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        torch.save({"state_dict": spotter.state_dict(), "charset": CHARSET},
                   output / "model.pt")

    example = heldout[0]
    metrics = {
        "model": {
            "name": "PageSpotter",
            "description": "one fully convolutional network; per-position character "
                           "presence and class at half resolution; no template, no "
                           "grid, no detector stage, no parser",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_stride": 2,
            "frame": [PAGE_WIDTH, PAGE_HEIGHT],
            "charset": len(CHARSET),
        },
        "data": {
            "train_documents": args.train_documents,
            "heldout_documents": args.heldout_documents,
            "train_pages": len(train_pages),
            "heldout_pages": len(heldout),
            "train_seed": args.seed,
            "heldout_seed": args.heldout_seed,
            "split": "by document: no held-out page belongs to a training document",
            "family": "portrait US letter at 2 px/pt, the renderer's columns, "
                      "9-10 pt Arial (Helvetica metrics), wrapped titles, "
                      "lower-case course codes",
            "degradation": "illumination, blur and JPEG; a quarter of training pages "
                           "clean; held-out pages all degraded",
            "targets": "presence on every printed glyph; class only on the four "
                       "measured fields",
            "example_gold": example["gold"][:2],
        },
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "optimizer": "AdamW 1e-3 one-cycle", "device": str(device),
                     "wall_seconds": elapsed,
                     "reproducibility": "seeded; cuDNN kernels are not deterministic"},
        "thresholds": {"peak": args.peak_threshold,
                       "min_confidence": args.min_confidence,
                       "tuned_on_evaluation_documents": False},
        "reading": {"views": 2,
                    "arithmetic": "float64: the network is read as a double "
                                  "copy, so the fields do not depend on whether "
                                  "a CPU or a GPU ran the convolutions",
                    "second_view": "shift (%.1f, %.1f) px, scale %.3f about the "
                                   "centre" % (SECOND_VIEW["dx"], SECOND_VIEW["dy"],
                                               SECOND_VIEW["scale"]),
                    "rule": "a field is emitted only where both readings produced "
                            "the same raw text; any disagreement is a refusal",
                    "unpaired": "rows only one reading saw are counted, never "
                                "emitted"},
        "heldout_documents": heldout_report,
        "heldout_single_view": single_report,
        "real_pages": real,
        "limitations": [
            "synthetic documents drawn in the real family; the only real page is "
            "the fixture's render, not a camera capture",
            "page furniture (titles, headings) is located but not transcribed",
            "characters are read one by one; a missed character usually loses its "
            "whole field, which is why refusal, not accuracy, is the safety property",
            "row grouping, the row pairing and the column split are decoding "
            "rules, not learned; a row is paired only where the pairing is "
            "unambiguous, so rows closer together than the decoding jitter are "
            "refused rather than confirmed against a neighbour",
        ],
    }
    if args.eval_only and previous:
        # Re-scoring must not rewrite the record of how the checkpoint was trained.
        metrics["training"] = previous.get("training", metrics["training"])
        for key in ("train_documents", "train_pages", "train_seed"):
            metrics["data"][key] = previous["data"].get(key, metrics["data"][key])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
