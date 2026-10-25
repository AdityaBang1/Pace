"""Read a PDF or an image into text, keeping the provenance of every page.

The deterministic readers expect text. This module is the only place that knows
where that text came from, so it keeps the answer next to the characters:

* ``text_layer`` — the PDF carried its own characters. The slices are the author's
  text and can be quoted back exactly.
* ``ocr`` — there was no text layer, so the page was rendered and a recogniser read
  the pixels. The characters are a *reading* of the page image, not the page.
  Values from such a page stay proposals that a person has to check.
* ``text_file`` — the input already was plain text.

OCR is optional on purpose. When no recogniser is installed, a page that needs one
raises :class:`OcrUnavailable` with the command that fixes it; it never returns an
empty string that would look like a blank page, and it never invents characters.
"""
from __future__ import annotations

import bisect
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Protocol

from .deskew import deskew_image, rotate_page
from .orientation import open_oriented_image

SCHEMA_VERSION = 1

TEXT_SUFFIXES = frozenset({".txt", ".text", ".md", ".csv"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"})
# iPhone photos are HEIC/HEIF by default, and Pillow has no decoder for them.
# They are named before the text fallback can produce a zero-row "success".
HEIF_SUFFIXES = frozenset({".heic", ".heif", ".hif"})
HEIF_BRANDS = frozenset({"heic", "heix", "hevc", "hevx", "heim", "heis", "mif1", "msf1"})
AVIF_BRANDS = frozenset({"avif", "avis"})

# A page with fewer characters than this is a scan with a header, a stamp, or
# nothing at all; its text layer cannot answer for the page.
MIN_TEXT_LAYER_CHARS = 24

# Rendered at 2.5x a 72 dpi page, i.e. about 180 dpi. Measured on the sample
# transcript: at 2x the recogniser lost two credit cells and misread a mark, at
# 2.5x every cell came back, and above 3x nothing improved.
DEFAULT_RENDER_SCALE = 2.5

# The recogniser's own default (0.5) drops isolated digits such as a credit of
# "1". Keeping boxes down to 0.3 recovers them; the scores are reported so a
# reader can still see which cells were marginal.
OCR_TEXT_SCORE = 0.3
OCR_BOX_THRESHOLD = 0.3

EXACT_METHODS = frozenset({"text_layer", "text_file"})

# A banner whose spaces were lost in recognition: "2025-2026Grade09Term2", or
# "2025-2026 Grade09 Term2".
_BANNER_REPAIR_RE = re.compile(
    r"(?<!\d)(\d{4})\s*-\s*(\d{4})\s*Grade\s*(\d{1,2})\s*Term\s*(\d{1,2})\b", re.IGNORECASE)

# Below this width a page is a thumbnail, and recognition starts losing digits.
MIN_RELIABLE_IMAGE_SIDE = 800


# Text is recognised best when a printed line is roughly 30 pixels tall. A page
# rendered at 2x has 22-pixel text and the recogniser both loses spaces inside a
# banner and misreads a mark; the same page scaled to a 2000-pixel long side reads
# every cell correctly. So images are lifted to that working size, never down.
OCR_TARGET_LONG_SIDE = 2000
OCR_MAX_UPSCALE = 3.0

OCR_INSTALL_HINT = (
    "install one with `pip install rapidocr-onnxruntime` (a small ONNX detector and "
    "recogniser, no service and no GPU) or put the `tesseract` executable on PATH"
)


class IngestError(RuntimeError):
    """The input could not be turned into text, and the message says why."""


class MissingDependency(IngestError):
    """An optional reader package is not installed."""


class OcrUnavailable(IngestError):
    """A page has no text layer and no recogniser is installed."""


@dataclass(frozen=True)
class OcrReading:
    """What one recogniser pass returned: text, its evidence boxes, and how it ran."""

    text: str
    detail: dict[str, Any] = field(default_factory=dict)
    # Each box is {"start", "end", "text", "score", "box": [x0, y0, x1, y1]}
    # with the offsets pointing into `text`, so a parsed value can be cropped back
    # out of the page image it was read from.
    spans: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PageText:
    """One page's text, where it came from, and its span in the joined document."""

    number: int
    text: str
    method: str
    start: int
    detail: dict[str, Any] = field(default_factory=dict)
    # Recognised boxes in rendered-image pixels; kept off the JSON summary because
    # they are evidence for the verifier, not a fact about the document.
    boxes: tuple[dict[str, Any], ...] = ()

    @property
    def end(self) -> int:
        return self.start + len(self.text)

    @property
    def exact(self) -> bool:
        return self.method in EXACT_METHODS

    def summary(self) -> dict[str, Any]:
        view = {
            "page": self.number,
            "method": self.method,
            "characters": len(self.text.strip()),
            "exact_source_text": self.exact,
        }
        if self.detail:
            view["detail"] = dict(self.detail)
        return view


@dataclass(frozen=True)
class Document:
    """A whole input, split into pages that each remember how they were read."""

    path: str
    kind: str
    pages: tuple[PageText, ...]
    warnings: tuple[str, ...] = ()
    render_scale: float = DEFAULT_RENDER_SCALE
    # The recogniser that produced the OCR pages, kept so a cell can be re-read
    # from its crop without the caller having to pass it back in.
    backend: Any = field(default=None, repr=False, compare=False)

    @property
    def text(self) -> str:
        # The two-character separator is what `_join` counted for each page after
        # the first, so every page start stays aligned with this string.
        return "\n\n".join(page.text for page in self.pages)

    @property
    def exact(self) -> bool:
        return all(page.exact for page in self.pages)

    def page_for(self, offset: int) -> PageText | None:
        """The page a character offset belongs to, so a value can be traced home."""
        if not self.pages:
            return None
        starts = [page.start for page in self.pages]
        index = bisect.bisect_right(starts, offset) - 1
        return self.pages[max(index, 0)]

    def page_image(self, page: PageText) -> Any:
        """Re-create the exact pixels a page was recognised from.

        Verification re-reads a cell from its own crop, which needs the rendered
        page. Keeping every page's image in memory would cost megabytes per scan,
        so the page is rendered again — one at a time — at the scale recorded when
        it was first read. The result is deterministic for a given file.
        """
        if self.kind == "image":
            _require("PIL.Image", "pillow", "ingest")
            # The same orientation resolution as the page read: the stored pixels
            # may be sideways, and a crop taken before turning them would land on
            # a different part of the page than the recogniser read.
            image = open_oriented_image(self.path).image
        elif self.kind == "pdf":
            pdfium = _require("pypdfium2", "pypdfium2", "ingest")
            document = pdfium.PdfDocument(self.path)
            try:
                image = document[page.number - 1].render(scale=self.render_scale)
                image = image.to_pil().convert("L")
            finally:
                document.close()
        else:  # pragma: no cover - text pages are never verified
            return None
        # The page was straightened before it was recognised. Re-create that exact
        # geometry from the angle recorded then, so a cell crop lands on the same
        # pixels the recogniser read instead of on the tilted original.
        angle = float(page.detail.get("deskew_degrees") or 0.0)
        if angle:
            image = rotate_page(image, angle)
        prepared, _ = prepare_for_ocr(image)
        return prepared

    def summary(self) -> dict[str, Any]:
        methods: dict[str, int] = {}
        for page in self.pages:
            methods[page.method] = methods.get(page.method, 0) + 1
        view: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "path": self.path,
            "kind": self.kind,
            "page_count": len(self.pages),
            "methods": methods,
            "exact_source_text": self.exact,
            "pages": [page.summary() for page in self.pages],
        }
        if not self.exact:
            view["note"] = (
                "Values from an OCR page are a reading of rendered pixels, not the "
                "author's characters: check them against the page image before use."
            )
        if self.warnings:
            view["warnings"] = list(self.warnings)
        return view


class OcrBackend(Protocol):
    """What an OCR backend has to provide.

    `read` recognises a whole page; `read_cell` recognises a crop containing one
    printed cell and no layout, which is how a doubtful value is checked. A backend
    that cannot crop returns None from `read_cell`, and the cell is reported as
    unchecked rather than silently trusted.
    """

    name: str

    def read(self, image: Any) -> OcrReading:
        ...

    def read_cell(self, image: Any) -> tuple[str, float] | None:
        ...


def _require(module: str, package: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - the message is the point
        raise MissingDependency(
            f"reading this file needs `{package}`, which is not installed; "
            f"install the ingest extra with `pip install -e .[{extra}]`"
        ) from exc


# --------------------------------------------------------------------------- #
# OCR backends
# --------------------------------------------------------------------------- #


def _cluster_rows(boxes: list[tuple[float, float, float, float, str, float]],
                  ) -> list[list[tuple[float, float, float, float, str, float]]]:
    """Group boxes into printed lines by their vertical centre."""
    if not boxes:
        return []
    height = median(box[3] - box[1] for box in boxes) or 1.0
    ordered = sorted(boxes, key=lambda box: ((box[1] + box[3]) / 2, box[0]))
    rows: list[list[tuple[float, float, float, float, str, float]]] = []
    centres: list[list[float]] = []
    for box in ordered:
        centre = (box[1] + box[3]) / 2
        if rows and abs(centre - sum(centres[-1]) / len(centres[-1])) <= height * 0.5:
            rows[-1].append(box)
            centres[-1].append(centre)
        else:
            rows.append([box])
            centres.append([centre])
    return rows


def boxes_to_text(boxes: Iterable[Iterable[Any]]) -> tuple[str, dict[str, Any]]:
    """Rebuild reading order and column gaps from recognised boxes."""
    text, detail, _ = boxes_to_text_with_spans(boxes)
    return text, detail


def boxes_to_text_with_spans(
    boxes: Iterable[Iterable[Any]],
) -> tuple[str, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Rebuild reading order, column gaps, and where each box landed.

    A recogniser returns boxes in no guaranteed order. Columns survive only if the
    horizontal gap between two boxes is turned back into spaces, because the
    transcript reader identifies a cell by its gap — without this a row reads as
    one run of words and the numbers cannot be assigned to their columns.

    The spans are the bridge back to the pixels: each one says which characters
    came from which box, so a parsed value can be cropped out of the page image
    and read again on its own.
    """
    cleaned = []
    for box, text, score in boxes:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        if not str(text).strip():
            continue
        cleaned.append((min(xs), min(ys), max(xs), max(ys), str(text), float(score)))
    if not cleaned:
        return "", {"lines": 0, "boxes": 0}, ()
    height = median(box[3] - box[1] for box in cleaned) or 1.0
    character = max(1.0, height * 0.55)
    lines: list[str] = []
    spans: list[dict[str, Any]] = []
    line_start = 0
    for row in _cluster_rows(cleaned):
        row.sort(key=lambda box: box[0])
        parts: list[str] = []
        cursor = 0
        right = row[0][0]
        for index, box in enumerate(row):
            if index:
                gap = box[0] - right
                spaces = 1 if gap <= character else min(12, int(round(gap / character)))
                parts.append(" " * max(spaces, 1))
                cursor += max(spaces, 1)
            start = cursor
            parts.append(box[4])
            cursor += len(box[4])
            spans.append({"start": line_start + start, "end": line_start + cursor,
                          "text": box[4], "score": round(box[5], 3),
                          "box": [round(box[0], 1), round(box[1], 1),
                                  round(box[2], 1), round(box[3], 1)]})
            right = max(right, box[2])
        line = "".join(parts).rstrip()
        lines.append(line)
        line_start += len(line) + 1  # the newline that joins the lines
    scores = [box[5] for box in cleaned]
    detail = {
        "lines": len(lines),
        "boxes": len(cleaned),
        "mean_confidence": round(sum(scores) / len(scores), 3),
        "min_confidence": round(min(scores), 3),
        "low_confidence_boxes": sum(1 for score in scores if score < 0.5),
    }
    return "\n".join(lines), detail, tuple(spans)


_RAPIDOCR_ENGINE: Any = None


class RapidOcrBackend:
    """Small in-process ONNX detector + recogniser (PP-OCR, 13.7 MB of models)."""

    name = "rapidocr"

    def __init__(self) -> None:
        module = _require("rapidocr_onnxruntime", "rapidocr-onnxruntime", "ocr")
        global _RAPIDOCR_ENGINE
        # Loading the ONNX sessions takes about a second; a process that reads a
        # document at a time should pay that once, not per page.
        if _RAPIDOCR_ENGINE is None:
            _RAPIDOCR_ENGINE = module.RapidOCR()
        self._engine = _RAPIDOCR_ENGINE

    def read(self, image: Any) -> tuple[str, dict[str, Any]]:
        import numpy as np

        # RapidOCR takes a path, bytes or a numpy array; the array is the one shape
        # that works for both a PDF page we just rendered and an opened image file.
        array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
        result, _ = self._engine(array, text_score=OCR_TEXT_SCORE,
                                 box_thresh=OCR_BOX_THRESHOLD)
        if not result:
            return OcrReading("", {"backend": self.name, "boxes": 0, "lines": 0})
        text, detail, spans = boxes_to_text_with_spans(result)
        detail["backend"] = self.name
        detail["text_score"] = OCR_TEXT_SCORE
        return OcrReading(text, detail, spans)

    def read_cell(self, image: Any) -> tuple[str, float] | None:
        """Recognise one cropped cell, skipping detection entirely.

        A cell crop has no layout to find: running the detector on two characters
        is both wasted work and a chance to drop the box. The recogniser alone is
        the same model that read the page, applied to the exact pixels of a value,
        at a size where the digits are unambiguous.
        """
        import numpy as np

        array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
        result, _ = self._engine.text_recognizer([array])
        if not result:
            return None
        text, score = result[0]
        return str(text), float(score)

    def read_cells(self, images: Iterable[Any]) -> list[tuple[str, float] | None]:
        """One recognition pass for all cell crops on a page."""
        import numpy as np

        arrays = [np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
                  for image in images]
        if not arrays:
            return []
        result, _ = self._engine.text_recognizer(arrays)
        readings = [(str(text), float(score)) for text, score in (result or [])]
        if len(readings) != len(arrays):  # pragma: no cover - defensive alignment
            return [None] * len(arrays)
        return readings


class TesseractBackend:
    """The `tesseract` executable, when it is on PATH."""

    name = "tesseract"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("tesseract") or ""
        if not self.executable:
            raise OcrUnavailable(
                "`tesseract` is not on PATH; install the Tesseract executable, or "
                "unset PACE_OCR_BACKEND to use the RapidOCR backend"
            )

    def read(self, image: Any) -> tuple[str, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="pace-ocr-") as folder:
            path = Path(folder) / "page.png"
            image.save(path, format="PNG")
            # --psm 6: one uniform block of text, which keeps the printed lines.
            process = subprocess.run(
                [self.executable, str(path), "stdout", "--psm", "6"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        if process.returncode != 0:
            raise IngestError(
                f"`tesseract` failed on this page (exit {process.returncode}): "
                f"{(process.stderr or '').strip()[:300]}"
            )
        text = process.stdout.replace("\f", "").strip("\n")
        return OcrReading(text, {"backend": self.name, "characters": len(text.strip())})

    def read_cell(self, image: Any) -> tuple[str, float] | None:
        with tempfile.TemporaryDirectory(prefix="pace-cell-") as folder:
            path = Path(folder) / "cell.png"
            image.save(path, format="PNG")
            # --psm 7: one text line, which is exactly what a cell crop is.
            process = subprocess.run(
                [self.executable, str(path), "stdout", "--psm", "7"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        if process.returncode != 0:
            return None
        text = process.stdout.strip("\n")
        return (text, 0.0) if text.strip() else None


def ocr_backend(name: str | None = None) -> OcrBackend:
    """Pick a recogniser, or explain what to install when there is none.

    Order: an explicit ``PACE_OCR_BACKEND`` wins; then RapidOCR (pip only, small,
    runs where the phone build can also run ONNX); then the tesseract executable.
    """
    import os

    choice = (name or os.environ.get("PACE_OCR_BACKEND") or "").strip().lower()
    if choice:
        if choice in {"rapidocr", "rapidocr_onnxruntime"}:
            return RapidOcrBackend()
        if choice == "tesseract":
            return TesseractBackend()
        raise IngestError(f"unknown OCR backend {choice!r}; use 'rapidocr' or 'tesseract'")
    try:
        return RapidOcrBackend()
    except MissingDependency:
        pass
    executable = shutil.which("tesseract")
    if executable:
        return TesseractBackend(executable)
    raise OcrUnavailable(
        "no OCR backend is installed, and this page has no text layer to read; "
        + OCR_INSTALL_HINT
    )


def ocr_backend_name() -> str | None:
    """The backend that would be used, or None. Never raises."""
    try:
        return ocr_backend().name
    except IngestError:
        return None


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def _join(pages: list[tuple[int, str, str, dict[str, Any], tuple[dict[str, Any], ...]]]
          ) -> tuple[PageText, ...]:
    joined: list[PageText] = []
    cursor = 0
    for number, text, method, detail, spans in pages:
        if joined:
            cursor += 2  # a blank line between pages, so no row straddles two
        joined.append(PageText(number=number, text=text, method=method,
                               start=cursor, detail=detail, boxes=spans))
        cursor += len(text)
    return tuple(joined)


def prepare_for_ocr(image: Any, target: int = OCR_TARGET_LONG_SIDE,
                    max_scale: float = OCR_MAX_UPSCALE) -> tuple[Any, float]:
    """Lift a small page image to the size the recogniser reads best, and say by how much."""
    longest = max(image.size)
    if longest <= 0 or longest >= target:
        return image, 1.0
    factor = min(max_scale, target / longest)
    if factor <= 1.02:
        return image, 1.0
    size = (max(1, int(round(image.width * factor))), max(1, int(round(image.height * factor))))
    return image.resize(size, _resample()), round(factor, 3)


def _resample() -> Any:
    pillow = _require("PIL.Image", "pillow", "ingest")
    return pillow.LANCZOS


def repair_ocr_spacing(text: str) -> tuple[str, list[str]]:
    """Put back the spaces a recogniser drops inside a printed label.

    A term banner read as ``2025-2026Grade09Term2`` is not a term, it is a banner
    whose spaces were lost, and the parser would otherwise treat the whole line as
    a school name. Only label-shaped runs are rewritten; a course code such as
    ``220725y`` is deliberately left alone because its digits and letter belong
    together. Each repair is reported, never silent.
    """
    repairs: list[str] = []

    def _banner(match: re.Match[str]) -> str:
        canonical = (f"{match.group(1)}-{match.group(2)} Grade {match.group(3)} "
                     f"Term {match.group(4)}")
        if canonical == match.group(0):
            return match.group(0)
        repairs.append(f"term banner spacing: {match.group(0)!r}")
        return canonical

    text = _BANNER_REPAIR_RE.sub(_banner, text)
    return text, repairs


def _remap_spans(spans: tuple[dict[str, Any], ...], old: str, new: str
                 ) -> tuple[dict[str, Any], ...]:
    """Move box offsets onto repaired text, dropping boxes whose characters moved.

    A spacing repair inserts or deletes characters, so every offset after it shifts.
    Boxes that covered the repaired characters are no longer exact evidence and are
    dropped; the rest keep pointing at the same printed cell.
    """
    if not spans or old == new:
        return spans
    from difflib import SequenceMatcher

    opcodes = SequenceMatcher(None, old, new, autojunk=False).get_opcodes()

    def mapped(offset: int, end: bool = False) -> int | None:
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal" and i1 <= offset <= i2:
                return j1 + (offset - i1)
            if tag == "insert" and offset == i1:
                return j2 if end else j1
            if tag == "replace":
                if offset == i1:
                    return j1
                if offset == i2:
                    return j2
                if i1 < offset < i2:
                    return None
        return None

    remapped = []
    for span in spans:
        start, stop = span["start"], span["end"]
        if any(tag != "equal" and not (i2 <= start or i1 >= stop)
               for tag, i1, i2, _j1, _j2 in opcodes):
            continue
        new_start = mapped(start)
        new_stop = mapped(stop, end=True)
        if new_start is None or new_stop is None or new_stop <= new_start:
            continue
        remapped.append({**span, "start": new_start, "end": new_stop})
    return tuple(remapped)


def _page_images(path: str, scale: float) -> Iterable[Any]:
    """Render each PDF page, lazily, so a text-layer document never pays for it."""
    pdfium = _require("pypdfium2", "pypdfium2", "ingest")
    document = pdfium.PdfDocument(path)
    try:
        for index in range(len(document)):
            yield document[index].render(scale=scale).to_pil().convert("L")
    finally:
        document.close()


def _pdf_pages(path: str, ocr: str, backend: OcrBackend | None, scale: float) -> tuple[
        list[tuple[int, str, str, dict[str, Any], tuple[dict[str, Any], ...]]], list[str]]:
    module = _require("pypdf", "pypdf", "ingest")
    reader = module.PdfReader(path)
    if reader.is_encrypted:
        raise IngestError("this PDF is encrypted; decrypt it before reading")
    pages: list[tuple[int, str, str, dict[str, Any], tuple[dict[str, Any], ...]]] = []
    warnings: list[str] = []
    images = None
    for index, page in enumerate(reader.pages, start=1):
        if ocr == "always":
            text, detail, method = "", {}, "ocr"
        else:
            text, detail, method = _text_layer(page, index, warnings), {}, "text_layer"
        if text.strip() and len(text.strip()) >= MIN_TEXT_LAYER_CHARS and ocr != "always":
            pages.append((index, text, method, detail, ()))
            continue
        if ocr == "never":
            if text.strip():
                pages.append((index, text, "text_layer",
                              {"sparse": True, "characters": len(text.strip())}, ()))
            else:
                pages.append((index, "", "text_layer", {"empty": True}, ()))
            continue
        if backend is None:
            if text.strip():
                pages.append((index, text, "text_layer",
                              {"sparse": True, "characters": len(text.strip())}, ()))
                warnings.append(
                    f"page {index} has only {len(text.strip())} characters of text "
                    "layer and no OCR backend is installed; it was kept as printed"
                )
                continue
            raise OcrUnavailable(
                f"page {index} of this PDF has no text layer (it is a scan or an "
                f"image) and no OCR backend is installed; {OCR_INSTALL_HINT}"
            )
        if images is None:
            images = _page_images(path, scale)
        image = next(images, None)
        if image is None:  # pragma: no cover - page count mismatch
            break
        image, angle = deskew_image(image)
        image, factor = prepare_for_ocr(image)
        reading = backend.read(image)
        recognised, repairs = repair_ocr_spacing(reading.text)
        ocr_detail = dict(reading.detail, upscale=factor)
        if angle:
            ocr_detail["deskew_degrees"] = angle
        if repairs:
            ocr_detail["repairs"] = repairs
        if not recognised.strip():
            warnings.append(f"page {index}: OCR returned no text; the page may be blank")
        spans = _remap_spans(reading.spans, reading.text, recognised)
        pages.append((index, recognised, "ocr", ocr_detail, spans))
    if not pages:
        raise IngestError("this PDF has no pages")
    return pages, warnings


def _text_layer(page: Any, number: int, warnings: list[str]) -> str:
    """Extract a page's own characters, keeping the printed layout when possible.

    `layout` mode puts the columns back where they were printed. The plain mode
    emits one text-showing operator per line, which splits a row's course code,
    title and mark onto separate lines and destroys the row structure entirely.
    """
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except (TypeError, ValueError) as exc:  # older pypdf without layout mode
        warnings.append(f"page {number}: this pypdf has no layout extraction ({exc})")
        return page.extract_text() or ""


def _image_page(path: str, ocr: str, backend: OcrBackend | None) -> tuple[
        list[tuple[int, str, str, dict[str, Any], tuple[dict[str, Any], ...]]], list[str]]:
    if ocr == "never":
        raise IngestError(
            "an image has no text layer; re-run with --ocr auto or --ocr always"
        )
    if backend is None:
        raise OcrUnavailable(
            f"this image has no text layer and no OCR backend is installed; {OCR_INSTALL_HINT}"
        )
    pillow = _require("PIL.Image", "pillow", "ingest")
    try:
        oriented = open_oriented_image(path)
    except (pillow.UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise IngestError(f"cannot read {path!r} as an image: {exc}") from exc
    image = oriented.image
    warnings: list[str] = []
    if oriented.frames > 1:
        warnings.append(
            f"this image has {oriented.frames} frames; only the first frame is read"
        )
    source_size = image.size
    image, angle = deskew_image(image)
    image, factor = prepare_for_ocr(image)
    reading = backend.read(image)
    text, repairs = repair_ocr_spacing(reading.text)
    detail = dict(reading.detail, upscale=factor, source_size=list(source_size),
                  **oriented.detail())
    if angle:
        detail["deskew_degrees"] = angle
    if repairs:
        detail["repairs"] = repairs
    if not text.strip():
        warnings.append("OCR returned no text; the image may be blank")
    if max(source_size) < MIN_RELIABLE_IMAGE_SIDE:
        warnings.append(
            f"this image is only {source_size[0]}x{source_size[1]} pixels; text that "
            "small is unreliable for recognition, so verify every value against a "
            "higher-resolution capture"
        )
    spans = _remap_spans(reading.spans, reading.text, text)
    return [(1, text, "ocr", detail, spans)], warnings


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _iso_bmff_brand(path: Path) -> str | None:
    """The major brand in an ISO-BMFF (ftyp) header, lower-cased."""
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return None
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("ascii", "ignore").strip().lower() or None


def _avif_supported() -> bool:
    try:
        from PIL import features
        return bool(features.check("avif"))
    except (ImportError, AttributeError):
        return False


def _looks_like_image(path: Path) -> bool:
    """A screenshot saved without an extension is still a screenshot."""
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return False
    return (head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff")
            or head.startswith(b"GIF8") or head.startswith(b"BM")
            or head.startswith(b"II*\x00") or head.startswith(b"MM\x00*")
            or (head.startswith(b"RIFF") and head[8:12] == b"WEBP"))


def _decode_bytes(data: bytes, label: str) -> tuple[str, str | None]:
    """Decode text, falling back to Windows-1252 when UTF-8 fails.

    A syllabus saved by an office suite is often Windows-1252. Refusing it is a
    dead end for the user; reading it with a visible note is not. Bytes that decode
    to a pile of control characters are not text at all, and are refused.
    """
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError as exc:
        try:
            text = data.decode("cp1252")
        except UnicodeDecodeError as fallback_exc:
            raise IngestError(
                f"{label} is not UTF-8 or Windows-1252 text; if it is a PDF or an "
                "image, use its real extension (.pdf, .png) so it can be read as one"
            ) from fallback_exc
        controls = sum(1 for character in text
                       if ord(character) < 32 and character not in "\r\n\t")
        if controls > max(4, len(text) // 100):
            raise IngestError(
                f"{label} is not readable text; if it is a PDF or an image, use its "
                "real extension (.pdf, .png) so it can be read as one"
            ) from exc
        return text, "cp1252"


def read_stdin_document(*, ocr: str = "auto") -> Document:
    """Read a transcript from stdin, so `... | pace parse-transcript -` works."""
    import sys

    data = sys.stdin.buffer.read()
    text, encoding = _decode_bytes(data, "stdin")
    if not text.strip():
        raise IngestError("stdin is empty")
    warnings = []
    if encoding:
        warnings.append("stdin was decoded as Windows-1252 because it is not UTF-8")
    if ocr != "auto":
        warnings.append("stdin is text, so --ocr does not apply")
    detail = {"encoding": encoding} if encoding else {}
    page = PageText(number=1, text=text, method="text_file", start=0, detail=detail)
    return Document(path="<stdin>", kind="text", pages=(page,), warnings=tuple(warnings))


def read_document(path: str | Path, *, ocr: str = "auto", backend: OcrBackend | None = None,
                  render_scale: float = DEFAULT_RENDER_SCALE) -> Document:
    """Read a text file, PDF or image into pages that remember their provenance.

    ``ocr`` is the policy for pages without a usable text layer: ``auto`` reads
    them from pixels when a backend exists, ``always`` reads every page from
    pixels, and ``never`` refuses instead of guessing.
    """
    if ocr not in {"auto", "always", "never"}:
        raise ValueError("ocr must be 'auto', 'always' or 'never'")
    if str(path) == "-":
        return read_stdin_document(ocr=ocr)
    target = Path(path)
    try:
        target.stat()
    except OSError as exc:
        raise IngestError(f"cannot read {str(path)!r}: {exc}") from exc
    if target.is_dir():
        # Opening a directory fails with a platform-specific permission/errno
        # message that reads like an access problem. It is an input mistake.
        raise IngestError(
            f"{str(path)!r} is a directory, not a document; pass the file inside it"
        )

    suffix = target.suffix.lower()
    brand = _iso_bmff_brand(target)
    kind = "pdf" if suffix == ".pdf" or _looks_like_pdf(target) else (
        "image" if suffix in IMAGE_SUFFIXES or _looks_like_image(target) else "text")
    if kind == "text" and brand in AVIF_BRANDS and _avif_supported():
        kind = "image"

    if kind == "text":
        # A phone's default photo format has no decoder here. Saying "not readable
        # text" would send the user to the wrong fix; name the format instead.
        if brand in HEIF_BRANDS or (brand is None and suffix in HEIF_SUFFIXES):
            label = f"HEIC/HEIF ({brand})" if brand else "HEIC/HEIF"
            raise IngestError(
                f"{target.name!r} is a {label} image. This build cannot decode "
                "HEIC/HEIF; export the photo as JPEG or PNG (on iPhone: Share > "
                "Save to Files as JPEG; on macOS: sips -s format jpeg) and re-run."
            )
        if brand in AVIF_BRANDS:
            raise IngestError(
                f"{target.name!r} is an AVIF image. This build cannot decode "
                "AVIF; export the photo as JPEG or PNG and re-run."
            )
        if brand:
            raise IngestError(
                f"{target.name!r} is an ISO-BMFF ({brand.upper()}) file, which this "
                "build cannot read as an image. Export the photo as JPEG or PNG "
                "and re-run."
            )
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise IngestError(f"cannot read {str(path)!r}: {exc}") from exc
        text, encoding = _decode_bytes(data, f"{target.name!r}")
        if not text.strip():
            raise IngestError(f"input file {str(path)!r} is empty")
        warnings = []
        if encoding:
            warnings.append(
                f"this file is not UTF-8; it was read as Windows-1252, so check "
                "accented characters"
            )
        if ocr != "auto":
            warnings.append("this input is text, so --ocr does not apply")
        detail = {"encoding": encoding} if encoding else {}
        page = PageText(number=1, text=text, method="text_file", start=0, detail=detail)
        return Document(path=str(target), kind="text", pages=(page,),
                        warnings=tuple(warnings))

    if ocr != "never" and backend is None:
        backend = ocr_backend()
    try:
        if kind == "image":
            pages, warnings = _image_page(str(target), ocr, backend)
        else:
            pages, warnings = _pdf_pages(str(target), ocr, backend, render_scale)
    except IngestError:
        raise
    except Exception as exc:
        # A malformed file is a user input, not a crash: the CLI names the file and
        # the reason instead of printing a traceback.
        label = "an image" if kind == "image" else "a PDF"
        raise IngestError(f"cannot read {str(path)!r} as {label}: {exc}") from exc
    document = Document(path=str(target), kind=kind, pages=_join(pages),
                        warnings=tuple(warnings), render_scale=render_scale,
                        backend=backend)
    if not document.text.strip():
        if ocr == "never":
            raise IngestError(
                "no page had a text layer, so there is nothing to read; re-run with "
                "--ocr auto to read the pages from pixels"
            )
        raise IngestError(
            "no readable text was found on any page; the input may be blank, or "
            "too low-resolution for OCR. Nothing is emitted rather than a guessed row."
        )
    return document


# --------------------------------------------------------------------------- #
# Handing the text to the deterministic reader
# --------------------------------------------------------------------------- #


def parse_document(document: Document, parsed: dict[str, Any] | None = None,
                   *, verify: bool = True) -> dict[str, Any]:
    """Parse a Document, then check and, where needed, re-read each OCR cell.

    The parser is unchanged: it sees the same text it always saw. This adds where
    that text came from, and — for pages read from pixels — a focused re-read of
    every value that can be wrong (code, mark, weight, credit). A value that cannot
    be confirmed is emitted as null with its reason; a wrong number is not kept.
    """
    import time

    from .transcript_extract import parse_transcript

    parsed = dict(parsed if parsed is not None else parse_transcript(document.text))
    parsed["source"] = document.summary()
    for key in ("course_records", "assessment_records", "attendance_records"):
        for record in parsed.get(key) or []:
            page = document.page_for(record.get("start", 0))
            if page is None:  # pragma: no cover - every document has a page
                continue
            record["page"] = page.number
            record["evidence_method"] = page.method
            record["evidence_exact"] = page.exact
    for term in parsed.get("terms") or []:
        page = document.page_for(term.get("start", 0))
        if page is not None:
            term["page"] = page.number

    if verify and not document.exact:
        from .ocr_verify import (distrust_unbalanced_credits, mark_unverified_values,
                                 refresh_checks, refresh_verification,
                                 unparsed_row_lines, verify_document)

        started = time.perf_counter()
        stats = verify_document(parsed, document)
        if stats.get("status") == "checked":
            refresh_checks(parsed)
            for reason in distrust_unbalanced_credits(parsed):
                parsed["source"].setdefault("warnings", []).append(reason)
            refresh_checks(parsed)
            # A credit refused by the page's arithmetic is one unread cell, not a
            # second counter, so the tallies are re-derived from the records.
            refresh_verification(parsed, stats)
            checks = parsed.get("checks") or {}
            if checks.get("distrusted_credit_rows"):
                stats["distrusted_credit_rows"] = checks["distrusted_credit_rows"]
            stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        missing_rows = unparsed_row_lines(parsed, document)
        for reason in missing_rows:
            parsed["source"].setdefault("warnings", []).append(reason)
        parsed.setdefault("checks", {})["unparsed_row_lines"] = len(missing_rows)
        if not parsed.get("course_records"):
            parsed["source"].setdefault("warnings", []).append(
                "no course rows could be read from this page; nothing is emitted "
                "rather than a guessed row"
            )
        parsed["verification"] = stats
        # Everything outside the four cell-checked fields is marked on the value
        # itself, so no emitted title, GPA or assessment can read as checked.
        mark_unverified_values(parsed)
    elif verify:
        # No crop stage runs on authored text, and `not_needed` is the statement
        # that says so — not a statement that every emitted value was checked.
        # Titles, term text, header names, the footer figures and assessment values
        # are parser output on this path too, so they carry the same per-value marks
        # an OCR record carries, written by the same function.
        from .ocr_verify import mark_unverified_values

        parsed["verification"] = {"status": "not_needed"}
        mark_unverified_values(parsed)

    checks = parsed.get("checks") or {}
    if (not document.exact and checks.get("reported_total_credits") is not None
            and checks.get("credit_sum_matches_reported_total") is False):
        # The footer is an independent number: when the rows disagree with it on a
        # page that was read from pixels, the likely cause is a misread cell.
        parsed["source"].setdefault("warnings", []).append(
            "the row credits do not add up to the printed total; on a page read by "
            "OCR that usually means a cell was misread, so check the page image"
        )
    return parsed


def summarize_source(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """The compact provenance block the CLI prints under ``--summary``."""
    source = parsed.get("source")
    if not source:
        return None
    summary = {
        "kind": source["kind"],
        "page_count": source["page_count"],
        "pages": [{"page": page["page"], "method": page["method"]}
                  for page in source["pages"]],
        "exact_source_text": source["exact_source_text"],
    }
    if source.get("warnings"):
        summary["warnings"] = list(source["warnings"])
    return summary
