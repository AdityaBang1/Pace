"""One learned artifact that finds the characters and reads them, with no grid.

This replaces the character-grid experiment's crutch. There is no template, no
column table and no cell geometry anywhere in this module: the network sees a page
image and predicts, for every position at half resolution, whether a printed
character is centred there and which character it is. The locations are learned
from the labels, not supplied by the renderer at inference.

Decoding is deliberately not recognition: peaks in the presence map are read in
reading order, joined, split on the printed commas and validated. A character the
model is not sure it located, or is not sure it read, makes its field ``None`` —
unread, with the raw characters kept — rather than a guess repaired by anything.

The public entry is :func:`detect_and_read`: page images in, one record per page
out. It reads each page twice — the second view is the same framing moved half an
output cell and enlarged — and keeps a field only where the two readings produced
the same characters. Rows one reading saw and the other did not are handed back
as ``unpaired`` rather than as values.
"""
from __future__ import annotations

import re
import string
from copy import deepcopy
from typing import Any, Iterable

import numpy as np

# What a printed transcript can contain: letters in either case, digits, the
# decimal point and the comma that separates fields. Real transcripts print
# lower-case course codes (``220725y``) beside upper-case headings, so the
# alphabet has to carry both cases.
CHARSET = string.ascii_letters + string.digits + ".,"
CHAR_TO_INDEX = {character: index for index, character in enumerate(CHARSET)}

# The page frame is a US-letter transcript at two pixels per printer's point —
# the raster this project's renderer produces and the capture lives at. The
# previously used 1152x640 landscape frame forced a portrait page into a
# 494-pixel-wide strip, shrinking 9 pt type to about 7 px.
PAGE_HEIGHT = 1584
PAGE_WIDTH = 1224
STRIDE = 2  # the head predicts at half resolution
SEPARATOR = ","
FIELDS = ("course_code", "mark", "weight", "credit")

# A field is emitted only when its weakest character passed both checks.
DEFAULT_PEAK_THRESHOLD = 0.35
DEFAULT_MIN_CONFIDENCE = 0.5

# The shipped reader's rule for a local course code: three to six digits and a
# one- or two-letter suffix (``220725y``, ``42075y``). The suffix is required so a
# year is never read as a code.
CODE_PATTERN = re.compile(r"^\d{3,6}[A-Za-z]{1,2}$")

# How far apart two printed characters may sit and still belong to the same
# column run: a multiple of the row's own character advance, capped so a row of
# short numbers cannot be clustered on its own field gaps.
COLUMN_GAP_FACTOR = 2.2
MIN_COLUMN_GAP = 12.0
MAX_COLUMN_GAP = 40.0

# Two peaks of the same character closer together than this fraction of the
# field's own character advance are one printed glyph detected twice, not two
# characters. The field that needs them is refused rather than repaired: a
# duplicated code suffix otherwise turns one legal code into another.
DUPLICATE_GAP_FACTOR = 0.75

# The second reading's view of the same page: the frame moved three pixels to the
# right and two down, then enlarged 1.5% about the centre. Small enough to still
# be that page, different enough that a glyph lands on another output cell and is
# resampled, so one view's decoding accident does not repeat in the other.
SECOND_VIEW = {"dx": 3.0, "dy": 2.0, "scale": 1.015}


def _torch() -> Any:
    import torch

    return torch


class PageSpotter:
    """A small U-Net-style network with a presence head and a character head.

    Output channels are ``1 + len(CHARSET)`` at stride 2: the first is the logit
    that a character is centred on that position, the rest are its logits.
    """

    def __init__(self, classes: int = len(CHARSET)) -> None:
        torch = _torch()
        from torch import nn

        def block(channels_in: int, channels_out: int, stride: int = 2) -> Any:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1,
                          bias=False),
                nn.BatchNorm2d(channels_out), nn.GELU(),
                nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out), nn.GELU(),
            )

        def up(channels_in: int, channels_out: int) -> Any:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out), nn.GELU(),
            )

        self.classes = classes
        self.net = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 96),  # stride 2, 4, 8
            up(96, 64), up(64, 48),                      # back to stride 2
            nn.Conv2d(48, 1 + classes, 1),
        )

    def __call__(self, batch: Any) -> Any:
        return self.net(batch)

    def parameters(self):
        """The network's parameters, under the name torch callers expect.

        A method, not a property: callers and torch utilities write
        ``network.parameters()``, and a property dressed as one raises instead of
        returning the parameters.
        """
        return self.net.parameters()

    def to(self, device: Any) -> "PageSpotter":
        self.net.to(device)
        return self

    def eval(self) -> "PageSpotter":
        self.net.eval()
        return self

    def double(self) -> "PageSpotter":
        self.net.double()
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state)


def page_tensor(image: Any) -> Any:
    """A page image as the network's input: one grey channel, zero mean."""
    torch = _torch()
    from PIL import Image

    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    grey = image.convert("L").resize((PAGE_WIDTH, PAGE_HEIGHT), Image.BILINEAR)
    array = np.asarray(grey, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) / 0.5)[None, None]


def second_view(image: Any) -> np.ndarray:
    """The same page through the second view, in the first view's frame."""
    from PIL import Image

    picture = image if hasattr(image, "convert") else Image.fromarray(
        np.asarray(image, dtype=np.uint8))
    picture = picture.convert("L")
    scale = SECOND_VIEW["scale"]
    dx, dy = SECOND_VIEW["dx"], SECOND_VIEW["dy"]
    centre_x, centre_y = (picture.width - 1) / 2.0, (picture.height - 1) / 2.0
    coefficients = (1.0 / scale, 0.0, centre_x - (dx + centre_x) / scale,
                    0.0, 1.0 / scale, centre_y - (dy + centre_y) / scale)
    moved = picture.transform(picture.size, Image.AFFINE, coefficients,
                              resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(moved, dtype=np.uint8)


def rows_from_second_view(pages: Iterable[list[dict[str, Any]]]) -> None:
    """Move a second reading's row positions back into the first view's frame."""
    scale = SECOND_VIEW["scale"]
    centre = (PAGE_HEIGHT - 1) / 2.0
    for records in pages:
        for record in records:
            record["row_y"] = (centre
                              + (record["row_y"] - centre - SECOND_VIEW["dy"]) / scale)


def targets_from_characters(characters: Iterable[tuple],
                            height: int = PAGE_HEIGHT, width: int = PAGE_WIDTH,
                            ) -> tuple[Any, Any]:
    """Presence and class targets from (character, x, y[, labelled]) tuples.

    A 3x3 patch of presence is set at each character's half-resolution cell, and
    the class is supervised at that cell when the character is part of a field
    the model must transcribe. A tuple may carry a fourth ``labelled`` flag: page
    furniture such as titles and headings is still printed ink, so its presence
    is supervised, but it is not a field character and gets no class target.
    """
    torch = _torch()

    rows, columns = height // STRIDE, width // STRIDE
    presence = torch.zeros((rows, columns), dtype=torch.float32)
    classes = torch.full((rows, columns), -100, dtype=torch.long)
    for entry in characters:
        character, x, y = entry[0], entry[1], entry[2]
        labelled = entry[3] if len(entry) > 3 else True
        cell_x, cell_y = int(round(x / STRIDE)), int(round(y / STRIDE))
        if not (0 <= cell_x < columns and 0 <= cell_y < rows):
            continue
        patch = (slice(max(0, cell_y - 1), cell_y + 2),
                 slice(max(0, cell_x - 1), cell_x + 2))
        presence[patch] = 1.0
        # The class is supervised across the whole presence patch, not only its
        # centre: a peak that lands a cell off must still be classified from a
        # position the model was trained on.
        if labelled:
            classes[patch] = CHAR_TO_INDEX[character]
    return presence, classes


def presence_loss(logits: Any, targets: Any) -> Any:
    """A weighted BCE: printed characters are a tiny fraction of the page."""
    torch = _torch()
    weight = torch.tensor(20.0, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight)


def character_loss(logits: Any, classes: Any) -> Any:
    """Cross-entropy only where a character was printed."""
    torch = _torch()
    return torch.nn.functional.cross_entropy(logits, classes, ignore_index=-100)


def _peaks(presence: np.ndarray, threshold: float) -> list[tuple[float, float, float]]:
    """Local maxima of the presence map, as (x, y, score) in input pixels.

    A peak must be the strict maximum of its 3x3 neighbourhood and at least
    ``threshold``, so one character produces one peak without a second detector.
    """
    padded = np.pad(presence, 1, mode="constant", constant_values=-1.0)
    neighbours = None
    for offset_y in (0, 1, 2):
        for offset_x in (0, 1, 2):
            if (offset_y, offset_x) == (1, 1):
                continue
            shifted = padded[offset_y:offset_y + presence.shape[0],
                             offset_x:offset_x + presence.shape[1]]
            neighbours = shifted if neighbours is None else np.maximum(neighbours, shifted)
    mask = (presence >= threshold) & (presence > neighbours)
    found = []
    for row, column in np.argwhere(mask):
        found.append((float(column * STRIDE), float(row * STRIDE),
                      float(presence[row, column])))
    return found


def estimate_skew(peaks: list[tuple[float, float, float]]) -> float:
    """The page's tilt, from the peak cloud alone.

    A camera or a rotation puts the characters of one printed line at different
    heights across the page. Projecting the peaks at the angle whose row profile
    is sharpest recovers the lines without being told the page is tilted.
    """
    if len(peaks) < 4:
        return 0.0
    xs = np.array([peak[0] for peak in peaks])
    ys = np.array([peak[1] for peak in peaks])
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-3.0, 3.001, 0.15):
        theta = np.radians(angle)
        projected = ys * np.cos(theta) - xs * np.sin(theta)
        bins = np.round((projected - projected.min()) / 6.0).astype(int)
        counts = np.bincount(bins).astype(float)
        score = float((counts ** 2).sum()) / len(peaks)
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def group_rows(peaks: list[tuple[float, float, float]]) -> list[list[tuple[float, float, float]]]:
    """Peaks in reading order, grouped into printed rows along the page's tilt.

    The tilt comes from :func:`estimate_skew`; the grouping tolerance is derived
    from the prediction, not from a template: half the median gap between distinct
    character heights.
    """
    if not peaks:
        return []
    angle = np.radians(estimate_skew(peaks))
    projected = [(x, y, score, y * np.cos(angle) - x * np.sin(angle))
                 for x, y, score in peaks]
    ordered = sorted(projected, key=lambda peak: (peak[3], peak[0]))
    heights = [peak[3] for peak in ordered]
    gaps = [later - earlier for earlier, later in zip(heights, heights[1:])
            if later - earlier > 1.0]
    tolerance = max(4.0, 0.5 * float(np.median(gaps))) if gaps else 4.0
    rows: list[list[tuple[float, float, float, float]]] = []
    for peak in ordered:
        if rows and abs(peak[3] - np.mean([item[3] for item in rows[-1]])) <= tolerance:
            rows[-1].append(peak)
        else:
            rows.append([peak])
    return [sorted(((x, y, score) for x, y, score, _ in row), key=lambda peak: peak[0])
            for row in rows]


def _validate(field: str, text: str) -> Any:
    """The typed value of a printed field, or None. Nothing is repaired."""
    if not text:
        return None
    if field == "course_code":
        return text if CODE_PATTERN.match(text) else None
    if (text.count(".") > 1 or not set(text) <= set(string.digits + ".")
            or text.startswith(".") or text.endswith(".")):
        # A point with no digit on one side is a character the model missed, not
        # a number: refusing it beats emitting ``0.`` as 0.0.
        return None
    try:
        number = float(text)
    except ValueError:  # pragma: no cover - the character test already excludes it
        return None
    if field == "mark":
        return str(int(number)) if number == int(number) and 0 <= number <= 100 else None
    if field == "weight" and "." not in text:
        # This family prints weights to four decimal places. A weight-shaped
        # string with no point is a character the model missed, not a weight, so
        # it is refused rather than emitted at the wrong order of magnitude.
        return None
    return number


def _column_runs(characters: list[tuple[str, float, float]]) -> list[list[int]]:
    """Split one printed row where the page leaves a column's worth of white.

    Returns runs of indices into ``characters``. The threshold comes from the row
    itself: the median of its closest half of character gaps is the advance
    inside a word, and a gap between columns is several advances. No column
    position is supplied — a row printed as comma-separated fields has no gap and
    stays one run, and a column layout splits even when the page's columns moved.
    """
    if not characters:
        return []
    xs = [x for _, _, x in characters]
    gaps = sorted(later - earlier for earlier, later in zip(xs, xs[1:]))
    if not gaps:
        return [[0]]
    sample = gaps[: max(1, len(gaps) // 2)]
    threshold = min(MAX_COLUMN_GAP,
                    max(MIN_COLUMN_GAP, COLUMN_GAP_FACTOR * float(np.median(sample))))
    runs: list[list[int]] = [[0]]
    for index in range(1, len(characters)):
        if xs[index] - xs[index - 1] > threshold:
            runs.append([])
        runs[-1].append(index)
    return runs


def _row_advance(characters: list[tuple[str, float, float]]) -> float:
    """The row's own character advance, or 0 when it is too short to estimate."""
    gaps = sorted(later - earlier for (_, _, earlier), (_, _, later)
                  in zip(characters, characters[1:]))
    return float(np.median(gaps)) if gaps else 0.0


def _duplicated_suffix(characters: list[tuple[str, float, float]], part: list[int],
                       row_advance: float) -> bool:
    """Is a code's closing letter one printed glyph detected twice?

    A duplicated final letter is the duplicate that stays legal: the code rule
    accepts a two-letter suffix, so ``733983w`` becomes ``733983ww`` and still
    validates — a different, confidently wrong code. Two peaks of the same letter
    closer than three-quarters of the run's own advance (close enough that peak
    jitter, not printed spacing, explains the distance) cannot be two printed
    characters, so the pair is one detection and the code is refused rather than
    repaired.

    The check is deliberately confined to the closing letters. A repeated digit
    inside a code can sit a little inside its advance on a degraded page while
    still being genuinely printed there; refusing those would cost correct codes,
    and a duplicated digit that stays legal is a separate failure this does not
    claim to catch.
    """
    if len(part) < 2:
        return False
    earlier, later = part[-2], part[-1]
    character = characters[earlier][0]
    if character != characters[later][0] or not character.isalpha():
        return False
    gaps = [characters[b][2] - characters[a][2] for a, b in zip(part, part[1:])]
    advance = float(np.median(gaps)) if len(part) >= 3 else row_advance
    return advance > 0 and gaps[-1] < DUPLICATE_GAP_FACTOR * advance


def decode_page(presence: Any, character_scores: Any,
                *, peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
                min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                ) -> list[dict[str, Any]]:
    """Peaks in, records out. No template, no repair.

    ``presence`` is a 2-D probability map; ``character_scores`` is the softmax over
    the character classes at the same resolution. A field whose weakest character
    is below ``min_confidence`` becomes ``None`` with the raw text kept beside it.

    A row printed as comma-separated fields is split on its commas. A row printed
    in columns — the real transcript layout — is split on its column gaps, and
    the course code and the three numeric columns are the first run and the last
    three. Any other shape is refused whole.
    """
    presence = np.asarray(presence, dtype=np.float32)
    scores = np.asarray(character_scores, dtype=np.float32)
    rows = group_rows(_peaks(presence, peak_threshold))
    records: list[dict[str, Any]] = []
    for peaks in rows:
        characters: list[tuple[str, float, float]] = []
        for x, y, located in peaks:
            cell_x, cell_y = int(round(x / STRIDE)), int(round(y / STRIDE))
            if not (0 <= cell_y < scores.shape[0] and 0 <= cell_x < scores.shape[1]):
                continue
            class_index = int(scores[cell_y, cell_x].argmax())
            characters.append((CHARSET[class_index],
                               located * float(scores[cell_y, cell_x, class_index]),
                               x))
        text = "".join(character for character, _, _ in characters)
        runs = _column_runs(characters)
        if len(runs) > len(FIELDS):
            # Code, title words, then the three numeric columns.
            parts = [runs[0], runs[-3], runs[-2], runs[-1]]
        elif len(runs) == len(FIELDS):
            parts = runs
        elif text.count(SEPARATOR) == len(FIELDS) - 1:
            parts = [[]]
            for index, (character, _, _) in enumerate(characters):
                if character == SEPARATOR:
                    parts.append([])
                else:
                    parts[-1].append(index)
        else:
            parts = runs
        record: dict[str, Any] = {field: None for field in FIELDS}
        confident: dict[str, str] = {}
        if len(parts) == len(FIELDS):
            advance = _row_advance(characters)
            for field, part in zip(FIELDS, parts):
                text = "".join(characters[index][0] for index in part)
                confidence = min((characters[index][1] for index in part), default=0.0)
                supported = not (field == "course_code"
                                 and _duplicated_suffix(characters, part, advance))
                if text and confidence >= min_confidence and supported:
                    confident[field] = text
            for field, text in confident.items():
                record[field] = _validate(field, text)
        record["row_text"] = "".join(character for character, _, _ in characters)
        # The raw text before validation, and where the row sits, are kept so a
        # second reading of the same page can be compared with this one.
        record["field_text"] = {field: confident.get(field) for field in FIELDS}
        record["row_y"] = float(np.mean([peak[1] for peak in peaks]))
        if any(record[field] is None for field in FIELDS):
            record["unread_cells"] = {field: record[field] is None for field in FIELDS}
        records.append(record)
    return records


def _pair_rows(first: list[dict[str, Any]], second: list[dict[str, Any]],
               *, row_tolerance: float = 4.0) -> list[int | None]:
    """Each first-view row's partner index, where only one pairing is possible.

    The two readings are of the same rigid page, so a printed row's two positions
    differ only by where the decoder put that row's centre — jitter, not layout.
    A pairing is used only when it is mutual and unique: the row's only candidate
    inside ``row_tolerance`` must also have that row as its only candidate.
    Anything else is an ambiguous neighbourhood, where a neighbour sits closer
    than the jitter and either could be the twin; such a row is left without a
    partner and so is refused, rather than certified against another printed
    row's text. A nearest-neighbour search cannot promise this: it always pairs
    when anything is in range, and hands out the partner a neighbouring row may
    need.
    """
    if first is second:
        # One reading compared with itself: every row is its own partner.
        return list(range(len(first)))
    partners: list[int | None] = [None] * len(first)
    candidates = [[index for index, other in enumerate(second)
                   if abs(other.get("row_y", 0.0) - record.get("row_y", 0.0))
                   <= row_tolerance]
                  for record in first]
    claimed = [[index for index, record in enumerate(first)
                if abs(other.get("row_y", 0.0) - record.get("row_y", 0.0))
                <= row_tolerance]
               for other in second]
    for index, inside in enumerate(candidates):
        if len(inside) != 1:
            continue
        partner = inside[0]
        if claimed[partner] == [index]:
            partners[index] = partner
    return partners


def _merge_views(first: list[dict[str, Any]], second: list[dict[str, Any]],
                 *, row_tolerance: float = 4.0
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep a field only where two readings agree character for character.

    ``first`` and ``second`` are records from two views of the same page in the
    same coordinate frame (the second view's row positions must already be mapped
    back). A field is emitted only when both readings produced the same raw text
    before validation and that text validates; any difference is a refusal. Rows
    are paired only where the pairing is unambiguous (see :func:`_pair_rows`); a
    row with no partner is refused whole. Nothing is voted on, repaired or chosen
    from the more confident view.
    """
    partners = _pair_rows(first, second, row_tolerance=row_tolerance)
    used = {index for index in partners if index is not None}

    merged: list[dict[str, Any]] = []
    for record, partner_index in zip(first, partners):
        partner = None if partner_index is None else second[partner_index]
        values: dict[str, Any] = {field: None for field in FIELDS}
        if partner is not None:
            for field in FIELDS:
                left = (record.get("field_text") or {}).get(field)
                right = (partner.get("field_text") or {}).get(field)
                if left and right and left == right:
                    values[field] = _validate(field, left)
        values["row_text"] = record.get("row_text", "")
        values["row_y"] = record.get("row_y", 0.0)
        values["paired"] = partner is not None
        if any(values[field] is None for field in FIELDS):
            values["unread_cells"] = {field: values[field] is None
                                      for field in FIELDS}
        merged.append(values)
    unpaired = [{"row_text": record.get("row_text", ""),
                 "row_y": record.get("row_y", 0.0), "view": "first"}
                for record, index in zip(first, partners) if index is None]
    unpaired += [{"row_text": record.get("row_text", ""),
                  "row_y": record.get("row_y", 0.0), "view": "second"}
                 for index, record in enumerate(second) if index not in used]
    return merged, unpaired


def agreeing_readings(first: list[dict[str, Any]], second: list[dict[str, Any]],
                      *, row_tolerance: float = 4.0) -> list[dict[str, Any]]:
    """The merged rows only: see :func:`_merge_views`."""
    merged, _ = _merge_views(first, second, row_tolerance=row_tolerance)
    return merged


def _device_of(network: Any) -> Any:
    """Where the weights live, whether they are exposed as method or property."""
    torch = _torch()

    parameters = getattr(network, "parameters", None)
    if callable(parameters):
        parameters = parameters()
    first = next(iter(parameters), None) if parameters is not None else None
    return first.device if first is not None else torch.device("cpu")


def reading_copy(network: Any) -> Any:
    """The network as the reading computes it: the same weights in float64.

    Float32 convolution is not portable. The same checkpoint and page differed by
    up to 1.6e-2 in logits between a GPU in cuDNN's TF32 mode and a CPU, and by
    2.9e-5 with TF32 off — while the closest two character classes came within
    3.8e-5 of a tie on the held-out and real pages. In float64 a CPU and a GPU
    agree to 7e-14, and a thread count change moves the logits by 3e-14: nine
    orders below that tie, so the same page and checkpoint read the same fields.

    The caller's model is copied, never converted in place. A stub with no
    ``double`` to call — the test doubles — is used as it is; a network that has
    one and fails while widening raises, because falling back to the float32 copy
    would quietly put the device dependence back.
    """
    copy = deepcopy(network)
    widen = getattr(copy, "double", None)
    if widen is None:
        return network
    widen()
    return copy


def _read_batch(network: Any, batch: Any, *, peak_threshold: float,
                min_confidence: float) -> list[list[dict[str, Any]]]:
    """One forward pass, one decode per page in the batch."""
    torch = _torch()

    with torch.no_grad():
        logits = network(batch)
        presence = torch.sigmoid(logits[:, 0])
        characters = logits[:, 1:].softmax(dim=1)
    return [decode_page(presence[index].cpu().numpy(),
                        characters[index].permute(1, 2, 0).cpu().numpy(),
                        peak_threshold=peak_threshold,
                        min_confidence=min_confidence)
            for index in range(logits.shape[0])]


def detect_and_read(network: Any, images: Iterable[Any], *,
                    views: int = 2,
                    batch_size: int = 2,
                    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
                    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                    ) -> list[dict[str, Any]]:
    """The public path: page images in, one record per page out.

    ``network`` is a :class:`PageSpotter` or the module it wraps; ``images`` are
    PIL images or ``uint8`` arrays, any size, in page order. ``views`` is 2 for
    the agreement reading and 1 for the single reading the experiment records as
    its before column; one view has nothing to compare, so each row is paired
    with itself. The network is read as a float64 copy (see
    :func:`reading_copy`), so a CPU and a GPU read the same page to the same
    fields.

    Each returned page is ``{"rows": [...], "unpaired": [...]}``. A row carries
    the four fields (``None`` where unread), ``unread_cells``, ``row_text`` and
    ``paired``. ``unpaired`` lists the rows only one reading saw, as raw text with
    the view that saw them: they are not values, and none is emitted from them.
    """
    if views not in (1, 2):
        raise ValueError("views must be 1 or 2")
    torch = _torch()
    from PIL import Image

    pictures = [image if hasattr(image, "convert")
                else Image.fromarray(np.asarray(image, dtype=np.uint8))
                for image in images]
    reading = reading_copy(network)
    device = _device_of(reading)
    pages: list[dict[str, Any]] = []
    for start in range(0, len(pictures), batch_size):
        chunk = pictures[start:start + batch_size]
        first_batch = torch.cat([page_tensor(picture)
                                 for picture in chunk]).double().to(device)
        first = _read_batch(reading, first_batch,
                            peak_threshold=peak_threshold,
                            min_confidence=min_confidence)
        if views == 2:
            second_batch = torch.cat([page_tensor(second_view(picture))
                                      for picture in chunk]).double().to(device)
            second = _read_batch(reading, second_batch,
                                 peak_threshold=peak_threshold,
                                 min_confidence=min_confidence)
            rows_from_second_view(second)
            pairs = zip(first, second)
        else:
            pairs = ((records, records) for records in first)
        for left, right in pairs:
            rows, unpaired = _merge_views(left, right)
            pages.append({"rows": rows, "unpaired": unpaired})
    return pages
