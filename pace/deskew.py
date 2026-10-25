"""Remove the rotation a photograph gives a printed page before recognition.

A recogniser groups characters into lines. When the page is tilted even a little,
a row's characters sit at different heights, the detector splits the line, and
the header can merge with the first course row — the parser then sees a table it
cannot map and recovers no rows at all. Rotation is cheap to undo and does not
change any character: the page is turned back, not filtered, so the pixels the
recogniser reads are still the page's own (interpolated) pixels.

The angle is estimated from the text itself, with a projection profile: a page
whose lines are horizontal concentrates its ink into narrow bands, so the row
sums of an ink mask are sharply peaked. Rotating the mask through a coarse then
fine range and keeping the angle with the sharpest profile recovers the tilt.

This runs before OCR, not after: it changes what the recogniser is given, and the
recognition, boxes and cell verification all describe the corrected image.
"""
from __future__ import annotations

from typing import Any

# More than this is a photo of a page lying at an angle, not a hand-held capture;
# the detector loses the page's structure long before it, and rotating further
# would only crop more of it.
SEARCH_LIMIT = 8.0
COARSE_STEP = 2.0
FINE_STEP = 0.25
# The angle is a property of the layout, so a small copy is enough to measure it.
SAMPLE_LONG_SIDE = 500
# Text is darker than the paper around it, measured against a local median so a
# gradient or a dark desk does not turn the background into "ink".
INK_DELTA = 20
MIN_INK_FRACTION = 0.001
MAX_INK_FRACTION = 0.35
# A correction smaller than this is measurement noise, not a tilted page.
MIN_CORRECTION = 0.5
# The aligned profile must be clearly sharper than the unrotated one, otherwise
# the page is either straight or has no text lines to align.
MIN_GAIN = 1.05


def _median_size(width: int, height: int) -> int:
    size = min(15, width, height)
    if size % 2 == 0:
        size -= 1
    return max(3, size)


def _ink_mask(image: Any) -> Any:
    """The page's text as a binary mask, robust to lighting and background."""
    import numpy as np
    from PIL import Image, ImageFilter

    gray = image.convert("L")
    width, height = gray.size
    longest = max(width, height)
    if longest <= 0:
        return None
    if longest > SAMPLE_LONG_SIDE:
        scale = SAMPLE_LONG_SIDE / longest
        gray = gray.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                           Image.BILINEAR)
    if min(gray.size) < 9:
        return None
    pixels = np.asarray(gray)
    paper = np.asarray(gray.filter(ImageFilter.MedianFilter(size=_median_size(*gray.size))))
    ink = pixels.astype(np.int16) < (paper.astype(np.int16) - INK_DELTA)
    fraction = float(ink.mean())
    if not MIN_INK_FRACTION <= fraction <= MAX_INK_FRACTION:
        return None
    return Image.fromarray((ink * 255).astype(np.uint8))


def _profile_sharpness(mask: Any, angle: float) -> float:
    """How tightly the ink is packed into rows after rotating the mask by `angle`."""
    import numpy as np
    from PIL import Image

    rotated = mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    width, height = mask.size
    left = (rotated.width - width) // 2
    top = (rotated.height - height) // 2
    # The same central window at every angle, so blank border rows added by the
    # expansion cannot make a rotation look sharper than it is.
    window = rotated.crop((left, top, left + width, top + height))
    profile = np.asarray(window, dtype=np.float32).sum(axis=1)
    return float(profile.var())


def estimate_skew(image: Any, *, limit: float = SEARCH_LIMIT) -> float:
    """The rotation (in degrees) that would straighten this page, or 0.0."""
    import numpy as np

    mask = _ink_mask(image)
    if mask is None:
        return 0.0
    straight = _profile_sharpness(mask, 0.0)
    coarse_angles = np.arange(-limit, limit + COARSE_STEP / 2, COARSE_STEP)
    coarse = [(float(angle), _profile_sharpness(mask, float(angle)))
              for angle in coarse_angles]
    best_angle, best_score = max(coarse, key=lambda pair: pair[1])
    fine_angles = np.arange(best_angle - COARSE_STEP,
                            best_angle + COARSE_STEP + FINE_STEP / 2, FINE_STEP)
    fine = [(float(angle), _profile_sharpness(mask, float(angle)))
            for angle in fine_angles]
    best_angle, best_score = max(fine, key=lambda pair: pair[1])
    if best_score <= straight * MIN_GAIN or abs(best_angle) < MIN_CORRECTION:
        return 0.0
    return round(best_angle, 2)


def rotate_page(image: Any, angle: float) -> Any:
    """Turn the page back by `angle` degrees, keeping all of it in the frame."""
    if not angle:
        return image
    from PIL import Image

    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def deskew_image(image: Any, *, limit: float = SEARCH_LIMIT) -> tuple[Any, float]:
    """Straighten a photographed page; returns the image and the applied angle."""
    angle = estimate_skew(image, limit=limit)
    if not angle:
        return image, 0.0
    return rotate_page(image, angle), angle
