"""Resolve the orientation a phone wrote in EXIF, once, for every consumer.

A phone usually stores its JPEG pixels in the sensor's orientation and records how
the viewer should turn them with an EXIF Orientation tag. A recogniser does not
read that tag: it reads the stored pixels. A sideways photo therefore produces
sideways text, no rows, and — worse — verification crops taken from a different
orientation than the page read.

This module is the single place that turns the tag into upright pixels. Both the
OCR read and the verification-crop replay call :func:`open_oriented_image`, so
they cannot disagree. The tag is recorded in the page's provenance rather than
silently discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ORIENTATION_TAG = 274

ORIENTATION_NAMES = {
    1: "normal",
    2: "mirror_horizontal",
    3: "rotate_180",
    4: "mirror_vertical",
    5: "transpose",
    6: "rotate_90_cw",
    7: "transverse",
    8: "rotate_270_cw",
}


@dataclass(frozen=True)
class OrientedImage:
    """The pixels a viewer would see, and the tag that produced them."""

    image: Any
    orientation: int | None
    frames: int

    def detail(self) -> dict[str, Any]:
        """Provenance for a page read, empty when the file needed no turning."""
        if self.orientation in (None, 1):
            return {}
        return {
            "exif_orientation": self.orientation,
            "exif_orientation_name": ORIENTATION_NAMES.get(self.orientation, "unknown"),
        }


def _read_orientation(handle: Any) -> int | None:
    try:
        value = handle.getexif().get(ORIENTATION_TAG)
    except (AttributeError, OSError, ValueError, SyntaxError):
        return None
    if value is None:
        return None
    try:
        orientation = int(value)
    except (TypeError, ValueError):
        return None
    return orientation if orientation in ORIENTATION_NAMES else None


def open_oriented_image(path: str) -> OrientedImage:
    """Open an image and return the pixels its EXIF orientation describes.

    The orientation is applied to every format Pillow can read, not only JPEG:
    PNG and WebP may carry the same tag. A malformed tag is left as stored rather
    than failing the read; the page's provenance then says no orientation was
    applied, so a consumer is not told a correction happened when it did not.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        frames = getattr(handle, "n_frames", 1)
        orientation = _read_orientation(handle)
        if orientation in (None, 1):
            image = handle.copy()
            orientation = None
        else:
            try:
                image = ImageOps.exif_transpose(handle)
            except (OSError, ValueError, SyntaxError):
                image = handle.copy()
                orientation = None
        image = image.convert("L")
    return OrientedImage(image=image, orientation=orientation, frames=frames)
