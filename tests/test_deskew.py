"""Coverage for straightening a photographed page before recognition."""
from pathlib import Path

import pytest

from pace.deskew import deskew_image, estimate_skew

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

CONTROL = Path("data/samples/transcript_term_marks.png")


def _photo(angle: float) -> Image.Image:
    """The page on a plain background, turned the way a camera would turn it."""
    with Image.open(CONTROL) as handle:
        page = handle.convert("L")
    margin = 60
    canvas = Image.new("L", (page.width + 2 * margin, page.height + 2 * margin), 210)
    canvas.paste(page, (margin, margin))
    return canvas.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=210)


@pytest.mark.skipif(not CONTROL.exists(), reason="the control page is unavailable")
def test_a_straight_page_is_left_alone():
    with Image.open(CONTROL) as handle:
        page = handle.convert("L")
    corrected, angle = deskew_image(page)

    assert angle == 0.0
    assert corrected is page


@pytest.mark.skipif(not CONTROL.exists(), reason="the control page is unavailable")
def test_a_four_degree_capture_is_measured_and_turned_back():
    corrected, angle = deskew_image(_photo(4))

    # The estimator reports the correction: a page turned +4 degrees needs -4.
    assert -5.0 <= angle <= -3.0
    assert abs(estimate_skew(corrected)) < 1.0
