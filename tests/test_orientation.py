"""Phone-photo orientation is resolved once, before both read and crop replay.

A phone stores pixels in the sensor's orientation and records the turn in EXIF.
The OCR read and the later verification crop each reopen the source, so they must
consume the same upright pixels or every cell box lands somewhere else.
"""
from pathlib import Path

import pytest

from pace.ingest import IngestError, ocr_backend_name, parse_document, read_document
from pace.orientation import ORIENTATION_TAG, open_oriented_image

pytest.importorskip("PIL")
from PIL import Image, ImageChops  # noqa: E402

CONTROL = Path("data/samples/transcript_term_marks.png")

# Stored pixels that a viewer turns into the upright page for each EXIF tag.
# These are the inverses of Pillow's `exif_transpose` operations: the two
# rotation cases (6 and 8) swap, the rest are their own inverse.
STORED_FROM_UPRIGHT = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def _upright() -> Image.Image:
    with Image.open(CONTROL) as handle:
        return handle.convert("L")


def _stored(orientation: int) -> Image.Image:
    upright = _upright()
    operation = STORED_FROM_UPRIGHT[orientation]
    return upright.copy() if operation is None else upright.transpose(operation)


def _save_with_orientation(path: Path, orientation: int, *, jpeg: bool) -> None:
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = orientation
    stored = _stored(orientation)
    if jpeg:
        stored.save(path, format="JPEG", quality=90, exif=exif)
    else:
        stored.save(path, format="PNG", exif=exif)


@pytest.mark.skipif(not CONTROL.exists(), reason="the control page is unavailable")
@pytest.mark.parametrize("orientation", sorted(STORED_FROM_UPRIGHT))
def test_every_exif_orientation_resolves_to_the_same_upright_pixels(tmp_path, orientation):
    path = tmp_path / f"photo_{orientation}.png"
    _save_with_orientation(path, orientation, jpeg=False)

    opened = open_oriented_image(str(path))

    assert opened.image.size == _upright().size
    assert ImageChops.difference(opened.image, _upright()).getbbox() is None
    if orientation == 1:
        assert opened.orientation is None
    else:
        assert opened.orientation == orientation


BACKEND = ocr_backend_name()


@pytest.mark.skipif(BACKEND is None, reason="no OCR backend installed on this machine")
@pytest.mark.skipif(not CONTROL.exists(), reason="the control page is unavailable")
@pytest.mark.parametrize("orientation", sorted(STORED_FROM_UPRIGHT))
def test_a_phone_jpeg_reads_and_verifies_in_every_orientation(tmp_path, orientation):
    """The real path: rotated or mirrored pixels, then eight rows and 32 crops."""
    path = tmp_path / f"IMG_{orientation}.jpg"
    _save_with_orientation(path, orientation, jpeg=True)

    document = read_document(path)
    parsed = parse_document(document)
    detail = document.pages[0].detail

    if orientation == 1:
        assert "exif_orientation" not in detail
    else:
        assert detail["exif_orientation"] == orientation
    assert document.pages[0].method == "ocr"
    assert len(parsed["course_records"]) == 8
    assert parsed["checks"]["computed_total_credits"] == 7.5
    assert parsed["checks"]["credit_sum_matches_reported_total"] is True
    # The focused re-reads crop every value from the page image. They only all
    # confirm when the crop path replayed the same upright pixels as the read.
    assert parsed["verification"]["cells_checked"] == 32
    assert parsed["verification"]["unread"] == 0


def test_a_heic_photo_names_the_format_and_the_fix(tmp_path):
    """iPhone's default format must not fall through to a zero-row text parse."""
    path = tmp_path / "IMG_0001.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")

    with pytest.raises(IngestError) as caught:
        read_document(path)

    message = str(caught.value)
    assert "HEIC/HEIF" in message
    assert "JPEG" in message and "PNG" in message
    assert "cannot decode" in message


def test_an_unknown_iso_bmff_file_names_its_brand_instead_of_parsing_as_text(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42")

    with pytest.raises(IngestError) as caught:
        read_document(path)

    message = str(caught.value)
    assert "MP42" in message
    assert "JPEG" in message or "PNG" in message


def test_the_cli_refuses_a_heic_photo_instead_of_emitting_zero_rows(tmp_path, capsys):
    from pace.cli import main

    path = tmp_path / "IMG_0002.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")

    with pytest.raises(SystemExit) as caught:
        main(["parse-transcript", str(path)])

    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "HEIC/HEIF" in error and "JPEG" in error
