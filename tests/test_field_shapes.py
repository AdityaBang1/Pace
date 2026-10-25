"""The shape layer may only fire when there is nothing left to interpret.

Every negative below is a value the model is better at than a regex: a capacity, a
policy sentence about the same room, a course number that looks like a room. The
positive list is the measured failure set -- values the model answered at 95-99%
confidence and got wrong.
"""
from __future__ import annotations

import json

import pytest

from pace.field_shapes import resolve

POSITIVES = {
    "Room 303": "class_meeting_room",
    "Rm. 12": "class_meeting_room",
    "Office 118": "class_meeting_room",
    "Dwinelle Hall, Room 210": "class_meeting_room",
    "Dr. Johnson": "instructor",
    "Prof. Allison Johnson": "instructor",
    "Johnson, Allison": "instructor",
    "Wed 2-4pm": "meeting_time",
    "MWF 10:00-10:50": "meeting_time",
    "TTh 1:00 PM - 2:30 PM": "meeting_time",
    "Period 4": "meeting_time",
    "Final Examination December 16": "exam_schedule",
    "Midterm: Oct 21": "exam_schedule",
    "3 credits": "credits",
    "4.0 units": "credits",
    "Fall 2026": "term",
}

NEGATIVES = [
    "Room capacity 30",
    "Office hours by appointment",
    "Wed 2-4pm or by appointment",
    "Course capacity 120",
    "Sexton 204",       # the same shape as a course number: "Biology 101"
    "Biology 101",
    "Tests 40%",
    "AP Chemistry",
    "Period 4 review session",
    "Fall 2026 syllabus",
    "Dr. Johnson's office hours",
    "",
]


@pytest.mark.parametrize("text,label", sorted(POSITIVES.items()))
def test_shapes_resolve_to_their_field(text, label):
    resolved = resolve(text)
    assert resolved is not None, text
    assert resolved["label"] == label
    assert resolved["source"] == "shape"
    assert resolved["reason"]


@pytest.mark.parametrize("text", NEGATIVES)
def test_ambiguous_values_are_left_to_the_model(text):
    assert resolve(text) is None


def test_cli_uses_the_shape_layer_and_can_be_turned_off(capsys):
    from pace.cli import main

    main(["classify-field", "Room 303"])
    shaped = json.loads(capsys.readouterr().out)
    assert shaped["label"] == "class_meeting_room"
    assert shaped["model"] == "shape"
    assert shaped["percent"] == 99.0

    # Without the layer the model is asked, which is what the ablation shows.
    main(["classify-field", "Room 303", "--no-shapes", "--device", "cpu"])
    model = json.loads(capsys.readouterr().out)
    assert model["model"] != "shape"
    assert 0.0 <= model["confidence"] <= 1.0
