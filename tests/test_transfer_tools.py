"""The tools that produced the v3 promotion, checked on tiny inputs.

Two things can silently go wrong in that pipeline: averaging checkpoints that do
not share a label menu (which produces a model that cannot be scored), and
composing SIS exports whose columns no longer reach the adapter. Both are cheap
to test and expensive to notice later.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path, weight: float, labels=None, config=None):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config or {"hidden": 8, "layers": 1},
        "labels": labels or ["a", "b"],
        "descriptions": ["first", "second"],
        "state_dict": {"w": torch.full((2, 2), weight), "n": torch.zeros(1, dtype=torch.long)},
        "temperature": 1.0,
        "parameter_count": 4,
    }, path / "model.pt")
    return path


def test_soup_averages_floating_tensors_and_copies_integers(tmp_path):
    soup_script = _script("soup_checkpoints")
    first = _checkpoint(tmp_path / "one", 1.0)
    second = _checkpoint(tmp_path / "two", 3.0)

    merged = soup_script.soup([first, second], [0.75, 0.25])

    assert torch.allclose(merged["state_dict"]["w"], torch.full((2, 2), 1.5))
    assert merged["state_dict"]["n"].dtype == torch.long
    assert merged["soup_weights"] == [0.75, 0.25]


def test_soup_refuses_a_different_label_menu(tmp_path):
    soup_script = _script("soup_checkpoints")
    first = _checkpoint(tmp_path / "one", 1.0)
    second = _checkpoint(tmp_path / "two", 3.0, labels=["a", "c"])

    with pytest.raises(SystemExit):
        soup_script.soup([first, second])


def test_soup_refuses_weights_that_do_not_sum_to_one(tmp_path):
    soup_script = _script("soup_checkpoints")
    first = _checkpoint(tmp_path / "one", 1.0)
    second = _checkpoint(tmp_path / "two", 3.0)

    with pytest.raises(SystemExit):
        soup_script.soup([first, second], [0.5, 0.2])


def test_composed_exports_reach_the_adapter_with_their_labels(tmp_path):
    builder = _script("build_sis_exports")
    from pace.sis_import import load_sis_examples

    groups = {
        "course:1": {"course_title": "Algebra I", "course_code": "MATH-101",
                     "program": "Mathematics", "term": "Fall 2026",
                     "class_meeting_room": "Room 12"},
        "course:2": {"course_title": "Geometry", "course_code": "MATH-102",
                     "class_section": "01"},
        # A group without a title is not a course and must not become one.
        "course:3": {"credits": "3"},
    }
    tables = builder.build(groups, max_courses=10, seed=1)
    builder.write_tables(tables, tmp_path / "export")

    rows = load_sis_examples(tmp_path / "export", school="Test", source="Infinite Campus")
    by_label: dict[str, list[str]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row["text"])

    # The title appears in both courses.csv and classes.csv, which is how the
    # interchange format repeats it; both rows are labelled the same way.
    assert set(by_label["course_title"]) == {"Algebra I", "Geometry"}
    assert "MATH-101" in by_label["course_code"]
    assert "Room 12" in by_label["class_meeting_room"]
    assert "Fall 2026" in by_label["term"]
    assert set(by_label) <= {"course_title", "course_code", "program", "term",
                             "class_meeting_room", "class_section"}
    assert all(row["group_id"] for row in rows)


def test_export_builder_skips_cells_that_are_sentences(tmp_path):
    builder = _script("build_sis_exports")
    long_text = "This course examines " + "many things " * 20
    groups = {"course:1": {"course_title": "Algebra I", "course_code": long_text,
                           "program": "Mathematics"}}
    tables = builder.build(groups, max_courses=10, seed=1)

    assert tables["courses"][0][5] == ""
