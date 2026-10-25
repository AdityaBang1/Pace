"""The mix corpus is only honest if composed rows never reach validation or test."""
from __future__ import annotations

import json
from pathlib import Path

from pace.data import save_jsonl
from pace.field_mix import build_mix


def _row(label, text, heading, source, group, school=None, source_field=""):
    row = {
        "group_id": group,
        "label": label,
        "text": text,
        "bare_text": text,
        "heading": heading,
        "source": source,
        "source_key": source,
        "source_field": source_field,
        "provenance": "fixture",
        "meta": {"school": school} if school else {},
    }
    return row


def _splits(tmp_path: Path, rows):
    split_dir = tmp_path / "splits"
    for name, predicate in (
        ("train", lambda r: r["group_id"].endswith(":train")),
        ("validation", lambda r: r["group_id"].endswith(":val")),
        ("test", lambda r: r["group_id"].endswith(":test")),
    ):
        save_jsonl(split_dir / f"{name}.jsonl", [r for r in rows if predicate(r)])
    return split_dir


def _fixture():
    return [
        _row("course_title", "Intro to Biology", "Class Name", "USC", "usc:train"),
        _row("course_title", "Statistical Inference", "title", "UW", "uw:train"),
        _row("course_title", "Thermodynamics", "Course Title", "MIT", "mit:train"),
        _row("office_hours", "Mon 2-4pm", "Office Hours", "schoolA", "syl-a:train",
             school="A University"),
        _row("office_hours", "Tue 10-12", "Instructor Office Hours", "schoolB", "syl-b:train",
             school="B University"),
        _row("office_hours", "Thu 1-2pm", "Office Hours", "schoolC", "syl-c:val",
             school="C University"),
        _row("office_hours", "Fri 9-10am", "Office Hours", "schoolD", "syl-d:test",
             school="D University"),
    ]


def test_composed_rows_never_enter_validation_or_test(tmp_path):
    rows = _fixture()
    splits, stats = build_mix(rows, _splits(tmp_path, rows), seed=7)
    assert stats["train_rows"] > 0
    assert all("family" in row for row in splits["train"])
    for name in ("validation", "test"):
        assert all("family" not in row for row in splits[name])
    assert {row["group_id"] for row in splits["test"]} == {"syl-d:test"}


def test_a_cross_row_takes_its_heading_from_a_different_source(tmp_path):
    rows = _fixture()
    splits, _ = build_mix(rows, _splits(tmp_path, rows), seed=3, cross_rate=1.0, bare_rate=0.0)
    cross = [row for row in splits["train"] if row["family"] == "cross"]
    assert cross, "no cross-source rows were composed"
    own_heading = {"USC": "Class Name", "UW": "title", "MIT": "Course Title",
                   "schoolA": "Office Hours", "schoolB": "Instructor Office Hours"}
    for row in cross:
        assert row["text"].split(":", 1)[0].strip() != own_heading[row["source_key"]], (
            "a value was paired with its own source's heading")


def test_composed_text_is_the_real_heading_and_value_without_doubling(tmp_path):
    rows = _fixture()
    splits, _ = build_mix(rows, _splits(tmp_path, rows), seed=5, cross_rate=0.0, bare_rate=0.0)
    natural = {row["text"] for row in splits["train"] if row["family"] == "natural"}
    assert "Class Name: Intro to Biology" in natural
    assert "Intro to Biology" not in natural
    # A value that already begins with its heading is kept once, not prefixed twice.
    doubled = _row("office_hours", "Office Hours: Mon 2-4pm", "Office Hours", "schoolA",
                   "syl-a:train", school="A University")
    twice, _ = build_mix([doubled], _splits(tmp_path, [doubled]), seed=5,
                         cross_rate=0.0, bare_rate=0.0)
    natural = [row["text"] for row in twice["train"] if row["family"] == "natural"]
    assert natural == ["Office Hours: Mon 2-4pm"]


def test_a_held_out_school_contributes_nothing_to_training(tmp_path):
    rows = _fixture()
    splits, stats = build_mix(rows, _splits(tmp_path, rows), seed=11,
                              holdout_schools=["A University"])
    assert stats["holdout_schools"] == ["A University"]
    assert all(row.get("meta", {}).get("school") != "A University"
               for row in splits["train"])
    assert any(row.get("meta", {}).get("school") == "A University" for row in splits["test"])


def test_the_per_label_cap_is_respected(tmp_path):
    rows = [_row("course_title", f"Title {i}", "Class Name", "USC", "usc:train")
            for i in range(20)]
    splits, _ = build_mix(rows, _splits(tmp_path, rows), seed=2, cross_rate=0.0,
                          bare_rate=0.0, caps={"natural": 5})
    assert sum(1 for row in splits["train"]
               if row["family"] == "natural" and row["label"] == "course_title") == 5


def test_no_row_is_repeated(tmp_path):
    rows = _fixture() + [
        _row("course_title", "Intro to Biology", "Class Name", "USC", "usc:train"),
    ]
    splits, _ = build_mix(rows, _splits(tmp_path, rows), seed=13, cross_rate=1.0)
    keys = [(row["text"].casefold(), row["label"]) for row in splits["train"]]
    assert len(keys) == len(set(keys))


def test_the_written_corpus_carries_its_stats(tmp_path):
    rows = _fixture()
    splits, stats = build_mix(rows, _splits(tmp_path, rows), seed=17)
    assert stats["train_by_family"].get("natural", 0) > 0
    assert stats["validation_rows"] == len(splits["validation"])
    assert json.dumps(stats)  # serialisable as written to corpus_stats.json
