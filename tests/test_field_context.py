"""Tests for the heading-aware corpus, the shared fold plan and the cascade.

The heading-aware corpus exists because a field's identity comes from the
heading a value sits under. These tests pin the three properties that make
the comparison honest: the heading is never invented, the two corpora can be
lined up row by row, and the cross-institution folds are the *same* folds for
both models.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pace.field_context import _heading_from_source_field, build_context_corpus
from pace.field_model import classify_section
from pace.field_splits import cross_institution_folds


def _row(text: str, label: str, group: str, source_field: str = "",
         source: str = "x/y", meta: dict | None = None) -> dict:
    return {
        "text": text, "label": label, "group_id": group, "source": source,
        "source_key": "x", "source_field": source_field,
        "text_sha256": text, "meta": meta or {},
    }


# ---------------------------------------------------------------------------
# Heading-aware corpus
# ---------------------------------------------------------------------------

def test_context_corpus_prefixes_the_natural_heading():
    rows = [_row("3.0 units", "credits", "g1", source_field="Units")]
    out, stats = build_context_corpus(rows)
    assert stats["rows"] == 1
    assert out[0]["text"] == "Units: 3.0 units"
    assert out[0]["heading"] == "Units"
    assert out[0]["bare_text"] == "3.0 units"
    assert out[0]["label"] == "credits"


def test_context_corpus_keeps_text_that_already_starts_with_the_heading():
    rows = [_row("Office Hours: Mon 2-4pm", "office_hours", "g1", source_field="Office Hours")]
    out, _ = build_context_corpus(rows)
    assert out[0]["text"] == "Office Hours: Mon 2-4pm"
    assert out[0]["bare_text"] == "Office Hours: Mon 2-4pm"


def test_context_corpus_drops_rows_with_no_heading_at_all():
    rows = [_row("3.0 units", "credits", "g1", source_field="")]
    out, stats = build_context_corpus(rows)
    assert out == []
    assert stats["rows_without_heading"] == 1


def test_context_corpus_removes_duplicate_heading_value_pairs():
    rows = [_row("3.0 units", "credits", "g1", source_field="Units"),
            _row("3.0 units", "credits", "g2", source_field="Units")]
    out, _ = build_context_corpus(rows)
    assert len(out) == 1


def test_qualifier_suffix_is_stripped_from_the_heading():
    assert _heading_from_source_field("Attendance (body)") == "Attendance"
    assert _heading_from_source_field("Attendance (as written)") == "Attendance"


# ---------------------------------------------------------------------------
# Shared cross-institution folds
# ---------------------------------------------------------------------------

def _syllabus_rows(schools: list[str], per_school: int = 2) -> list[dict]:
    rows = []
    for school in schools:
        for index in range(per_school):
            rows.append({
                "text": f"{school} section {index}", "label": "office_hours",
                "group_id": f"syllabus-doc:{school}-{index}",
                "source": "a/b", "source_key": "a", "source_field": "Office Hours",
                "text_sha256": f"{school}{index}",
                "meta": {"school": school},
            })
    return rows


def test_folds_are_disjoint_across_universities():
    rows = _syllabus_rows(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    folds = cross_institution_folds(rows, folds=3)
    seen: set[str] = set()
    for fold in folds:
        schools = set(fold["holdout_schools"])
        assert not (schools & seen)
        seen |= schools
    assert len(seen) == 6


def test_validation_rows_never_come_from_a_held_out_university():
    rows = _syllabus_rows(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    folds = cross_institution_folds(rows, folds=3)
    for fold in folds:
        heldout = set(fold["holdout_schools"])
        train_schools = {row["meta"]["school"] for row in fold["train"]}
        validation_schools = {row["meta"]["school"] for row in fold["validation"]}
        test_schools = {row["meta"]["school"] for row in fold["test"]}
        assert not (validation_schools & heldout)
        assert not (train_schools & heldout)
        assert test_schools <= heldout


def test_a_shared_school_plan_gives_identical_folds_for_different_corpora():
    """Two corpora with different row counts must be measured on the same folds."""
    plan = [["alpha", "beta"], ["gamma", "delta"]]
    big = _syllabus_rows(["alpha", "beta", "gamma", "delta"], per_school=4)
    small = _syllabus_rows(["alpha", "beta", "gamma", "delta"], per_school=1)
    folds_big = cross_institution_folds(big, folds=2, school_plan=plan)
    folds_small = cross_institution_folds(small, folds=2, school_plan=plan)
    assert [f["holdout_schools"] for f in folds_big] == [f["holdout_schools"] for f in folds_small]
    # The two corpora hold different numbers of documents per university, but
    # every document of the smaller corpus that exists in the larger one must
    # land in the same fold.
    for big_fold, small_fold in zip(folds_big, folds_small):
        assert set(small_fold["holdout_documents"]) <= set(big_fold["holdout_documents"])


def test_a_fold_plan_that_omits_a_university_leaves_it_in_training():
    rows = _syllabus_rows(["alpha", "beta", "gamma"])
    folds = cross_institution_folds(rows, folds=1, school_plan=[["alpha"]])
    trained = {row["meta"]["school"] for row in folds[0]["train"]}
    assert trained == {"beta", "gamma"}


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def test_paired_key_lines_up_a_bare_row_with_its_context_row():
    from pace.field_pair import _key

    bare = {"group_id": "syllabus-doc:a-1", "text": "3.0 units", "label": "credits"}
    context = {"group_id": "syllabus-doc:a-1", "text": "Units: 3.0 units",
               "bare_text": "3.0 units", "label": "credits"}
    assert _key(bare) == _key(context)


def test_paired_key_separates_different_labels():
    from pace.field_pair import _key

    one = {"group_id": "g", "text": "x", "label": "credits"}
    two = {"group_id": "g", "text": "x", "label": "meeting_days"}
    assert _key(one) != _key(two)


# ---------------------------------------------------------------------------
# Heading-first cascade
# ---------------------------------------------------------------------------

class _StubEngine:
    """A field engine that answers from a fixed table instead of a model."""

    def __init__(self, table: dict[str, tuple[str, float]]):
        self.table = table

    def classify(self, text: str, top_k: int = 5) -> dict:
        label, confidence = self.table[text]
        return {"label": label, "task": "t", "confidence": confidence,
                "percent": round(confidence * 100, 2), "top": [], "probabilities": {}}


def test_cascade_keeps_a_confident_heading_prediction():
    heading = _StubEngine({"Office Hours": ("office_hours", 0.97)})
    context = _StubEngine({"Office Hours: Mon 2-4pm": ("office_hours", 0.9)})
    result = classify_section("Office Hours", "Mon 2-4pm", heading, context, threshold=0.8)
    assert result["source"] == "heading"
    assert result["fallback"] is False
    assert result["label"] == "office_hours"


def test_cascade_falls_back_when_the_heading_is_unfamiliar():
    heading = _StubEngine({"Zzz Novel Heading": ("other", 0.12)})
    context = _StubEngine({"Zzz Novel Heading: some text": ("course_content", 0.66)})
    result = classify_section("Zzz Novel Heading", "some text", heading, context, threshold=0.8)
    assert result["source"] == "heading+value"
    assert result["fallback"] is True
    assert result["label"] == "course_content"


def test_cascade_without_a_context_model_uses_the_heading():
    heading = _StubEngine({"Grading": ("grading_policy", 0.55)})
    result = classify_section("Grading", "A-F", heading, None, threshold=0.8)
    assert result["source"] == "heading"
    assert result["label"] == "grading_policy"


def test_cascade_reports_nothing_when_there_is_no_heading_and_no_context_model():
    heading = _StubEngine({})
    result = classify_section("", "some text", heading, None)
    assert result["source"] == "none"
    assert result["label"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _checkpoint_dir() -> Path | None:
    artifact = Path("artifacts/field-classifier-minilm")
    if (artifact / "model.pt").exists():
        return artifact
    for candidate in sorted(artifact.glob("seed-*")):
        if (candidate / "model.pt").exists():
            return candidate
    return None


@pytest.mark.skipif(_checkpoint_dir() is None, reason="trained field artifact is not present")
def test_classify_section_cli_reads_a_heading_and_a_value(capsys):
    from pace.cli import main

    main(["classify-section", "--heading", "Office Hours",
          "--value", "Wednesdays 3-4 PM",
          "--checkpoint", str(_checkpoint_dir()),
          "--context-checkpoint", "artifacts/field-classifier-context",
          "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "office_hours"
    assert payload["source"] in {"heading", "heading+value"}
    assert payload["heading_text"] == "Office Hours"
    assert 0.0 <= payload["confidence"] <= 1.0
