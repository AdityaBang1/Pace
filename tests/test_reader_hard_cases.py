"""Adversarial inputs the shipped reader must survive without lying about them.

These are invariants, not accuracy claims: a distribution that sums to one, the
same answer for the same input, a defined answer for text longer than the state
window, and no crash on script it cannot read. Accuracy on hard rows is measured
elsewhere (`artifacts/jev-field-v4/abstention.json`), because a test that asserts a
label on an ambiguous string just pins today's guess in place.
"""
from __future__ import annotations

from pathlib import Path

import pytest

CHECKPOINT = Path("artifacts/jev-field-v4")
needs_reader = pytest.mark.skipif(not (CHECKPOINT / "model.pt").exists(),
                                  reason="v4 checkpoint is not present")


@pytest.fixture(scope="module")
def engine():
    from pace.jev_model import JevEngine

    return JevEngine(CHECKPOINT, device="cpu")


@needs_reader
def test_every_input_produces_a_proper_distribution(engine):
    texts = ["", "   ", "\n\t ", "4", "2:00", "TBA", "----", "…", "課程大綱", "หลักสูตร",
             "a" * 5000, "Attendance required. " * 40]
    rows = engine.probabilities(texts)
    assert len(rows) == len(texts)
    for row in rows:
        assert set(row) == set(engine.labels)
        assert abs(sum(row.values()) - 1.0) < 1e-4
        assert all(0.0 <= value <= 1.0 for value in row.values())


@needs_reader
def test_the_same_text_gets_the_same_answer_twice(engine):
    text = "Assessment: two midterms (20% each) and a final exam (60%)."
    first = engine.probabilities([text])[0]
    second = engine.probabilities([text])[0]
    assert first == second


@needs_reader
def test_text_longer_than_the_state_window_is_truncated_not_rejected(engine):
    short = "Grading: homework 40%, final 60%."
    long = short + " " + "Additional course policy text. " * 400
    assert engine.classify(short)["label"] in engine.labels
    assert engine.classify(long)["label"] in engine.labels
    ids, attention, state_mask = engine._encode([long])
    assert ids.shape[1] <= engine.max_state + 64
    assert int(state_mask.sum()) <= engine.max_state


@needs_reader
def test_a_value_with_no_header_still_gets_a_field(engine):
    result = engine.classify("Wednesdays 14:00-16:00, Room 312")
    assert result["label"] in engine.labels
    ranked = [item["label"] for item in result["top"]]
    assert len(ranked) == len(set(ranked)), "top-k must not repeat a label"


@needs_reader
def test_the_menu_can_be_rewritten_without_retraining(engine):
    written = {"thesis_deadline": "the date a thesis draft must be submitted",
               "lab_fee": "the amount charged for laboratory materials"}
    scores = engine.score_against(["Draft due 14 March 2025"], written)[0]
    assert set(scores) == set(written)
    assert abs(sum(scores.values()) - 1.0) < 1e-4


@needs_reader
def test_the_abstention_threshold_actually_separates_on_bare_values(engine):
    """A threshold is only worth publishing if accepted rows are more accurate."""
    rows = []
    path = Path("data/processed/field_mix/test.jsonl")
    if not path.exists():
        pytest.skip("mix test set is not present")
    import json

    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 1200:
                break
            row = json.loads(line)
            if (row.get("bare_text") or "").strip() and row["label"] in engine.labels:
                rows.append(row)
    predictions = engine.classify_many([row["bare_text"] for row in rows])
    correct = [item["label"] == row["label"] for item, row in zip(predictions, rows)]
    confidence = [item["confidence"] for item in predictions]
    accepted = [ok for ok, value in zip(correct, confidence) if value >= 0.9]
    assert len(accepted) > 0.8 * len(correct), "the threshold must keep most rows"
    assert sum(accepted) / len(accepted) >= sum(correct) / len(correct)
