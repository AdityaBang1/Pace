"""v4 exists because the reader was reading column headers instead of values.

These tests guard the two claims that made the swap worthwhile: the checkpoint
loads and answers a bare value without a header, and it is at least as accurate on
bare values as the checkpoint it replaced. They skip when the local artifacts are
absent, like the other trained-artifact tests in this suite.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

V2 = Path("artifacts/jev-field-v2")
V4 = Path("artifacts/jev-field-v4")
MIX_TEST = Path("data/processed/field_mix/test.jsonl")


def _rows(limit: int = 300) -> list[dict]:
    rows = []
    with MIX_TEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("bare_text") or "").strip():
                rows.append(row)
    random.Random(11).shuffle(rows)
    return rows[:limit]


@pytest.mark.skipif(not (V4 / "model.pt").exists(), reason="v4 checkpoint is not present")
def test_the_shipped_reader_answers_a_bare_value_without_a_header():
    from pace.jev_model import JevEngine

    engine = JevEngine(V4, device="cpu")
    result = engine.classify("MATH 102 Calculus II")
    assert result["label"] in engine.labels
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    assert result["confidence"] >= 0.5


@pytest.mark.skipif(not (V2 / "model.pt").exists() or not (V4 / "model.pt").exists()
                    or not MIX_TEST.exists(), reason="both checkpoints and the mix test set are required")
def test_v4_is_at_least_as_good_as_v2_on_values_without_headers():
    from pace.jev_model import JevEngine

    rows = [dict(row, text=row["bare_text"]) for row in _rows()]
    labels = {row["label"] for row in rows}
    scores = {}
    for name, path in (("v2", V2), ("v4", V4)):
        engine = JevEngine(path, device="cpu")
        predictions = [item["label"] for item in engine.classify_many([row["text"] for row in rows])]
        correct = [prediction == row["label"] for prediction, row in zip(predictions, rows)]
        scores[name] = sum(correct) / len(correct)
    assert labels, "the sample must carry labels"
    # The measured gap on 7,571 rows is +4.8 accuracy points; a 300-row sample
    # cannot resolve a point, so allow a small negative tolerance rather than
    # making this test flaky.
    assert scores["v4"] >= scores["v2"] - 0.02, scores
    assert scores["v4"] >= 0.85, scores


@pytest.mark.skipif(not (V4 / "header_reliance.json").exists(), reason="measurement is not present")
def test_the_recorded_header_gap_is_smaller_than_the_one_it_replaced():
    v4 = json.loads((V4 / "header_reliance.json").read_text(encoding="utf-8"))
    v2_path = V2 / "header_reliance.json"
    assert v4["with_header"]["accuracy"] > v4["bare_value"]["accuracy"]
    assert v4["accuracy_drop_without_header"] < 0.10
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        assert v4["bare_value"]["accuracy"] > v2["bare_value"]["accuracy"]
