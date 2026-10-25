"""The leave-one-source-out benchmark must not silently train on a fraction of its budget."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_source_holdout", ROOT / "scripts" / "build_source_holdout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(counts: dict[str, int]) -> list[dict]:
    rows = []
    for label, count in counts.items():
        rows.extend({"text": f"{label} {index}", "label": label} for index in range(count))
    return rows


def test_sampling_fills_the_budget_when_labels_are_rare():
    module = _module()
    # 30 labels seen twice each and one label with 5,000 rows: a naive per-label
    # share would return 60 rows and train a much weaker model than intended.
    pool = _rows({"rare": 2, **{f"common{index}": 200 for index in range(4)},
                  "frequent": 5000})
    sampled = module._sample(pool, 1000, seed=3)
    assert len(sampled) == 1000
    labels = {row["label"] for row in sampled}
    assert {"rare", "frequent"} <= labels, "the rare label must survive sampling"


def test_sampling_keeps_every_row_when_the_pool_is_smaller_than_the_budget():
    module = _module()
    pool = _rows({"a": 3, "b": 4})
    assert len(module._sample(pool, 1000, seed=1)) == 7


def test_the_default_folds_are_distinct_publishers():
    module = _module()
    assert len(set(module.DEFAULT_SOURCES)) == len(module.DEFAULT_SOURCES)
    assert "anordkvist/gu-course-syllabus" in module.DEFAULT_SOURCES
