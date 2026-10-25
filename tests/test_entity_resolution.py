"""The matcher must never merge two different courses.

False merges corrupt a record silently, so the shipped threshold is chosen from
the measured false-merge rate and a level/number conflict vetoes a merge outright.
These tests fix both the behaviour and the bar.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path("artifacts/entity-resolution")


@pytest.fixture(scope="module")
def matcher():
    if not (ARTIFACT / "matcher.joblib").exists():
        pytest.skip("entity-resolution checkpoint not built")
    from pace.entity_resolution import CourseMatcher

    return CourseMatcher.load(ARTIFACT)


@pytest.mark.parametrize("left,right", [
    ("AP Chem", "AP Chemistry"),
    ("Chem AP", "AP Chemistry"),
    ("AP Chemistry", "Advanced Placement Chemistry"),
    ("Introduction to Statistics", "Intro to Statistics"),
])
def test_real_aliases_merge(matcher, left, right):
    assert matcher.decide(left, right).decision == "same"


@pytest.mark.parametrize("left,right", [
    ("AP Calculus AB", "AP Calculus BC"),
    ("Calculus I", "Calculus II"),
    ("Chemistry Honors", "AP Chemistry"),
    ("TEACHING WORLD LANGUAGES (K-8)", "TEACHING WORLD LANGUAGES (6-12)"),
    ("Biology", "AP Biology"),
])
def test_near_misses_never_merge(matcher, left, right):
    # "uncertain" is allowed for the genuinely ambiguous pair; "same" is not.
    assert matcher.decide(left, right).decision in {"different", "uncertain"}


def test_the_measured_false_merge_rate_stays_low():
    metrics_path = ARTIFACT / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("entity-resolution metrics not built")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["test"]["false_merge_rate"] <= 0.01
    assert metrics["test"]["recall"] >= 0.85
    assert metrics["institution_holdout"]["false_merge_rate"] <= 0.01
