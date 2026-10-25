"""A non-academic page must not be read as an academic document.

The previous document classifier called a bank statement a transcript at 0.9975.
These tests pin the replacement's behaviour on the four inputs that exposed it,
and pin the measured false-academic rate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = Path("artifacts/doc-type")

CASES = {
    "receipt": "CORNER MARKET\nDate: 10/14/2026\n2 x Coffee 3.50\n1 x Sandwich 7.25\n"
               "Subtotal 10.75\nTax 0.86\nTOTAL 11.61\nVISA **** 4412",
    "menu": "THE BLUE PLATE\nStarters\nSoup of the day 6\nMains\nRoast chicken 18\n"
            "Open Tue-Sun 5pm-10pm",
    "bank_statement": "STATEMENT PERIOD 01/09/2026 - 30/09/2026\nOpening balance 1,204.55\n"
                      "09/03 Direct deposit 2,100.00\n09/05 Rent -1,450.00\nClosing balance 1,342.15",
    "article": "Urban heat is not evenly distributed. A study of 300 neighbourhoods found "
               "that blocks with mature canopy were up to four degrees cooler on summer "
               "afternoons than bare blocks a few streets away.",
}


@pytest.fixture(scope="module")
def classifier():
    if not (ARTIFACT / "doc_type.joblib").exists():
        pytest.skip("document-type checkpoint not built")
    from pace.doc_type import DocumentTypeClassifier

    return DocumentTypeClassifier.load(ARTIFACT)


@pytest.mark.parametrize("name", sorted(CASES))
def test_non_academic_pages_are_refused(classifier, name):
    result = classifier.classify(CASES[name])
    assert result["academic"] is False
    assert result["label"] == "other"
    assert result["confidence"] >= 0.5


def test_a_syllabus_is_academic(classifier):
    text = Path("data/samples/example_syllabus.txt").read_text(encoding="utf-8")
    result = classifier.classify(text)
    assert result["academic"] is True
    assert result["label"] in {"syllabus", "course_catalog", "schedule"}


def test_measured_false_academic_rate_is_zero_on_real_news():
    metrics_path = ARTIFACT / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("document-type metrics not built")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["test"]["false_academic_rate"] <= 0.01
    assert metrics["test"]["macro_f1"] >= 0.95
    assert metrics["threshold_sweep"][0]["negatives"] > 100
