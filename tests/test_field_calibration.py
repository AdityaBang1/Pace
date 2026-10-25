"""Tests for the calibration and operating-point tool."""
from __future__ import annotations

import numpy as np
from pytest import approx

from pace.field_calibration import (error_upper_bound, expected_calibration_error,
                                    operating_point, reliability_bins, threshold_sweep)


def test_error_bound_needs_evidence_not_just_a_good_ratio():
    """0 errors in 100 rows cannot certify 99% precision; 300 can."""
    assert error_upper_bound(0, 100) == approx(0.0295, abs=0.001)
    assert error_upper_bound(0, 300) == approx(0.00995, abs=0.001)
    assert error_upper_bound(0, 300) < 0.01
    assert error_upper_bound(0, 100) > 0.01


def test_bound_is_monotone_in_errors():
    assert error_upper_bound(1, 100) > error_upper_bound(0, 100)
    assert error_upper_bound(10, 100) > error_upper_bound(1, 100)
    assert error_upper_bound(0, 0) is None


def test_reliability_bins_recover_a_perfectly_calibrated_predictor():
    # 90% confidence and 90% correct in one bin, 60% and 60% in another.
    confidence = np.array([0.90] * 100 + [0.60] * 100)
    correct = np.array([True] * 90 + [False] * 10 + [True] * 60 + [False] * 40)
    bins = {row["bin"]: row for row in reliability_bins(confidence, correct)}
    assert bins["0.9-1.0"]["gap"] == approx(0.0, abs=1e-9)
    assert bins["0.6-0.7"]["gap"] == approx(0.0, abs=1e-9)
    assert expected_calibration_error(confidence, correct) == approx(0.0, abs=1e-9)


def test_reliability_bins_expose_overconfidence():
    confidence = np.full(50, 0.99)
    correct = np.array([True] * 25 + [False] * 25)
    bins = reliability_bins(confidence, correct)
    top = bins[-1]
    assert top["rows"] == 50
    assert top["gap"] == approx(0.49, abs=1e-9)
    assert expected_calibration_error(confidence, correct) == approx(0.49, abs=1e-9)


def test_threshold_sweep_counts_surviving_errors_and_abstention():
    confidence = np.array([0.99, 0.96, 0.80, 0.55])
    correct = np.array([True, False, True, False])
    sweep = {row["threshold"]: row for row in threshold_sweep(confidence, correct)}
    at_99 = sweep[0.99]
    assert at_99["accepted_rows"] == 1
    assert at_99["coverage"] == 0.25
    assert at_99["precision"] == 1.0
    assert at_99["false_positive_rate"] == 0.0
    at_50 = sweep[0.50]
    assert at_50["accepted_rows"] == 4
    assert at_50["precision"] == 0.5
    assert at_50["false_positive_rate"] == 0.5
    assert at_50["abstention_rate"] == 0.0


def test_operating_point_picks_the_lowest_threshold_that_clears_the_bar():
    sweep = [
        {"threshold": 0.5, "precision": 0.80, "coverage": 1.0, "certified_precision_lower": 0.7},
        {"threshold": 0.9, "precision": 0.995, "coverage": 0.5, "certified_precision_lower": 0.99},
        {"threshold": 0.99, "precision": 1.0, "coverage": 0.1, "certified_precision_lower": 0.95},
    ]
    point = operating_point(sweep, target_precision=0.99)
    assert point["threshold"] == 0.9
    assert point["certified"] is True


def test_operating_point_reports_the_best_it_could_do_when_the_bar_is_unreachable():
    sweep = [
        {"threshold": 0.5, "precision": 0.60, "coverage": 1.0, "certified_precision_lower": 0.55},
        {"threshold": 0.9, "precision": 0.75, "coverage": 0.4, "certified_precision_lower": 0.65},
    ]
    point = operating_point(sweep, target_precision=0.99)
    assert point["threshold"] is None
    assert "no threshold" in point["note"]
    assert point["best_observed"]["precision"] == 0.75
