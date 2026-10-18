import numpy as np
import pytest

from experiments.dataset_stats import (
    DOCUMENT_BENCHMARK_SCENARIOS,
    SUPG_SCENARIOS,
    _calibration_bias,
    _expected_calibration_error,
    _pr_auc,
    _saturation_rate,
    _stats_from_population,
)


def test_metric_helpers():
    # Perfectly separated and calibrated
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([False, False, True, True])

    assert _pr_auc(scores, labels) == pytest.approx(1.0)
    assert _calibration_bias(scores, labels) == pytest.approx(0.0)
    assert _expected_calibration_error(scores, labels) == pytest.approx(0.0)
    assert _saturation_rate(scores) == pytest.approx(1.0)

    # Overconfident proxy: scores high, labels false
    scores_over = np.array([0.8, 0.8, 0.8, 0.8])
    labels_over = np.array([False, False, False, False])
    # PR-AUC for all false labels returns nan
    assert np.isnan(_pr_auc(scores_over, labels_over))
    assert _calibration_bias(scores_over, labels_over) == pytest.approx(0.8)
    assert _expected_calibration_error(scores_over, labels_over) == pytest.approx(0.8)
    assert _saturation_rate(scores_over) == pytest.approx(0.0)

    # Underconfident proxy: scores low, labels true
    scores_under = np.array([0.2, 0.2, 0.2, 0.2])
    labels_under = np.array([True, True, True, True])
    assert _calibration_bias(scores_under, labels_under) == pytest.approx(-0.8)
    assert _expected_calibration_error(scores_under, labels_under) == pytest.approx(0.8)


def test_stats_from_population():
    scores = np.array([0.0, 0.5, 1.0])
    labels = np.array([False, True, True])
    res = _stats_from_population(scores, labels, n_total=5)

    assert res["N_total"] == 5
    assert res["N_missing"] == 2
    assert res["pos_rate"] == pytest.approx(2.0 / 3.0)
    assert res["sat_rate"] == pytest.approx(2.0 / 3.0)
    assert res["cal_bias"] == pytest.approx(0.5 - (2.0 / 3.0))


def test_scenario_registries_non_empty():
    assert "onto" in SUPG_SCENARIOS
    assert "imagenet" in SUPG_SCENARIOS

    assert "pubmed" in DOCUMENT_BENCHMARK_SCENARIOS
    assert "wiki" in DOCUMENT_BENCHMARK_SCENARIOS
