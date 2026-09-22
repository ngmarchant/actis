"""Unit tests for multi-metric costing, Population dataclass,
and cascade evaluation logic.
"""

import numpy as np
import pytest
from datasets import Dataset

from experiments.data_prep.labeling import save_dataset
from experiments.runners import (
    BargainPRRunner,
    TrialResult,
    compute_precision_recall,
    evaluate_cascade_trial,
    summarize_runner_trials,
)
from experiments.scenarios import (
    Benign,
    Population,
    TabularDataset,
    compute_ideal_oracle_call_rate,
)


def test_population_dataclass_basics():
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([False, True, True])
    oracle_costs = {"monetary": np.array([0.01, 0.02, 0.03])}
    proxy_costs = {"monetary": np.array([0.001, 0.001, 0.001])}

    pop = Population(
        scores=scores,
        labels=labels,
        oracle_costs=oracle_costs,
        proxy_costs=proxy_costs,
    )

    assert len(pop) == 3
    # Test unpacking
    s, lbl = pop
    np.testing.assert_array_equal(s, scores)
    np.testing.assert_array_equal(lbl, labels)

    # Test slicing
    sub_pop = pop.slice(np.array([0, 2]))
    assert len(sub_pop) == 2
    np.testing.assert_array_equal(sub_pop.scores, [0.1, 0.9])
    np.testing.assert_array_equal(sub_pop.labels, [False, True])
    np.testing.assert_array_equal(sub_pop.oracle_costs["monetary"], [0.01, 0.03])
    np.testing.assert_array_equal(sub_pop.proxy_costs["monetary"], [0.001, 0.001])


def test_population_compute_trial_costs():
    N = 4
    scores = np.linspace(0.1, 0.9, N)
    labels = np.array([False, False, True, True])
    oracle_costs = {
        "monetary": np.array([0.10, 0.20, 0.30, 0.40]),
        "input_tokens": np.array([100, 200, 300, 400]),
    }
    proxy_costs = {
        "monetary": np.array([0.01, 0.01, 0.01, 0.01]),
        "input_tokens": np.array([10, 10, 10, 10]),
    }
    pop = Population(
        scores=scores,
        labels=labels,
        oracle_costs=oracle_costs,
        proxy_costs=proxy_costs,
    )

    # Oracle queried on items 1 and 3 (2 items queried out of 4)
    oracle_queried_mask = np.array([False, True, False, True])
    costs = pop.compute_trial_costs(oracle_queried_mask)

    # Check oracle costs
    assert costs["oracle"]["num_calls"] == 2.0
    assert costs["oracle"]["call_rate"] == 0.5
    assert costs["oracle"]["monetary"] == pytest.approx(0.20 + 0.40)
    assert costs["oracle"]["input_tokens"] == pytest.approx(200 + 400)

    # Check proxy costs (always incurred across all N items)
    assert costs["proxy"]["num_calls"] == 4.0
    assert costs["proxy"]["call_rate"] == 1.0
    assert costs["proxy"]["monetary"] == pytest.approx(0.04)
    assert costs["proxy"]["input_tokens"] == pytest.approx(40)


def test_population_empty_costs():
    pop = Population(
        scores=np.array([0.2, 0.8]),
        labels=np.array([False, True]),
    )
    costs = pop.compute_trial_costs(np.array([True, False]))
    assert costs["oracle"] == {"num_calls": 1.0, "call_rate": 0.5}
    assert costs["proxy"] == {"num_calls": 2.0, "call_rate": 1.0}


def test_compute_precision_recall():
    preds = np.array([True, True, False, False])
    labels = np.array([True, False, True, False])
    prec, rec = compute_precision_recall(preds, labels)
    assert prec == 0.5
    assert rec == 0.5


def test_evaluate_cascade_trial_calibration_override():
    """Verify that calibration samples queried from the oracle use their ground-truth
    label rather than the cascade prediction.
    """
    scores = np.array([0.1, 0.5, 0.9, 0.9])
    # Item 0: score=0.1 (reject region, would predict False), ground truth is True
    # Item 1: score=0.5 (uncertain region, routes to oracle), ground truth is False
    # Item 2: score=0.9 (accept region, would predict True), ground truth is False
    # Item 3: score=0.9 (accept region, uncalibrated), ground truth is True
    labels = np.array([True, False, False, True])

    tau_neg = 0.3
    tau_pos = 0.7
    calib_indices = np.array([0, 2])  # calibration set includes items 0 and 2

    pop = Population(scores=scores, labels=labels)
    res = evaluate_cascade_trial(
        population=pop,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        calib_indices=calib_indices,
    )

    # Expected predictions:
    # Item 0 (calibrated, reject region): oracle label -> True
    # Item 1 (uncalibrated, uncertain region [0.3, 0.7]): oracle sent -> True routing,
    # label is False
    # Item 2 (calibrated, accept region): oracle label -> False
    # Item 3 (uncalibrated, accept region): cascade prediction -> True
    # Preds should be [True, False, False, True]
    # Positives predicted: items 0 and 3
    # Ground truth positives: items 0 and 3
    # Therefore, recall = 1.0, precision = 1.0!
    assert res.recall == 1.0
    assert res.precision == 1.0

    # Oracle queried items: 0 (calib), 1 (uncertain), 2 (calib) -> total 3 of 4
    assert res.cost["oracle"]["num_calls"] == 3.0
    assert res.cost["oracle"]["call_rate"] == 0.75
    assert res.cost["proxy"]["num_calls"] == 4.0
    assert res.cost["proxy"]["call_rate"] == 1.0


def test_bargain_pr_queried_indices_override():
    """Verify BargainPRRunner overrides predictions for all queried indices
    with oracle labels.
    """
    rng = np.random.default_rng(42)
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    labels = np.array([False, False, True, True, True])
    pop = Population(scores=scores, labels=labels)

    runner = BargainPRRunner(
        name="test_bargain",
        window_size=2,
        num_thresholds=5,
        sample_step=2,
    )

    result = runner.run_trial(
        population=pop,
        gamma_R=0.8,
        gamma_P=0.8,
        delta=0.05,
        rng=rng,
    )

    assert isinstance(result, TrialResult)
    assert "oracle" in result.cost
    assert "proxy" in result.cost
    assert "num_calls" in result.cost["oracle"]
    assert "call_rate" in result.cost["oracle"]
    assert result.cost["proxy"]["num_calls"] == len(scores)
    assert result.cost["proxy"]["call_rate"] == 1.0


def test_summarize_runner_trials_hierarchical_cost_and_include_raw():
    scenario = Benign()
    runner = BargainPRRunner(name="test_runner")

    results = [
        TrialResult(
            precision=0.95,
            recall=0.92,
            tau_pos=0.8,
            tau_neg=0.2,
            runtime=0.05,
            cost={
                "oracle": {
                    "num_calls": 20.0,
                    "call_rate": 0.2,
                    "monetary": 0.50,
                },
                "proxy": {
                    "num_calls": 100.0,
                    "call_rate": 1.0,
                    "monetary": 0.05,
                },
            },
        ),
        TrialResult(
            precision=0.90,
            recall=0.88,
            tau_pos=0.75,
            tau_neg=0.25,
            runtime=0.06,
            cost={
                "oracle": {
                    "num_calls": 30.0,
                    "call_rate": 0.3,
                    "monetary": 0.75,
                },
                "proxy": {
                    "num_calls": 100.0,
                    "call_rate": 1.0,
                    "monetary": 0.05,
                },
            },
        ),
    ]

    # Test with include_raw=True
    summary_with_raw = summarize_runner_trials(
        runner=runner,
        results=results,
        scenario=scenario,
        gamma_R=0.9,
        gamma_P=0.9,
        delta=0.05,
        pop_size=100,
        ideal_oracle_rate=0.15,
        include_raw=True,
    )

    assert "raw" in summary_with_raw["precision"]
    assert "raw" in summary_with_raw["recall"]
    assert "raw" in summary_with_raw["runtime"]
    assert "raw" in summary_with_raw["cost"]["oracle"]["call_rate"]
    assert summary_with_raw["cost"]["oracle"]["call_rate"]["mean"] == pytest.approx(
        0.25
    )
    assert summary_with_raw["cost"]["oracle"]["monetary"]["mean"] == pytest.approx(
        0.625
    )
    assert summary_with_raw["cost"]["proxy"]["call_rate"]["mean"] == pytest.approx(1.0)
    # Check that legacy keys are absent
    assert "mean_total_oracle_calls" not in summary_with_raw
    assert "mean_total_oracle_rate" not in summary_with_raw

    # Test with include_raw=False
    summary_no_raw = summarize_runner_trials(
        runner=runner,
        results=results,
        scenario=scenario,
        gamma_R=0.9,
        gamma_P=0.9,
        delta=0.05,
        pop_size=100,
        ideal_oracle_rate=0.15,
        include_raw=False,
    )

    assert "raw" not in summary_no_raw["precision"]
    assert "raw" not in summary_no_raw["recall"]
    assert "raw" not in summary_no_raw["runtime"]
    assert "raw" not in summary_no_raw["cost"]["oracle"]["call_rate"]
    assert summary_no_raw["cost"]["oracle"]["call_rate"]["mean"] == pytest.approx(0.25)


def test_tabular_dataset_cost_parsing(tmp_path):
    data_file = tmp_path / "test_data.parquet"
    ds = Dataset.from_dict(
        {
            "id": [0, 1, 2],
            "content": ["a", "b", "c"],
            "label": [True, False, True],
            "proxy_score": [0.9, 0.1, 0.8],
            "oracle_cost": [
                {"monetary": 0.05, "input_tokens": 100, "output_tokens": 10},
                {"monetary": 0.05, "input_tokens": 110, "output_tokens": 12},
                {"monetary": 0.05, "input_tokens": 90, "output_tokens": 8},
            ],
            "proxy_cost": [
                {"monetary": 0.001, "input_tokens": 50, "output_tokens": 1},
                {"monetary": 0.001, "input_tokens": 55, "output_tokens": 1},
                {"monetary": 0.001, "input_tokens": 45, "output_tokens": 1},
            ],
        }
    )
    save_dataset(ds, data_file, format="parquet")

    scenario = TabularDataset(name="test_tabular_costs", data_path=data_file)
    pop = scenario.generate_population()

    assert isinstance(pop, Population)
    assert len(pop) == 3
    assert "monetary" in pop.oracle_costs
    assert "input_tokens" in pop.oracle_costs
    assert "output_tokens" in pop.oracle_costs
    assert "monetary" in pop.proxy_costs

    np.testing.assert_array_equal(pop.oracle_costs["input_tokens"], [100, 110, 90])
    np.testing.assert_array_equal(pop.proxy_costs["output_tokens"], [1, 1, 1])


def test_compute_ideal_oracle_call_rate():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    # Perfect separation: ideal rate should be 0.0
    rate = compute_ideal_oracle_call_rate(scores, labels, gamma_R=1.0, gamma_P=1.0)
    assert rate == pytest.approx(0.0)
