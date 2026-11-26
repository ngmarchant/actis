# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for threshold grid generation, validation, and upper anchor fallback."""

import numpy as np
import pytest

from actis import (
    ACTIS,
    ACTISFinite,
    PopulationSampler,
    compute_pr_proposal,
    compute_prior_and_target_var,
    quantile_power_law_grid,
    validate_thresholds,
)
from actis.finite import FiniteSampleCascadeConfSeqs


def test_validate_thresholds_bounds_and_order():
    arr = validate_thresholds([0.0, 0.5, 1.0])
    np.testing.assert_allclose(arr, [0.0, 0.5, 1.0])

    for invalid in ([0.0, 0.5, np.inf], [-0.1, 0.5], [0.5, 1.1], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
            validate_thresholds(invalid)

    with pytest.raises(ValueError, match="strictly ascending"):
        validate_thresholds([0.0, 0.5, 0.5, 1.0])


def test_quantile_power_law_grid_bounds():
    scores = np.array([0.0, 0.2, 0.5, 1.0, 1.0])
    grid = quantile_power_law_grid(scores, num_thresholds=10)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)
    assert np.all(np.isfinite(grid))

    empty_grid = quantile_power_law_grid([], num_thresholds=10)
    np.testing.assert_allclose(empty_grid, [0.0, 1.0])

    single_score_grid = quantile_power_law_grid([0.42], num_thresholds=10)
    np.testing.assert_allclose(single_score_grid, [0.42])

    single_threshold_grid = quantile_power_law_grid(scores, num_thresholds=1)
    np.testing.assert_allclose(single_threshold_grid, [0.0])

    with pytest.raises(ValueError, match="num_thresholds.*positive integer"):
        quantile_power_law_grid(scores, num_thresholds=0)


def test_finite_confseq_requires_max_weight_ge_upper_for_distinct_upper_grid():
    with pytest.raises(ValueError, match="max_weight_ge_upper"):
        FiniteSampleCascadeConfSeqs(
            gamma_P=0.9,
            gamma_R=0.9,
            delta=0.05,
            thresholds=np.array([0.2, 0.5]),
            thresholds_upper=np.array([0.3, 0.6, 0.9]),
            max_weight_ge=np.array([2.0, 1.5]),
        )


@pytest.mark.parametrize(
    "conf_seq,sampling_method",
    [
        ("asymptotic", "is"),
        ("asymptotic", "wor"),
        ("finite", "is"),
        ("finite", "wor"),
    ],
)
def test_upper_anchor_prevents_fp_on_saturated_max_scores(conf_seq, sampling_method):
    """When many records saturate at score=1.0 and contain false positives such that
    precision at tau_upper=1.0 is below gamma_P, ACTIS must fall back to tau_upper=inf
    and achieve precision=1.0.
    """
    rng = np.random.default_rng(12345)
    # 500 records: 300 at score=0.0 (all False), 200 at score=1.0 (150 True, 50 False)
    # Precision at tau_upper=1.0 (with tau_lower=0.0) is 150 / (150 + 50) = 0.75 < 0.95.
    scores = np.concatenate([np.zeros(300), np.ones(200)])
    labels = np.concatenate(
        [np.zeros(300, dtype=bool), np.ones(150, dtype=bool), np.zeros(50, dtype=bool)]
    )
    pop_size = len(scores)
    gamma_R = 0.90
    gamma_P = 0.95
    delta = 0.05

    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 10)))
    thresholds_upper = quantile_power_law_grid(scores, num_thresholds=20)

    q = None
    weights = None
    max_weight_ge = None
    max_weight_lt = None
    max_weight_ge_upper = None
    if sampling_method == "is":
        q = compute_pr_proposal(
            scores=scores,
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            alpha=0.4,
        )
        weights = 1.0 / (q * pop_size)
        if conf_seq == "finite":
            max_weight_ge = np.array(
                [
                    weights[scores >= tau].max()
                    if np.any(scores >= tau)
                    else float(weights.max())
                    for tau in thresholds
                ]
            )
            max_weight_lt = np.array(
                [
                    weights[scores < tau].max() if np.any(scores < tau) else 0.0
                    for tau in thresholds
                ]
            )
            max_weight_ge_upper = np.array(
                [
                    weights[scores >= tau].max() if np.any(scores >= tau) else 0.0
                    for tau in thresholds_upper
                ]
            )
        pop_size_param = None
    else:
        pop_size_param = pop_size

    sampler = PopulationSampler(
        pop_size=pop_size,
        replace=(sampling_method != "wor"),
        p=q,
        rng=rng,
    )

    tuner: ACTIS | ACTISFinite
    if conf_seq == "asymptotic":
        v_0 = compute_prior_and_target_var(
            scores=scores,
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=weights,
        )
        tuner = ACTIS(
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size_param,
            v_0=v_0,
        )
    else:
        tuner = ACTISFinite(
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size_param,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
        )

    batch_size = 50
    max_sample_size = 100
    idx = sampler.sample(batch_size)
    res = tuner.add_samples(
        indices=idx,
        scores=scores[idx],
        labels=labels[idx],
        weights=weights[idx] if weights is not None else None,
        population_scores=scores,
    )
    while sampler.sample_count < max_sample_size:
        next_batch = min(batch_size, max_sample_size - sampler.sample_count)
        should_continue, _ = tuner.should_continue_sampling(
            population_scores=scores,
            batch_size=next_batch,
            max_sample_size=max_sample_size,
        )
        if not should_continue:
            break
        next_idx = sampler.sample(next_batch)
        res = tuner.add_samples(
            indices=next_idx,
            scores=scores[next_idx],
            labels=labels[next_idx],
            weights=weights[next_idx] if weights is not None else None,
            population_scores=scores,
        )

    pred_pos = (scores >= res.tau_upper) | (
        (scores >= res.tau_lower) & (scores < res.tau_upper) & labels
    )
    tp = int(np.sum(pred_pos & labels))
    precision = tp / int(np.sum(pred_pos))
    recall = tp / int(np.sum(labels))

    assert np.isposinf(res.tau_upper)
    assert precision == pytest.approx(1.0)
    assert recall >= 0.90
