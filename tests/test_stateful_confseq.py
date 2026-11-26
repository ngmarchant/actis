# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

import math

import numpy as np
import pytest

from actis import ACTIS, ACTISFinite
from actis.confseq import BettingSupermartingale, GaussianMixtureSupermartingale


class TestGaussianMixtureSupermartingale:
    @pytest.mark.parametrize("m", [0.0, 0.2, -0.1])
    @pytest.mark.parametrize("v_0", [0.01, 0.1, 1.0, 5.0])
    def test_batch_vs_full_parity(self, m, v_0):
        rng = np.random.default_rng(42)
        x = rng.normal(loc=m + 0.05, scale=0.5, size=500)

        # Full single-shot evaluation
        mart_full = GaussianMixtureSupermartingale(m=m, v_0=v_0)
        mart_full.update(x)
        w_full = mart_full.wealth()

        # Multi-batch incremental evaluation
        mart_step = GaussianMixtureSupermartingale(m=m, v_0=v_0)
        chunk_sizes = [50, 100, 150, 200]
        idx = 0
        for sz in chunk_sizes:
            mart_step.update(x[idx : idx + sz])
            idx += sz

        assert np.isclose(w_full, mart_step.wealth(), rtol=1e-12, atol=1e-12)

    def test_empty_update(self):
        mart = GaussianMixtureSupermartingale(m=0.0, v_0=1.0)
        mart.update([])
        assert mart.wealth() == 1.0


class TestBettingSupermartingale:
    @pytest.mark.parametrize("pop_size", [None, 1000])
    @pytest.mark.parametrize("horizon", [None, 300])
    @pytest.mark.parametrize("alpha", [0.01, 0.05])
    @pytest.mark.parametrize("m", [0.0, 0.2, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_batch_vs_full_and_streaming_parity(
        self, pop_size, horizon, alpha, m, reverse
    ):
        rng = np.random.default_rng(123)
        x = rng.uniform(0.0, 1.0, size=300)

        mart_full = BettingSupermartingale(
            m=m, alpha=alpha, pop_size=pop_size, horizon=horizon, reverse=reverse
        )
        mart_full.update(x)

        mart_chunks = BettingSupermartingale(
            m=m, alpha=alpha, pop_size=pop_size, horizon=horizon, reverse=reverse
        )
        idx = 0
        for sz in (30, 70, 100, 100):
            mart_chunks.update(x[idx : idx + sz])
            idx += sz

        mart_stream = BettingSupermartingale(
            m=m, alpha=alpha, pop_size=pop_size, horizon=horizon, reverse=reverse
        )
        for val in x:
            mart_stream.update([val])

        assert np.isclose(
            mart_chunks.wealth(), mart_full.wealth(), rtol=1e-12, atol=1e-12
        )
        assert np.isclose(
            mart_stream.wealth(), mart_full.wealth(), rtol=1e-12, atol=1e-12
        )

    def test_exact_first_two_steps(self):
        # Unclipped fixed-horizon case
        x = [0.8, 0.6]
        mart = BettingSupermartingale(m=0.5, alpha=0.1, horizon=1000)
        mart.update(x)

        lam1 = math.sqrt(2.0 * math.log(10.0) / (1000.0 * 0.25))
        mu_hat_1 = (0.5 + 0.8) / 2.0
        sigma2_1 = (0.25 + (0.8 - mu_hat_1) ** 2) / 2.0
        lam2 = math.sqrt(2.0 * math.log(10.0) / (1000.0 * sigma2_1))
        expected = (1.0 + lam1 * (0.8 - 0.5)) * (1.0 + lam2 * (0.6 - 0.5))

        assert np.isclose(mart.wealth(), expected, rtol=1e-12, atol=1e-12)

        # Clipped anytime case (trunc_scale=0.5, m=0.5 -> lambda clipped to 1.0)
        mart_clipped = BettingSupermartingale(m=0.5, alpha=0.05, horizon=None)
        mart_clipped.update([0.8, 0.2])
        assert np.isclose(mart_clipped.wealth(), 1.3 * 0.7, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("pop_size", [None, 500])
    @pytest.mark.parametrize("m", [0.0, 0.3, 0.7, 1.0])
    def test_reverse_symmetry(self, pop_size, m):
        rng = np.random.default_rng(456)
        x = rng.uniform(0.0, 1.0, size=200)

        mart_fwd = BettingSupermartingale(
            m=1.0 - m, alpha=0.05, pop_size=pop_size, horizon=None, reverse=False
        )
        mart_fwd.update(1.0 - x[:80])
        mart_fwd.update(1.0 - x[80:])

        mart_rev = BettingSupermartingale(
            m=m, alpha=0.05, pop_size=pop_size, horizon=None, reverse=True
        )
        mart_rev.update(x[:80])
        mart_rev.update(x[80:])

        assert np.isclose(mart_rev.wealth(), mart_fwd.wealth(), rtol=1e-12, atol=1e-12)

    def test_null_and_alternative_behavior(self):
        rng = np.random.default_rng(789)
        alpha = 0.05

        mart_null = BettingSupermartingale(m=0.5, alpha=alpha)
        x_null = rng.binomial(1, 0.5, size=200).astype(float)
        mart_null.update(x_null)
        assert mart_null.wealth() < 1.0 / alpha

        mart_alt = BettingSupermartingale(m=0.3, alpha=alpha)
        x_alt = rng.binomial(1, 0.7, size=200).astype(float)
        mart_alt.update(x_alt)
        assert mart_alt.wealth() > 1.0 / alpha

    def test_empty_update_and_wor_impossible_null(self):
        mart = BettingSupermartingale(m=0.5, alpha=0.05)
        mart.update([])
        assert mart.wealth() == 1.0

        # Pop size 4, null mean 0.25 => at most 1 positive item in population.
        # Observing two 1.0s makes conditional null mean mu_t == 0.0 on step 2, and a
        # third 1.0 makes mu_t < 0.0 (impossible under H0).
        mart_wor = BettingSupermartingale(m=0.25, alpha=0.05, pop_size=4)
        mart_wor.update([1.0, 1.0, 1.0])
        assert math.isinf(mart_wor.wealth())


class TestACTISTunerStateParity:
    @pytest.mark.parametrize("tuner_cls", [ACTISFinite, ACTIS])
    def test_streaming_batches_vs_oneshot(self, tuner_cls):
        rng = np.random.default_rng(42)
        N = 1000
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)
        pop_scores = np.concatenate([scores, scores])

        thresholds = np.linspace(0.1, 0.9, 20)
        thresholds_upper = np.linspace(0.1, 0.99, 100)
        extra_kwargs = {"v_0": 0.01} if tuner_cls is ACTIS else {}

        # 1. Single-shot tuner
        tuner_oneshot = tuner_cls(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            **extra_kwargs,
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
        )
        _, diag_oneshot = tuner_oneshot.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=len(pop_scores),
        )

        # 2. Multi-batch streaming tuner
        tuner_stream = tuner_cls(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            **extra_kwargs,
        )
        batch_size = 100
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i : i + batch_size],
                labels=labels[i : i + batch_size],
            )
        _, diag_stream = tuner_stream.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=len(pop_scores),
        )

        # Check identical conservative and optimistic thresholds
        assert res_oneshot.tau_upper == res_stream.tau_upper
        assert res_oneshot.tau_lower == res_stream.tau_lower
        assert diag_oneshot.tau_upper_opt == diag_stream.tau_upper_opt
        assert diag_oneshot.tau_lower_opt == diag_stream.tau_lower_opt

    @pytest.mark.parametrize("tuner_cls", [ACTISFinite, ACTIS])
    def test_streaming_with_importance_sampling(self, tuner_cls):
        rng = np.random.default_rng(999)
        N = 1200
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)
        pop_scores = np.concatenate([scores, scores])

        thresholds = np.linspace(0.1, 0.9, 15)
        thresholds_upper = np.linspace(0.1, 0.99, 50)

        # Generate weights
        q = np.sqrt(scores + 0.05)
        q = q / q.sum()
        weights = 1.0 / (q * N)

        if tuner_cls is ACTIS:
            extra_kwargs = {"v_0": 0.01}
        else:
            extra_kwargs = {
                "max_weight_ge": np.array(
                    [
                        weights[scores >= tau].max()
                        if np.any(scores >= tau)
                        else weights.max()
                        for tau in thresholds
                    ]
                ),
                "max_weight_lt": np.array(
                    [
                        weights[scores < tau].max() if np.any(scores < tau) else 0.0
                        for tau in thresholds
                    ]
                ),
                "max_weight_ge_upper": np.array(
                    [
                        weights[scores >= tau].max()
                        if np.any(scores >= tau)
                        else weights.max()
                        for tau in thresholds_upper
                    ]
                ),
            }

        # 1. Oneshot
        tuner_oneshot = tuner_cls(
            gamma_P=0.75,
            gamma_R=0.75,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            **extra_kwargs,
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
            weights=weights,
        )
        _, diag_oneshot = tuner_oneshot.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=len(pop_scores),
        )

        # 2. Multi-batch
        tuner_stream = tuner_cls(
            gamma_P=0.75,
            gamma_R=0.75,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            **extra_kwargs,
        )
        batch_size = 150
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i : i + batch_size],
                labels=labels[i : i + batch_size],
                weights=weights[i : i + batch_size],
            )
        _, diag_stream = tuner_stream.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=len(pop_scores),
        )

        assert res_oneshot.tau_upper == res_stream.tau_upper
        assert res_oneshot.tau_lower == res_stream.tau_lower
        assert diag_oneshot.tau_upper_opt == diag_stream.tau_upper_opt
        assert diag_oneshot.tau_lower_opt == diag_stream.tau_lower_opt

    def test_streaming_without_replacement(self):
        rng = np.random.default_rng(777)
        N = 1000
        pop_size = 5000
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)
        pop_scores = np.tile(scores, 5)

        thresholds = np.linspace(0.1, 0.9, 20)
        thresholds_upper = np.linspace(0.1, 0.99, 100)

        tuner_oneshot = ACTISFinite(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size,
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
        )
        _, diag_oneshot = tuner_oneshot.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=pop_size,
        )

        tuner_stream = ACTISFinite(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size,
        )
        batch_size = 100
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i : i + batch_size],
                labels=labels[i : i + batch_size],
            )
        _, diag_stream = tuner_stream.should_continue_sampling(
            population_scores=pop_scores,
            batch_size=50,
            max_sample_size=pop_size,
        )

        assert res_oneshot.tau_upper == res_stream.tau_upper
        assert res_oneshot.tau_lower == res_stream.tau_lower
        assert diag_oneshot.tau_upper_opt == diag_stream.tau_upper_opt
        assert diag_oneshot.tau_lower_opt == diag_stream.tau_lower_opt
