import numpy as np
import pytest
from confseq.betting import diversified_betting_mart
from confseq.betting_strategies import lambda_predmix_eb

from actis.confseq import BettingSupermartingale, GaussianMixtureSupermartingale
from actis.tuner import ACTIS


def _eval_confseq_reference(
    x,
    m,
    alpha,
    pop_size=None,
    horizon=None,
    trunc_scale=0.5,
    m_trunc=True
) -> float:
    lambdas_fns = [
        lambda data, mean: lambda_predmix_eb(data, alpha=alpha, fixed_n=horizon)
    ]
    wealth_process = diversified_betting_mart(
        x,
        m,
        alpha=alpha,
        lambdas_fns_positive=lambdas_fns,
        N=pop_size,
        convex_comb=True,
        theta=1,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )
    return float(wealth_process[-1])


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
            mart_step.update(x[idx:idx + sz])
            idx += sz

        assert np.isclose(w_full, mart_step.wealth(), rtol=1e-12, atol=1e-12)

    def test_empty_update(self):
        mart = GaussianMixtureSupermartingale(m=0.0, v_0=1.0)
        mart.update([])
        assert mart.wealth() == 1.0


class TestBettingSupermartingale:

    @pytest.mark.parametrize("pop_size", [None, 1000])
    @pytest.mark.parametrize("horizon", [None, 300])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    @pytest.mark.parametrize("m", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_confseq_library_parity(self, pop_size, horizon, alpha, m):
        rng = np.random.default_rng(123)
        x = rng.uniform(0.0, 1.0, size=300)

        # Reference from confseq library
        ref_wealth = _eval_confseq_reference(
            x, m=m, alpha=alpha, pop_size=pop_size, horizon=horizon
        )

        # Incremental stateful BettingSupermartingale in multiple uneven chunks
        mart = BettingSupermartingale(
            m=m, alpha=alpha, pop_size=pop_size, horizon=horizon
        )
        chunk_sizes = [30, 70, 100, 100]
        idx = 0
        for sz in chunk_sizes:
            mart.update(x[idx:idx + sz])
            idx += sz

        assert np.isclose(mart.wealth(), ref_wealth, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("pop_size", [None, 500])
    @pytest.mark.parametrize("m", [0.0, 0.3, 0.7, 1.0])
    def test_reverse_betting_parity(self, pop_size, m):
        rng = np.random.default_rng(456)
        x = rng.uniform(0.0, 1.0, size=200)

        # Reverse testing H0: mean >= m by testing (1 - x) against (1 - m)
        ref_rev_wealth = _eval_confseq_reference(
            1.0 - x, m=1.0 - m, alpha=0.05, pop_size=pop_size, horizon=None
        )

        mart_rev = BettingSupermartingale(
            m=m, alpha=0.05, pop_size=pop_size, horizon=None, reverse=True
        )
        mart_rev.update(x[:80])
        mart_rev.update(x[80:])

        assert np.isclose(mart_rev.wealth(), ref_rev_wealth, rtol=1e-10, atol=1e-10)

    def test_single_item_streaming(self):
        rng = np.random.default_rng(789)
        x = rng.uniform(0.1, 0.9, size=50)

        ref_wealth = _eval_confseq_reference(x, m=0.5, alpha=0.05)

        mart = BettingSupermartingale(m=0.5, alpha=0.05)
        for val in x:
            mart.update([val])

        assert np.isclose(mart.wealth(), ref_wealth, rtol=1e-10, atol=1e-10)


class TestACTISTunerStateParity:

    @pytest.mark.parametrize("conf_seq", ["finite", "asymptotic"])
    def test_streaming_batches_vs_oneshot(self, conf_seq):
        rng = np.random.default_rng(42)
        N = 1000
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)

        thresholds = np.linspace(0.1, 0.9, 20)
        thresholds_upper = np.linspace(0.1, 0.99, 100)

        # 1. Single-shot ACTIS
        tuner_oneshot = ACTIS(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            conf_seq=conf_seq,
            v_0=0.01,
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
        )
        opt_oneshot = tuner_oneshot.compute_optimistic_thresholds()

        # 2. Multi-batch streaming ACTIS
        tuner_stream = ACTIS(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            conf_seq=conf_seq,
            v_0=0.01,
        )
        batch_size = 100
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i:i + batch_size],
                labels=labels[i:i + batch_size],
            )
        opt_stream = tuner_stream.compute_optimistic_thresholds()

        # Check identical conservative thresholds
        assert res_oneshot.tau_pos == res_stream.tau_pos
        assert res_oneshot.tau_neg == res_stream.tau_neg
        assert opt_oneshot == opt_stream

    @pytest.mark.parametrize("conf_seq", ["finite", "asymptotic"])
    def test_streaming_with_importance_sampling(self, conf_seq):
        rng = np.random.default_rng(999)
        N = 1200
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)

        thresholds = np.linspace(0.1, 0.9, 15)
        thresholds_upper = np.linspace(0.1, 0.99, 50)

        # Generate weights
        q = np.sqrt(scores + 0.05)
        q = q / q.sum()
        weights = 1.0 / (q * N)

        max_weight_ge = np.array([
            weights[scores >= tau].max() if np.any(scores >= tau) else weights.max()
            for tau in thresholds
        ])
        max_weight_lt = np.array([
            weights[scores < tau].max() if np.any(scores < tau) else 0.0
            for tau in thresholds
        ])
        max_weight_ge_upper = np.array([
            weights[scores >= tau].max() if np.any(scores >= tau) else weights.max()
            for tau in thresholds_upper
        ])

        # 1. Oneshot
        tuner_oneshot = ACTIS(
            gamma_P=0.75,
            gamma_R=0.75,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            conf_seq=conf_seq,
            v_0=0.01,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
            weights=weights,
        )
        opt_oneshot = tuner_oneshot.compute_optimistic_thresholds()

        # 2. Multi-batch
        tuner_stream = ACTIS(
            gamma_P=0.75,
            gamma_R=0.75,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=None,
            conf_seq=conf_seq,
            v_0=0.01,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
        )
        batch_size = 150
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i:i + batch_size],
                labels=labels[i:i + batch_size],
                weights=weights[i:i + batch_size],
            )
        opt_stream = tuner_stream.compute_optimistic_thresholds()

        assert res_oneshot.tau_pos == res_stream.tau_pos
        assert res_oneshot.tau_neg == res_stream.tau_neg
        assert opt_oneshot == opt_stream

    def test_streaming_without_replacement(self):
        rng = np.random.default_rng(777)
        N = 1000
        pop_size = 5000
        scores = rng.uniform(0.0, 1.0, size=N)
        labels = rng.binomial(1, scores).astype(bool)

        thresholds = np.linspace(0.1, 0.9, 20)
        thresholds_upper = np.linspace(0.1, 0.99, 100)

        tuner_oneshot = ACTIS(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size,
            conf_seq="finite",
        )
        res_oneshot = tuner_oneshot.add_samples(
            indices=np.arange(N),
            scores=scores,
            labels=labels,
        )
        opt_oneshot = tuner_oneshot.compute_optimistic_thresholds()

        tuner_stream = ACTIS(
            gamma_P=0.70,
            gamma_R=0.70,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size,
            conf_seq="finite",
        )
        batch_size = 100
        for i in range(0, N, batch_size):
            res_stream = tuner_stream.add_samples(
                indices=np.arange(i, i + batch_size),
                scores=scores[i:i + batch_size],
                labels=labels[i:i + batch_size],
            )
        opt_stream = tuner_stream.compute_optimistic_thresholds()

        assert res_oneshot.tau_pos == res_stream.tau_pos
        assert res_oneshot.tau_neg == res_stream.tau_neg
        assert opt_oneshot == opt_stream

