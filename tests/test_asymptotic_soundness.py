import math

import numpy as np
import pytest

from actis import (
    ACTIS,
    AsymptoticValidityDiagnostics,
    CascadeThresholds,
    PriorAndTargetVar,
    compute_prior_and_target_var,
)
from actis.confseq import GaussianMixtureSupermartingale
from actis.tuner import AsymptoticCascadeConfSeqs
from experiments.runners import ACTISRunner
from experiments.scenarios import Population


class TestGaussianMixtureAccumulator:
    def test_v_t_property(self):
        mart = GaussianMixtureSupermartingale(m=0.0, v_0=1.0)
        assert mart.v_t == 0.0

        mart.update([2.0])
        assert mart.v_t == 0.0

        mart.update([4.0])
        # x = [2.0, 4.0], mean = 3.0, sum of sq diffs = (2-3)^2 + (4-3)^2 = 2.0
        assert np.isclose(mart.v_t, 2.0)

        # Update with more points
        mart.update([6.0, 8.0])
        # x = [2, 4, 6, 8], mean = 5, sum of sq diffs = 9 + 1 + 1 + 9 = 20.0
        assert np.isclose(mart.v_t, 20.0)


class TestComputePriorVar:
    def test_empty_scores_raises_value_error(self):
        with pytest.raises(ValueError, match="scores.*must be a non-empty 1D array"):
            compute_prior_and_target_var(
                scores=[], gamma_R=0.8, gamma_P=0.8, delta=0.05
            )

        with pytest.raises(ValueError, match="scores.*must be a non-empty 1D array"):
            compute_prior_and_target_var(
                scores=np.zeros((0, 2)), gamma_R=0.8, gamma_P=0.8, delta=0.05
            )

    def test_scalar_fallback(self):
        rng = np.random.default_rng(123)
        scores = rng.beta(2, 5, size=500)

        v_0_scalar = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=None,
        )
        assert v_0_scalar.prior_R(0) > 0.0

    def test_per_threshold_dict(self):
        rng = np.random.default_rng(123)
        scores = rng.beta(2, 5, size=500)
        thresholds = np.linspace(0.0, 0.9, 9)
        thresholds_upper = np.linspace(0.2, 1.0, 15)

        # 1. When thresholds_upper is provided
        v_0 = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
        )

        # All values should be strictly positive and finite
        assert all([v_0.prior_R(k) > 0.0 for k in range(len(thresholds))])
        assert all([math.isfinite(v_0.prior_R(k)) for k in range(len(thresholds))])
        assert all(
            [
                v_0.prior_P(k_upper, k_lower) > 0.0
                for k_upper, k_lower in zip(
                    range(len(thresholds_upper)), range(len(thresholds))
                )
            ]
        )
        assert all(
            [
                math.isfinite(v_0.prior_P(k_upper, k_lower))
                for k_upper, k_lower in zip(
                    range(len(thresholds_upper)), range(len(thresholds))
                )
            ]
        )

        # 2. When thresholds_upper is None, defaults to shape (num_lower, num_lower)
        v_0_default = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=thresholds,
        )

        assert isinstance(v_0_default, PriorAndTargetVar)
        assert v_0_default._prior_P.shape == (len(thresholds), len(thresholds))

    def test_soundness_floor_prevents_false_rejection(self):
        delta = 0.05
        gamma_R = 0.8
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        thresholds = np.array([0.5, 0.7])

        v_0 = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=0.8,
            delta=delta,
            thresholds=thresholds,
        )

        delta_R = delta / 2.0

        # The zero-variance safety floor prevents premature false rejection
        # on small spurious runs (t <= 2 positives) before empirical variance
        # accumulates.
        for k in range(len(thresholds)):
            mart = GaussianMixtureSupermartingale(m=0.0, v_0=v_0.prior_R(k))
            # Maximum positive increment:
            step_val = 1.0 - gamma_R
            for step in range(1, 3):
                mart.update([step_val])
                # At step <= 2 under zero variance, wealth MUST not exceed 1 / delta_R
                assert mart.wealth() <= (1.0 / delta_R) + 1e-9, (
                    f"Spurious rejection at step {step} for threshold {thresholds[k]}"
                )

    def test_symmetrical_variance_estimation(self):
        rng = np.random.default_rng(42)
        scores = rng.beta(2, 5, size=200)
        thresholds = np.array([0.2, 0.4, 0.6, 0.8])
        v_0 = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.85,
            gamma_P=0.75,
            delta=0.05,
            thresholds=thresholds,
            n_0=1.0,
        )

        # Variances should be strictly positive, non-infinite
        assert all([v_0.prior_R(k) > 0.0 for k in range(len(thresholds))])
        assert all([math.isfinite(v_0.prior_R(k)) for k in range(len(thresholds))])
        assert all(
            [
                v_0.prior_P(k_upper, k_lower) > 0.0
                for k_upper, k_lower in zip(
                    range(len(thresholds)), range(len(thresholds))
                )
            ]
        )
        assert all(
            [
                math.isfinite(v_0.prior_P(k_upper, k_lower))
                for k_upper, k_lower in zip(
                    range(len(thresholds)), range(len(thresholds))
                )
            ]
        )


class TestAsymptoticCascadeConfSeqs:
    def test_v_0_input_formats(self):
        thresholds = np.array([0.2, 0.5, 0.8])
        K = len(thresholds)

        # 1. Scalar v_0
        cs_scalar = AsymptoticCascadeConfSeqs(
            thresholds=thresholds,
            thresholds_upper=thresholds,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            v_0=1.5,
        )
        mart_r0 = cs_scalar._create_mart_R(0)
        mart_p0 = cs_scalar._create_mart_P(0, 0)

        assert isinstance(mart_r0, GaussianMixtureSupermartingale)
        assert mart_r0.v_0 == 1.5
        assert isinstance(mart_p0, GaussianMixtureSupermartingale)
        assert mart_p0.v_0 == 1.5

        # 2. Tuple of arrays (v_0_recall, v_0_precision)
        v_0_r = np.array([0.5, 1.0, 1.5])
        v_0_p = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        cs_tuple = AsymptoticCascadeConfSeqs(
            thresholds=thresholds,
            thresholds_upper=thresholds,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            v_0=PriorAndTargetVar(v_0_r, v_0_p),
        )
        for k in range(K):
            mart_k = cs_tuple._create_mart_R(k)
            assert isinstance(mart_k, GaussianMixtureSupermartingale)
            assert mart_k.v_0 == v_0_r[k]
            mart_p_k = cs_tuple._create_mart_P(k, k)
            assert isinstance(mart_p_k, GaussianMixtureSupermartingale)
            assert mart_p_k.v_0 == v_0_p[k, k]


class TestACTISAsymptoticDiagnostics:
    def test_diagnostics_structure(self):
        rng = np.random.default_rng(42)
        N = 300
        scores = rng.beta(2, 5, size=N)
        labels = (scores > 0.4).astype(int)

        thresholds = np.quantile(scores, [0.2, 0.5, 0.8])
        v_0 = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=thresholds,
        )

        tuner = ACTIS(
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=thresholds,
            conf_seq="asymptotic",
            v_0=v_0,
        )

        # Initial small batch (e.g. 10 samples)
        calib_res = tuner.add_samples(
            indices=np.arange(10),
            scores=scores[:10],
            labels=labels[:10],
        )

        assert isinstance(calib_res, CascadeThresholds)
        # Small sample size has not certified candidate thresholds;
        # falls back to anchors
        assert calib_res.tau_neg == thresholds[0]
        assert calib_res.tau_pos == thresholds[-1]
        assert calib_res.asymptotic_diag_R is None
        assert calib_res.asymptotic_diag_P is None
        assert calib_res.num_samples == 10

        # Candidate non-anchor threshold k=1 is not valid under small sample size
        diag_k1 = tuner.conf_seqs.asymptotic_diag_R(1)
        assert diag_k1 is not None and not diag_k1.is_valid()

        # Add remaining samples to reach full set
        calib_res_full = tuner.add_samples(
            indices=np.arange(10, N),
            scores=scores[10:N],
            labels=labels[10:N],
        )
        assert calib_res_full.num_samples == N
        assert calib_res_full.asymptotic_diag_R is not None
        assert calib_res_full.asymptotic_diag_R.is_valid() is True
        assert calib_res_full.asymptotic_diag_R.v_t > 0.0

    def test_auto_min_positives(self):
        tuner = ACTIS(
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=np.array([0.2, 0.5]),
            conf_seq="asymptotic",
            pop_size=1000,
        )
        tuner.add_samples(
            indices=np.arange(100),
            scores=np.linspace(0.1, 0.9, 100),
            labels=np.ones(100, dtype=int),
        )

        # "auto" min_positives should compute min(30, max(5, ceil(0.01 * 1000))) = 10
        should_continue, diag = tuner.should_continue_sampling(
            population_scores=np.linspace(0.1, 0.9, 1000),
            batch_size=50,
        )
        assert diag.min_positives == 10
        assert diag.warmup_needed is False

    def test_delta_dependent_cap(self):
        # delta = 0.05 -> delta_R = 0.025 -> ceil(8 * log(40)) = 30
        tuner_05 = ACTIS(
            gamma_R=0.8, gamma_P=0.8, delta=0.05, thresholds=[0.5], pop_size=100000
        )
        assert tuner_05._resolve_min_positives() == 30

        # delta = 0.10 -> delta_R = 0.05 -> ceil(8 * log(20)) = 24
        tuner_10 = ACTIS(
            gamma_R=0.8, gamma_P=0.8, delta=0.10, thresholds=[0.5], pop_size=100000
        )
        assert tuner_10._resolve_min_positives() == 24

        # delta = 0.01 -> delta_R = 0.005 -> ceil(8 * log(200)) = 43
        tuner_01 = ACTIS(
            gamma_R=0.8, gamma_P=0.8, delta=0.01, thresholds=[0.5], pop_size=100000
        )
        assert tuner_01._resolve_min_positives() == 43

    def test_p_floor_explicit_and_auto(self):
        # Explicit p_floor = 0.001 on pop_size = 10000 -> ceil(0.001 * 10000) = 10
        tuner_exp = ACTIS(
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=[0.5],
            pop_size=10000,
            p_floor=0.001,
        )
        assert tuner_exp._resolve_min_positives() == 10

        # Auto p_floor from rare scores with mean 0.001 (like ImageNet)
        # eff_p = min(0.01, max(1e-4, 0.5 * 0.001)) = 0.0005
        # min_pos = min(30, max(5, ceil(0.0005 * 10000))) = 5
        rare_scores = np.full(10000, 0.001)
        tuner_auto = ACTIS(
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            thresholds=[0.5],
            pop_size=10000,
            p_floor="auto",
        )
        assert tuner_auto._resolve_min_positives(scores=rare_scores) == 5


class TestACTISRunnerIntegration:
    def test_auto_resolution_small_dataset(self):
        rng = np.random.default_rng(999)
        N = 500
        scores = rng.beta(2, 5, size=N)
        labels = (scores > 0.3).astype(int)

        runner = ACTISRunner(
            name="test_actis_auto",
            conf_seq="asymptotic",
            adaptive=True,
            initial_sample_size="auto",
            min_positives="auto",
            v_0="auto",
        )

        pop = Population(scores=scores, labels=labels)
        trial_result = runner.run_trial(
            population=pop,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            rng=rng,
        )

        assert trial_result.cost["oracle"]["num_calls"] > 0
        assert trial_result.cost["oracle"]["num_calls"] <= N
        assert 0.0 <= trial_result.recall <= 1.0
        assert 0.0 <= trial_result.precision <= 1.0
        assert trial_result.tau_pos is not None
        assert trial_result.tau_neg is not None


class TestRefinedAsymptoticDiagnostics:
    def test_target_variance_scaling(self):
        # Setup: v_0_target = 0.10, V_t = 0.01 ->
        # V_t / v_0_target = 0.10 >= 0.05 (valid)
        recall_diag = AsymptoticValidityDiagnostics(
            v_t=0.01,
            v_0=0.10,
            num_true_positives=72,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        precision_diag = AsymptoticValidityDiagnostics(
            v_t=0.01,
            v_0=0.10,
            num_true_positives=72,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Precision",
            delta=0.025,
        )
        assert recall_diag.is_valid() is True
        assert len(recall_diag.diagnose()) == 0
        assert recall_diag.v_t / recall_diag.v_0 == pytest.approx(0.10)

        assert precision_diag.is_valid() is True
        assert len(precision_diag.diagnose()) == 0
        assert precision_diag.v_t / precision_diag.v_0 == pytest.approx(0.10)

        # Now if V_t is below 0.05 * v_0_target (e.g. 0.002):
        diag_low = AsymptoticValidityDiagnostics(
            v_t=0.002,
            v_0=0.10,
            num_true_positives=72,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_low.is_valid() is False
        issues = diag_low.diagnose()
        assert any("Recall empirical variance ratio" in s for s in issues)

    def test_exact_binomial_consistency(self):
        # Target: gamma_R = 0.95, delta_R = 0.025, p0 = 1 - gamma_R = 0.05
        # 1. At k_FN = 0, n_TP = 43 gives p = 0.95^43 = 0.1102 > 0.025
        # (Gaussian CS discrete blind spot)
        diag_43 = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=43,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_43.is_valid() is False
        assert diag_43.p_value is not None
        assert diag_43.p_value > 0.025
        assert any("Recall binomial p-value" in s for s in diag_43.diagnose())

        # 2. At k_FN = 0, n_TP = 71 gives p = 0.0262 > 0.025 (fails)
        diag_71 = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=71,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_71.is_valid() is False
        assert diag_71.p_value is not None
        assert diag_71.p_value > 0.025

        # 3. At k_FN = 0, n_TP = 72 gives p = 0.0249 <= 0.025 (passes)
        diag_72 = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=72,
            num_false_negatives_or_positives=0,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_72.is_valid() is True
        assert diag_72.p_value is not None
        assert diag_72.p_value <= 0.025

        # 4. At k_FN = 1: n_TP = 108 fails (p = 0.0255), n_TP = 109 passes (p = 0.0242)
        diag_108 = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=108,
            num_false_negatives_or_positives=1,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_108.is_valid() is False
        assert diag_108.p_value is not None
        assert diag_108.p_value > 0.025

        diag_109 = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=109,
            num_false_negatives_or_positives=1,
            null_failure_prob=0.05,
            metric="Recall",
            delta=0.025,
        )
        assert diag_109.is_valid() is True
        assert diag_109.p_value is not None
        assert diag_109.p_value <= 0.025

        # 5. Precision exact binomial check symmetry
        # 5 false positives with n_TP=72 -> p = binom.cdf(5, 77, 0.05) = 0.80 > 0.025
        diag_p_fail = AsymptoticValidityDiagnostics(
            v_t=1.0,
            v_0=1.0,
            num_true_positives=72,
            num_false_negatives_or_positives=5,
            null_failure_prob=0.05,
            metric="Precision",
            delta=0.025,
        )
        assert diag_p_fail.is_valid() is False
        assert diag_p_fail.p_value is not None
        assert diag_p_fail.p_value > 0.025
        assert any("Precision binomial p-value" in s for s in diag_p_fail.diagnose())

    def test_deterministic_anchor_exemptions(self):
        thresholds = np.array([0.0, 0.5, 1.0])
        conf_seqs = AsymptoticCascadeConfSeqs(
            gamma_P=0.95,
            gamma_R=0.95,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds,
        )
        # For lower anchor k=0 (tau_lower = 0.0), recall diagnostic is None (exempt)
        assert conf_seqs.asymptotic_diag_R(k_lower=0) is None

        # For upper anchor k=len(thresholds_upper)-1 (tau_upper = 1.0),
        # precision diagnostic is None (exempt)
        last_k = len(thresholds) - 1
        assert conf_seqs.asymptotic_diag_P(k_upper=last_k, k_lower=0) is None

    def test_compute_prior_and_target_var_structure(self):
        scores = np.linspace(0.1, 0.9, 100)
        thresholds = np.array([0.3, 0.6])
        thresholds_upper = np.array([0.5, 0.8])

        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.9,
            gamma_P=0.9,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
        )
        assert isinstance(res, PriorAndTargetVar)

        scalar_res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.9,
            gamma_P=0.9,
            delta=0.05,
            thresholds=None,
        )
        assert isinstance(scalar_res, PriorAndTargetVar)
        assert scalar_res.prior_R(0) > 0.0

    def test_check_asymptotic_validity_via_conf_seqs(self):
        thresholds = np.array([0.0, 0.4, 0.8])
        thresholds_upper = np.array([0.2, 0.6, 1.0])
        conf_seqs = AsymptoticCascadeConfSeqs(
            gamma_P=0.95,
            gamma_R=0.95,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
        )

        # 80 positive items with score 0.5 (above lower=0.4, below upper=0.6)
        # and 120 negative items with score 0.1 (below lower=0.4)
        scores = np.concatenate([np.full(80, 0.5), np.full(120, 0.1)])
        labels = np.concatenate([np.ones(80, dtype=bool), np.zeros(120, dtype=bool)])
        conf_seqs.add_samples(scores=scores, labels=labels)

        # Candidate: k_upper = 1 (tau_upper=0.6), k_lower = 1 (tau_lower=0.4)
        # num_tp = 80, num_fn = 0, num_fp = 0
        # At n_tp = 80, exact binomial p_val <= 0.025 (passes)
        recall_diag = conf_seqs.asymptotic_diag_R(k_lower=1)
        precision_diag = conf_seqs.asymptotic_diag_P(k_upper=1, k_lower=1)
        is_valid = (recall_diag is None or recall_diag.is_valid()) and (
            precision_diag is None or precision_diag.is_valid()
        )
        assert is_valid is True
        if recall_diag is not None:
            p_value = recall_diag.p_value
            assert p_value is not None and p_value <= 0.025
            assert recall_diag.num_true_positives == 80
            assert recall_diag.num_false_negatives_or_positives == 0
        if precision_diag is not None:
            p_value = precision_diag.p_value
            assert p_value is not None and p_value <= 0.025
            assert precision_diag.num_true_positives == 80
            assert precision_diag.num_false_negatives_or_positives == 0

        # Now test with only 43 positive samples (where exact binomial fails)
        conf_seqs_small = AsymptoticCascadeConfSeqs(
            gamma_P=0.95,
            gamma_R=0.95,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
        )
        scores_small = np.concatenate([np.full(43, 0.5), np.full(60, 0.1)])
        labels_small = np.concatenate(
            [np.ones(43, dtype=bool), np.zeros(60, dtype=bool)]
        )
        conf_seqs_small.add_samples(scores=scores_small, labels=labels_small)
        recall_diag = conf_seqs_small.asymptotic_diag_R(k_lower=1)
        precision_diag = conf_seqs_small.asymptotic_diag_P(k_upper=1, k_lower=1)
        is_valid_small = (recall_diag is None or recall_diag.is_valid()) and (
            precision_diag is None or precision_diag.is_valid()
        )
        assert not is_valid_small
        if recall_diag is not None:
            p_value = recall_diag.p_value
            assert p_value is not None and p_value > 0.025
            assert any("Recall binomial p-value" in s for s in recall_diag.diagnose())

    def test_positive_jumps_only_b_R_and_b_P(self):
        N = 100
        # Items below tau=0.5 are negative (score 0.0) with large weights (100.0)
        # Items above tau=0.5 are positive (score 0.8) with small weights (1.0)
        scores = np.where(np.linspace(0.0, 1.0, N) < 0.5, 0.0, 0.8)
        weights = np.where(scores < 0.5, 100.0, 1.0)
        gamma_R = 0.95
        delta_R = 0.025
        log_delta_R_inv = math.log(1.0 / delta_R)

        thresholds = np.array([0.5])
        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=0.95,
            delta=0.05,
            thresholds=thresholds,
            weights=weights,
        )

        # For threshold tau=0.5, w_max_ge is 1.0 (from items with scores >= 0.5).
        # Positive jump bound should be (1 - gamma_R) * 1.0 = 0.05.
        # If negative jumps were included, it would be:
        # gamma_R * 100.0 = 95.0 (v_0_floor > 4000).
        expected_floor = (2.0 * ((1.0 - gamma_R) * 1.0) ** 2) / log_delta_R_inv
        # Prior variance floor should not be contaminated by 100.0 from negative items
        assert res.prior_R(0) < 1.0
        assert res.prior_R(0) >= expected_floor

    def test_precision_null_failure_prob_proxy_and_geometric_floor(self):
        N = 1000
        scores = np.linspace(0.01, 0.99, N)
        # Proposal strongly biased towards high scores:
        # q(x) proportional to s(x)^2 + 0.05
        weights = 1.0 / (scores**2 + 0.05)
        weights = weights / np.sum(weights) * N  # normalize so mean weight is ~1

        gamma_P = 0.95
        gamma_R = 0.95
        delta = 0.05
        thresholds = np.array([0.2, 0.5])
        thresholds_upper = np.array([0.6, 0.9])

        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=weights,
        )

        # Check that null_failure_prob_P is bounded in (0, 0.5]
        for k_u in range(len(thresholds_upper)):
            for k_l in range(len(thresholds)):
                p0_P = res.null_failure_prob_P(k_u, k_l)
                assert 0.0 < p0_P <= 0.5

        # For matched thresholds (tau_u = tau_l = 0.5), proxy weighting ensures
        # True Positives receive higher proposal mass than False Positives,
        # so p0_P < 1 - gamma_P
        res_matched = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=np.array([0.5]),
            thresholds_upper=np.array([0.5]),
            weights=weights,
        )
        assert res_matched.null_failure_prob_P(0, 0) < (1.0 - gamma_P)

        # Verify geometric floor enforcement under adversarial proxy scores
        # Suppose proxy scores for candidate upper are near 0.99, but q is constant
        const_weights = np.ones(N)
        res_const = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=const_weights,
        )
        # Under uniform proposal, ratio_P_worst = 1.0, so eta_P >= 1.0,
        # null_failure_prob_P should be exactly (1 - gamma_P) = 0.05
        for k_u in range(len(thresholds_upper)):
            for k_l in range(len(thresholds)):
                assert res_const.null_failure_prob_P(k_u, k_l) == pytest.approx(
                    1.0 - gamma_P, abs=1e-5
                )

    def test_compute_prior_and_target_var_uniform_fast_path_equivalence(self):
        rng = np.random.default_rng(42)
        N = 1000
        scores = rng.beta(2, 5, size=N)
        thresholds = np.linspace(0.1, 0.9, 10)
        thresholds_upper = np.linspace(0.2, 0.95, 12)

        res_none = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.9,
            gamma_P=0.85,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=None,
        )

        res_ones = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.9,
            gamma_P=0.85,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=np.ones(N),
        )

        np.testing.assert_allclose(
            res_none._prior_R, res_ones._prior_R, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_none._prior_P, res_ones._prior_P, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_none._target_R, res_ones._target_R, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_none._target_P, res_ones._target_P, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res_none._null_failure_prob_R,
            res_ones._null_failure_prob_R,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            res_none._null_failure_prob_P,
            res_ones._null_failure_prob_P,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compute_prior_and_target_var_numerical_stability_extreme_tails(self):
        # Extreme scores close to 1.0 and 0.0
        N = 2000
        scores = np.concatenate(
            [
                np.full(1000, 1.0 - 1e-14),
                np.full(1000, 1e-14),
            ]
        )
        thresholds = np.array([0.0, 1e-15, 0.5, 1.0 - 1e-15, 1.0])
        thresholds_upper = np.array([0.5, 1.0 - 1e-15, 1.0])

        q = scores**2 + 0.05
        q /= q.sum()
        weights = 1.0 / (N * q)

        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=0.95,
            gamma_P=0.90,
            delta=0.05,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=weights,
        )

        # Ensure no NaNs, Infs, or negative numbers anywhere
        assert np.all(np.isfinite(res._prior_R)) and np.all(res._prior_R > 0)
        assert np.all(np.isfinite(res._prior_P)) and np.all(res._prior_P > 0)
        assert np.all(np.isfinite(res._target_R)) and np.all(res._target_R >= 0)
        assert np.all(np.isfinite(res._target_P)) and np.all(res._target_P >= 0)
        assert np.all(np.isfinite(res._null_failure_prob_R)) and np.all(
            res._null_failure_prob_R >= 0
        )
        assert np.all(np.isfinite(res._null_failure_prob_P)) and np.all(
            res._null_failure_prob_P > 0
        )


class TestBoundaryNullVarianceBounds:
    def test_precision_null_cap_and_theoretical_ceiling(self):
        # High positive prevalence, but compressed proxy scores (mean ~0.26)
        rng = np.random.default_rng(42)
        N = 1000
        scores = rng.beta(2, 6, size=N) * 0.7  # max score < 0.7
        thresholds = np.linspace(0.0, 0.6, 20)
        thresholds_upper = np.linspace(0.1, 0.65, 50)
        gamma_R = 0.95
        gamma_P = 0.95
        delta = 0.05
        eff_n_0 = min(1000.0, max(50.0, 0.1 * N))

        # Unweighted
        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=None,
        )

        # 1. Target variance must never exceed theoretical ceiling
        # eff_n_0 * (1 - gamma) * max_weight
        v_0_max_R = eff_n_0 * (1.0 - gamma_R) * 1.0
        v_0_max_P = eff_n_0 * (1.0 - gamma_P) * 1.0
        assert np.all(res._target_R <= v_0_max_R + 1e-12)
        assert np.all(res._target_P <= v_0_max_P + 1e-12)

        # 2. Weighted (uniform) should match unweighted bitwise
        res_weighted = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=np.ones(N),
        )
        np.testing.assert_allclose(
            res._target_R, res_weighted._target_R, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res._target_P, res_weighted._target_P, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res._prior_R, res_weighted._prior_R, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            res._prior_P, res_weighted._prior_P, rtol=1e-12, atol=1e-12
        )

    def test_weighted_theoretical_ceiling(self):
        # Non-uniform importance weights
        rng = np.random.default_rng(123)
        N = 500
        scores = rng.uniform(0.1, 0.9, size=N)
        q = scores + 0.1
        q /= q.sum()
        weights = 1.0 / (N * q)

        thresholds = np.linspace(0.1, 0.8, 15)
        thresholds_upper = np.linspace(0.2, 0.85, 25)
        gamma_R = 0.90
        gamma_P = 0.90
        delta = 0.05
        eff_n_0 = min(1000.0, max(50.0, 0.1 * N))

        res = compute_prior_and_target_var(
            scores=scores,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            weights=weights,
        )

        order = np.argsort(scores)
        weights_sorted = weights[order]
        idx_lower = np.searchsorted(scores[order], thresholds, side="left")
        suffix_max_weight = np.append(
            np.maximum.accumulate(weights_sorted[::-1])[::-1],
            np.max(weights),
        )
        w_max_lower = suffix_max_weight[idx_lower]

        v_0_max_R = eff_n_0 * (1.0 - gamma_R) * w_max_lower
        v_0_max_P = eff_n_0 * (1.0 - gamma_P) * w_max_lower[np.newaxis, :]

        assert np.all(res._target_R <= v_0_max_R + 1e-12)
        assert np.all(res._target_P <= v_0_max_P + 1e-12)
