#!/usr/bin/env python3
"""
Generic Filter Runners & Evaluation Harness for Semantic Database Filtering

Defines abstract base runners and concrete implementations for:
- ACTIS (`ACTISRunner`)
- BARGAIN-PR (`BargainPRRunner`)
- LOTUS (`LotusOriginalRunner`)

Provides a unified evaluation suite (`run_evaluation_suite`) for paired Monte Carlo
benchmarking.
"""
import math
import random
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import lotus
import numpy as np
from BARGAIN.models.AbstractModels import Oracle, Proxy
from BARGAIN.process.BARGAIN_PR import BARGAIN_PR
from lotus.sem_ops.cascade_utils import (
    importance_sampling,
    learn_cascade_thresholds,
)
from lotus.types import CascadeArgs
from numpy.typing import NDArray

from actis import ACTIS, ProposalMethod, compute_pr_proposal
from actis.sampler import PopulationSampler
from actis.threshold_grid import quantile_power_law_grid
from actis.tuner import compute_prior_var
from experiments.scenarios import BaseScenario

_SCALEDOC_DIR = (
    Path(__file__).resolve().parent.parent / "externals" / "ScaleDoc" / "src"
)
_HAS_SCALEDOC = False
if _SCALEDOC_DIR.exists():
    if str(_SCALEDOC_DIR) not in sys.path:
        sys.path.insert(0, str(_SCALEDOC_DIR))
    try:
        from cascade import calibrate_sampling, select_sim_filterB, smooth_distr

        _HAS_SCALEDOC = True
    except ImportError:
        _HAS_SCALEDOC = False


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def compute_precision_recall(
    pred_mask: BoolArray,
    labels: BoolArray
) -> tuple[float, float]:
    total_positives = np.sum(labels)
    tp = np.sum(pred_mask & labels)
    fp = np.sum(pred_mask & (~labels))
    recall = float(tp / total_positives) if total_positives > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    return precision, recall


@dataclass
class TrialResult:
    """Stores the evaluation metrics from a single Monte Carlo trial."""
    recall: float
    precision: float
    total_oracle_calls: int
    total_oracle_rate: float
    calibration_calls: int = 0
    deployment_calls: int = 0
    tau_pos: float | None = None
    tau_neg: float | None = None
    is_asymptotically_valid: bool | None = None


@dataclass(kw_only=True)
class BaseFilterRunner(ABC):
    """Abstract base class for semantic filtering methods."""
    name: str = ""

    @abstractmethod
    def run_trial(
        self,
        scores: FloatArray,
        labels: BoolArray,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        """
        Executes filtering over a dataset for a single trial.

        Args:
            scores: Proxy scores across the entire dataset.
            labels: Binary ground-truth oracle outputs across the entire dataset.
            gamma_R: Target recall constraint.
            gamma_P: Target precision constraint.
            delta: Allowed failure probability that the constraints are met.
            rng: Random number generator for trial-specific sampling.

        Returns:
            TrialResult containing realized recall, precision, and total oracle cost.
        """
        pass


def evaluate_cascade_trial(
    scores: FloatArray,
    labels: BoolArray,
    tau_pos: float,
    tau_neg: float,
    calib_indices: set[int] | list[int] | NDArray[np.int64],
    is_asymptotically_valid: bool | None = None,
) -> TrialResult:
    pop_size = len(scores)
    pred_mask = (scores >= tau_pos) | ((scores >= tau_neg) & labels)
    precision, recall = compute_precision_recall(pred_mask, labels)

    # Oracle call accounting: count unique oracle calls across calibration and
    # deployment
    unique_calib = set(calib_indices)
    oracle_sent = (scores >= tau_neg) & (scores < tau_pos)
    calib_calls = len(unique_calib)
    pilot_mask = np.zeros(pop_size, dtype=bool)
    if calib_calls > 0:
        pilot_mask[list(unique_calib)] = True
    dep_calls = int(np.sum(oracle_sent & (~pilot_mask)))
    total_calls = calib_calls + dep_calls
    return TrialResult(
        recall=recall,
        precision=precision,
        total_oracle_calls=total_calls,
        total_oracle_rate=total_calls / pop_size if pop_size > 0 else 0.0,
        calibration_calls=calib_calls,
        deployment_calls=dep_calls,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        is_asymptotically_valid=is_asymptotically_valid,
    )


@dataclass(kw_only=True)
class ACTISRunner(BaseFilterRunner):
    """Runner for ACTIS"""

    num_thresholds: int = 50
    """Number of candidate thresholds to consider when tuning thresholds. Used for both
    lower and upper thresholds if `num_thresholds_upper` is not specified, otherwise
    just for the lower threshold."""

    num_thresholds_upper: int | None = 500
    """Number of candidate thresholds to consider when tuning the upper threshold. If
    None, `num_thresholds` is used for both lower and upper thresholds."""

    sampling_method: Literal["wor", "is"] = "is"
    """Method to use when sampling dataset items to label."""

    alpha: float | None = 0.4
    """Defensive mixing weight for importance sampling proposal distribution. If None
    or 0.0, no defensive mixing is applied."""

    proposal_method: ProposalMethod = "snr_balanced"
    """Method to use when computing the importance sampling proposal distribution."""

    power_law_quantiles: bool = True
    """Whether to use power-law quantiles for the upper threshold grid when
    `num_thresholds_upper` is specified."""

    conf_seq: Literal["finite", "asymptotic"] = "asymptotic"
    """Confidence sequence type to use."""

    v_0: float | tuple[FloatArray, FloatArray] | Literal["auto"] = "auto"
    """Prior variance for asymptotic Gaussian mixture supermartingale if `conf_seq` is
    "asymptotic"."""

    adaptive: bool = True
    """Whether to use adaptive sampling expansion."""

    initial_sample_size: int | Literal["auto"] = "auto"
    """Initial sample size. If `adaptive` is True, this is the size of the initial
    batch. If `adaptive` is False, this is the fixed sample size. If "auto", dynamically
    scales to the dataset size."""

    batch_size: int = 100
    """Batch size for adaptive sampling expansion. Ignored if `adaptive` is False."""

    max_sample_size: int | float | None = 0.5
    """Maximum sample size for adaptive sampling expansion. If a float in (0, 1], it is
    interpreted as a fraction of the dataset size. If an integer >= 1, it is interpreted
    as an absolute sample size. Ignored if None or `adaptive` is False."""

    min_positives: int | Literal["auto"] = "auto"
    """Minimum number of positive labels to observe before stopping adaptive sampling.
    If "auto", dynamically calibrated based on delta, pop_size, and proxy scores."""

    p_floor: float | Literal["auto"] = "auto"
    """Lower-bound positive prevalence for population scaling of `min_positives`.
    If "auto" (default), dynamically estimated from population proxy scores. If float,
    specifies the prevalence floor directly (e.g. 0.01 or 0.001)."""

    def __post_init__(self):
        if self.name == "":
            self.name = "actis_adaptive" if self.adaptive else "actis_static"

    def run_trial(
        self,
        scores: FloatArray,
        labels: BoolArray,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        pop_size = len(scores)

        # Resolve auto initial_sample_size
        if self.initial_sample_size == "auto":
            init_sample_size = min(
                pop_size,
                max(1, min(1000, max(50, int(np.ceil(0.10 * pop_size)))))
            )
        else:
            init_sample_size = min(pop_size, int(self.initial_sample_size))

        max_sample_size: int | None = None
        if isinstance(self.max_sample_size, float) and \
            (0.0 < self.max_sample_size <= 1.0):
            max_sample_size = max(1, int(self.max_sample_size * pop_size))
        elif isinstance(self.max_sample_size, int) and self.max_sample_size >= 1:
            max_sample_size = min(self.max_sample_size, pop_size)
        elif self.max_sample_size is None and self.sampling_method == "wor":
            max_sample_size = pop_size

        num_thresholds = min(self.num_thresholds, pop_size)
        thresholds = np.quantile(
            scores,
            np.linspace(0, 1, num_thresholds)
        )
        thresholds = np.unique(thresholds)
        thresholds_upper = None
        if self.num_thresholds_upper is not None:
            num_thresholds_upper = min(self.num_thresholds_upper, pop_size)
            if self.power_law_quantiles:
                thresholds_upper = quantile_power_law_grid(
                    scores,
                    num_thresholds=num_thresholds_upper
                )
            else:
                thresholds_upper = np.quantile(
                    scores,
                    np.linspace(0, 1, num_thresholds_upper)
                )
                thresholds_upper = np.unique(thresholds_upper)

        q = None
        weights = None
        max_weight_ge = None
        max_weight_lt = None
        max_weight_ge_upper = None

        if self.sampling_method == "is":
            q = compute_pr_proposal(
                scores=scores,
                gamma_P=gamma_P,
                gamma_R=gamma_R,
                thresholds=thresholds,
                thresholds_upper=thresholds_upper,
                alpha=self.alpha,
                method=self.proposal_method
            )

            weights = 1.0 / (q * pop_size)

            if self.conf_seq == "finite":
                max_weight_ge = np.array([
                    weights[scores >= tau].max()
                    if np.any(scores >= tau)
                    else float(weights.max())
                    for tau in thresholds
                ])

                max_weight_lt = np.array([
                    weights[scores < tau].max()
                    if np.any(scores < tau)
                    else 0.0
                    for tau in thresholds
                ])

                max_weight_ge_upper = None
                if thresholds_upper is not None:
                    max_weight_ge_upper = np.array([
                        weights[scores >= tau].max()
                        if np.any(scores >= tau)
                        else float(weights.max())
                        for tau in thresholds_upper
                    ])

            pop_size_param = None
        elif self.sampling_method == "wor":
            pop_size_param = pop_size
        else:
            raise ValueError(f"Unknown sampling_method: '{self.sampling_method}'")

        sampler = PopulationSampler(
            pop_size=pop_size,
            replace=self.sampling_method != "wor",
            p=q,
            rng=rng,
        )

        if self.v_0 is None or self.v_0 == "auto":
            v_0_val = compute_prior_var(
                scores=scores,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                thresholds=thresholds,
                thresholds_upper=thresholds_upper,
                weights=weights,
            )
        elif self.v_0 in ("global", "global_calibrated"):
            v_0_val = compute_prior_var(
                scores=scores,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                thresholds=None,
                weights=weights,
            )
        else:
            v_0_val = self.v_0

        tuner = ACTIS(
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size_param,
            horizon=None if self.adaptive else init_sample_size,
            conf_seq=self.conf_seq,
            v_0=v_0_val,
            min_positives=self.min_positives,
            p_floor=self.p_floor,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
        )

        def get_batch_size(requested: int) -> int:
            remaining = float("inf")
            if max_sample_size is not None:
                remaining = min(remaining, max_sample_size - sampler.sample_count)
            if self.sampling_method == "wor":
                remaining = min(remaining, pop_size - sampler.sample_count)
            if math.isinf(remaining):
                return max(0, requested)
            return max(0, min(requested, int(remaining)))

        # Draw initial batch
        init_batch_size = get_batch_size(init_sample_size)
        batch_idx = sampler.sample(init_batch_size)

        calib_res = tuner.add_samples(
            indices=batch_idx,
            scores=scores[batch_idx],
            labels=labels[batch_idx],
            weights=weights[batch_idx] if weights is not None else None,
            population_scores=scores,
        )

        # Adaptive expansion using value-of-information policy (expected net oracle
        # savings > 0)
        if self.adaptive:
            while True:
                next_batch_size = get_batch_size(self.batch_size)
                if next_batch_size <= 0:
                    break

                should_continue, _ = tuner.should_continue_sampling(
                    population_scores=scores,
                    batch_size=next_batch_size,
                    max_sample_size=max_sample_size,
                )
                if not should_continue:
                    break

                next_batch_idx = sampler.sample(next_batch_size)
                calib_res = tuner.add_samples(
                    indices=next_batch_idx,
                    scores=scores[next_batch_idx],
                    labels=labels[next_batch_idx],
                    weights=weights[next_batch_idx] if weights is not None else None,
                    population_scores=scores,
                )

        return evaluate_cascade_trial(
            scores=scores,
            labels=labels,
            tau_pos=calib_res.tau_pos,
            tau_neg=calib_res.tau_neg,
            calib_indices=tuner.seen_indices,
            is_asymptotically_valid=calib_res.is_asymptotically_valid,
        )


@dataclass(kw_only=True)
class LotusRunner(BaseFilterRunner):
    """Runner for legacy LOTUS using learn_cascade_thresholds."""

    sampling_percentage: float = 0.1

    def __post_init__(self):
        if self.name == "":
            self.name = "lotus"

    def run_trial(
        self,
        scores: FloatArray,
        labels: BoolArray,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:

        cascade_args = CascadeArgs(
            recall_target=gamma_R,
            precision_target=gamma_P,
            failure_probability=delta,
            sampling_percentage=self.sampling_percentage,
            cascade_IS_random_seed=int(rng.integers(0, 2**31 - 1))
        )

        sample_idx, weights = importance_sampling(
            proxy_scores=scores.tolist(),
            cascade_args=cascade_args,
        )

        s_scores = scores[sample_idx].tolist()
        s_oracle = labels[sample_idx].tolist()

        try:
            (tau_pos, tau_neg), _ = learn_cascade_thresholds(
                proxy_scores=s_scores,
                oracle_outputs=s_oracle,
                sample_correction_factors=weights,
                cascade_args=cascade_args,
            )
        except Exception as e:
            lotus.logger.error(f"Legacy LOTUS calibration error: {e}")
            tau_pos, tau_neg = 1.0, 0.0

        return evaluate_cascade_trial(
            scores=scores,
            labels=labels,
            tau_pos=tau_pos,
            tau_neg=tau_neg,
            calib_indices=sample_idx,
        )


class VectorizedProxy(Proxy):
    """Fast vectorized proxy wrapper for population arrays."""

    def __init__(self, scores: FloatArray):
        super().__init__(verbose=False, max_workers=1)
        self.scores = np.asarray(scores, dtype=np.float64)

    def proxy_func(self, input: Any) -> tuple[int, float]:
        idx = int(input)
        return 1, float(self.scores[idx])

    def get_preds_and_scores(
        self,
        indxs: list[int],
        data_records: list[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        idx_arr = np.asarray(indxs, dtype=int)
        return np.ones(len(idx_arr), dtype=int), self.scores[idx_arr]

    def reset(self) -> None:
        super().reset()


class VectorizedOracle(Oracle):
    """Fast vectorized oracle wrapper that tracks all unique queried items."""

    def __init__(self, labels: BoolArray):
        super().__init__(verbose=False, max_workers=1)
        self.labels = np.asarray(labels, dtype=bool)
        self.queried_indices: set[int] = set()

    def oracle_func(self, input: Any, proxy_output: Any) -> tuple[bool, int]:
        idx = int(input)
        label = int(self.labels[idx])
        self.queried_indices.add(idx)
        return (label == proxy_output), label

    def get_number_preds(self) -> int:
        return len(self.queried_indices)

    def get_pred(
        self,
        data_records: list[Any],
        indxs: list[int] | None = None) -> np.ndarray:
        if indxs is None:
            idx_arr = np.asarray(data_records, dtype=int)
        else:
            idx_arr = np.asarray(indxs, dtype=int)
        preds = self.labels[idx_arr].astype(int)
        self.queried_indices.update(idx_arr.tolist())
        return preds

    def reset(self) -> None:
        super().reset()
        self.queried_indices.clear()


@dataclass(kw_only=True)
class BargainPRRunner(BaseFilterRunner):
    """Runner for BARGAIN_PR"""

    window_size: int = 50
    """Sliding window size for precision threshold search"""

    num_thresholds: int = 20
    """Number of thresholds for fallback precision threshold search"""

    sample_step: int = 10
    """Number of oracle labels to request per batch"""

    def __post_init__(self):
        if self.name == "":
            self.name = "bargain_pr"

    def run_trial(
        self,
        scores: FloatArray,
        labels: BoolArray,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        if gamma_P != gamma_R:
            raise ValueError(
                "BargainPRRunner only supports `gamma_P == gamma_R`."
            )

        pop_size = len(scores)

        proxy_model = VectorizedProxy(scores)
        oracle_model = VectorizedOracle(labels)

        trial_seed = int(rng.integers(0, 2**31 - 1))
        bargain = BARGAIN_PR(
            proxy=proxy_model,
            oracle=oracle_model,
            delta=delta,
            target=gamma_R,
            W=self.window_size,
            M=self.num_thresholds,
            sample_step=self.sample_step,
            verbose=False,
            seed=trial_seed,
        )

        data_records = np.arange(pop_size)
        pred_positives = bargain.process(data_records)  # ty: ignore[invalid-argument-type]

        pred_mask = np.zeros(pop_size, dtype=bool)
        if len(pred_positives) > 0:
            pred_mask[pred_positives] = True

        precision, recall = compute_precision_recall(pred_mask, labels)

        total_calls = len(oracle_model.queried_indices)

        return TrialResult(
            recall=recall,
            precision=precision,
            total_oracle_calls=total_calls,
            total_oracle_rate=total_calls / pop_size,
        )


if _HAS_SCALEDOC:

    @dataclass(kw_only=True)
    class ScaleDocRunner(BaseFilterRunner):
        """
        Runner for ScaleDoc (https://github.com/Seurgul/ScaleDoc).

        Calibrates model cascade thresholds on proxy confidence scores using
        ScaleDoc's calibration workflow (stratified sampling, jittering, moving average
        smoothing, and Pareto-frontier threshold search).
        """

        sample_rate: float = 0.05
        """Calibration sampling ratio (default 0.05, matching ScaleDoc's
        config.json)."""

        sample_size: int | None = None
        """Optional fixed calibration sample size. If specified and > 0, sample_rate
        is computed as min(1.0, sample_size / pop_size)."""

        num_bins: int = 64
        """Number of bins for score histogram (default 64, matching ScaleDoc)."""

        window_size: int = 5
        """Window size for moving average smoothing of score distributions
        (default 5)."""

        def __post_init__(self):
            if self.name == "":
                self.name = "scaledoc"

        def run_trial(
            self,
            scores: FloatArray,
            labels: BoolArray,
            gamma_R: float,
            gamma_P: float,
            delta: float,
            rng: np.random.Generator,
        ) -> TrialResult:
            pop_size = len(scores)

            # ScaleDoc only supports a target F1 score gamma_F, not separate precision
            # and recall targets. In order to compare, setting:
            # gamma_F = max(2*gamma_P / (1 + gamma_P), 2*gamma_R / (1 + gamma_R))
            # guarantees that precision >= gamma_P and recall >= gamma_R.
            f1_p = (2.0 * gamma_P) / (1.0 + gamma_P)
            f1_r = (2.0 * gamma_R) / (1.0 + gamma_R)
            gamma_F = max(f1_p, f1_r)

            sample_rate = self.sample_rate
            if self.sample_size is not None and pop_size > 0:
                sample_rate = min(1.0, self.sample_size / pop_size)

            # Seed global random and np.random for reproducibility across trials,
            # as ScaleDoc's cascade functions internally invoke legacy np.random.
            trial_seed = int(rng.integers(0, 2**31 - 1))
            random.seed(trial_seed)
            np.random.seed(trial_seed)

            calib_samples: list[int] | NDArray[np.int64] = []
            try:
                hist, bins = np.histogram(scores, bins=self.num_bins)
                pos_idx = np.where(labels)[0]
                neg_idx = np.where(~labels)[0]

                samples = calibrate_sampling(
                    sample_rate, hist, bins, scores, pos_idx, neg_idx
                )
                calib_samples = samples

                bins_center = np.array(
                    [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
                )
                pos_ = np.array([j for j in samples if j in pos_idx])
                neg_ = np.array([j for j in samples if j in neg_idx])

                pos_cos_sample = (
                    scores[pos_] if pos_.shape[0] > 0 else np.array([])
                )
                neg_cos_sample = (
                    scores[neg_] if neg_.shape[0] > 0 else np.array([])
                )
                hist_pos_sample, _ = np.histogram(pos_cos_sample, bins=bins)
                hist_neg_sample, _ = np.histogram(neg_cos_sample, bins=bins)

                # 1. Jittering
                rand1 = np.random.choice(
                    [0, 0.1, 0.2], size=hist_pos_sample.shape, p=[0.6, 0.3, 0.1]
                )
                rand2 = np.random.choice(
                    [0, 0.1, 0.2], size=hist_neg_sample.shape, p=[0.6, 0.3, 0.1]
                )
                hist_pos_sample = hist_pos_sample + rand1
                hist_neg_sample = hist_neg_sample + rand2

                # 2. Smoothing
                _, neg_hist_sample_ma = smooth_distr(
                    bins, hist_neg_sample, window_size=self.window_size
                )
                _, pos_hist_sample_ma = smooth_distr(
                    bins, hist_pos_sample, window_size=self.window_size
                )

                steps = bins[::2]

                le, re, _ = select_sim_filterB(
                    steps=steps,
                    x=bins_center,
                    y_pos=pos_hist_sample_ma,
                    y_neg=neg_hist_sample_ma,
                    l_s=bins[0],
                    r_s=bins[-1],
                    target_acc=gamma_F,
                )

                # Map bounds to original bin edges as done in ScaleDoc's apply_bounds
                le_matches = np.where(steps == le)[0]
                le_step_idx = (
                    int(le_matches[0])
                    if len(le_matches) > 0
                    else int(np.argmin(np.abs(steps - le)))
                )
                le_idx = max(0, le_step_idx * 2 - 1)

                re_matches = np.where(steps == re)[0]
                re_step_idx = (
                    int(re_matches[0])
                    if len(re_matches) > 0
                    else int(np.argmin(np.abs(steps - re)))
                )
                re_idx = min(len(bins) - 1, re_step_idx * 2 + 1)

                tau_neg = float(bins[le_idx])
                tau_pos = float(bins[re_idx])
            except Exception as e:
                warnings.warn(f"ScaleDoc calibration error: {e}")
                tau_pos = float(np.max(scores)) if pop_size > 0 else 1.0
                tau_neg = float(np.min(scores)) if pop_size > 0 else 0.0

            return evaluate_cascade_trial(
                scores=scores,
                labels=labels,
                tau_pos=tau_pos,
                tau_neg=tau_neg,
                calib_indices=calib_samples,
            )


def summarize_runner_trials(
    runner: BaseFilterRunner,
    results: list[TrialResult],
    scenario: BaseScenario,
    exp_name: str,
    gamma_R: float,
    gamma_P: float,
    delta: float,
    pop_size: int,
    ideal_oracle_rate: float,
) -> dict[str, Any]:
    rec_arr = np.array([r.recall for r in results])
    prec_arr = np.array([r.precision for r in results])
    rate_arr = np.array([r.total_oracle_rate for r in results])
    calls_arr = np.array([r.total_oracle_calls for r in results])
    rec_fail = rec_arr < gamma_R
    prec_fail = prec_arr < gamma_P
    joint_fail = rec_fail | prec_fail
    summary = {
        "scenario": scenario.name,
        "description": scenario.description,
        "experiment_name": exp_name,
        "runner_params": asdict(runner),
        "num_trials": len(results),
        "pop_size": pop_size,
        "gamma_R": gamma_R,
        "gamma_P": gamma_P,
        "promised_delta": delta,
        "joint_failure_rate": float(np.mean(joint_fail)),
        "recall_failure_rate": float(np.mean(rec_fail)),
        "precision_failure_rate": float(np.mean(prec_fail)),
        "coverage_guaranteed": bool(np.mean(joint_fail) <= delta),
        "mean_true_recall": float(np.mean(rec_arr)),
        "std_true_recall": float(np.std(rec_arr)),
        "5th_percentile_recall": float(np.percentile(rec_arr, 5)),
        "min_true_recall": float(np.min(rec_arr)),
        "mean_true_precision": float(np.mean(prec_arr)),
        "std_true_precision": float(np.std(prec_arr)),
        "5th_percentile_precision": float(np.percentile(prec_arr, 5)),
        "min_true_precision": float(np.min(prec_arr)),
        "mean_total_oracle_rate": float(np.mean(rate_arr)),
        "std_total_oracle_rate": float(np.std(rate_arr)),
        "se_total_oracle_rate": float(np.std(rate_arr) / np.sqrt(len(results))),
        "5th_percentile_oracle_rate": float(np.percentile(rate_arr, 5)),
        "25th_percentile_oracle_rate": float(np.percentile(rate_arr, 25)),
        "75th_percentile_oracle_rate": float(np.percentile(rate_arr, 75)),
        "95th_percentile_oracle_rate": float(np.percentile(rate_arr, 95)),
        "mean_total_oracle_calls": float(np.mean(calls_arr)),
        "ideal_oracle_call_rate": float(ideal_oracle_rate),
        "raw_recalls": rec_arr.tolist(),
        "raw_precisions": prec_arr.tolist(),
        "raw_total_oracle_rates": rate_arr.tolist(),
    }

    tau_poses = [r.tau_pos for r in results if r.tau_pos is not None]
    tau_negs = [r.tau_neg for r in results if r.tau_neg is not None]
    if tau_poses:
        summary["mean_tau_pos"] = float(np.mean(tau_poses))
    if tau_negs:
        summary["mean_tau_neg"] = float(np.mean(tau_negs))

    valid_flags = [
        r.is_asymptotically_valid for r in results
        if r.is_asymptotically_valid is not None
    ]
    if valid_flags:
        summary["asymptotic_validity_rate"] = float(np.mean(valid_flags))

    return summary


def run_evaluation_suite(
    scenario: BaseScenario,
    runners: list[BaseFilterRunner],
    num_trials: int,
    pop_size: int | None,
    gamma_R: float,
    gamma_P: float,
    delta: float,
    seed: int,
    exp_name: str = "comparative",
) -> list[dict[str, Any]]:
    """
    Executes paired Monte Carlo trials across configured runners on identical population
    instances.
    """
    import time

    ss = np.random.SeedSequence(seed)
    pop_ss, *trial_seeds = ss.spawn(1 + num_trials)

    pop_rng = np.random.default_rng(pop_ss)
    pop_scores, pop_oracle = scenario.generate_population(
        pop_size,
        rng=pop_rng,
        gamma_P=gamma_P,
        gamma_R=gamma_R,
    )
    actual_pop_size = len(pop_scores)
    total_pop_positives = np.sum(pop_oracle)

    if total_pop_positives == 0:
        raise ValueError(f"Scenario '{scenario.name}' generated 0 positives.")

    ideal_oracle_call_rate = scenario.compute_ideal_oracle_call_rate(
        pop_scores, pop_oracle, gamma_R, gamma_P
    )

    runner_results: dict[str, list[TrialResult]] = {r.name: [] for r in runners}

    start_suite_time = time.time()
    print(
        f"\n[{scenario.name.upper()}] Starting {num_trials} trials across "
        f"{len(runners)} runner(s)...",
        flush=True,
    )

    log_freq = max(1, num_trials // 10)

    for trial_idx, trial_seed in enumerate(trial_seeds):

        for runner in runners:
            runner_rng = np.random.default_rng(trial_seed)

            res = runner.run_trial(
                scores=pop_scores,
                labels=pop_oracle,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                rng=runner_rng,
            )

            runner_results[runner.name].append(res)

        if (trial_idx + 1) % log_freq == 0 or (trial_idx + 1) == num_trials:
            elapsed = time.time() - start_suite_time
            rate = (trial_idx + 1) / elapsed
            remaining = (num_trials - (trial_idx + 1)) / rate if rate > 0 else 0.0
            print(
                f"  [{scenario.name.upper()}] Trial {trial_idx + 1}/{num_trials} "
                f"complete ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)",
                flush=True,
            )

    summary_results = []
    for runner in runners:
        summary = summarize_runner_trials(
            runner=runner,
            results=runner_results[runner.name],
            scenario=scenario,
            exp_name=exp_name,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            pop_size=actual_pop_size,
            ideal_oracle_rate=ideal_oracle_call_rate,
        )
        summary_results.append(summary)

    return summary_results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Prints a formatted comparison table for evaluation results."""
    if not results:
        return
    scen = results[0]["scenario"]
    print("\n" + "=" * 90, flush=True)
    print(
        f" SCENARIO: {scen.upper()} (Target gamma_R={results[0]['gamma_R']:.2f}, "
        f"gamma_P={results[0]['gamma_P']:.2f}, "
        f"delta={results[0]['promised_delta']:.2f})",
        flush=True
    )
    print("=" * 90, flush=True)
    fmt_header = "{:<32} | {:<12} | {:<12} | {:<12} | {:<16}"
    fmt_row = "{:<32} | {:<12.3f} | {:<12.3f} | {:<12.3f} | {:<15.2f}%"
    print(
        fmt_header.format("Method", "Joint Fail", "Rec Fail", "Prec Fail",
        "Mean Oracle Rate"),
        flush=True
    )
    print("-" * 90, flush=True)
    for r in results:
        print(fmt_row.format(
            r["runner_params"]["name"],
            r["joint_failure_rate"],
            r["recall_failure_rate"],
            r["precision_failure_rate"],
            r["mean_total_oracle_rate"] * 100,
        ), flush=True)
    print(
        f" Ideal Optimal Oracle Call Rate: "
        f"{results[0]['ideal_oracle_call_rate'] * 100:.2f}%",
        flush=True
    )
    print("=" * 90, flush=True)

