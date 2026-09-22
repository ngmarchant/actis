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

import hashlib
import json
import math
import random
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Sequence

import lotus
import numpy as np
from BARGAIN.models.AbstractModels import Oracle, Proxy
from BARGAIN.process.BARGAIN_PR import BARGAIN_PR
from lotus.sem_ops.cascade_utils import (
    importance_sampling,
    learn_cascade_thresholds,
)
from lotus.types import CascadeArgs
from numpy.typing import ArrayLike, NDArray

from actis import ACTIS, ProposalMethod, compute_pr_proposal
from actis.sampler import PopulationSampler
from actis.threshold_grid import quantile_power_law_grid
from actis.tuner import PriorAndTargetVar, compute_prior_and_target_var
from experiments.scenarios import (
    BaseScenario,
    Population,
    compute_ideal_oracle_call_rate,
)

_SCALEDOC_DIR = (
    Path(__file__).resolve().parent.parent / "externals" / "ScaleDoc" / "src"
)
_HAS_SCALEDOC = False
if _SCALEDOC_DIR.exists():
    if str(_SCALEDOC_DIR) not in sys.path:
        sys.path.insert(0, str(_SCALEDOC_DIR))
    try:
        from cascade import (  # ty: ignore[unresolved-import]
            calibrate_sampling,
            select_sim_filterB,
            smooth_distr,
        )

        _HAS_SCALEDOC = True
    except ImportError:
        _HAS_SCALEDOC = False


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def compute_precision_recall(
    preds: BoolArray, labels: BoolArray
) -> tuple[float, float]:
    total_positives = np.sum(labels)
    tp = np.sum(preds & labels)
    fp = np.sum(preds & (~labels))
    recall = float(tp / total_positives) if total_positives > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    return precision, recall


@dataclass
class TrialResult:
    """Stores the evaluation metrics from a single Monte Carlo trial."""

    recall: float
    precision: float
    cost: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration_calls: int = 0
    deployment_calls: int = 0
    tau_pos: float | None = None
    tau_neg: float | None = None
    is_asymptotically_valid: bool | None = None
    runtime: float | None = None


@dataclass(kw_only=True)
class BaseFilterRunner(ABC):
    """Abstract base class for semantic filtering methods."""

    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializes runner configuration to a JSON-compatible dictionary."""
        res = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, Path):
                res[f.name] = str(val)
            elif isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                res[f.name] = val
            else:
                res[f.name] = str(val)
        return res

    def config_hash(self, length: int = 8) -> str:
        """Deterministic hash of runner configuration."""
        d = self.to_dict()
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]

    @abstractmethod
    def run_trial(
        self,
        population: Population,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        """
        Executes filtering over a dataset for a single trial.

        Args:
            population: Population object containing proxy scores, oracle labels, and
                costs.
            gamma_R: Target recall constraint.
            gamma_P: Target precision constraint.
            delta: Allowed failure probability that the constraints are met.
            rng: Random number generator for trial-specific sampling.

        Returns:
            TrialResult containing realized recall, precision, and cost metrics.
        """
        pass


def evaluate_cascade_trial(
    population: Population,
    tau_pos: float,
    tau_neg: float,
    calib_indices: set[int] | list[int] | NDArray[np.int64],
) -> TrialResult:
    scores = population.scores
    labels = population.labels
    pop_size = len(population)
    calib_set = set(calib_indices)
    calib_calls = len(calib_set)
    calib_mask = np.zeros(pop_size, dtype=bool)
    if calib_calls > 0:
        calib_mask[list(calib_set)] = True

    # Items not sampled for calibration are routed to cascade thresholds
    cascade_pred = (scores >= tau_pos) | ((scores >= tau_neg) & labels)
    # Final predictions: true oracle label for calibration items, cascade prediction for
    # the rest
    preds = np.where(calib_mask, labels, cascade_pred)
    precision, recall = compute_precision_recall(preds, labels)

    oracle_sent = (scores >= tau_neg) & (scores < tau_pos)
    dep_calls = int(np.sum(oracle_sent & (~calib_mask)))
    oracle_queried_mask = calib_mask | oracle_sent

    cost = population.compute_trial_costs(oracle_queried_mask)

    return TrialResult(
        recall=recall,
        precision=precision,
        cost=cost,
        calibration_calls=calib_calls,
        deployment_calls=dep_calls,
        tau_pos=tau_pos,
        tau_neg=tau_neg,
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

    alpha: float | None = 0.0
    """Defensive mixing weight for importance sampling proposal distribution. If None
    or 0.0, no defensive mixing is applied."""

    proposal_method: ProposalMethod = "snr_balanced"
    """Method to use when computing the importance sampling proposal distribution."""

    tail_tolerance: float = 0.10
    """Dimensionless tail uncertainty tolerance for prevalence-adaptive clipping of
    importance sampling proposal distribution."""

    power_law_quantiles: bool = True
    """Whether to use power-law quantiles for the upper threshold grid when
    `num_thresholds_upper` is specified."""

    conf_seq: Literal["finite", "asymptotic"] = "asymptotic"
    """Confidence sequence type to use."""

    v_0: float | PriorAndTargetVar | Literal["auto"] = "auto"
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

    enable_asymptotic_protection: bool = True
    """Whether to enable heuristic protection for anytime-valid FWER control when
    operating in the non-asymptotic regime."""

    variance_ratio_bound: float = 0.05
    """Bound on the ratio of the variance of the test statistic to the variance of the
    null distribution. This is used to control the false discovery rate."""

    def __post_init__(self):
        if self.name == "":
            self.name = "actis_adaptive" if self.adaptive else "actis_static"

    def run_trial(
        self,
        population: Population,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        scores = population.scores
        labels = population.labels
        pop_size = len(population)

        # Resolve auto initial_sample_size
        if self.initial_sample_size == "auto":
            init_sample_size = min(
                pop_size, max(1, min(1000, max(50, int(np.ceil(0.10 * pop_size)))))
            )
        else:
            init_sample_size = min(pop_size, int(self.initial_sample_size))

        max_sample_size: int | None = None
        if isinstance(self.max_sample_size, float) and (
            0.0 < self.max_sample_size <= 1.0
        ):
            max_sample_size = max(1, int(self.max_sample_size * pop_size))
        elif isinstance(self.max_sample_size, int) and self.max_sample_size >= 1:
            max_sample_size = min(self.max_sample_size, pop_size)
        elif self.max_sample_size is None and self.sampling_method == "wor":
            max_sample_size = pop_size

        num_thresholds = min(self.num_thresholds, pop_size)
        thresholds = np.quantile(scores, np.linspace(0, 1, num_thresholds))
        thresholds = np.unique(thresholds)
        thresholds_upper = None
        if self.num_thresholds_upper is not None:
            num_thresholds_upper = min(self.num_thresholds_upper, pop_size)
            if self.power_law_quantiles:
                thresholds_upper = quantile_power_law_grid(
                    scores, num_thresholds=num_thresholds_upper
                )
            else:
                thresholds_upper = np.quantile(
                    scores, np.linspace(0, 1, num_thresholds_upper)
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
                method=self.proposal_method,
                tail_tolerance=self.tail_tolerance,
            )

            weights = 1.0 / (q * pop_size)

            if self.conf_seq == "finite":
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

                max_weight_ge_upper = None
                if thresholds_upper is not None:
                    max_weight_ge_upper = np.array(
                        [
                            weights[scores >= tau].max()
                            if np.any(scores >= tau)
                            else float(weights.max())
                            for tau in thresholds_upper
                        ]
                    )

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
            v_0 = compute_prior_and_target_var(
                scores=scores,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                thresholds=thresholds,
                thresholds_upper=thresholds_upper,
                weights=weights,
            )
        elif self.v_0 in ("global", "global_calibrated"):
            v_0 = compute_prior_and_target_var(
                scores=scores,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                thresholds=None,
                weights=weights,
            )
        else:
            v_0 = self.v_0

        tuner = ACTIS(
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size_param,
            horizon=None if self.adaptive else init_sample_size,
            conf_seq=self.conf_seq,
            v_0=v_0,
            min_positives=self.min_positives,
            p_floor=self.p_floor,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
            enable_asymptotic_protection=self.enable_asymptotic_protection,
            variance_ratio_bound=self.variance_ratio_bound,
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
            population=population,
            tau_pos=calib_res.tau_pos,
            tau_neg=calib_res.tau_neg,
            calib_indices=tuner.seen_indices,
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
        population: Population,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        scores = population.scores
        labels = population.labels

        cascade_args = CascadeArgs(
            recall_target=gamma_R,
            precision_target=gamma_P,
            failure_probability=delta,
            sampling_percentage=self.sampling_percentage,
            cascade_IS_max_sample_range=len(population),
            cascade_IS_random_seed=int(rng.integers(0, 2**31 - 1)),
        )

        sample_idx, weights = importance_sampling(
            proxy_scores=scores.tolist(),
            cascade_args=cascade_args,
        )

        s_scores = scores[sample_idx].tolist()
        s_oracle = labels[sample_idx].tolist()
        s_weights = weights[sample_idx]

        try:
            (tau_pos, tau_neg), _ = learn_cascade_thresholds(
                proxy_scores=s_scores,
                oracle_outputs=s_oracle,
                sample_correction_factors=s_weights,
                cascade_args=cascade_args,
            )
        except Exception as e:
            lotus.logger.error(f"Legacy LOTUS calibration error: {e}")
            tau_pos, tau_neg = 1.0, 0.0

        return evaluate_cascade_trial(
            population=population,
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
        self, indxs: list[int], data_records: list[Any]
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
        self, data_records: list[Any], indxs: list[int] | None = None
    ) -> np.ndarray:
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
        population: Population,
        gamma_R: float,
        gamma_P: float,
        delta: float,
        rng: np.random.Generator,
    ) -> TrialResult:
        if gamma_P != gamma_R:
            raise ValueError("BargainPRRunner only supports `gamma_P == gamma_R`.")

        scores = population.scores
        labels = population.labels
        pop_size = len(population)

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

        preds = np.zeros(pop_size, dtype=bool)
        if len(pred_positives) > 0:
            preds[pred_positives] = True

        queried_indices = oracle_model.queried_indices
        oracle_queried_mask = np.zeros(pop_size, dtype=bool)
        if len(queried_indices) > 0:
            q_arr = np.fromiter(queried_indices, dtype=int)
            preds[q_arr] = labels[q_arr]
            oracle_queried_mask[q_arr] = True

        precision, recall = compute_precision_recall(preds, labels)

        cost = population.compute_trial_costs(oracle_queried_mask)

        return TrialResult(
            recall=recall,
            precision=precision,
            cost=cost,
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
            population: Population,
            gamma_R: float,
            gamma_P: float,
            delta: float,
            rng: np.random.Generator,
        ) -> TrialResult:
            scores = population.scores
            labels = population.labels
            pop_size = len(population)

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
                pos_ = samples[labels[samples]]
                neg_ = samples[~labels[samples]]

                pos_cos_sample = scores[pos_] if pos_.shape[0] > 0 else np.array([])
                neg_cos_sample = scores[neg_] if neg_.shape[0] > 0 else np.array([])
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
                population=population,
                tau_pos=tau_pos,
                tau_neg=tau_neg,
                calib_indices=calib_samples,
            )


def summarize_runner_trials(
    runner: BaseFilterRunner,
    results: list[TrialResult],
    scenario: BaseScenario,
    gamma_R: float,
    gamma_P: float,
    delta: float,
    pop_size: int,
    ideal_oracle_rate: float,
    include_raw: bool = True,
) -> dict[str, Any]:
    rec_arr = np.array([r.recall for r in results])
    prec_arr = np.array([r.precision for r in results])
    rec_fail = rec_arr < gamma_R
    prec_fail = prec_arr < gamma_P
    joint_fail = rec_fail | prec_fail

    def metric_dict(vals: ArrayLike, include_raw: bool) -> dict[str, Any]:
        vals = np.asarray(vals)
        d = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "se": float(np.std(vals) / np.sqrt(len(vals))),
            "max": float(np.max(vals)),
            "min": float(np.min(vals)),
        }
        if include_raw:
            d["raw"] = vals.tolist()
        return d

    summary: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "scenario": scenario.to_dict(),
        "runner": runner.to_dict(),
        "num_trials": len(results),
        "pop_size": pop_size,
        "gamma_R": gamma_R,
        "gamma_P": gamma_P,
        "promised_delta": delta,
        "joint_failure": metric_dict(joint_fail, include_raw),
        "recall_failure": metric_dict(rec_fail, include_raw),
        "precision_failure": metric_dict(prec_fail, include_raw),
        "recall": metric_dict(rec_arr, include_raw),
        "precision": metric_dict(prec_arr, include_raw),
        "ideal_oracle_call_rate": float(ideal_oracle_rate),
    }

    # Hierarchical cost summary
    summary["cost"] = {"oracle": {}, "proxy": {}}
    if results and results[0].cost:
        for model_role in ("oracle", "proxy"):
            metric_keys = results[0].cost.get(model_role, {}).keys()
            for k in metric_keys:
                vals = np.array(
                    [r.cost[model_role].get(k, 0.0) for r in results],
                    dtype=np.float64,
                )
                summary["cost"][model_role][k] = metric_dict(vals, include_raw)

    tau_poses = [r.tau_pos for r in results if r.tau_pos is not None]
    tau_negs = [r.tau_neg for r in results if r.tau_neg is not None]
    if tau_poses:
        summary["tau_pos"] = metric_dict(tau_poses, False)
    if tau_negs:
        summary["tau_neg"] = metric_dict(tau_negs, False)

    valid_flags = [
        r.is_asymptotically_valid
        for r in results
        if r.is_asymptotically_valid is not None
    ]
    if valid_flags:
        summary["asymptotic_validity"] = metric_dict(valid_flags, False)

    runtimes = [r.runtime for r in results if r.runtime is not None]
    if runtimes:
        summary["runtime"] = metric_dict(runtimes, include_raw)

    return summary


def run_evaluation_suite(
    scenario: BaseScenario,
    runners: Sequence[BaseFilterRunner],
    num_trials: int,
    pop_size: int | None,
    gamma_R: float,
    gamma_P: float,
    delta: float,
    seed: int,
    include_raw: bool = True,
    results_dir: Path | str | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Executes paired Monte Carlo trials across configured runners on identical population
    instances.
    """
    if pop_size is not None:
        scenario.pop_size = pop_size
    if scenario.seed is None:
        scenario.seed = seed

    if gamma_R == gamma_P:
        filename = f"delta_{delta}_gamma_{gamma_R}_trials_{num_trials}.json"
    else:
        filename = (
            f"delta_{delta}_gammaR_{gamma_R}_gammaP_{gamma_P}_trials_{num_trials}.json"
        )

    cached_results: dict[str, dict[str, Any]] = {}
    runners_to_run: list[BaseFilterRunner] = []

    for runner in runners:
        if results_dir is not None:
            subpath = (
                scenario.results_subpath()
                if hasattr(scenario, "results_subpath")
                else Path(f"{scenario.name}_{scenario.config_hash()}")
            )
            target_dir = (
                Path(results_dir) / subpath / f"{runner.name}_{runner.config_hash()}"
            )
            target_file = target_dir / filename
        else:
            target_file = None

        if target_file is not None and skip_existing and target_file.exists():
            try:
                with open(target_file, "r") as f:
                    cached_results[runner.name] = json.load(f)
                print(
                    f"  [SKIP] {runner.name} on {scenario.name} "
                    f"(cached in {target_file})",
                    flush=True,
                )
            except Exception as e:
                warnings.warn(f"Failed to read cache {target_file}: {e}. Re-running.")
                runners_to_run.append(runner)
        else:
            runners_to_run.append(runner)

    if not runners_to_run:
        return [cached_results[r.name] for r in runners]

    ss = np.random.SeedSequence(seed)
    trial_seeds = ss.spawn(num_trials)

    pop = scenario.generate_population(
        gamma_P=gamma_P,
        gamma_R=gamma_R,
    )
    actual_pop_size = len(pop)
    total_pop_positives = np.sum(pop.labels)

    if total_pop_positives == 0:
        raise ValueError(f"Scenario '{scenario.name}' generated 0 positives.")

    ideal_oracle_call_rate = compute_ideal_oracle_call_rate(
        pop.scores, pop.labels, gamma_R, gamma_P
    )

    runner_results: dict[str, list[TrialResult]] = {r.name: [] for r in runners_to_run}

    start_suite_time = time.time()
    print(
        f"\n[{scenario.name.upper()}] Starting {num_trials} trials across "
        f"{len(runners_to_run)} runner(s)...",
        flush=True,
    )

    log_freq = max(1, num_trials // 10)

    for trial_idx, trial_seed in enumerate(trial_seeds):
        for runner in runners_to_run:
            runner_rng = np.random.default_rng(trial_seed)

            t0 = time.perf_counter()
            res = runner.run_trial(
                population=pop,
                gamma_R=gamma_R,
                gamma_P=gamma_P,
                delta=delta,
                rng=runner_rng,
            )
            res.runtime = time.perf_counter() - t0

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

    for runner in runners_to_run:
        summary = summarize_runner_trials(
            runner=runner,
            results=runner_results[runner.name],
            scenario=scenario,
            gamma_R=gamma_R,
            gamma_P=gamma_P,
            delta=delta,
            pop_size=actual_pop_size,
            ideal_oracle_rate=ideal_oracle_call_rate,
            include_raw=include_raw,
        )
        if results_dir is not None:
            subpath = (
                scenario.results_subpath()
                if hasattr(scenario, "results_subpath")
                else Path(f"{scenario.name}_{scenario.config_hash()}")
            )
            target_dir = (
                Path(results_dir) / subpath / f"{runner.name}_{runner.config_hash()}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / filename
            tmp_file = target_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(summary, f, indent=2)
            tmp_file.replace(target_file)
            print(
                f"  [SAVED] {runner.name} on {scenario.name} -> {target_file}",
                flush=True,
            )

        cached_results[runner.name] = summary

    return [cached_results[r.name] for r in runners]


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Prints a formatted comparison table for evaluation results."""
    if not results:
        return
    scen = results[0]["scenario"]
    scen_name = scen.get("name", "") if isinstance(scen, dict) else str(scen)
    print("\n" + "=" * 90, flush=True)
    print(
        f" SCENARIO: {scen_name.upper()} (Target gamma_R={results[0]['gamma_R']:.2f}, "
        f"gamma_P={results[0]['gamma_P']:.2f}, "
        f"delta={results[0]['promised_delta']:.2f})",
        flush=True,
    )
    print("=" * 90, flush=True)
    fmt_header = "{:<32} | {:<12} | {:<12} | {:<12} | {:<16}"
    fmt_row = "{:<32} | {:<12.3f} | {:<12.3f} | {:<12.3f} | {:<15.2f}%"
    print(
        fmt_header.format(
            "Method", "Joint Fail", "Rec Fail", "Prec Fail", "Mean Oracle Rate"
        ),
        flush=True,
    )
    print("-" * 90, flush=True)
    for r in results:
        runner_info = r.get("runner")
        runner_name = (
            runner_info.get("name", "unknown")
            if isinstance(runner_info, dict)
            else str(runner_info)
        )
        oracle_rate = (
            r.get("cost", {}).get("oracle", {}).get("call_rate", {}).get("mean", 0.0)
        )
        print(
            fmt_row.format(
                runner_name,
                r["joint_failure"]["mean"],
                r["recall_failure"]["mean"],
                r["precision_failure"]["mean"],
                oracle_rate * 100,
            ),
            flush=True,
        )
    print(
        f" Ideal Optimal Oracle Call Rate: "
        f"{results[0]['ideal_oracle_call_rate'] * 100:.2f}%",
        flush=True,
    )
    print("=" * 90, flush=True)
