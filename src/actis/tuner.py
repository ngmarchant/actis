import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .confseq import (
    BettingSupermartingale,
    GaussianMixtureSupermartingale,
    TestSupermartingale,
)
from .threshold_grid import validate_thresholds


@dataclass
class AsymptoticValidityDiagnostics:
    """Diagnostics for assessing whether the asymptotic confidence sequence guarantee is
    reliable."""
    v_t_recall: float
    v_t_precision: float
    v_0_recall: float
    v_0_precision: float
    num_positives: int
    min_positives: int
    num_samples: int

    def is_valid(self) -> bool:
        return len(self.diagnose()) == 0

    def diagnose(self) -> list[str]:
        """Returns a list of diagnostic messages for the asymptotic validity check."""
        issues = []
        ratio_recall = self.v_t_recall / max(self.v_0_recall, 1e-9)
        ratio_precision = self.v_t_precision / max(self.v_0_precision, 1e-9)
        if self.v_t_recall <= 0.0:
            issues.append(
                "Recall empirical variance accumulator V_t is 0.0 (too few positive "
                "events for CLT)."
            )
        elif ratio_recall < 0.1:
            issues.append(
                f"Recall empirical variance ratio V_t / v_0 is low ({ratio_recall:.3f} "
                f"< 0.1)."
            )

        if self.v_t_precision <= 0.0:
            issues.append(
                "Precision empirical variance accumulator V_t is 0.0."
            )
        elif ratio_precision < 0.1:
            issues.append(
                f"Precision empirical variance ratio V_t / v_0 is low "
                f"({ratio_precision:.3f} < 0.1)."
            )

        if self.num_positives < self.min_positives:
            issues.append(
                f"Observed positives ({self.num_positives}) below recommended threshold"
                f" ({self.min_positives})."
            )

        if self.num_samples < 30:
            issues.append(
                f"Total calibration sample size ({self.num_samples}) below recommended "
                f"minimum of 30."
            )

        return issues


@dataclass
class CascadeThresholds:
    """Stores calibrated thresholds and associated diagnostics."""
    tau_pos: float
    tau_neg: float
    num_samples: int
    predicted_oracle_rate: float | None = None
    is_asymptotically_valid: bool | None = None
    asymptotic_diagnostics: AsymptoticValidityDiagnostics | None = None

    @property
    def tau_upper(self) -> float:
        return self.tau_pos

    @property
    def tau_lower(self) -> float:
        return self.tau_neg


@dataclass
class StoppingDiagnostics:
    """Diagnostics for the stopping decision of the ACTIS tuner."""
    n_draws: int
    n_seen: int
    N_rem: int
    num_positives: int
    min_positives: int
    warmup_needed: bool
    r_curr: float
    r_opt: float
    marginal_batch_savings: float
    marginal_batch_cost: float
    threshold_savings: float
    tau_pos_curr: float
    tau_neg_curr: float
    tau_pos_opt: float
    tau_neg_opt: float


class CascadeConfSeqs(ABC):
    """Abstract collection of stateful confidence sequences for two-threshold cascade
    threshold calibration."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: NDArray[np.float64],
        thresholds_upper: NDArray[np.float64],
    ) -> None:
        r"""
        Args:
            gamma_P: Target precision.
            gamma_R: Target recall.
            delta: Probability that the tuning procedure fails to achieve the target
                precision and recall.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples. If `thresholds_upper` is None, the same grid
                is used for tuning the upper and lower thresholds. Otherwise, this grid
                is used for the lower threshold and `thresholds_upper` is used for the
                upper threshold.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
        """
        if gamma_P <= 0.0 or gamma_P >= 1.0:
            raise ValueError("Parameter `gamma_P` must be in (0, 1).")
        self.gamma_P = float(gamma_P)

        if gamma_R <= 0.0 or gamma_R >= 1.0:
            raise ValueError("Parameter `gamma_R` must be in (0, 1).")
        self.gamma_R = float(gamma_R)

        if delta <= 0.0 or delta >= 1.0:
            raise ValueError("Parameter `delta` must be in (0, 1).")
        self.delta = float(delta)

        self.delta_R = self.delta / 2.0
        self.delta_P = self.delta / 2.0
        self.thresholds = thresholds
        self.thresholds_upper = thresholds_upper

        # Sample history
        self.scores: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.labels: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        self.weights: NDArray[np.float64] | None = None

        # Supermartingale caches
        self._recall_marts: dict[tuple[int, bool], TestSupermartingale] = {}
        self._precision_marts: dict[int, TestSupermartingale] = {}
        self._precision_opt_marts: dict[int, TestSupermartingale] = {}
        self._last_k_lower: int | None = None
        self._last_k_lower_opt: int | None = None

    @property
    def recall_wealth_threshold(self) -> float:
        return 1.0 / self.delta_R

    @property
    def precision_wealth_threshold(self) -> float:
        return len(self.thresholds) / self.delta_P

    def add_samples(
        self,
        scores: Sequence[float] | NDArray[np.float64],
        labels: Sequence[bool] | NDArray[np.bool_],
        weights: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> None:
        r"""Update the internal sample history

        Args:
            scores: Proxy scores in [0, 1] for the samples.
            labels: Binary (True/False) oracle outputs for the samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.
        """
        self.scores = np.concatenate([self.scores, scores])
        self.labels = np.concatenate([self.labels, labels])

        if self.weights is None:
            if weights is not None:
                if len(weights) != len(scores):
                    raise ValueError("`weights` must have the same length as `scores`.")
                self.weights = np.asarray(weights, dtype=np.float64)
        else:
            if weights is None:
                raise ValueError(
                    "If `weights` were previously provided, they must be provided "
                    "for all subsequent samples."
                )
            if len(weights) != len(scores):
                raise ValueError("`weights` must have the same length as `scores`.")
            self.weights = np.concatenate([self.weights, weights])

    def is_recall_satisfied(self, k: int) -> bool:
        """Conservative: rejects null hypothesis: mu <= 0 at level delta_R."""
        wealth = self._get_recall_mart(k, reverse=False).wealth()
        return wealth > self.recall_wealth_threshold

    def is_recall_plausible(self, k: int) -> bool:
        """Optimistic: fails to reject null hypothesis: mu >= 0 at level delta_R."""
        wealth = self._get_recall_mart(k, reverse=True).wealth()
        return wealth <= self.recall_wealth_threshold

    def is_precision_satisfied(self, k_upper: int, k_lower: int) -> bool:
        """Conservative: rejects null hypothesis: mu <= 0 at level
        delta_P / len(thresholds)."""
        wealth = self._get_precision_mart(k_upper, k_lower, reverse=False).wealth()
        return wealth > self.precision_wealth_threshold

    def is_precision_plausible(self, k_upper: int, k_lower: int) -> bool:
        """Optimistic: fails to reject null hypothesis: mu >= 0 at level
        delta_P / len(thresholds)."""
        wealth = self._get_precision_mart(k_upper, k_lower, reverse=True).wealth()
        return wealth <= self.precision_wealth_threshold

    def _get_unprocessed_samples(
        self, start_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64] | None]:
        """Returns the slice of historical samples from start_idx to the latest."""
        return (
            self.scores[start_idx:],
            self.labels[start_idx:],
            self.weights[start_idx:] if self.weights is not None else None,
        )

    def _get_recall_mart(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        """Returns an up-to-date recall test supermartingale for threshold index k."""
        key = (k, reverse)
        if key not in self._recall_marts:
            self._recall_marts[key] = self._create_recall_mart(k, reverse=reverse)
        mart = self._recall_marts[key]

        if mart.t < len(self.scores):
            scores, labels, weights = self._get_unprocessed_samples(mart.t)
            rvs = self._get_recall_rvs(k, scores, labels, weights)
            mart.update(rvs)

        return mart

    def _get_precision_mart(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        """Returns an up-to-date precision test supermartingale for threshold pair
        (k_upper, k_lower)."""
        if not reverse:
            if k_lower != self._last_k_lower:
                self._precision_marts.clear()
                self._last_k_lower = k_lower
            cache = self._precision_marts
        else:
            if k_lower != self._last_k_lower_opt:
                self._precision_opt_marts.clear()
                self._last_k_lower_opt = k_lower
            cache = self._precision_opt_marts
        if k_upper not in cache:
            cache[k_upper] = self._create_precision_mart(
                k_upper,
                k_lower,
                reverse=reverse
            )
        mart = cache[k_upper]

        if mart.t < len(self.scores):
            scores, labels, weights = self._get_unprocessed_samples(mart.t)
            rvs = self._get_precision_rvs(k_upper, k_lower, scores, labels, weights)
            mart.update(rvs)

        return mart

    @abstractmethod
    def _get_recall_rvs(
        self,
        k: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _create_recall_mart(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        pass

    @abstractmethod
    def _get_precision_rvs(
        self,
        k_upper: int,
        k_lower: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _create_precision_mart(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        pass

    def check_asymptotic_validity(
        self,
        k_upper: int,
        k_lower: int,
        num_samples: int,
        min_positives: int = 30,
    ) -> tuple[bool | None, AsymptoticValidityDiagnostics | None]:
        """Checks asymptotic validity of the confidence sequences for the selected
        thresholds.

        Default implementation for non-asymptotic confidence sequences returns
        (True, None). Subclasses implementing asymptotic sequences should override this
        method to inspect their empirical variance accumulators, prior variance
        parameters, and degrees of freedom.
        """
        return True, None


class FiniteSampleCascadeConfSeqs(CascadeConfSeqs):
    """Finite-sample betting supermartingale strategy for [0, 1]-bounded observations
    under uniform or importance sampling."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: NDArray[np.float64],
        thresholds_upper: NDArray[np.float64],
        pop_size: int | None = None,
        horizon: int | None = None,
        max_weight_ge: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_lt: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None
    ) -> None:
        r"""
        Args:
            gamma_P: Target precision.
            gamma_R: Target recall.
            delta: Probability that the tuning procedure fails to achieve the target
                precision and recall.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples. If `thresholds_upper` is None, the same grid
                is used for tuning the upper and lower thresholds. Otherwise, this grid
                is used for the lower threshold and `thresholds_upper` is used for the
                upper threshold.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            pop_size: If sampling is done uniformly without replacement, this parameter
                specifies the population size. This parameter must be set to None if
                sampling is done with replacement.
            horizon: Fixed sample size for tuning the betting strategy. If None
                (default), uses anytime-valid predictable bets for adaptive stopping.
            max_weight_ge: If using importance sampling, it is necessary to provide
                upper bounds on the importance weight $w(x)$ over items $x$ in the
                dataset with proxy score $s(x)$ _above_ threshold $\tau$, for each
                $\tau$ in `thresholds`. Specifically, an array aligned with `thresholds`
                must be provided where the $k$-th entry is $\max_{x: s(x) >= \tau} w(x)$
                for $\tau$ equal to `thresholds[k]`. This argument is not required if
                sampling is done uniformly with/without replacement.
            max_weight_lt: Tighter confidence intervals can be obtained by providing
                upper bounds on the importance weight $w(x)$ over items $x$ in the
                dataset with proxy score $s(x)$ _below_ threshold $\tau$, for each
                $\tau$ in `thresholds`. Specifically, an array aligned with `thresholds`
                must be provided where the $k$-th entry is $\max_{x: s(x) < \tau} w(x)$
                for $\tau$ equal to `thresholds[k]`. If None, a looser bound using
                `max_weight_ge` will be used. This argument is not required if
                sampling is done uniformly with/without replacement.
            max_weight_ge_upper: If using importance sampling and `thresholds_upper` is
                provided, it is necessary to provide `max_weight_ge` where `thresholds`
                is replaced by `thresholds_upper`. This argument is not required if
                sampling is done uniformly with/without replacement.
        """
        super().__init__(gamma_P, gamma_R, delta, thresholds, thresholds_upper)
        if pop_size is not None and (pop_size <= 0 or pop_size != int(pop_size)):
            raise ValueError("Parameter `pop_size` must be a positive integer.")
        self.pop_size = pop_size

        if horizon is not None and (horizon <= 0 or horizon != int(horizon)):
            raise ValueError("Parameter `horizon` must be a positive integer.")
        self.horizon = horizon

        self.max_weight_ge = None
        self.max_weight_lt = None
        self.max_weight_ge_upper = None

        if max_weight_ge is not None:
            max_weight_ge = np.asarray(max_weight_ge, dtype=np.float64)
            if len(max_weight_ge) != len(self.thresholds):
                raise ValueError(
                    "`thresholds` and `max_weight_ge` must have the same length."
                )
            self.max_weight_ge = max_weight_ge

        if max_weight_lt is not None:
            max_weight_lt = np.asarray(max_weight_lt, dtype=np.float64)
            if len(max_weight_lt) != len(self.thresholds):
                raise ValueError(
                    "`thresholds` and `max_weight_lt` must have the same length."
                )
            self.max_weight_lt = max_weight_lt

        if max_weight_ge_upper is not None:
            max_weight_ge_upper = np.asarray(max_weight_ge_upper, dtype=np.float64)
            if len(max_weight_ge_upper) != len(self.thresholds_upper):
                raise ValueError(
                    "`thresholds_upper` and `max_weight_ge_upper` must have the same "
                    "length."
                )
            self.max_weight_ge_upper = max_weight_ge_upper
        else:
            self.max_weight_ge_upper = self.max_weight_ge

    def _get_recall_shift_width(
        self,
        k: int
    ) -> tuple[float, float]:
        max_weight_ge_k = (
            float(self.max_weight_ge[k]) if self.max_weight_ge is not None else 1.0
        )
        if self.max_weight_lt is not None:
            max_weight_lt_k = float(self.max_weight_lt[k])
        elif k == 0:
            max_weight_lt_k = 0.0
        else:
            first_max_weight_ge = (
                float(self.max_weight_ge[0]) if self.max_weight_ge is not None else 1.0
            )
            max_weight_lt_k = first_max_weight_ge

        shift_R = self.gamma_R * max_weight_lt_k
        width_R = shift_R + (1.0 - self.gamma_R) * max_weight_ge_k
        return shift_R, width_R

    def _get_precision_shift_width(
        self,
        k_upper: int,
        k_lower: int
    ) -> tuple[float, float]:
        max_weight_ge_upper_k = (
            float(self.max_weight_ge_upper[k_upper])
            if self.max_weight_ge_upper is not None
            else 1.0
        )
        max_weight_ge_lower = (
            float(self.max_weight_ge[k_lower])
            if self.max_weight_ge is not None
            else 1.0
        )
        shift_P = self.gamma_P * max_weight_ge_upper_k
        width_P = shift_P + (1.0 - self.gamma_P) * max_weight_ge_lower
        return shift_P, width_P

    def _create_recall_mart(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        shift_R, width_R = self._get_recall_shift_width(k)
        target_mean = (shift_R / width_R) if width_R > 0 else 0.0
        return BettingSupermartingale(
            m=target_mean,
            alpha=self.delta_R,
            population_size=self.pop_size,
            horizon=self.horizon,
            reverse=reverse,
        )

    def _get_recall_rvs(
        self,
        k: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        tau = self.thresholds[k]
        is_above_tau = (scores >= tau).astype(np.float64)
        unscaled_rvs = labels * (is_above_tau - self.gamma_R)
        if weights is not None:
            unscaled_rvs *= weights
        shift_R, width_R = self._get_recall_shift_width(k)
        if width_R > 0:
            return (unscaled_rvs + shift_R) / width_R
        return np.zeros_like(unscaled_rvs)

    def _create_precision_mart(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        shift_P, width_P = self._get_precision_shift_width(k_upper, k_lower)
        target_mean = (shift_P / width_P) if width_P > 0 else 0.0
        return BettingSupermartingale(
            m=target_mean,
            alpha=self.delta_P / len(self.thresholds),
            population_size=self.pop_size,
            horizon=self.horizon,
            reverse=reverse,
        )

    def _get_precision_rvs(
        self,
        k_upper: int,
        k_lower: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        tau_upper = self.thresholds_upper[k_upper]
        tau_lower = self.thresholds[k_lower]
        is_true_pos = labels & (scores >= tau_lower)
        is_false_pos = (~labels) & (scores >= tau_upper)
        unscaled_rvs = (
            (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos
        )
        if weights is not None:
            unscaled_rvs *= weights
        shift_P, width_P = self._get_precision_shift_width(k_upper, k_lower)
        if width_P > 0:
            return (unscaled_rvs + shift_P) / width_P
        return np.zeros_like(unscaled_rvs)


def compute_prior_var(
    scores: ArrayLike,
    gamma_R: float,
    gamma_P: float,
    delta: float = 0.05,
    thresholds: Sequence[float] | NDArray[np.float64] | None = None,
    thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
    weights: ArrayLike | None = None,
    n_0: float = 1.0,
) -> float | tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Computes prior variance parameters $v_0$ for the asymptotic Gaussian mixture
    supermartingales used in ACTIS cascade tuning.

    Args:
        scores: Proxy scores in $[0, 1]$ across the population dataset.
        gamma_R: Target recall constraint in $(0, 1)$.
        gamma_P: Target precision constraint in $(0, 1)$.
        delta: Total allowed family-wise error rate in $(0, 1)$ (default: 0.05).
        thresholds: Candidate grid of thresholds in $[0, 1]$. If `thresholds_upper` is
            None, this function assumes the same grid is used for tuning the upper and
            lower thresholds. Otherwise, this function assumes the grid is used for the
            lower threshold and `thresholds_upper` is used for the upper threshold. If
            provided, returns a tuple of tailored prior variance arrays
            `(v_0_recall, v_0_precision)`.
        thresholds_upper: Optional candidate grid for tuning the upper threshold.
            Defaults to `thresholds`.
        weights: Optional importance weights $w(x) = 1 / (N \cdot q(x))$.
        n_0: Effective prior sample weight (default: 1.0 pseudo-observation).

    Returns:
        If `thresholds` is provided, a tuple of two arrays:
            - 1D array of shape `(num_lower,)` containing $v_0$ for each candidate lower
              threshold.
            - 2D array of shape `(num_upper, num_lower)` containing $v_0$ for each
              (upper, lower) threshold pair.
        If `thresholds` is None, returns a scalar float $v_0$ suitable for a single
        global test.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Parameter `scores` must be a non-empty 1D array.")
    N = len(scores)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
    else:
        weights = np.ones(N, dtype=np.float64)

    # Split family-wise error rate delta equally between recall and precision
    delta_R = delta / 2.0
    delta_P = delta / 2.0

    log_delta_R_inv = math.log(1.0 / delta_R)
    max_weight = float(np.max(weights))

    # Fallback when no threshold grid is specified
    if thresholds is None:
        b = max(1.0 - gamma_R, 1.0 - gamma_P) * max_weight
        v_0_floor = (2.0 * b**2) / log_delta_R_inv
        weighted_pos_mass = float(np.mean(scores * weights))
        sigma2 = max(weighted_pos_mass, 1.0 / N) * gamma_R * (1.0 - gamma_R)
        v_0_target = n_0 * sigma2
        return float(max(v_0_floor, v_0_target))

    thresholds = np.asarray(thresholds, dtype=np.float64)
    M_lower = len(thresholds)

    # Bonferroni correction for precision across candidate lower thresholds
    delta_P_corrected = delta_P / max(1, M_lower)
    log_delta_P_inv = math.log(1.0 / delta_P_corrected)

    if thresholds_upper is not None:
        thresholds_upper = np.asarray(thresholds_upper, dtype=np.float64)
    else:
        thresholds_upper = thresholds

    # Sort items by proxy score to enable O(1) prefix/suffix query per threshold
    order = np.argsort(scores)
    scores_sorted = scores[order]
    weights_sorted = weights[order]

    # Precompute weighted label probabilities using proxy scores as surrogates
    weighted_pos_prob = scores_sorted * weights_sorted
    weighted_neg_prob = (1.0 - scores_sorted) * weights_sorted

    # Suffix accumulations for items with s_i >= tau (reverse cumsum):
    # Appending terminal values ensures valid indices when thresholds exceed max score.
    suffix_max_weight = np.append(
        np.maximum.accumulate(weights_sorted[::-1])[::-1],
        max_weight
    )
    suffix_sum_pos_prob = np.append(
        np.cumsum(weighted_pos_prob[::-1])[::-1],
        0.0
    )
    suffix_sum_neg_prob = np.append(
        np.cumsum(weighted_neg_prob[::-1])[::-1],
        0.0
    )

    # Total expected positive mass in the population
    total_pos_mass = suffix_sum_pos_prob[0] / N

    # -------------------------------------------------------------------------
    # 1. Recall prior variance per candidate lower threshold
    # -------------------------------------------------------------------------
    idx_lower = np.searchsorted(scores_sorted, thresholds, side="left")
    w_max_lower = suffix_max_weight[idx_lower]

    # Positive probability mass above and below each candidate lower threshold
    pos_mass_ge_lower = suffix_sum_pos_prob[idx_lower] / N
    pos_mass_lt_lower = np.maximum(0.0, total_pos_mass - pos_mass_ge_lower)

    # Safety floor for recall
    b_R = (1.0 - gamma_R) * w_max_lower
    v_0_floor_recall = (2.0 * b_R**2) / log_delta_R_inv

    # Estimated second moment of the recall margin increment
    sigma2_recall = (
        (1.0 - gamma_R)**2 * pos_mass_ge_lower + (gamma_R ** 2) * pos_mass_lt_lower
    )
    v_0_target_recall = n_0 * sigma2_recall
    v_0_recall = np.maximum(v_0_floor_recall, v_0_target_recall)

    # -------------------------------------------------------------------------
    # 2. Precision prior variance per (upper, lower) threshold pair
    # -------------------------------------------------------------------------
    # Safety floor for precision (governed by positive increments above each lower
    # threshold)
    b_P = (1.0 - gamma_P) * w_max_lower
    v_0_floor_precision = (2.0 * b_P**2) / log_delta_P_inv

    # Negative probability mass above each candidate upper threshold
    idx_upper = np.searchsorted(scores_sorted, thresholds_upper, side="left")
    neg_mass_ge_upper = suffix_sum_neg_prob[idx_upper] / N
    neg_var_upper = (gamma_P ** 2) * neg_mass_ge_upper
    pos_var_lower = (1.0 - gamma_P)**2 * pos_mass_ge_lower

    # 2D second moment matrix: sum of false-positive variance (from upper threshold)
    # and true-positive variance (from lower threshold) via NumPy broadcasting
    total_var = neg_var_upper[:, np.newaxis] + pos_var_lower[np.newaxis, :]
    v_0_target_precision = n_0 * total_var
    v_0_precision = np.maximum(v_0_floor_precision[np.newaxis, :], v_0_target_precision)

    return (v_0_recall, v_0_precision)


class AsymptoticCascadeConfSeqs(CascadeConfSeqs):
    """Asymptotic Gaussian mixture test supermartingale strategy (Howard et al. 2021).
    """

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: NDArray[np.float64],
        thresholds_upper: NDArray[np.float64],
        v_0: float | tuple[NDArray[np.float64], NDArray[np.float64]] = 0.01,
    ) -> None:
        r"""
        Args:
            gamma_P: Target precision.
            gamma_R: Target recall.
            delta: Probability that the tuning procedure fails to achieve the target
                precision and recall.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples. If `thresholds_upper` is None, the same grid
                is used for tuning the upper and lower thresholds. Otherwise, this grid
                is used for the lower threshold and `thresholds_upper` is used for the
                upper threshold.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            v_0: Prior variance for the Gaussian mixture supermartingale. Can be a
                scalar float (applied globally across all tests) or a tuple
                `(v_0_recall, v_0_precision)` containing per-threshold arrays.
                The `v_0_recall` array should have the same shape as `thresholds` and
                the `v_0_precision` array should have shape `(len(thresholds_upper),
                len(thresholds))`. Defaults to 0.01.
        """
        super().__init__(gamma_P, gamma_R, delta, thresholds, thresholds_upper)
        if isinstance(v_0, (tuple, list)):
            if len(v_0) != 2:
                raise ValueError(
                    "`v_0` must be a tuple of 2 arrays: (v_0_recall, v_0_precision)."
                )
            if len(v_0[0]) != len(thresholds):
                raise ValueError(
                    "`v_0[0]` must have the same length as `thresholds`."
                )
            self.v_0_recall = np.asarray(v_0[0], dtype=np.float64)

            if np.shape(v_0[1]) != (len(thresholds_upper), len(thresholds)):
                raise ValueError(
                    "`v_0[1]` must have shape `(len(thresholds_upper), "
                    "len(thresholds))`."
                )
            self.v_0_precision = np.asarray(v_0[1], dtype=np.float64)
        else:
            self.v_0_recall = v_0
            self.v_0_precision = v_0
        self.v_0 = v_0

    def _create_recall_mart(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        if isinstance(self.v_0_recall, np.ndarray):
            v_0 = float(self.v_0_recall[k])
        else:
            v_0 = float(self.v_0_recall)
        return GaussianMixtureSupermartingale(m=0.0, v_0=v_0, reverse=reverse)

    def _get_recall_rvs(
        self,
        k: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        tau = self.thresholds[k]
        is_above_tau = (scores >= tau).astype(np.float64)
        rvs = labels * (is_above_tau - self.gamma_R)
        if weights is not None:
            rvs *= weights
        return rvs

    def _create_precision_mart(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        if isinstance(self.v_0_precision, np.ndarray):
            v_0 = float(self.v_0_precision[k_upper, k_lower])
        else:
            v_0 = float(self.v_0_precision)
        return GaussianMixtureSupermartingale(m=0.0, v_0=v_0, reverse=reverse)

    def _get_precision_rvs(
        self,
        k_upper: int,
        k_lower: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        tau_upper = self.thresholds_upper[k_upper]
        tau_lower = self.thresholds[k_lower]
        is_true_pos = labels & (scores >= tau_lower)
        is_false_pos = (~labels) & (scores >= tau_upper)
        rvs = (
            (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos
        )
        if weights is not None:
            rvs *= weights
        return rvs

    def check_asymptotic_validity(
        self,
        k_upper: int,
        k_lower: int,
        num_samples: int,
        min_positives: int = 30,
    ) -> tuple[bool, AsymptoticValidityDiagnostics]:
        mart_R = self._get_recall_mart(k_lower, reverse=False)
        mart_P = self._get_precision_mart(k_upper, k_lower, reverse=False)

        assert isinstance(mart_R, GaussianMixtureSupermartingale)
        assert isinstance(mart_P, GaussianMixtureSupermartingale)

        diagnostics = AsymptoticValidityDiagnostics(
            v_t_recall=float(mart_R.v_t),
            v_t_precision=float(mart_P.v_t),
            v_0_recall=float(mart_R.v_0),
            v_0_precision=float(mart_P.v_0),
            num_positives=int(np.sum(self.labels)),
            min_positives=min_positives,
            num_samples=num_samples,
        )

        return diagnostics.is_valid(), diagnostics


class ACTIS:
    r"""Stateful anytime-valid threshold tuner for cascades using confidence sequences.
    """

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: Sequence[float] | NDArray[np.float64],
        thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
        pop_size: int | None = None,
        horizon: int | None = None,
        conf_seq: Literal["finite", "asymptotic"] = "finite",
        v_0: float | tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        min_positives: int | Literal["auto"] = "auto",
        p_floor: float | Literal["auto"] = "auto",
        max_weight_ge: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_lt: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None,
    ):
        r"""
        Args:
            gamma_P: Target precision.
            gamma_R: Target recall.
            delta: Probability that the tuning procedure fails to achieve the target
                precision and recall.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples. If `thresholds_upper` is None, the same grid
                is used for tuning the upper and lower thresholds. Otherwise, this grid
                is used for the lower threshold and `thresholds_upper` is used for the
                upper threshold.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            pop_size: If `conf_seq='finite'` and sampling is done uniformly without
                replacement, this parameter specifies the population size. This
                parameter must be set to None if sampling is done with replacement or if
                `conf_seq='asymptotic'`.
            horizon: If `conf_seq='finite'`, this parameter specifies the fixed sample
                size for tuning the betting strategy. If None (default), uses
                anytime-valid predictable bets for adaptive stopping.
            conf_seq: 'finite' (default) for finite-sample betting supermartingales or
                'asymptotic' for Gaussian mixture supermartingales.
            v_0: If `conf_seq='asymptotic'`, this parameter specifies the prior variance
                for the Gaussian mixture supermartingale. Can be a scalar float (applied
                globally across all tests) or a tuple `(v_0_recall, v_0_precision)`
                containing per-threshold arrays.
                The `v_0_recall` array should have the same shape as `thresholds` and
                the `v_0_precision` array should have shape `(len(thresholds_upper),
                len(thresholds))` or `(len(thresholds),)`. Defaults to 0.01.
            min_positives: Minimum number of positive oracle calls required for the
                asymptotic confidence sequence to be considered valid and for early
                stopping warmup. If "auto" (default), dynamically calibrated based on
                delta, pop_size, and proxy scores.
            p_floor: Assumed or estimated lower-bound positive prevalence for population
                scaling of `min_positives`. If "auto" (default), dynamically estimated
                from population proxy scores (discounted by 0.5 with floor 1e-4 and cap
                0.01). If a float, specifies the prevalence floor directly (e.g. 0.01
                or 0.001).
            max_weight_ge: If using importance sampling with `conf_seq='finite'`, it is
                necessary to provide upper bounds on the importance weight $w(x)$ over
                items $x$ in the dataset with proxy score $s(x)$ _above_ threshold
                $\tau$, for each $\tau$ in `thresholds`. Specifically, an array
                aligned with `thresholds` must be provided where the $k$-th entry is
                $\max_{x: s(x) >= \tau} w(x)$ for $\tau$ equal to `thresholds[k]`. This
                argument is not required for `conf_seq='asymptotic'` or if sampling is
                done uniformly with/without replacement.
            max_weight_lt: If using importance sampling with `conf_seq='finite'`,
                tighter confidence intervals can be obtained by providing upper bounds
                on the importance weight $w(x)$ over items $x$ in the dataset with proxy
                score $s(x)$ _below_ threshold $\tau$, for each $\tau$ in `thresholds`.
                Specifically, an array aligned with `thresholds` must be provided where
                the $k$-th entry is $\max_{x: s(x) < \tau} w(x)$ for $\tau$ equal to
                `thresholds[k]`. If None, a looser bound using `max_weight_ge` will be
                used. This argument is not required for `conf_seq='asymptotic'` or if
                sampling is done uniformly with/without replacement.
            max_weight_ge_upper: If using importance sampling with `conf_seq='finite'`
                and `thresholds_upper` is provided, it is necessary to provide
                `max_weight_ge` where `thresholds` is replaced by `thresholds_upper`.
                This argument is not required for `conf_seq='asymptotic'` or if
                sampling is done uniformly with/without replacement.
        """
        if pop_size is not None and (pop_size <= 0 or pop_size != int(pop_size)):
            raise ValueError("Parameter `pop_size` must be a positive integer.")
        self.pop_size = pop_size

        self.gamma_P = gamma_P
        self.gamma_R = gamma_R
        self.delta = delta
        self.conf_seq = conf_seq
        self.min_positives = min_positives
        if p_floor != "auto" and (p_floor <= 0.0 or p_floor >= 1.0):
            raise ValueError("Parameter `p_floor` must be in (0, 1) or 'auto'.")
        self.p_floor: float | Literal["auto"] = p_floor
        if v_0 is not None:
            if isinstance(v_0, (list, tuple)):
                if np.any(v_0[0] <= 0.0) or np.any(v_0[1] <= 0.0):
                    raise ValueError("Parameter `v_0` must be non-negative.")
            elif v_0 <= 0.0:
                raise ValueError("Parameter `v_0` must be non-negative.")
        self.v_0 = v_0 if v_0 is not None else 0.01

        self.thresholds = validate_thresholds(thresholds, "thresholds")
        self.thresholds_upper = (
            validate_thresholds(thresholds_upper, "thresholds_upper")
            if thresholds_upper is not None
            else self.thresholds
        )

        if self.conf_seq == "asymptotic":
            self.conf_seqs = AsymptoticCascadeConfSeqs(
                gamma_P=gamma_P,
                gamma_R=gamma_R,
                delta=delta,
                thresholds=self.thresholds,
                thresholds_upper=self.thresholds_upper,
                v_0=self.v_0,
            )
        elif self.conf_seq == "finite":
            self.conf_seqs = FiniteSampleCascadeConfSeqs(
                gamma_P=gamma_P,
                gamma_R=gamma_R,
                delta=delta,
                thresholds=self.thresholds,
                thresholds_upper=self.thresholds_upper,
                pop_size=pop_size,
                horizon=horizon,
                max_weight_ge=max_weight_ge,
                max_weight_lt=max_weight_lt,
                max_weight_ge_upper=max_weight_ge_upper
            )
        else:
            raise ValueError(f"Unknown `conf_seq` mode: {self.conf_seq}")

        self.seen_indices: set[int] = set()

        self.current_thresholds: CascadeThresholds = CascadeThresholds(
            tau_pos=1.0,
            tau_neg=0.0,
            num_samples=0,
            predicted_oracle_rate=None,
        )

    def _resolve_min_positives(
        self,
        N: int | None = None,
        scores: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> int:
        r"""Resolves the minimum number of positive oracle observations required
        for asymptotic validity and warmup.
        """
        if self.min_positives != "auto":
            return int(self.min_positives)

        delta_R = self.delta / 2.0
        cap = max(5, int(np.ceil(np.round(8.0 * math.log(1.0 / delta_R), 9))))

        pop_size = self.pop_size if self.pop_size is not None else N
        if pop_size is None:
            return cap

        if self.p_floor == "auto":
            if scores is not None and len(scores) > 0:
                scores = np.asarray(scores, dtype=np.float64)
                mean_score = np.mean(scores)
                eff_p = min(0.01, max(1e-4, 0.5 * mean_score))
            else:
                eff_p = 0.01
        else:
            eff_p = float(self.p_floor)

        return min(cap, max(5, int(np.ceil(np.round(eff_p * pop_size, 9)))))

    def add_samples(
        self,
        indices: Sequence[int] | NDArray[np.int_],
        scores: Sequence[float] | NDArray[np.float64],
        labels: Sequence[bool] | NDArray[np.bool_],
        weights: Sequence[float] | NDArray[np.float64] | None = None,
        population_scores: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> CascadeThresholds:
        r"""Ingests a batch of labeled samples, updates the test supermartingale wealth
        and returns updated CascadeThresholds

        Args:
            indices: Indices for the samples in the population dataset.
            scores: Proxy scores in [0, 1] for the samples.
            labels: Binary (True/False) oracle outputs for the samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.
            population_scores: Optional proxy scores in [0, 1] for the population
                dataset. These are used to estimate the expected oracle rate of the
                cascade. If None, the expected oracle rate is not computed.

        Returns:
            Updated CascadeThresholds with new thresholds and diagnostics.
        """

        if len(indices) != len(scores) or len(scores) != len(labels):
            raise ValueError(
                "`indices`, `scores`, and `labels` must all have the same length."
            )

        if self.pop_size is not None:
            unique_indices = set(indices)
            if len(unique_indices) < len(indices):
                raise ValueError(
                    "Duplicate indices encountered within the sample."
                )
            duplicates = unique_indices.intersection(self.seen_indices)
            if len(duplicates) > 0:
                raise ValueError(
                    f"Encountered {len(duplicates)} duplicate indices "
                    "previously seen in without-replacement sampling."
                )

        self.seen_indices.update(indices)
        self.conf_seqs.add_samples(scores, labels, weights=weights)

        tau_upper, tau_lower, k_upper, k_lower = self._tune_thresholds()

        predicted_oracle_rate = None
        if population_scores is not None:
            predicted_oracle_rate = self.predict_oracle_rate(
                population_scores,
                tau_upper=tau_upper,
                tau_lower=tau_lower,
            )

        min_pos = self._resolve_min_positives(
            N=len(population_scores) if population_scores is not None else None,
            scores=population_scores,
        )
        is_asymptotically_valid, asymptotic_diagnostics = (
            self.conf_seqs.check_asymptotic_validity(
                k_upper=k_upper,
                k_lower=k_lower,
                num_samples=len(self.seen_indices),
                min_positives=min_pos,
            )
        )

        self.current_thresholds = CascadeThresholds(
            tau_pos=tau_upper,
            tau_neg=tau_lower,
            num_samples=len(self.seen_indices),
            predicted_oracle_rate=predicted_oracle_rate,
            is_asymptotically_valid=is_asymptotically_valid,
            asymptotic_diagnostics=asymptotic_diagnostics,
        )

        return self.current_thresholds

    def _tune_thresholds(self) -> tuple[float, float, int, int]:
        r"""Evaluates the test supermartingales across candidate threshold grids to find
        the optimal pair of thresholds."""
        # Forward sequential scan stopping at first failure of recall target
        k_lower = 0
        tau_lower = float(self.thresholds[0])

        for k, tau in enumerate(self.thresholds):
            if self.conf_seqs.is_recall_satisfied(k):
                k_lower = k
                tau_lower = float(tau)
            else:
                # Stop at first failure
                break

        # Backward sequential scan stopping at first failure of precision target
        k_upper = len(self.thresholds_upper) - 1
        tau_upper = float(self.thresholds_upper[-1])

        for k in range(len(self.thresholds_upper) - 1, -1, -1):
            tau_k = float(self.thresholds_upper[k])
            if tau_k < tau_lower:
                break

            if self.conf_seqs.is_precision_satisfied(k, k_lower):
                k_upper = k
                tau_upper = tau_k
            else:
                # Stop at first failure
                break

        return tau_upper, tau_lower, k_upper, k_lower

    def _compute_optimistic_thresholds(self) -> tuple[float, float]:
        r"""Computes the best-case (optimistic) candidate thresholds within the anytime
        confidence bounds on recall and precision margins via binary search."""
        if len(self.seen_indices) == 0:
            return float(self.thresholds_upper[-1]), float(self.thresholds[0])

        # Optimistic recall threshold via binary search
        low_neg, high_neg = 0, len(self.thresholds) - 1
        opt_neg_idx = 0
        while low_neg <= high_neg:
            mid_k = (low_neg + high_neg) // 2
            if self.conf_seqs.is_recall_plausible(mid_k):
                opt_neg_idx = mid_k
                low_neg = mid_k + 1
            else:
                high_neg = mid_k - 1

        tau_neg_opt = float(self.thresholds[opt_neg_idx])

        # Optimistic precision threshold via binary search

        start_k = int(np.searchsorted(self.thresholds_upper, tau_neg_opt, side="left"))
        low_pos, high_pos = start_k, len(self.thresholds_upper) - 1
        opt_pos_idx = len(self.thresholds_upper) - 1

        while low_pos <= high_pos:
            mid_k = (low_pos + high_pos) // 2
            if self.conf_seqs.is_precision_plausible(mid_k, opt_neg_idx):
                opt_pos_idx = mid_k
                high_pos = mid_k - 1
            else:
                low_pos = mid_k + 1

        tau_pos_opt = float(self.thresholds_upper[opt_pos_idx])
        return tau_pos_opt, tau_neg_opt

    compute_optimistic_thresholds = _compute_optimistic_thresholds

    def predict_oracle_rate(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        tau_upper: float | None = None,
        tau_lower: float | None = None,
    ) -> float:
        r"""Predicts the oracle call rate over the full population given thresholds.

        Args:
            population_scores: Proxy scores in [0, 1] for the population dataset.
            tau_upper: Optional upper threshold. If None, uses the current internal
                upper threshold.
            tau_lower: Optional lower threshold. If None, uses the current internal
                lower threshold.

        Returns:
            The predicted oracle call rate as a float between 0 and 1.
        """
        if tau_upper is None:
            tau_upper = self.current_thresholds.tau_pos
        if tau_lower is None:
            tau_lower = self.current_thresholds.tau_neg

        if tau_lower > tau_upper:
            return 1.0

        population_scores = np.asarray(population_scores, dtype=np.float64)
        unsure = (population_scores >= tau_lower) & (population_scores < tau_upper)
        return float(np.mean(unsure))

    def should_continue_sampling(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        batch_size: int,
        max_sample_size: int | None = None,
        min_expected_savings: float = 0.0
    ) -> tuple[bool, StoppingDiagnostics]:
        r"""Determines whether continuing to sample is justified based on the
        Uncertainty Gap between currently accepted conservative thresholds
        and the best-case (optimistic) thresholds within the confidence bounds.

        Args:
            population_scores: Proxy scores in [0, 1] for the population dataset.
            batch_size: Number of samples to acquire in the next batch.
            max_sample_size: Optional maximum number of samples to acquire. If None,
                defaults to the population size (if known) or unlimited.
            min_expected_savings: Minimum expected savings threshold for continuing
                sampling. If the expected savings from acquiring the next batch is
                below this threshold, sampling will stop.

        Returns:
            A tuple containing:
                - A boolean indicating whether to continue sampling.
                - A StoppingDiagnostics object containing detailed information about
                  the stopping decision.
        """
        population_scores = np.asarray(population_scores, dtype=np.float64)
        N = len(population_scores) if self.pop_size is None else self.pop_size
        n_draws = len(self.conf_seqs.scores)
        n_seen = len(self.seen_indices)
        N_rem = max(0, N - n_seen)

        if max_sample_size is None:
            max_sample_size = N

        min_positives = (
            self._resolve_min_positives(N=N, scores=population_scores)
            if self.min_positives == "auto"
            else self.min_positives
        )

        num_positives = int(np.sum(self.conf_seqs.labels))
        if self.conf_seq == "asymptotic":
            asymp_valid = (
                self.current_thresholds.is_asymptotically_valid
                if self.current_thresholds.is_asymptotically_valid is not None
                else False
            )
            warmup_needed = (not asymp_valid) or (num_positives < min_positives)
        else:
            warmup_needed = num_positives < min_positives

        if N_rem == 0 or n_draws >= max_sample_size:
            diagnostics = StoppingDiagnostics(
                n_draws=n_draws,
                n_seen=n_seen,
                N_rem=N_rem,
                num_positives=num_positives,
                min_positives=min_positives,
                warmup_needed=warmup_needed,
                r_curr=self.current_thresholds.predicted_oracle_rate or 0.0,
                r_opt=self.current_thresholds.predicted_oracle_rate or 0.0,
                marginal_batch_savings=0.0,
                marginal_batch_cost=0.0,
                threshold_savings=0.0,
                tau_pos_curr=self.current_thresholds.tau_pos,
                tau_neg_curr=self.current_thresholds.tau_neg,
                tau_pos_opt=self.current_thresholds.tau_pos,
                tau_neg_opt=self.current_thresholds.tau_neg,
            )
            return False, diagnostics

        # Conservative thresholds and oracle rate
        tau_pos_curr = self.current_thresholds.tau_pos
        tau_neg_curr = self.current_thresholds.tau_neg
        r_curr = self.predict_oracle_rate(
            population_scores,
            tau_upper=tau_pos_curr,
            tau_lower=tau_neg_curr,
        )

        # Optimistic thresholds (within confidence bounds) and oracle rate
        tau_pos_opt, tau_neg_opt = self._compute_optimistic_thresholds()
        r_opt = self.predict_oracle_rate(
            population_scores,
            tau_upper=tau_pos_opt,
            tau_lower=tau_neg_opt,
        )

        # Marginal Value of Information for the next batch
        marginal_batch_savings = (
            (batch_size / (2.0 * max(1.0, float(n_draws))))
            * max(0.0, r_curr - r_opt)
            * N_rem
        )

        # Marginal cost of acquiring the next batch
        marginal_batch_cost = batch_size * (1.0 - r_curr)
        threshold_savings = max(marginal_batch_cost, min_expected_savings)

        should_continue = (
            (marginal_batch_savings > threshold_savings) or warmup_needed
        ) and (n_draws < max_sample_size)

        diagnostics = StoppingDiagnostics(
            n_draws=n_draws,
            n_seen=n_seen,
            N_rem=N_rem,
            num_positives=num_positives,
            min_positives=min_positives,
            warmup_needed=warmup_needed,
            r_curr=r_curr,
            r_opt=r_opt,
            marginal_batch_savings=marginal_batch_savings,
            marginal_batch_cost=marginal_batch_cost,
            threshold_savings=threshold_savings,
            tau_pos_curr=tau_pos_curr,
            tau_neg_curr=tau_neg_curr,
            tau_pos_opt=tau_pos_opt,
            tau_neg_opt=tau_neg_opt,
        )
        return should_continue, diagnostics
