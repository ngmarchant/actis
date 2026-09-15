import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from .confseq import (
    BettingSupermartingale,
    GaussianMixtureSupermartingale,
    TestSupermartingale,
)
from .threshold_grid import validate_thresholds


@dataclass
class AsymptoticValidityDiagnostics:
    """Stores diagnostics for assessing whether the asymptotic confidence sequence
    guarantee is reliable."""
    v_t: float
    v_0: float
    rho_max: float
    num_true_positives: int
    num_false_negatives_or_positives: int
    null_failure_prob: float
    metric: str = field(default="Recall")
    thresholds: tuple[float, ...] | None = None
    variance_ratio_bound: float = 0.05
    max_jump_ratio_bound: float = 0.5
    delta: float = 0.05
    p_value: float | None = None

    def is_valid(self) -> bool:
        """Returns True if all diagnostic checks passed."""
        return len(self.diagnose()) == 0

    def diagnose(self) -> list[str]:
        """Returns the list of diagnostic warning messages."""
        issues: list[str] = []

        ratio = self.v_t / max(self.v_0, 1e-16)
        if self.v_t <= 0.0:
            issues.append(
                f"{self.metric} empirical variance accumulator V_t is 0.0 (too few "
                f"positive events for CLT)."
            )
        elif ratio < self.variance_ratio_bound:
            issues.append(
                f"{self.metric} empirical variance ratio V_t / v_0 is low "
                f"({ratio:.4g} < {self.variance_ratio_bound:.4g})."
            )
        if self.rho_max > self.max_jump_ratio_bound:
            issues.append(
                f"{self.metric} maximum jump dispersion ratio rho_max is high "
                f"({self.rho_max:.4g} > {self.max_jump_ratio_bound:.4g}, "
                f"violates Lindeberg condition)."
            )
        n_active = self.num_true_positives + self.num_false_negatives_or_positives
        self.p_value = float(
            scipy.special.bdtr(
                self.num_false_negatives_or_positives,
                n_active,
                self.null_failure_prob,
            )
        )
        if self.p_value > self.delta:
            issues.append(
                f"{self.metric} binomial p-value ({self.p_value:.4g}) exceeds "
                f"delta ({self.delta:.4g}), indicating insufficient finite-sample "
                f"evidence under H0: {self.metric} <= {self.null_failure_prob:.4g}."
            )

        return issues


@dataclass
class CascadeThresholds:
    """Stores calibrated thresholds and associated diagnostics."""
    tau_pos: float
    tau_neg: float
    num_samples: int
    predicted_oracle_rate: float | None = None
    asymptotic_diag_R: AsymptoticValidityDiagnostics | None = None
    asymptotic_diag_P: AsymptoticValidityDiagnostics | None = None

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
        self._marts_R: dict[tuple[int, bool], TestSupermartingale] = {}
        self._marts_P: dict[int, TestSupermartingale] = {}
        self._opt_marts_P: dict[int, TestSupermartingale] = {}
        self._last_k_lower: int | None = None
        self._last_k_lower_opt: int | None = None

    @property
    def wealth_threshold_R(self) -> float:
        return 1.0 / self.delta_R

    @property
    def wealth_threshold_P(self) -> float:
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
        wealth = self._get_mart_R(k, reverse=False).wealth()
        return wealth > self.wealth_threshold_R

    def is_recall_plausible(self, k: int) -> bool:
        """Optimistic: fails to reject null hypothesis: mu >= 0 at level delta_R."""
        wealth = self._get_mart_R(k, reverse=True).wealth()
        return wealth <= self.wealth_threshold_R

    def is_precision_satisfied(self, k_upper: int, k_lower: int) -> bool:
        """Conservative: rejects null hypothesis: mu <= 0 at level
        delta_P / len(thresholds)."""
        wealth = self._get_mart_P(k_upper, k_lower, reverse=False).wealth()
        return wealth > self.wealth_threshold_P

    def is_precision_plausible(self, k_upper: int, k_lower: int) -> bool:
        """Optimistic: fails to reject null hypothesis: mu >= 0 at level
        delta_P / len(thresholds)."""
        wealth = self._get_mart_P(k_upper, k_lower, reverse=True).wealth()
        return wealth <= self.wealth_threshold_P

    def _get_unprocessed_samples(
        self, start_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64] | None]:
        """Returns the slice of historical samples from start_idx to the latest."""
        return (
            self.scores[start_idx:],
            self.labels[start_idx:],
            self.weights[start_idx:] if self.weights is not None else None,
        )

    def _get_mart_R(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        """Returns an up-to-date recall test supermartingale for threshold index k."""
        key = (k, reverse)
        if key not in self._marts_R:
            self._marts_R[key] = self._create_mart_R(k, reverse=reverse)
        mart = self._marts_R[key]

        if mart.t < len(self.scores):
            scores, labels, weights = self._get_unprocessed_samples(mart.t)
            rvs = self._get_rvs_R(k, scores, labels, weights)
            mart.update(rvs)

        return mart

    def _get_mart_P(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        """Returns an up-to-date precision test supermartingale for threshold pair
        (k_upper, k_lower)."""
        if not reverse:
            if k_lower != self._last_k_lower:
                self._marts_P.clear()
                self._last_k_lower = k_lower
            cache = self._marts_P
        else:
            if k_lower != self._last_k_lower_opt:
                self._opt_marts_P.clear()
                self._last_k_lower_opt = k_lower
            cache = self._opt_marts_P
        if k_upper not in cache:
            cache[k_upper] = self._create_mart_P(
                k_upper,
                k_lower,
                reverse=reverse
            )
        mart = cache[k_upper]

        if mart.t < len(self.scores):
            scores, labels, weights = self._get_unprocessed_samples(mart.t)
            rvs = self._get_rvs_P(k_upper, k_lower, scores, labels, weights)
            mart.update(rvs)

        return mart

    @abstractmethod
    def _get_rvs_R(
        self,
        k: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _create_mart_R(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        pass

    @abstractmethod
    def _get_rvs_P(
        self,
        k_upper: int,
        k_lower: int,
        scores: NDArray[np.float64],
        labels: NDArray[np.bool_],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def _create_mart_P(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        pass

    def asymptotic_diag_R(
        self,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        """Checks asymptotic validity of the recall confidence sequences for the
        selected threshold.

        Default implementation for non-asymptotic confidence sequences returns
        None.
        """
        return None

    def asymptotic_diag_P(
        self,
        k_upper: int,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        """Checks asymptotic validity of the precision confidence sequence for the
        selected thresholds.

        Default implementation for non-asymptotic confidence sequences returns
        None.
        """
        return None


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

    def _get_shift_width_R(
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

    def _get_shift_width_P(
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

    def _create_mart_R(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        shift_R, width_R = self._get_shift_width_R(k)
        target_mean = (shift_R / width_R) if width_R > 0 else 0.0
        return BettingSupermartingale(
            m=target_mean,
            alpha=self.delta_R,
            pop_size=self.pop_size,
            horizon=self.horizon,
            reverse=reverse,
        )

    def _get_rvs_R(
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
        shift_R, width_R = self._get_shift_width_R(k)
        if width_R > 0:
            return (unscaled_rvs + shift_R) / width_R
        return np.zeros_like(unscaled_rvs)

    def _create_mart_P(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        shift_P, width_P = self._get_shift_width_P(k_upper, k_lower)
        target_mean = (shift_P / width_P) if width_P > 0 else 0.0
        return BettingSupermartingale(
            m=target_mean,
            alpha=self.delta_P / len(self.thresholds),
            pop_size=self.pop_size,
            horizon=self.horizon,
            reverse=reverse,
        )

    def _get_rvs_P(
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
        shift_P, width_P = self._get_shift_width_P(k_upper, k_lower)
        if width_P > 0:
            return (unscaled_rvs + shift_P) / width_P
        return np.zeros_like(unscaled_rvs)


@dataclass
class PriorAndTargetVar:
    """Tuple of (v_0_R, v_0_P) that also holds expected process target
    variances (v_0_target_R, v_0_target_P)."""

    def __init__(
        self,
        prior_R: float | NDArray[np.float64],
        prior_P: float | NDArray[np.float64] | None = None,
        target_R: float | NDArray[np.float64] | None = None,
        target_P: float | NDArray[np.float64] | None = None,
        null_failure_prob_R: float | NDArray[np.float64] | None = None,
        null_failure_prob_P: float | NDArray[np.float64] | None = None,
    ) -> None:
        self._prior_R = np.asarray(prior_R, dtype=np.float64)
        if np.any(self._prior_R <= 0.0):
            raise ValueError("All entries of prior_R must be positive.")

        if prior_P is not None:
            self._prior_P = np.asarray(prior_P, dtype=np.float64)
            if np.any(self._prior_P <= 0.0):
                raise ValueError("All entries of prior_P must be positive.")
        else:
            self._prior_P = self._prior_R

        if target_R is not None:
            self._target_R = np.asarray(target_R, dtype=np.float64)
            if np.any(self._target_R < 0.0):
                raise ValueError("All entries of target_R must be non-negative.")
            if np.shape(self._target_R) != np.shape(self._prior_R):
                raise ValueError(
                    "target_R must have the same shape as prior_R."
                )
        else:
            self._target_R = self._prior_R

        if target_P is not None:
            self._target_P = np.asarray(target_P, dtype=np.float64)
            if np.any(self._target_P < 0.0):
                raise ValueError(
                    "All entries of target_P must be non-negative."
                )
            if np.shape(self._target_P) != np.shape(self._prior_P):
                raise ValueError(
                    "target_P must have the same shape as prior_P."
                )
        else:
            self._target_P = self._prior_P

        self._null_failure_prob_R = (
            np.asarray(null_failure_prob_R, dtype=np.float64)
            if null_failure_prob_R is not None
            else None
        )
        self._null_failure_prob_P = (
            np.asarray(null_failure_prob_P, dtype=np.float64)
            if null_failure_prob_P is not None
            else None
        )

    def prior_R(self, k: int) -> float:
        if self._prior_R.shape == ():
            return float(self._prior_R)
        return float(self._prior_R[k])

    def prior_P(self, k_upper: int, k_lower: int) -> float:
        if self._prior_P.shape == ():
            return float(self._prior_P)
        return float(self._prior_P[k_upper, k_lower])

    def target_R(self, k: int) -> float:
        if self._target_R.shape == ():
            return float(self._target_R)
        return float(self._target_R[k])

    def target_P(self, k_upper: int, k_lower: int) -> float:
        if self._target_P.shape == ():
            return float(self._target_P)
        return float(self._target_P[k_upper, k_lower])

    def null_failure_prob_R(self, k: int) -> float | None:
        if self._null_failure_prob_R is None:
            return None
        if self._null_failure_prob_R.shape == ():
            return float(self._null_failure_prob_R)
        return float(self._null_failure_prob_R[k])

    def null_failure_prob_P(self, k_upper: int, k_lower: int = 0) -> float | None:
        if self._null_failure_prob_P is None:
            return None
        if self._null_failure_prob_P.shape == ():
            return float(self._null_failure_prob_P)
        if self._null_failure_prob_P.ndim == 1:
            return float(self._null_failure_prob_P[k_upper])
        return float(self._null_failure_prob_P[k_upper, k_lower])


def compute_prior_and_target_var(
    scores: ArrayLike,
    gamma_R: float,
    gamma_P: float,
    delta: float,
    thresholds: Sequence[float] | NDArray[np.float64] | None = None,
    thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
    weights: ArrayLike | None = None,
    n_0: float | Literal["auto"] = "auto",
) -> PriorAndTargetVar:
    r"""Computes prior variance parameters (with safety floor) and expected cumulative
    process target variances for the asymptotic Gaussian mixture supermartingales used
    in ACTIS cascade tuning.

    Args:
        scores: Proxy scores in $[0, 1]$ across the population dataset.
        gamma_R: Target recall constraint in $(0, 1)$.
        gamma_P: Target precision constraint in $(0, 1)$.
        delta: Total allowed family-wise error rate in $(0, 1)$.
        thresholds: Candidate grid of thresholds in $[0, 1]$. If `thresholds_upper` is
            None, this function assumes the same grid is used for tuning the upper and
            lower thresholds. Otherwise, this function assumes the grid is used for the
            lower threshold and `thresholds_upper` is used for the upper threshold. If
            provided, returns a `PriorAndTargetVar` containing tailored prior variance
            arrays `(v_0_R, v_0_P)` and target variances
            `(v_0_target_R, v_0_target_P)`.
        thresholds_upper: Optional candidate grid for tuning the upper threshold.
            Defaults to `thresholds`.
        weights: Optional importance weights $w(x) = 1 / (N \cdot q(x))$.
        n_0: Effective prior sample horizon. If "auto" (default), dynamically scaled
            to min(1000.0, max(50.0, 0.10 * N)).

    Returns:
        A `PriorAndTargetVar` instance.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Parameter `scores` must be a non-empty 1D array.")
    N = len(scores)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        max_weight = float(np.max(weights))
    else:
        max_weight = 1.0

    if n_0 == "auto":
        eff_n_0 = min(1000.0, max(50.0, 0.1 * N))
    else:
        eff_n_0 = float(n_0)

    # Split family-wise error rate delta equally between recall and precision
    delta_R = delta / 2.0
    delta_P = delta / 2.0

    log_delta_R_inv = math.log(1.0 / delta_R)

    # Fallback when no threshold grid is specified
    if thresholds is None:
        b = max(1.0 - gamma_R, 1.0 - gamma_P) * max_weight
        v_0_floor = (2.0 * b**2) / log_delta_R_inv
        if weights is not None:
            weighted_pos_mass = float(np.mean(scores * weights))
        else:
            weighted_pos_mass = float(np.mean(scores))
        sigma2 = max(weighted_pos_mass, 1.0 / N) * gamma_R * (1.0 - gamma_R)
        v_0_target = eff_n_0 * sigma2
        return PriorAndTargetVar(
            max(v_0_floor, v_0_target),
            target_R=v_0_target,
            null_failure_prob_R=1.0 - gamma_R,
            null_failure_prob_P=1.0 - gamma_P,
        )

    thresholds = np.asarray(thresholds, dtype=np.float64)
    M_lower = len(thresholds)

    # Bonferroni correction for precision across candidate lower thresholds
    delta_P_corrected = delta_P / max(1, M_lower)
    log_delta_P_inv = math.log(1.0 / delta_P_corrected)

    # Sort items by proxy score to enable O(1) prefix/suffix query per threshold
    order = np.argsort(scores)
    scores_sorted = scores[order]

    idx_lower = np.searchsorted(scores_sorted, thresholds, side="left")
    if thresholds_upper is None or thresholds_upper is thresholds:
        thresholds_upper = thresholds
        idx_upper = idx_lower
    else:
        thresholds_upper = np.asarray(thresholds_upper, dtype=np.float64)
        idx_upper = np.searchsorted(scores_sorted, thresholds_upper, side="left")

    M_upper = len(thresholds_upper)

    if weights is None:
        prefix_sum_pos = (
            np.concatenate(([0.0], np.cumsum(scores_sorted))) * max_weight
        )
        suffix_sum_pos = (
            np.append(np.cumsum(scores_sorted[::-1])[::-1], 0.0) * max_weight
        )

        pos_mass_ge_lower = suffix_sum_pos[idx_lower] / N
        pos_mass_lt_lower = prefix_sum_pos[idx_lower] / N

        b_R = (1.0 - gamma_R) * max_weight
        v_0_floor_R = (2.0 * b_R**2) / log_delta_R_inv
        sigma2_R = (
            (1.0 - gamma_R)**2 * pos_mass_ge_lower + (gamma_R**2) * pos_mass_lt_lower
        )
        v_0_target_R = eff_n_0 * sigma2_R
        v_0_R = np.maximum(v_0_floor_R, v_0_target_R)
        null_failure_prob_R = np.full(M_lower, 1 - gamma_R, dtype=np.float64)

        pos_mass_ge_upper = (
            suffix_sum_pos[idx_upper] / N
            if idx_upper is not idx_lower
            else pos_mass_ge_lower
        )
        neg_mass_ge_upper = np.maximum(
            0.0,
            (N - idx_upper) / N * max_weight - pos_mass_ge_upper
        )
        neg_var_upper = (gamma_P**2) * neg_mass_ge_upper
        pos_var_lower = (1.0 - gamma_P)**2 * pos_mass_ge_lower

        b_P = (1.0 - gamma_P) * max_weight
        v_0_floor_P = (2.0 * b_P**2) / log_delta_P_inv

        total_var = neg_var_upper[:, np.newaxis] + pos_var_lower[np.newaxis, :]
        v_0_target_P = eff_n_0 * total_var
        v_0_P = np.maximum(v_0_floor_P, v_0_target_P)
        null_failure_prob_P = np.full((M_upper, M_lower), 1 - gamma_P, dtype=np.float64)

        return PriorAndTargetVar(
            prior_R=v_0_R,
            prior_P=v_0_P,
            target_R=v_0_target_R,
            target_P=v_0_target_P,
            null_failure_prob_R=null_failure_prob_R,
            null_failure_prob_P=null_failure_prob_P,
        )

    weights_sorted = weights[order]

    # Proposal probability per sorted item: w_i = 1 / (N * q_i) => q_i = 1 / (N * w_i)
    q_sorted = 1.0 / (N * weights_sorted)

    # Suffix accumulations for items with s_i >= tau (reverse cumsum):
    # Appending terminal values ensures valid indices when thresholds exceed max score.
    suffix_max_weight = np.append(
        np.maximum.accumulate(weights_sorted[::-1])[::-1],
        max_weight,
    )
    w_max_lower = suffix_max_weight[idx_lower]

    # Precompute weighted label probabilities via direct prefix and suffix sums
    # to avoid catastrophic cancellation in both lower and upper tails
    weighted_pos_prob = scores_sorted * weights_sorted
    prefix_sum_pos_prob = np.concatenate(([0.0], np.cumsum(weighted_pos_prob)))
    suffix_sum_pos_prob = np.append(
        np.cumsum(weighted_pos_prob[::-1])[::-1],
        0.0,
    )

    pos_mass_ge_lower = suffix_sum_pos_prob[idx_lower] / N
    pos_mass_lt_lower = prefix_sum_pos_prob[idx_lower] / N

    # Safety floor for recall (incorporates positive jumps only)
    b_R = (1.0 - gamma_R) * w_max_lower
    v_0_floor_R = (2.0 * b_R**2) / log_delta_R_inv

    # Estimated second moment of the recall margin increment
    sigma2_R = (
        (1.0 - gamma_R)**2 * pos_mass_ge_lower + (gamma_R ** 2) * pos_mass_lt_lower
    )
    v_0_target_R = eff_n_0 * sigma2_R
    v_0_R = np.maximum(v_0_floor_R, v_0_target_R)

    # Proposal-adjusted null failure probability for recall via partition density:
    # Direct prefix and suffix sums of q avoid subtractive cancellation in Q_ge
    prefix_q = np.concatenate(([0.0], np.cumsum(q_sorted)))
    suffix_q = np.append(np.cumsum(q_sorted[::-1])[::-1], 0.0)

    N_lt_lower = idx_lower.astype(np.float64)
    N_ge_lower = (N - idx_lower).astype(np.float64)

    Q_lt_lower = prefix_q[idx_lower]
    Q_ge_lower = suffix_q[idx_lower]

    q_bar_lt_lower = np.where(
        N_lt_lower > 0,
        Q_lt_lower / np.maximum(N_lt_lower, 1.0),
        1.0 / N,
    )
    q_bar_ge_lower = np.where(
        N_ge_lower > 0,
        Q_ge_lower / np.maximum(N_ge_lower, 1.0),
        1.0 / N,
    )

    eta_R = np.clip(
        q_bar_lt_lower / np.maximum(q_bar_ge_lower, 1e-12),
        0.0,
        1.0,
    )
    null_failure_prob_R = (
        (1.0 - gamma_R) * eta_R / (gamma_R + (1.0 - gamma_R) * eta_R)
    )
    null_failure_prob_R = np.clip(null_failure_prob_R, 0.0, 1.0 - gamma_R)

    # Precision prior and target variance:
    # Direct suffix sum of weighted negative mass to prevent catastrophic cancellation
    weighted_neg_prob = (1.0 - scores_sorted) * weights_sorted
    suffix_sum_neg_prob = np.append(
        np.cumsum(weighted_neg_prob[::-1])[::-1],
        0.0,
    )
    neg_mass_ge_upper = suffix_sum_neg_prob[idx_upper] / N
    neg_var_upper = (gamma_P ** 2) * neg_mass_ge_upper
    pos_var_lower = (1.0 - gamma_P)**2 * pos_mass_ge_lower

    # Safety floor for precision (incorporates positive jumps only)
    b_P = (1.0 - gamma_P) * w_max_lower[np.newaxis, :]
    v_0_floor_P = (2.0 * b_P**2) / log_delta_P_inv

    # 2D second moment matrix: sum of false-positive variance (from upper threshold)
    # and true-positive variance (from lower threshold) via NumPy broadcasting
    total_var = neg_var_upper[:, np.newaxis] + pos_var_lower[np.newaxis, :]
    v_0_target_P = eff_n_0 * total_var
    v_0_P = np.maximum(v_0_floor_P, v_0_target_P)

    q_fp = q_sorted * (1.0 - scores_sorted)
    q_tp = q_sorted * scores_sorted
    n_fp_proxy = 1.0 - scores_sorted
    n_tp_proxy = scores_sorted

    suffix_q_fp = np.append(np.cumsum(q_fp[::-1])[::-1], 0.0)
    suffix_n_fp = np.append(np.cumsum(n_fp_proxy[::-1])[::-1], 0.0)
    suffix_q_tp = np.append(np.cumsum(q_tp[::-1])[::-1], 0.0)
    suffix_n_tp = np.append(np.cumsum(n_tp_proxy[::-1])[::-1], 0.0)

    q_fp_upper = suffix_q_fp[idx_upper]
    n_fp_upper = suffix_n_fp[idx_upper]
    q_bar_fp_upper = np.where(
        n_fp_upper > 1e-12,
        q_fp_upper / np.maximum(n_fp_upper, 1e-12),
        1.0 / N,
    )

    q_tp_lower = suffix_q_tp[idx_lower]
    n_tp_lower = suffix_n_tp[idx_lower]
    q_bar_tp_lower = np.where(
        n_tp_lower > 1e-12,
        q_tp_lower / np.maximum(n_tp_lower, 1e-12),
        1.0 / N,
    )

    ratio_P_proxy = q_bar_fp_upper[:, np.newaxis] / np.maximum(
        q_bar_tp_lower[np.newaxis, :], 1e-12
    )

    # Unconditional geometric worst-case bounds across valid partitions:
    suffix_min_q = np.append(np.minimum.accumulate(q_sorted[::-1])[::-1], 1.0 / N)
    suffix_max_q = np.append(np.maximum.accumulate(q_sorted[::-1])[::-1], 1.0 / N)
    q_min_upper = suffix_min_q[idx_upper]
    q_max_upper = suffix_max_q[idx_upper]
    q_min_lower = suffix_min_q[idx_lower]
    q_max_lower = suffix_max_q[idx_lower]

    ratio_P_min = q_min_upper[:, np.newaxis] / np.maximum(
        q_max_lower[np.newaxis, :], 1e-12
    )
    ratio_P_max = q_max_upper[:, np.newaxis] / np.maximum(
        q_min_lower[np.newaxis, :], 1e-12
    )

    eta_P = np.clip(ratio_P_proxy, ratio_P_min, ratio_P_max)
    null_failure_prob_P = (
        (1.0 - gamma_P) * eta_P / (gamma_P + (1.0 - gamma_P) * eta_P)
    )
    null_failure_prob_P = np.clip(null_failure_prob_P, 1e-6, 0.5)

    return PriorAndTargetVar(
        prior_R=v_0_R,
        prior_P=v_0_P,
        target_R=v_0_target_R,
        target_P=v_0_target_P,
        null_failure_prob_R=null_failure_prob_R,
        null_failure_prob_P=null_failure_prob_P,
    )


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
        v_0: float | PriorAndTargetVar = 0.01,
        max_jump_ratio_bound: float = 0.50,
        variance_ratio_bound: float = 0.05,
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
                scalar float or a `PriorAndTargetVar` instance. Defaults to 0.01.
            max_jump_ratio_bound: Maximum allowable Hall & Heyde Lindeberg
                jump-to-variance ratio $\rho_{\max} = \max_i (X_i - \bar{X})^2 / V_t$.
                Defaults to 0.50.
            variance_ratio_bound: Maximum allowable ratio of cumulative process
                variance to prior variance $V_t / v_0$. Defaults to 0.05.
        """
        super().__init__(gamma_P, gamma_R, delta, thresholds, thresholds_upper)
        if isinstance(v_0, PriorAndTargetVar):
            self.v_0 = v_0
        else:
            self.v_0 = PriorAndTargetVar(prior_R=v_0)
        self.max_jump_ratio_bound = max_jump_ratio_bound
        self.variance_ratio_bound = variance_ratio_bound

        # Update when adding samples to speed up asymptotic diagnostics
        self._num_tp = np.zeros(len(self.thresholds), dtype=np.int64)
        self._num_fn = np.zeros(len(self.thresholds), dtype=np.int64)
        self._num_fp = np.zeros(len(self.thresholds_upper), dtype=np.int64)

    def add_samples(
        self,
        scores: Sequence[float] | NDArray[np.float64],
        labels: Sequence[bool] | NDArray[np.bool_],
        weights: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.bool_)
        super().add_samples(scores, labels, weights)

        pos_scores = scores[labels]
        if len(pos_scores) > 0:
            self._num_tp += np.sum(pos_scores[:, None] >= self.thresholds, axis=0)
            self._num_fn += np.sum(pos_scores[:, None] < self.thresholds, axis=0)

        neg_scores = scores[~labels]
        if len(neg_scores) > 0:
            self._num_fp += np.sum(neg_scores[:, None] >= self.thresholds_upper, axis=0)

    def _create_mart_R(
        self,
        k: int,
        reverse: bool = False
    ) -> TestSupermartingale:
        v_0 = float(self.v_0.prior_R(k))
        return GaussianMixtureSupermartingale(m=0.0, v_0=v_0, reverse=reverse)

    def _get_rvs_R(
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

    def _create_mart_P(
        self,
        k_upper: int,
        k_lower: int,
        reverse: bool = False,
    ) -> TestSupermartingale:
        v_0 = float(self.v_0.prior_P(k_upper, k_lower))
        return GaussianMixtureSupermartingale(m=0.0, v_0=v_0, reverse=reverse)

    def _get_rvs_P(
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

    def asymptotic_diag_R(
        self,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        if k_lower == 0:
            return None

        tau_lower = self.thresholds[k_lower]
        num_true_positives = self._num_tp[k_lower]
        num_false_negatives = self._num_fn[k_lower]

        mart = self._get_mart_R(k_lower, reverse=False)
        assert isinstance(mart, GaussianMixtureSupermartingale)

        null_failure_prob = self.v_0.null_failure_prob_R(k_lower)
        if null_failure_prob is None:
            null_failure_prob = 1.0 - self.gamma_R

        return AsymptoticValidityDiagnostics(
            v_t=mart.v_t,
            v_0=self.v_0.target_R(k_lower),
            rho_max=mart.rho_max,
            num_true_positives=int(num_true_positives),
            num_false_negatives_or_positives=int(num_false_negatives),
            null_failure_prob=null_failure_prob,
            metric="Recall",
            thresholds=(tau_lower,),
            variance_ratio_bound=self.variance_ratio_bound,
            max_jump_ratio_bound=self.max_jump_ratio_bound,
            delta=self.delta_R,
        )

    def asymptotic_diag_P(
        self,
        k_upper: int,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        if k_upper == len(self.thresholds_upper) - 1:
            return None

        tau_lower = self.thresholds[k_lower]
        tau_upper = self.thresholds_upper[k_upper]
        num_false_positives = self._num_fp[k_upper]
        num_true_positives = self._num_tp[k_lower]

        mart = self._get_mart_P(k_upper, k_lower, reverse=False)
        assert isinstance(mart, GaussianMixtureSupermartingale)

        v_0_target = self.v_0.target_P(k_upper, k_lower)
        null_failure_prob = self.v_0.null_failure_prob_P(k_upper, k_lower)
        if null_failure_prob is None:
            null_failure_prob = 1.0 - self.gamma_P

        delta = self.delta_P
        return AsymptoticValidityDiagnostics(
            v_t=mart.v_t,
            v_0=v_0_target,
            rho_max=mart.rho_max,
            num_true_positives=int(num_true_positives),
            num_false_negatives_or_positives=int(num_false_positives),
            null_failure_prob=null_failure_prob,
            metric="Precision",
            thresholds=(tau_lower, tau_upper),
            variance_ratio_bound=self.variance_ratio_bound,
            max_jump_ratio_bound=self.max_jump_ratio_bound,
            delta=delta
        )


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
        v_0: float | PriorAndTargetVar = 0.01,
        min_positives: int | Literal["auto"] = "auto",
        p_floor: float | Literal["auto"] = "auto",
        max_weight_ge: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_lt: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None,
        enable_asymptotic_protection: bool = True,
        variance_ratio_bound: float = 0.05,
        max_jump_ratio_bound: float = 0.50,
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
                globally across all tests), or a `PriorAndTargetVar` object containing
                per-test variances. Defaults to 0.01.
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
            enable_asymptotic_protection: If `conf_seq='asymptotic'` and True, enables
                heuristic protection for anytime-valid FWER control when operating in
                the non-asymptotic regime. This arugment has no effect when
                `conf_seq='finite'`.
            variance_ratio_bound: If `conf_seq='asymptotic'`, this parameter specifies
                the maximum allowable ratio of cumulative process variance to prior
                variance $V_t / v_0$. Defaults to 0.05. This argument has no effect when
                `conf_seq='finite'` or when `enable_asymptotic_protection` is False.
            max_jump_ratio_bound: If `conf_seq='asymptotic'`, this parameter specifies
                the maximum allowable jump-to-variance ratio. Defaults to 0.50. This
                argument has no effect when `conf_seq='finite'` or when
                `enable_asymptotic_protection` is False.
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
                v_0=v_0,
                max_jump_ratio_bound=max_jump_ratio_bound,
                variance_ratio_bound=variance_ratio_bound,
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

        self.enable_asymptotic_protection = enable_asymptotic_protection

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

        min_req = max(5, int(np.ceil(np.round(eff_p * pop_size, 9))))
        return min(pop_size, min(cap, min_req))

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

        tau_upper, tau_lower, diag_R, diag_P = self._tune_thresholds()

        predicted_oracle_rate = None
        if population_scores is not None:
            predicted_oracle_rate = self.predict_oracle_rate(
                population_scores,
                tau_upper=tau_upper,
                tau_lower=tau_lower,
            )

        self.current_thresholds = CascadeThresholds(
            tau_pos=tau_upper,
            tau_neg=tau_lower,
            num_samples=len(self.seen_indices),
            predicted_oracle_rate=predicted_oracle_rate,
            asymptotic_diag_R=diag_R,
            asymptotic_diag_P=diag_P
        )

        return self.current_thresholds

    def _tune_thresholds(self) -> tuple[
        float,
        float,
        AsymptoticValidityDiagnostics | None,
        AsymptoticValidityDiagnostics | None
    ]:
        r"""Evaluates the test supermartingales across candidate threshold grids to find
        the optimal pair of thresholds."""
        # Forward sequential scan stopping at first failure of recall target
        k_lower = 0
        tau_lower = float(self.thresholds[0])
        diag_R = None
        for k in range(len(self.thresholds)):
            if not self.conf_seqs.is_recall_satisfied(k):
                break

            d = None
            if self.enable_asymptotic_protection:
                d = self.conf_seqs.asymptotic_diag_R(k)
                if d is not None and not d.is_valid():
                    if k_lower == 0:
                        diag_R = d
                    break

            k_lower = k
            tau_lower = float(self.thresholds[k])
            diag_R = d

        # Backward sequential scan stopping at first failure of precision target
        k_upper = len(self.thresholds_upper) - 1
        tau_upper = float(self.thresholds_upper[-1])
        diag_P = None
        for k in range(len(self.thresholds_upper) - 1, -1, -1):
            tau_k = float(self.thresholds_upper[k])

            if (tau_k < tau_lower) or \
                not self.conf_seqs.is_precision_satisfied(k, k_lower):
                break

            d = None
            if self.enable_asymptotic_protection:
                d = self.conf_seqs.asymptotic_diag_P(k, k_lower)
                if d is not None and not d.is_valid():
                    if k_upper == len(self.thresholds_upper) - 1:
                        diag_P = d
                    break

            k_upper = k
            tau_upper = tau_k
            diag_P = d

        return (
            tau_upper,
            tau_lower,
            diag_R,
            diag_P
        )

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
            tau_upper = self.current_thresholds.tau_upper
        if tau_lower is None:
            tau_lower = self.current_thresholds.tau_lower

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
        # if self.conf_seq == "asymptotic" and self.enable_asymptotic_protection:
        #     asymp_valid = False
        #     if ((self.current_thresholds.asymptotic_diag_R is None or
        #         self.current_thresholds.asymptotic_diag_R.is_valid()) and
        #         (self.current_thresholds.asymptotic_diag_P is None or
        #         self.current_thresholds.asymptotic_diag_P.is_valid())):
        #         asymp_valid = True

        #     warmup_needed = (not asymp_valid) or (num_positives < min_positives)
        # else:
        #     warmup_needed = num_positives < min_positives
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
