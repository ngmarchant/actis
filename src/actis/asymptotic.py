# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Asymptotic Gaussian-mixture confidence sequences and ACTIS tuner."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray

from .confseq import GaussianMixtureSupermartingale, TestSupermartingale
from .tuner import CascadeConfSeqs, CascadeTuner


@dataclass(frozen=True)
class AsymptoticValidityDiagnostics:
    """Diagnostics for assessing whether asymptotic confidence guarantees hold.

    Attributes:
        v_t: Empirical quadratic variation accumulator $V_t$.
        v_0: Target cumulative process variance $v_0$ for scaling checks.
        num_true_positives: Observed true positive count at the candidate threshold.
        num_false_negatives_or_positives: Observed false negatives (for Recall) or
            false positives (for Precision).
        null_failure_prob: Boundary null error probability used in the exact binomial
            check.
        metric: Metric name (`"Recall"` or `"Precision"`).
        thresholds: Candidate threshold tuple associated with this check.
        variance_ratio_bound: Minimum required ratio $V_t / v_0$.
        delta: Significance level for the exact one-sided binomial test.
        p_value: Computed binomial p-value when `enable_binomial_check` is True.
        enable_variance_ratio_check: Whether to verify
            `v_t / v_0 >= variance_ratio_bound`.
        enable_binomial_check: Whether to run the exact one-sided binomial check.
    """

    v_t: float
    v_0: float
    num_true_positives: int
    num_false_negatives_or_positives: int
    null_failure_prob: float
    metric: str = field(default="Recall")
    thresholds: tuple[float, ...] | None = None
    variance_ratio_bound: float = 0.05
    delta: float = 0.05
    p_value: float | None = field(init=False, default=None)
    enable_variance_ratio_check: bool = True
    enable_binomial_check: bool = True

    def __post_init__(self) -> None:
        """Computes the exact one-sided binomial p-value upon initialization."""
        if self.enable_binomial_check:
            n_active = self.num_true_positives + self.num_false_negatives_or_positives
            p_value = float(
                scipy.special.bdtr(
                    self.num_false_negatives_or_positives,
                    n_active,
                    self.null_failure_prob,
                )
            )
            object.__setattr__(self, "p_value", p_value)

    def is_valid(self) -> bool:
        """Returns True if all diagnostic checks passed."""
        return len(self.diagnose()) == 0

    def diagnose(self) -> list[str]:
        """Returns the list of diagnostic warning messages."""
        issues: list[str] = []

        if self.enable_variance_ratio_check:
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

        if (
            self.enable_binomial_check
            and self.p_value is not None
            and self.p_value > self.delta
        ):
            issues.append(
                f"{self.metric} binomial p-value ({self.p_value:.4g}) exceeds "
                f"delta ({self.delta:.4g}), indicating insufficient finite-sample "
                f"evidence under H0: {self.metric} <= "
                f"{1.0 - self.null_failure_prob:.4g}."
            )

        return issues


class PriorAndTargetVar:
    """Stores prior variances, target variances, and null failure probabilities.

    Provides indexed access across recall (`k`) and precision (`k_upper`, `k_lower`)
    threshold grids while supporting scalar, 1D, and 2D array inputs.
    """

    def __init__(
        self,
        prior_R: float | NDArray[np.float64],
        prior_P: float | NDArray[np.float64] | None = None,
        target_R: float | NDArray[np.float64] | None = None,
        target_P: float | NDArray[np.float64] | None = None,
        null_failure_prob_R: float | NDArray[np.float64] | None = None,
        null_failure_prob_P: float | NDArray[np.float64] | None = None,
    ) -> None:
        """Initializes prior, target variance, and null probability containers.

        Args:
            prior_R: Positive scalar or 1D array of prior variances for recall tests.
            prior_P: Optional positive scalar or 2D array of prior variances for
                precision tests. Defaults to `prior_R`.
            target_R: Optional non-negative expected process target variances for
                recall tests. Defaults to `prior_R`.
            target_P: Optional non-negative expected process target variances for
                precision tests. Defaults to `prior_P`.
            null_failure_prob_R: Optional boundary null false-negative probabilities
                for recall binomial checks.
            null_failure_prob_P: Optional boundary null false-positive probabilities
                for precision binomial checks.

        Raises:
            ValueError: If `prior_R` or `prior_P` has non-positive entries, if
                `target_R` or `target_P` has negative entries, or if target shapes
                do not match the corresponding prior shapes.
        """
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
                raise ValueError("target_R must have the same shape as prior_R.")
        else:
            self._target_R = self._prior_R

        if target_P is not None:
            self._target_P = np.asarray(target_P, dtype=np.float64)
            if np.any(self._target_P < 0.0):
                raise ValueError("All entries of target_P must be non-negative.")
            if np.shape(self._target_P) != np.shape(self._prior_P):
                raise ValueError("target_P must have the same shape as prior_P.")
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
        """Returns the recall prior variance for lower threshold index `k`."""
        if self._prior_R.shape == ():
            return float(self._prior_R)
        return float(self._prior_R[k])

    def prior_P(self, k_upper: int, k_lower: int) -> float:
        """Returns the precision prior variance for `(k_upper, k_lower)`."""
        if self._prior_P.shape == ():
            return float(self._prior_P)
        return float(self._prior_P[k_upper, k_lower])

    def target_R(self, k: int) -> float:
        """Returns the recall expected target variance for lower threshold index `k`."""
        if self._target_R.shape == ():
            return float(self._target_R)
        return float(self._target_R[k])

    def target_P(self, k_upper: int, k_lower: int) -> float:
        """Returns the precision expected target variance for `(k_upper, k_lower)`."""
        if self._target_P.shape == ():
            return float(self._target_P)
        return float(self._target_P[k_upper, k_lower])

    def null_failure_prob_R(self, k: int) -> float | None:
        """Returns the recall null failure probability at index `k`, if set."""
        if self._null_failure_prob_R is None:
            return None
        if self._null_failure_prob_R.shape == ():
            return float(self._null_failure_prob_R)
        return float(self._null_failure_prob_R[k])

    def null_failure_prob_P(self, k_upper: int, k_lower: int = 0) -> float | None:
        """Returns the precision null failure probability for `(k_upper, k_lower)`."""
        if self._null_failure_prob_P is None:
            return None
        if self._null_failure_prob_P.shape == ():
            return float(self._null_failure_prob_P)
        if self._null_failure_prob_P.ndim == 1:
            return float(self._null_failure_prob_P[k_upper])
        return float(self._null_failure_prob_P[k_upper, k_lower])

    def __repr__(self) -> str:
        """Returns a compact string representation of stored shapes and scalars."""

        def _fmt(arr: NDArray[np.float64]) -> str:
            if arr.shape == ():
                return f"{float(arr):.4g}"
            return f"array(shape={arr.shape})"

        return (
            f"PriorAndTargetVar(prior_R={_fmt(self._prior_R)}, "
            f"prior_P={_fmt(self._prior_P)}, "
            f"target_R={_fmt(self._target_R)}, "
            f"target_P={_fmt(self._target_P)})"
        )


def compute_prior_and_target_var(
    scores: ArrayLike,
    gamma_P: float,
    gamma_R: float,
    delta: float,
    thresholds: Sequence[float] | NDArray[np.float64] | None = None,
    thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
    weights: ArrayLike | None = None,
    n_0: float | Literal["auto"] = "auto",
    enable_variance_floor: bool = True,
) -> PriorAndTargetVar:
    r"""Computes prior and target variances for asymptotic ACTIS tuning.

    Estimates expected cumulative process target variances (with optional single-step
    safety floor) for the Gaussian mixture supermartingales used in ACTIS cascade
    calibration.

    Args:
        scores: Proxy scores in $[0, 1]$ across the population dataset.
        gamma_P: Target precision constraint in $(0, 1)$.
        gamma_R: Target recall constraint in $(0, 1)$.
        delta: Total allowed family-wise error rate in $(0, 1)$.
        thresholds: Candidate grid of thresholds in $[0, 1]$. If `thresholds_upper` is
            None, this grid is used for both upper and lower thresholds; otherwise, it
            is used for the lower threshold. If None, computes scalar fallback values.
        thresholds_upper: Optional candidate grid for tuning the upper threshold.
            Defaults to `thresholds`.
        weights: Optional importance weights $w(x) = 1 / (N \cdot q(x))$.
        n_0: Effective prior sample horizon. If `"auto"` (default), scales as
            $\min(1000.0, \max(50.0, 0.10 \cdot N))$.
        enable_variance_floor: If True (default), enforces a single-step safety floor
            $v^{\text{floor}}$ on prior variances to prevent single-sample crossings.

    Returns:
        A `PriorAndTargetVar` instance containing calibrated priors, target variances,
        and null failure probabilities.

    Raises:
        ValueError: If `scores` is not a non-empty 1D array.
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
        v_0_target = max(1e-12, eff_n_0 * sigma2)
        prior_val = max(v_0_floor, v_0_target) if enable_variance_floor else v_0_target
        return PriorAndTargetVar(
            prior_val,
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
        prefix_sum_pos = np.concatenate(([0.0], np.cumsum(scores_sorted))) * max_weight
        suffix_sum_pos = (
            np.append(np.cumsum(scores_sorted[::-1])[::-1], 0.0) * max_weight
        )

        pos_mass_ge_lower = suffix_sum_pos[idx_lower] / N
        pos_mass_lt_lower = prefix_sum_pos[idx_lower] / N

        # Boundary null cap for Recall: under H_0 (Recall <= gamma_R),
        # false negative mass cannot exceed (1 - gamma_R)/gamma_R * true positive mass.
        # This prevents proxy score inaccuracies from inflating expected process
        # variance.
        null_fn_cap = ((1.0 - gamma_R) / gamma_R) * pos_mass_ge_lower
        pos_mass_lt_lower = np.minimum(pos_mass_lt_lower, null_fn_cap)

        b_R = (1.0 - gamma_R) * max_weight
        v_0_floor_R = (2.0 * b_R**2) / log_delta_R_inv
        sigma2_R = (1.0 - gamma_R) ** 2 * pos_mass_ge_lower + (
            gamma_R**2
        ) * pos_mass_lt_lower
        v_0_target_R = np.maximum(1e-12, eff_n_0 * sigma2_R)
        v_0_R = (
            np.maximum(v_0_floor_R, v_0_target_R)
            if enable_variance_floor
            else v_0_target_R
        )
        null_failure_prob_R = np.full(M_lower, 1 - gamma_R, dtype=np.float64)

        pos_mass_ge_upper = (
            suffix_sum_pos[idx_upper] / N
            if idx_upper is not idx_lower
            else pos_mass_ge_lower
        )
        neg_mass_ge_upper = np.maximum(
            0.0, (N - idx_upper) / N * max_weight - pos_mass_ge_upper
        )

        # Boundary null cap for Precision: under H_0 (Precision <= gamma_P),
        # false positive mass cannot exceed (1 - gamma_P)/gamma_P * true positive mass.
        # Capping neg_mass_ge_upper eliminates proxy distortion (e.g. compressed scores)
        # while preserving exact boundary null validity.
        null_fp_cap = ((1.0 - gamma_P) / gamma_P) * pos_mass_ge_lower[np.newaxis, :]
        neg_mass_ge_upper_2d = np.minimum(neg_mass_ge_upper[:, np.newaxis], null_fp_cap)
        neg_var_upper = (gamma_P**2) * neg_mass_ge_upper_2d
        pos_var_lower = (1.0 - gamma_P) ** 2 * pos_mass_ge_lower

        b_P = (1.0 - gamma_P) * max_weight
        v_0_floor_P = (2.0 * b_P**2) / log_delta_P_inv

        total_var = neg_var_upper + pos_var_lower[np.newaxis, :]
        v_0_target_P = np.maximum(1e-12, eff_n_0 * total_var)
        v_0_P = (
            np.maximum(v_0_floor_P, v_0_target_P)
            if enable_variance_floor
            else v_0_target_P
        )
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
    w_max_ge_upper = suffix_max_weight[idx_upper]

    # Extremal partition weights for importance sampling disparity ratios:
    # 1. Dropped partition: s < tau_lo
    prefix_max_weight = np.concatenate(([1.0], np.maximum.accumulate(weights_sorted)))
    w_max_lt_lower = prefix_max_weight[idx_lower]

    # 2. Retained partition: s >= tau_lo
    suffix_min_weight = np.append(
        np.minimum.accumulate(weights_sorted[::-1])[::-1],
        max_weight,
    )
    w_min_ge_lower = suffix_min_weight[idx_lower]

    # Disparity ratios:
    rho_R = w_max_lt_lower / np.maximum(w_min_ge_lower, 1e-12)
    rho_P = w_max_ge_upper[:, np.newaxis] / np.maximum(
        w_min_ge_lower[np.newaxis, :], 1e-12
    )

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

    # Boundary null cap for Recall incorporating partition weight disparity ratio rho_R:
    null_fn_cap = ((1.0 - gamma_R) / gamma_R) * rho_R * pos_mass_ge_lower
    pos_mass_lt_lower = np.minimum(pos_mass_lt_lower, null_fn_cap)

    # Safety floor for recall (incorporates positive jumps only)
    b_R = (1.0 - gamma_R) * w_max_lower
    v_0_floor_R = (2.0 * b_R**2) / log_delta_R_inv

    # Estimated second moment of the recall margin increment
    sigma2_R = (1.0 - gamma_R) ** 2 * pos_mass_ge_lower + (
        gamma_R**2
    ) * pos_mass_lt_lower
    v_0_target_R = np.maximum(1e-12, eff_n_0 * sigma2_R)
    v_0_R = (
        np.maximum(v_0_floor_R, v_0_target_R) if enable_variance_floor else v_0_target_R
    )

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
    null_failure_prob_R = (1.0 - gamma_R) * eta_R / (gamma_R + (1.0 - gamma_R) * eta_R)
    null_failure_prob_R = np.clip(null_failure_prob_R, 0.0, 1.0 - gamma_R)

    # Precision prior and target variance:
    # Direct suffix sum of weighted negative mass to prevent catastrophic cancellation
    weighted_neg_prob = (1.0 - scores_sorted) * weights_sorted
    suffix_sum_neg_prob = np.append(
        np.cumsum(weighted_neg_prob[::-1])[::-1],
        0.0,
    )
    neg_mass_ge_upper = suffix_sum_neg_prob[idx_upper] / N

    # Boundary null cap for Precision incorporating partition weight disparity ratio
    # rho_P:
    null_fp_cap = ((1.0 - gamma_P) / gamma_P) * rho_P * pos_mass_ge_lower[np.newaxis, :]
    neg_mass_ge_upper_2d = np.minimum(neg_mass_ge_upper[:, np.newaxis], null_fp_cap)
    neg_var_upper = (gamma_P**2) * neg_mass_ge_upper_2d
    pos_var_lower = (1.0 - gamma_P) ** 2 * pos_mass_ge_lower

    # Safety floor for precision (incorporates positive jumps only)
    b_P = (1.0 - gamma_P) * w_max_lower[np.newaxis, :]
    v_0_floor_P = (2.0 * b_P**2) / log_delta_P_inv

    # 2D second moment matrix: sum of false-positive variance (from upper threshold)
    # and true-positive variance (from lower threshold) via NumPy broadcasting
    total_var = neg_var_upper + pos_var_lower[np.newaxis, :]
    v_0_target_P = np.maximum(1e-12, eff_n_0 * total_var)
    v_0_P = (
        np.maximum(v_0_floor_P, v_0_target_P) if enable_variance_floor else v_0_target_P
    )

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
    null_failure_prob_P = (1.0 - gamma_P) * eta_P / (gamma_P + (1.0 - gamma_P) * eta_P)
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
    """Asymptotic Gaussian mixture supermartingale strategy (Howard et al., 2021)."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: NDArray[np.float64],
        thresholds_upper: NDArray[np.float64],
        v_0: float | PriorAndTargetVar = 0.01,
        variance_ratio_bound: float = 0.05,
        enable_variance_ratio_check: bool = True,
        enable_binomial_check: bool = True,
    ) -> None:
        r"""Initializes the asymptotic cascade confidence sequences.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$.
            thresholds: Sorted 1D grid of candidate lower thresholds in $[0, 1]$,
                determined independently of samples.
            thresholds_upper: Sorted 1D grid of candidate upper thresholds in $[0, 1]$,
                determined independently of samples.
            v_0: Prior variance for the Gaussian mixture supermartingale. Can be a
                scalar float or a `PriorAndTargetVar` instance (default: 0.01).
            variance_ratio_bound: Minimum required ratio of cumulative empirical
                variance to target prior variance $V_t / v_0$ (default: 0.05).
            enable_variance_ratio_check: If True (default), verifies empirical
                quadratic variation accumulation $V_t / v_0 \ge \rho$.
            enable_binomial_check: If True (default), verifies candidate thresholds
                against an exact one-sided binomial test under the boundary null.
        """
        super().__init__(gamma_P, gamma_R, delta, thresholds, thresholds_upper)
        if isinstance(v_0, PriorAndTargetVar):
            self.v_0 = v_0
        else:
            self.v_0 = PriorAndTargetVar(prior_R=v_0)
        self.variance_ratio_bound = variance_ratio_bound
        self.enable_variance_ratio_check = enable_variance_ratio_check
        self.enable_binomial_check = enable_binomial_check

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
        r"""Updates sample history and running TP/FN/FP counts for diagnostics.

        Args:
            scores: Proxy scores in $[0, 1]$ for the samples.
            labels: Binary (True/False) oracle outputs for the samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.
        """
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

    def _create_mart_R(self, k: int, reverse: bool = False) -> TestSupermartingale:
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
        rvs = np.asarray(
            (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos,
            dtype=np.float64,
        )
        if weights is not None:
            rvs *= weights
        return rvs

    def asymptotic_diag_R(
        self,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        """Builds recall asymptotic validity diagnostics for threshold `k_lower`.

        Returns None for the lower anchor index `k_lower == 0`.
        """
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
            num_true_positives=int(num_true_positives),
            num_false_negatives_or_positives=int(num_false_negatives),
            null_failure_prob=null_failure_prob,
            metric="Recall",
            thresholds=(tau_lower,),
            variance_ratio_bound=self.variance_ratio_bound,
            delta=self.delta_R,
            enable_variance_ratio_check=self.enable_variance_ratio_check,
            enable_binomial_check=self.enable_binomial_check,
        )

    def asymptotic_diag_P(
        self,
        k_upper: int,
        k_lower: int,
    ) -> AsymptoticValidityDiagnostics | None:
        """Builds precision asymptotic validity diagnostics for `(k_upper, k_lower)`."""
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
            num_true_positives=int(num_true_positives),
            num_false_negatives_or_positives=int(num_false_positives),
            null_failure_prob=null_failure_prob,
            metric="Precision",
            thresholds=(tau_lower, tau_upper),
            variance_ratio_bound=self.variance_ratio_bound,
            delta=delta,
            enable_variance_ratio_check=self.enable_variance_ratio_check,
            enable_binomial_check=self.enable_binomial_check,
        )


class ACTIS(CascadeTuner):
    r"""Stateful anytime-valid cascade tuner using asymptotic confidence sequences."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: Sequence[float] | NDArray[np.float64],
        thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
        pop_size: int | None = None,
        v_0: float | PriorAndTargetVar = 0.01,
        min_positives: int | Literal["auto"] = "auto",
        p_floor: float | Literal["auto"] = "auto",
        enable_variance_ratio_check: bool = True,
        enable_binomial_check: bool = True,
        variance_ratio_bound: float = 0.05,
    ) -> None:
        r"""Initializes the asymptotic ACTIS cascade threshold tuner.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$ that the tuning procedure
                fails to satisfy the precision and recall targets.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            pop_size: Optional population dataset size. When provided, enables duplicate
                index checks in `add_samples`, scales `"auto"` `min_positives`, and
                bounds remaining unlabeled items in `should_continue_sampling`.
            v_0: Prior variance for the Gaussian mixture supermartingale. Can be a
                scalar float or a `PriorAndTargetVar` instance (default: 0.01).
            min_positives: Minimum number of positive oracle observations required
                during early stopping warmup in `should_continue_sampling`.
            p_floor: Assumed or estimated lower-bound positive prevalence for
                population scaling of `min_positives`.
            enable_variance_ratio_check: If True (default), verifies empirical
                quadratic variation accumulation $V_t / v_0 \ge \rho$.
            enable_binomial_check: If True (default), verifies candidate thresholds
                against an exact one-sided binomial test.
            variance_ratio_bound: Minimum required ratio of cumulative empirical
                variance to target prior variance $V_t / v_0$ (default: 0.05).
        """
        super().__init__(
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=thresholds,
            thresholds_upper=thresholds_upper,
            pop_size=pop_size,
            min_positives=min_positives,
            p_floor=p_floor,
        )
        self.enable_variance_ratio_check = enable_variance_ratio_check
        self.enable_binomial_check = enable_binomial_check
        self._conf_seqs = AsymptoticCascadeConfSeqs(
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=self.thresholds,
            thresholds_upper=self.thresholds_upper,
            v_0=v_0,
            variance_ratio_bound=variance_ratio_bound,
            enable_variance_ratio_check=enable_variance_ratio_check,
            enable_binomial_check=enable_binomial_check,
        )

    @property
    def conf_seqs(self) -> AsymptoticCascadeConfSeqs:
        """Returns the asymptotic cascade confidence sequences."""
        return self._conf_seqs

    def _diagnose_recall_candidate(
        self, k_lower: int
    ) -> AsymptoticValidityDiagnostics | None:
        if not (self.enable_variance_ratio_check or self.enable_binomial_check):
            return None
        return self._conf_seqs.asymptotic_diag_R(k_lower)

    def _diagnose_precision_candidate(
        self, k_upper: int, k_lower: int
    ) -> AsymptoticValidityDiagnostics | None:
        if not (self.enable_variance_ratio_check or self.enable_binomial_check):
            return None
        return self._conf_seqs.asymptotic_diag_P(k_upper, k_lower)
