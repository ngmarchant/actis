# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Finite-sample betting confidence sequences and ACTISFinite tuner."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .confseq import BettingSupermartingale, TestSupermartingale
from .tuner import CascadeConfSeqs, CascadeTuner


class FiniteSampleCascadeConfSeqs(CascadeConfSeqs):
    """Finite-sample betting supermartingale strategy for cascade calibration."""

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
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> None:
        r"""Initializes the finite-sample cascade confidence sequences.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$.
            thresholds: Sorted 1D grid of candidate lower thresholds in $[0, 1]$,
                determined independently of samples.
            thresholds_upper: Sorted 1D grid of candidate upper thresholds in $[0, 1]$,
                determined independently of samples.
            pop_size: Population dataset size when sampling uniformly without
                replacement. Must be None when sampling with replacement.
            horizon: Fixed sample size horizon for tuning the betting strategy. If None
                (default), uses anytime-valid predictable bets for adaptive stopping.
            max_weight_ge: Upper bounds on importance weights $w(x)$ over items with
                proxy score $s(x) \ge \tau$ for each $\tau$ in `thresholds`. Required
                for importance sampling.
            max_weight_lt: Optional upper bounds on $w(x)$ over items with proxy score
                $s(x) < \tau$ for each $\tau$ in `thresholds`.
            max_weight_ge_upper: Upper bounds on $w(x)$ over items with $s(x) \ge \tau$
                for each $\tau$ in `thresholds_upper`. Required when `max_weight_ge` is
                provided and `thresholds_upper` differs from `thresholds`.

        Raises:
            ValueError: If `pop_size` or `horizon` is not a positive integer, or if
                weight bound arrays have mismatched lengths or omit
                `max_weight_ge_upper` when `thresholds_upper` differs from `thresholds`.
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
            if (
                self.max_weight_ge is not None
                and self.thresholds_upper is not self.thresholds
                and (
                    len(self.thresholds_upper) != len(self.thresholds)
                    or not np.array_equal(self.thresholds_upper, self.thresholds)
                )
            ):
                raise ValueError(
                    "`max_weight_ge_upper` must be provided when `thresholds_upper` "
                    "differs from `thresholds` and `max_weight_ge` is set."
                )
            self.max_weight_ge_upper = self.max_weight_ge

    def _get_shift_width_R(self, k: int) -> tuple[float, float]:
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

    def _get_shift_width_P(self, k_upper: int, k_lower: int) -> tuple[float, float]:
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

    def _create_mart_R(self, k: int, reverse: bool = False) -> TestSupermartingale:
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
        unscaled_rvs = np.asarray(
            (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos,
            dtype=np.float64,
        )
        if weights is not None:
            unscaled_rvs *= weights
        shift_P, width_P = self._get_shift_width_P(k_upper, k_lower)
        if width_P > 0:
            return np.asarray((unscaled_rvs + shift_P) / width_P, dtype=np.float64)
        return np.zeros_like(unscaled_rvs, dtype=np.float64)


class ACTISFinite(CascadeTuner):
    r"""Stateful anytime-valid cascade tuner using finite-sample betting sequences."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: Sequence[float] | NDArray[np.float64],
        thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
        pop_size: int | None = None,
        horizon: int | None = None,
        min_positives: int | Literal["auto"] = "auto",
        p_floor: float | Literal["auto"] = "auto",
        max_weight_ge: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_lt: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> None:
        r"""Initializes the finite-sample betting ACTIS cascade threshold tuner.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$ that the tuning procedure
                fails to satisfy the precision and recall targets.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            pop_size: Population dataset size when sampling uniformly without
                replacement. Enables hypergeometric betting in `BettingSupermartingale`,
                so must be None when sampling with replacement.
            horizon: Fixed sample size for tuning the betting strategy. If None
                (default), uses anytime-valid predictable bets for adaptive stopping.
            min_positives: Minimum number of positive oracle observations required
                during early stopping warmup in `should_continue_sampling`.
            p_floor: Assumed or estimated lower-bound positive prevalence for
                population scaling of `min_positives`.
            max_weight_ge: Upper bounds on importance weight $w(x)$ over items with
                proxy score $s(x) \ge \tau$ for each $\tau$ in `thresholds`.
            max_weight_lt: Optional upper bounds on $w(x)$ over items with proxy score
                $s(x) < \tau$ for each $\tau$ in `thresholds`.
            max_weight_ge_upper: Upper bounds on $w(x)$ over items with $s(x) \ge \tau$
                for each $\tau$ in `thresholds_upper`.
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
        self._conf_seqs = FiniteSampleCascadeConfSeqs(
            gamma_P=gamma_P,
            gamma_R=gamma_R,
            delta=delta,
            thresholds=self.thresholds,
            thresholds_upper=self.thresholds_upper,
            pop_size=pop_size,
            horizon=horizon,
            max_weight_ge=max_weight_ge,
            max_weight_lt=max_weight_lt,
            max_weight_ge_upper=max_weight_ge_upper,
        )

    @property
    def conf_seqs(self) -> FiniteSampleCascadeConfSeqs:
        """Returns the finite-sample cascade confidence sequences."""
        return self._conf_seqs
