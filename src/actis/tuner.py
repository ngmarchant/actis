# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared base classes and diagnostics for adaptive cascade threshold tuners."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from .confseq import TestSupermartingale
from .threshold_grid import validate_thresholds

if TYPE_CHECKING:
    from .asymptotic import AsymptoticValidityDiagnostics


@dataclass
class CascadeThresholds:
    """Stores calibrated cascade thresholds and associated diagnostics.

    Attributes:
        tau_upper: Upper threshold; items with score >= `tau_upper` are accepted
            without querying the oracle (`inf` disables proxy-positive bypass).
        tau_lower: Lower threshold; items with score < `tau_lower` are discarded
            without querying the oracle.
        num_samples: Number of unique population indices labeled so far
            (`len(seen_indices)`), reflecting unique oracle calls since repeated
            draws reuse cached oracle labels.
        predicted_oracle_rate: Estimated fraction of the population routed to the
            oracle under `(tau_lower, tau_upper)`, if population scores were provided.
        asymptotic_diag_R: Recall asymptotic validity diagnostics, if applicable.
        asymptotic_diag_P: Precision asymptotic validity diagnostics, if applicable.
    """

    tau_upper: float
    tau_lower: float
    num_samples: int
    predicted_oracle_rate: float | None = None
    asymptotic_diag_R: AsymptoticValidityDiagnostics | None = None
    asymptotic_diag_P: AsymptoticValidityDiagnostics | None = None


@dataclass
class StoppingDiagnostics:
    """Diagnostics for the adaptive stopping decision of a cascade tuner.

    Attributes:
        n_draws: Total number of sample draws ingested so far (including repeats).
        n_seen: Number of unique population indices labeled so far.
        N_rem: Remaining unlabeled items in the population.
        num_positives: Total positive oracle labels observed across ingested draws.
        min_positives: Minimum positive oracle count required before early stopping.
        warmup_needed: Whether `num_positives < min_positives`.
        r_curr: Predicted oracle call rate at the current conservative thresholds.
        r_opt: Predicted oracle call rate at the optimistic thresholds.
        marginal_batch_savings: Estimated reduction in remaining oracle calls from
            acquiring the next batch.
        marginal_batch_cost: Oracle call cost of acquiring the next batch.
        threshold_savings: Required savings threshold `max(cost, min_expected_savings)`.
        tau_upper_curr: Current conservative upper threshold.
        tau_lower_curr: Current conservative lower threshold.
        tau_upper_opt: Optimistic upper threshold within confidence bounds.
        tau_lower_opt: Optimistic lower threshold within confidence bounds.
    """

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
    tau_upper_curr: float
    tau_lower_curr: float
    tau_upper_opt: float
    tau_lower_opt: float


class CascadeConfSeqs(ABC):
    """Abstract confidence sequence collection for two-threshold cascade calibration."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: NDArray[np.float64],
        thresholds_upper: NDArray[np.float64],
    ) -> None:
        r"""Initializes the cascade confidence sequences.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$.
            thresholds: Sorted 1D grid of candidate lower thresholds in $[0, 1]$,
                determined independently of samples.
            thresholds_upper: Sorted 1D grid of candidate upper thresholds in $[0, 1]$,
                determined independently of samples.

        Raises:
            ValueError: If `gamma_P`, `gamma_R`, or `delta` is not in $(0, 1)$.
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
        r"""Returns the wealth threshold $1 / \delta_R$ for certifying recall."""
        return 1.0 / self.delta_R

    @property
    def wealth_threshold_P(self) -> float:
        """Returns the Bonferroni-corrected wealth threshold for precision."""
        return len(self.thresholds) / self.delta_P

    def add_samples(
        self,
        scores: Sequence[float] | NDArray[np.float64],
        labels: Sequence[bool] | NDArray[np.bool_],
        weights: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> None:
        r"""Updates the internal sample history.

        Args:
            scores: Proxy scores in $[0, 1]$ for the samples.
            labels: Binary (True/False) oracle outputs for the samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.

        Raises:
            ValueError: If `weights` length mismatches `scores` or is omitted after
                having been provided for earlier samples.
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
        """Conservative check rejecting H0: mu <= 0 at level delta_P / M_lower."""
        wealth = self._get_mart_P(k_upper, k_lower, reverse=False).wealth()
        return wealth > self.wealth_threshold_P

    def is_precision_plausible(self, k_upper: int, k_lower: int) -> bool:
        """Optimistic check failing to reject H0: mu >= 0 at level delta_P / M_lower."""
        wealth = self._get_mart_P(k_upper, k_lower, reverse=True).wealth()
        return wealth <= self.wealth_threshold_P

    def _get_unprocessed_samples(
        self, start_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64] | None]:
        """Returns the slice of historical samples from `start_idx` to the latest."""
        return (
            self.scores[start_idx:],
            self.labels[start_idx:],
            self.weights[start_idx:] if self.weights is not None else None,
        )

    def _get_mart_R(self, k: int, reverse: bool = False) -> TestSupermartingale:
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
        self, k_upper: int, k_lower: int, reverse: bool = False
    ) -> TestSupermartingale:
        """Returns an up-to-date precision supermartingale for `(k_upper, k_lower)`."""
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
            cache[k_upper] = self._create_mart_P(k_upper, k_lower, reverse=reverse)
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
    def _create_mart_R(self, k: int, reverse: bool = False) -> TestSupermartingale:
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


class CascadeTuner(ABC):
    """Abstract base class for stateful anytime-valid cascade threshold tuners."""

    def __init__(
        self,
        gamma_P: float,
        gamma_R: float,
        delta: float,
        thresholds: Sequence[float] | NDArray[np.float64],
        thresholds_upper: Sequence[float] | NDArray[np.float64] | None = None,
        pop_size: int | None = None,
        min_positives: int | Literal["auto"] = "auto",
        p_floor: float | Literal["auto"] = "auto",
    ) -> None:
        r"""Initializes the cascade threshold tuner.

        Args:
            gamma_P: Target precision in $(0, 1)$.
            gamma_R: Target recall in $(0, 1)$.
            delta: Joint failure probability in $(0, 1)$ that the tuning procedure
                fails to satisfy the precision and recall targets.
            thresholds: Grid of candidate thresholds in $[0, 1]$, determined
                independently of samples. If `thresholds_upper` is None, the same grid
                is used for both upper and lower thresholds; otherwise, this grid is
                used for the lower threshold.
            thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$
                for the upper threshold. If None, defaults to `thresholds`.
            pop_size: Optional population dataset size. When provided, enables duplicate
                index checks in `add_samples`, scales `"auto"` `min_positives`, and
                bounds remaining unlabeled items in `should_continue_sampling`.
            min_positives: Minimum number of positive oracle observations required
                before early stopping is permitted in `should_continue_sampling`. If
                `"auto"` (default), dynamically calibrated from `delta`, `pop_size`,
                and proxy scores.
            p_floor: Assumed or estimated lower-bound positive prevalence for
                population scaling of `min_positives` in $(0, 1)$ or `"auto"`
                (default).

        Raises:
            ValueError: If `pop_size` is not a positive integer, `p_floor` is not in
                $(0, 1)$ or `"auto"`, or `thresholds`/`thresholds_upper` are invalid.
        """
        if pop_size is not None and (pop_size <= 0 or pop_size != int(pop_size)):
            raise ValueError("Parameter `pop_size` must be a positive integer.")
        self.pop_size = pop_size

        self.gamma_P = gamma_P
        self.gamma_R = gamma_R
        self.delta = delta
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

        self.seen_indices: set[int] = set()

        self.current_thresholds: CascadeThresholds = CascadeThresholds(
            tau_upper=float("inf"),
            tau_lower=float(self.thresholds[0]),
            num_samples=0,
            predicted_oracle_rate=None,
        )

        self._cached_pop_scores = None
        self._cached_sorted_scores: NDArray[np.float64] | None = None
        self._cached_mean_score: float | None = None

    @property
    @abstractmethod
    def conf_seqs(self) -> CascadeConfSeqs:
        """Returns the confidence sequences used to certify candidate thresholds."""
        pass

    def _diagnose_recall_candidate(
        self, k_lower: int
    ) -> AsymptoticValidityDiagnostics | None:
        return None

    def _diagnose_precision_candidate(
        self, k_upper: int, k_lower: int
    ) -> AsymptoticValidityDiagnostics | None:
        return None

    def _ensure_population_scores_cached(
        self,
        scores: Sequence[float] | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""Ensures sorted population scores and the mean proxy score are cached.

        Returns:
            A tuple of `(sorted_scores, mean_score)`.
        """
        if (
            self._cached_pop_scores is not scores
            or self._cached_sorted_scores is None
            or self._cached_mean_score is None
        ):
            arr = np.asarray(scores, dtype=np.float64)
            self._cached_pop_scores = scores
            self._cached_sorted_scores = np.sort(arr)
            self._cached_mean_score = float(np.mean(arr))

        return self._cached_sorted_scores, self._cached_mean_score

    def _resolve_min_positives(
        self,
        N: int | None = None,
        scores: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> int:
        r"""Resolves the minimum positive oracle count required for warmup."""
        if self.min_positives != "auto":
            return int(self.min_positives)

        delta_R = self.delta / 2.0
        cap = max(5, int(np.ceil(np.round(8.0 * math.log(1.0 / delta_R), 9))))

        pop_size = self.pop_size if self.pop_size is not None else N
        if pop_size is None:
            return cap

        if self.p_floor == "auto":
            if scores is not None and len(scores) > 0:
                _, mean_score = self._ensure_population_scores_cached(scores)
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
        r"""Ingests a batch of labeled samples and updates calibrated thresholds.

        Updates test supermartingale wealth processes, records unique seen population
        indices, and returns the updated `CascadeThresholds`.

        Args:
            indices: Indices for the samples in the population dataset.
            scores: Proxy scores in $[0, 1]$ for the samples.
            labels: Binary (True/False) oracle outputs for the samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.
            population_scores: Optional proxy scores in $[0, 1]$ for the population
                dataset, used to estimate the expected oracle rate of the cascade. If
                None, the expected oracle rate is not computed.

        Returns:
            Updated `CascadeThresholds` with calibrated thresholds and diagnostics.

        Raises:
            ValueError: If `indices`, `scores`, and `labels` lengths differ, or if
                duplicate indices are encountered when `pop_size` is set.
        """
        if len(indices) != len(scores) or len(scores) != len(labels):
            raise ValueError(
                "`indices`, `scores`, and `labels` must all have the same length."
            )

        if self.pop_size is not None:
            unique_indices = set(indices)
            if len(unique_indices) < len(indices):
                raise ValueError("Duplicate indices encountered within the sample.")
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
            tau_upper=tau_upper,
            tau_lower=tau_lower,
            num_samples=len(self.seen_indices),
            predicted_oracle_rate=predicted_oracle_rate,
            asymptotic_diag_R=diag_R,
            asymptotic_diag_P=diag_P,
        )

        return self.current_thresholds

    def _tune_thresholds(
        self,
    ) -> tuple[
        float,
        float,
        AsymptoticValidityDiagnostics | None,
        AsymptoticValidityDiagnostics | None,
    ]:
        r"""Selects the tightest certified lower and upper cascade thresholds."""
        # Forward sequential scan stopping at first failure of recall target
        k_lower = 0
        tau_lower = float(self.thresholds[0])
        diag_R = None
        for k in range(len(self.thresholds)):
            if not self.conf_seqs.is_recall_satisfied(k):
                break

            d = self._diagnose_recall_candidate(k)
            if d is not None and not d.is_valid():
                if k_lower == 0:
                    diag_R = d
                break

            k_lower = k
            tau_lower = float(self.thresholds[k])
            diag_R = d

        # Backward sequential scan stopping at first failure of precision target.
        # Falls back to +inf (disabling proxy-positive bypass, Precision = 1) if no
        # candidate in `thresholds_upper` is certified.
        k_upper: int | None = None
        tau_upper = float("inf")
        diag_P = None
        for k in range(len(self.thresholds_upper) - 1, -1, -1):
            tau_k = float(self.thresholds_upper[k])

            if (tau_k < tau_lower) or not self.conf_seqs.is_precision_satisfied(
                k, k_lower
            ):
                break

            d = self._diagnose_precision_candidate(k, k_lower)
            if d is not None and not d.is_valid():
                if k_upper is None:
                    diag_P = d
                break

            k_upper = k
            tau_upper = tau_k
            diag_P = d

        return (tau_upper, tau_lower, diag_R, diag_P)

    def _compute_optimistic_thresholds(self) -> tuple[float, float]:
        r"""Computes optimistic candidate thresholds within confidence bounds."""
        if len(self.seen_indices) == 0:
            return float("inf"), float(self.thresholds[0])

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

        tau_lower_opt = float(self.thresholds[opt_neg_idx])

        # Optimistic precision threshold via binary search
        start_k = int(
            np.searchsorted(self.thresholds_upper, tau_lower_opt, side="left")
        )
        low_pos, high_pos = start_k, len(self.thresholds_upper) - 1
        opt_pos_idx: int | None = None

        while low_pos <= high_pos:
            mid_k = (low_pos + high_pos) // 2
            if self.conf_seqs.is_precision_plausible(mid_k, opt_neg_idx):
                opt_pos_idx = mid_k
                high_pos = mid_k - 1
            else:
                low_pos = mid_k + 1

        tau_upper_opt = (
            float(self.thresholds_upper[opt_pos_idx])
            if opt_pos_idx is not None
            else float("inf")
        )
        return tau_upper_opt, tau_lower_opt

    def predict_oracle_rate(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        tau_upper: float | None = None,
        tau_lower: float | None = None,
    ) -> float:
        r"""Predicts the oracle call rate over the full population given thresholds.

        Returns `1.0` conservatively if `tau_lower > tau_upper`, and `0.0` if
        `population_scores` is empty.

        Args:
            population_scores: Proxy scores in $[0, 1]$ for the population dataset.
            tau_upper: Optional upper threshold. If None, uses the current internal
                upper threshold.
            tau_lower: Optional lower threshold. If None, uses the current internal
                lower threshold.

        Returns:
            The predicted oracle call rate as a float in $[0, 1]$.
        """
        if tau_upper is None:
            tau_upper = self.current_thresholds.tau_upper
        if tau_lower is None:
            tau_lower = self.current_thresholds.tau_lower

        if tau_lower > tau_upper:
            return 1.0

        sorted_scores, _ = self._ensure_population_scores_cached(population_scores)
        n = len(sorted_scores)
        if n == 0:
            return 0.0

        idx_lower = int(np.searchsorted(sorted_scores, tau_lower, side="left"))
        idx_upper = int(np.searchsorted(sorted_scores, tau_upper, side="left"))
        return float((idx_upper - idx_lower) / n)

    def should_continue_sampling(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        batch_size: int,
        max_sample_size: int | None = None,
        min_expected_savings: float = 0.0,
    ) -> tuple[bool, StoppingDiagnostics]:
        r"""Determines whether continuing to sample is economically justified.

        Evaluates the uncertainty gap between currently accepted conservative
        thresholds and optimistic thresholds within the confidence bounds.

        Args:
            population_scores: Proxy scores in $[0, 1]$ for the population dataset.
            batch_size: Number of samples to acquire in the next batch.
            max_sample_size: Optional maximum number of sample draws to acquire. If
                None, defaults to the population size.
            min_expected_savings: Minimum expected savings threshold for continuing
                sampling. If the expected savings from acquiring the next batch is
                below this threshold, sampling will stop.

        Returns:
            A tuple containing:
                - A boolean indicating whether to continue sampling.
                - A `StoppingDiagnostics` object detailing the stopping decision.
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
                tau_upper_curr=self.current_thresholds.tau_upper,
                tau_lower_curr=self.current_thresholds.tau_lower,
                tau_upper_opt=self.current_thresholds.tau_upper,
                tau_lower_opt=self.current_thresholds.tau_lower,
            )
            return False, diagnostics

        # Conservative thresholds and oracle rate
        tau_upper_curr = self.current_thresholds.tau_upper
        tau_lower_curr = self.current_thresholds.tau_lower
        r_curr = self.predict_oracle_rate(
            population_scores,
            tau_upper=tau_upper_curr,
            tau_lower=tau_lower_curr,
        )

        # Optimistic thresholds (within confidence bounds) and oracle rate
        tau_upper_opt, tau_lower_opt = self._compute_optimistic_thresholds()
        r_opt = self.predict_oracle_rate(
            population_scores,
            tau_upper=tau_upper_opt,
            tau_lower=tau_lower_opt,
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
            tau_upper_curr=tau_upper_curr,
            tau_lower_curr=tau_lower_curr,
            tau_upper_opt=tau_upper_opt,
            tau_lower_opt=tau_lower_opt,
        )
        return should_continue, diagnostics
