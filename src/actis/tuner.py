from collections.abc import Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .confseq import eval_asymptotic_wealth, eval_betting_wealth


@dataclass
class CascadeThresholds:
    """Stores calibrated thresholds and associated diagnostics."""
    tau_pos: float
    tau_neg: float
    num_samples: int
    predicted_oracle_rate: float | None = None

    @property
    def tau_upper(self) -> float:
        return self.tau_pos

    @property
    def tau_lower(self) -> float:
        return self.tau_neg


class ACTIS:
    r"""Stateful anytime-valid threshold tuner for LOTUS cascades using confidence
    sequences.

    Supports both finite-sample betting supermartingales (`conf_seq='finite'`) and
    asymptotic Gaussian mixture test supermartingales (`conf_seq='asymptotic'`).

    If `conf_seq='finite'`, observations and proxy scores are assumed to be bounded
    in the unit interval [0, 1]. If `conf_seq='asymptotic'`, observations are not
    required to be in [0, 1] and only require finite variance.
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
        v0: float = 0.01,
        max_weight_ge: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_lt: Sequence[float] | NDArray[np.float64] | None = None,
        max_weight_ge_upper: Sequence[float] | NDArray[np.float64] | None = None,
    ):
        r"""
        Args:
            gamma_P: Probability of a positive false alarm.
            gamma_R: Probability of a negative false alarm.
            delta: Failure probability.
            thresholds: Grid of candidate thresholds in [0, 1], determined independently
                of the sample. If `thresholds_upper` is None, the same grid is used for
                tuning the upper and lower thresholds. Otherwise, this grid is used for
                the lower threshold and `thresholds_upper` is used for the upper
                threshold.
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
                'asymptotic' for Gaussian mixture supermartingales (Howard et al. 2021,
                which avoid the w_max penalty under importance sampling).
            v0: Prior variance parameter for `conf_seq='asymptotic'` (default 0.01).
            max_weight_ge: If using importance sampling with `conf_seq='finite'`, it is
                necessary to provide upper bounds on the importance weight $w(x)$ for
                each $\tau$ in `thresholds`, $\max_{x: s(x) >= \tau} w(x)$. The bound
                must be taken over all items $x$ in the dataset. Not required for
                `conf_seq='asymptotic'`
            max_weight_lt: If using importance sampling with `conf_seq='finite'`,
                tighter confidence intervals can be obtained by providing upper bounds
                $\max_{x: s(x) < \tau} w(x)$ for each $\tau$ in `thresholds`. If None, a
                looser bound using `max_weight_ge` is used. Not required for
                `conf_seq='asymptotic'`.
            max_weight_ge_upper: If using importance sampling with `conf_seq='finite'`
                and `thresholds_upper` is provided, it is necessary to provide
                `max_weight_ge` for the upper threshold grid. That is, provide upper
                bounds $\max_{x: s(x) >= \tau} w(x)$ for each $\tau$ in
                `thresholds_upper`. Not required for `conf_seq='asymptotic'`.
        """
        self.gamma_P = gamma_P
        self.gamma_R = gamma_R
        self.delta = delta
        self.pop_size = pop_size
        self.horizon = horizon
        self.conf_seq = conf_seq
        self.v0 = v0

        thresholds = np.asarray(thresholds, dtype=np.float64)
        self.M_lower = len(thresholds)
        self.thresholds, sort_idx = np.unique(thresholds, return_index=True)
        if len(sort_idx) < self.M_lower:
            raise ValueError("`thresholds` must contain unique values.")

        if thresholds_upper is not None:
            thresholds_upper = np.asarray(thresholds_upper, dtype=np.float64)
            self.M_upper = len(thresholds_upper)
            self.thresholds_upper, sort_idx_upper = np.unique(
                thresholds_upper,
                return_index=True
            )
            if len(sort_idx_upper) < self.M_upper:
                raise ValueError("`thresholds_upper` must contain unique values.")
        else:
            self.thresholds_upper = self.thresholds
            self.M_upper = self.M_lower
            sort_idx_upper = sort_idx

        self.max_weight_ge = None
        self.max_weight_lt = None
        self.max_weight_ge_upper = None

        if self.conf_seq == "finite":
            if max_weight_ge is not None:
                max_weight_ge = np.asarray(max_weight_ge, dtype=np.float64)
                if len(max_weight_ge) != self.M_lower:
                    raise ValueError(
                        "`thresholds` and `max_weight_ge` must have the same length."
                    )
                self.max_weight_ge = max_weight_ge[sort_idx]

            if max_weight_lt is not None:
                max_weight_lt = np.asarray(max_weight_lt, dtype=np.float64)
                if len(max_weight_lt) != self.M_lower:
                    raise ValueError(
                        "`thresholds` and `max_weight_lt` must have the same length."
                    )
                self.max_weight_lt = max_weight_lt[sort_idx]

            if thresholds_upper is not None:
                if max_weight_ge_upper is not None:
                    max_weight_ge_upper = np.asarray(max_weight_ge_upper, dtype=np.float64)
                    if len(max_weight_ge_upper) != self.M_upper:
                        raise ValueError(
                            "`thresholds_upper` and `max_weight_ge_upper` must have the same length."
                        )
                    self.max_weight_ge_upper = max_weight_ge_upper[sort_idx_upper]
            else:
                self.max_weight_ge_upper = self.max_weight_ge

        self.gamma_R = gamma_R
        self.gamma_P = gamma_P
        self.delta = delta
        self.delta_R = self.delta / 2
        self.delta_P = self.delta / 2

        # State tracking
        self.seen_indices: set[int] = set()
        self.scores_list: list[float] = []
        self.labels_list: list[bool] = []
        self.weights_list: list[float] = []
        self.using_weights = False

        self.current_thresholds: CascadeThresholds = CascadeThresholds(
            tau_pos=1.0,
            tau_neg=0.0,
            num_samples=0,
            predicted_oracle_rate=None,
        )

    def add_samples(
        self,
        indices: Sequence[int] | NDArray[np.integer],
        scores: Sequence[float] | NDArray[np.float64],
        labels: Sequence[bool] | NDArray[np.bool_],
        weights: Sequence[float] | NDArray[np.float64] | None = None,
        population_scores: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> CascadeThresholds:
        r"""Ingests a batch of labeled samples, updates the test supermartingale wealth
        and returns updated CascadeThresholds

        Args:
            indices: Population indices for the new samples.
            scores: Proxy scores in [0, 1] for the new samples.
            labels: Binary (True/False) oracle outputs for the new samples.
            weights: Optional importance weights $w(x) = p(x) / q(x)$ for the sample.
            population_scores: Optional population proxy scores in [0, 1] for estimating
                the expected oracle rate of the cascade. If None, the expected oracle
                rate is not computed.

        Returns:
            Updated CascadeThresholds with new thresholds and diagnostics.
        """

        if len(indices) != len(scores) or len(scores) != len(labels):
            raise ValueError(
                "`indices`, `scores`, and `labels` must all have the same "
                "length."
            )

        if self.pop_size is not None:
            indices_set = set(indices)
            if len(indices_set) < len(indices):
                raise ValueError(
                    "Duplicate sample indices encountered within the new "
                    "batch."
                )
            duplicates = indices_set.intersection(self.seen_indices)
            if len(duplicates) > 0:
                raise ValueError(
                    f"Encountered {len(duplicates)} duplicate sample indices "
                    "previously seen in without-replacement sampling."
                )

        self.seen_indices.update(indices)
        self.scores_list.extend(scores)
        self.labels_list.extend(labels)

        if weights is not None:
            self.using_weights = True
            if len(weights) != len(scores):
                raise ValueError(
                    "`weights` must have the same length as `scores`."
                )
            self.weights_list.extend(weights)
        else:
            if self.using_weights:
                raise ValueError(
                    "Cannot omit `weights` after previously providing weights."
                )
            self.weights_list.extend(repeat(1.0, len(scores)))

        tau_upper, tau_lower = self._tune_thresholds()

        predicted_oracle_rate = None
        if population_scores is not None:
            predicted_oracle_rate = self.predict_oracle_rate(
                population_scores,
                tau_upper=tau_upper,
                tau_lower=tau_lower
            )

        self.current_thresholds = CascadeThresholds(
            tau_pos=tau_upper,
            tau_neg=tau_lower,
            num_samples=len(self.seen_indices),
            predicted_oracle_rate=predicted_oracle_rate,
        )
        return self.current_thresholds

    def _tune_thresholds(
        self,
    ) -> tuple[float, float]:
        r"""Evaluates the test supermartingales across candidate threshold grids to find
        the optimal pair of thresholds

        Returns:
            Optimal pair of thresholds: (tau_pos, tau_neg)
        """
        all_scores = np.asarray(self.scores_list, dtype=np.float64)
        all_labels = np.asarray(self.labels_list, dtype=bool)
        all_weights = np.asarray(self.weights_list, dtype=np.float64)

        # 1. Recall check (conservative lower bound)
        def check_recall_target(k: int) -> bool:
            tau = self.thresholds[k]
            is_above_tau = (all_scores >= tau).astype(np.float64)
            unscaled_rvs = all_weights * all_labels * (is_above_tau - self.gamma_R)

            if self.conf_seq == "asymptotic":
                wealth = eval_asymptotic_wealth(unscaled_rvs, m=0.0, prior_var=self.v0)
                return wealth > (1.0 / self.delta_R)
            else:
                max_weight_ge_k = (
                    float(self.max_weight_ge[k])
                    if self.max_weight_ge is not None
                    else 1.0
                )
                if self.max_weight_lt is not None:
                    max_weight_lt_k = float(self.max_weight_lt[k])
                elif k == 0:
                    max_weight_lt_k = 0.0
                else:
                    first_max_weight_ge = (
                        float(self.max_weight_ge[0])
                        if self.max_weight_ge is not None
                        else 1.0
                    )
                    max_weight_lt_k = first_max_weight_ge

                shift_R = self.gamma_R * max_weight_lt_k
                width_R = shift_R + (1.0 - self.gamma_R) * max_weight_ge_k

                if width_R > 0:
                    f_R = (unscaled_rvs + shift_R) / width_R
                    target_mean = shift_R / width_R
                    wealth = eval_betting_wealth(
                        f_R,
                        m=target_mean,
                        alpha=self.delta_R,
                        population_size=self.pop_size,
                        horizon=self.horizon,
                    )
                    return wealth > (1.0 / self.delta_R)
                return True

        tau_lower = float(self.thresholds[0])
        max_weight_ge_lower = (
            float(self.max_weight_ge[0])
            if self.max_weight_ge is not None
            else 1.0
        )
        for k in range(self.M_lower):
            if check_recall_target(k):
                tau_lower = float(self.thresholds[k])
                if self.max_weight_ge is not None:
                    max_weight_ge_lower = float(self.max_weight_ge[k])
            else:
                break

        # 2. Precision check (conservative lower bound)
        def check_precision_target(k: int) -> bool:
            tau_upper = self.thresholds_upper[k]
            is_true_pos = all_labels & (all_scores >= tau_lower)
            is_false_pos = (~all_labels) & (all_scores >= tau_upper)
            unscaled_rvs = all_weights * (
                (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos
            )

            if self.conf_seq == "asymptotic":
                wealth = eval_asymptotic_wealth(unscaled_rvs, m=0.0, prior_var=self.v0)
                return wealth > (float(self.M_lower) / self.delta_P)
            else:
                max_weight_ge_upper_k = (
                    float(self.max_weight_ge_upper[k])
                    if self.max_weight_ge_upper is not None
                    else 1.0
                )

                shift_P = self.gamma_P * max_weight_ge_upper_k
                width_P = shift_P + (1.0 - self.gamma_P) * max_weight_ge_lower

                if width_P > 0:
                    f_P = (unscaled_rvs + shift_P) / width_P
                    target_mean = shift_P / width_P
                    wealth = eval_betting_wealth(
                        f_P,
                        m=target_mean,
                        alpha=self.delta_P / self.M_lower,
                        population_size=self.pop_size,
                        horizon=self.horizon,
                    )
                    return wealth > (float(self.M_lower) / self.delta_P)
                return True

        tau_upper = float(self.thresholds_upper[-1])
        for k in range(self.M_upper - 1, -1, -1):
            if self.thresholds_upper[k] < tau_lower:
                break
            if check_precision_target(k):
                tau_upper = float(self.thresholds_upper[k])
            else:
                break

        return tau_upper, tau_lower

    def compute_optimistic_thresholds(self) -> tuple[float, float]:
        r"""Computes the best-case (optimistic) candidate thresholds within the anytime
        confidence bounds on recall and precision margins.

        For each candidate threshold tau:
          - Conservative threshold acceptance uses Lower Confidence Bound / test of H0: mu <= 0.
          - Optimistic threshold plausibility uses Upper Confidence Bound / test of H0: mu >= 0.

        Returns:
            (tau_pos_opt, tau_neg_opt)
        """
        n_draws = len(self.scores_list)
        if n_draws == 0:
            return float(self.thresholds_upper[-1]), float(self.thresholds[0])

        all_scores = np.asarray(self.scores_list, dtype=np.float64)
        all_labels = np.asarray(self.labels_list, dtype=bool)
        all_weights = (
            np.asarray(self.weights_list, dtype=np.float64)
            if self.weights_list
            else np.ones(n_draws, dtype=np.float64)
        )

        # 1. Optimistic Recall Threshold (tau_neg_opt)
        opt_neg_idx = 0
        for k, tau in enumerate(self.thresholds):
            is_above_tau = (all_scores >= tau).astype(np.float64)
            unscaled_rvs = all_weights * all_labels * (is_above_tau - self.gamma_R)

            if self.conf_seq == "asymptotic":
                mu_hat = float(np.mean(unscaled_rvs))
                v_n = float(np.sum((unscaled_rvs - mu_hat) ** 2))
                denom_log = 0.5 * np.log(1.0 + v_n / self.v0)
                rad = (1.0 / n_draws) * np.sqrt(
                    2.0 * (self.v0 + v_n) * (np.log(1.0 / self.delta_R) + denom_log)
                )
                # Upper Confidence Bound on recall margin
                if mu_hat + rad >= 0.0:
                    opt_neg_idx = k
                else:
                    break
            else:
                max_weight_ge_k = (
                    float(self.max_weight_ge[k])
                    if self.max_weight_ge is not None
                    else 1.0
                )
                if self.max_weight_lt is not None:
                    max_weight_lt_k = float(self.max_weight_lt[k])
                elif k == 0:
                    max_weight_lt_k = 0.0
                else:
                    first_max_weight_ge = (
                        float(self.max_weight_ge[0])
                        if self.max_weight_ge is not None
                        else 1.0
                    )
                    max_weight_lt_k = first_max_weight_ge

                shift_R = self.gamma_R * max_weight_lt_k
                width_R = shift_R + (1.0 - self.gamma_R) * max_weight_ge_k

                if width_R > 0:
                    f_R = (unscaled_rvs + shift_R) / width_R
                    target_mean = shift_R / width_R
                    # Lower-tailed non-rejection test: H0: mean >= target_mean
                    wealth_rev = eval_betting_wealth(
                        1.0 - f_R,
                        m=1.0 - target_mean,
                        alpha=self.delta_R,
                        population_size=self.pop_size,
                        horizon=self.horizon,
                    )
                    if wealth_rev <= (1.0 / self.delta_R):
                        opt_neg_idx = k
                    else:
                        break
                else:
                    opt_neg_idx = k

        tau_neg_opt = float(self.thresholds[opt_neg_idx])

        # 2. Optimistic Precision Threshold (tau_pos_opt)
        opt_pos_idx = self.M_upper - 1
        for k in range(self.M_upper - 1, -1, -1):
            tau = self.thresholds_upper[k]
            if tau < tau_neg_opt:
                break
            is_true_pos = all_labels & (all_scores >= tau_neg_opt)
            is_false_pos = (~all_labels) & (all_scores >= tau)
            unscaled_rvs = all_weights * (
                (1.0 - self.gamma_P) * is_true_pos - self.gamma_P * is_false_pos
            )

            if self.conf_seq == "asymptotic":
                mu_hat = float(np.mean(unscaled_rvs))
                v_n = float(np.sum((unscaled_rvs - mu_hat) ** 2))
                denom_log = 0.5 * np.log(1.0 + v_n / self.v0)
                rad = (1.0 / n_draws) * np.sqrt(
                    2.0 * (self.v0 + v_n) *
                    (np.log(self.M_lower / self.delta_P) + denom_log)
                )
                # Upper Confidence Bound on precision margin
                if mu_hat + rad >= 0.0:
                    opt_pos_idx = k
                else:
                    break
            else:
                max_weight_ge_up = (
                    float(self.max_weight_ge_upper[k])
                    if self.max_weight_ge_upper is not None
                    else 1.0
                )
                max_weight_ge_low = (
                    float(self.max_weight_ge[opt_neg_idx])
                    if self.max_weight_ge is not None
                    else 1.0
                )
                shift_P = self.gamma_P * max_weight_ge_up
                width_P = shift_P + (1.0 - self.gamma_P) * max_weight_ge_low

                if width_P > 0:
                    f_P = (unscaled_rvs + shift_P) / width_P
                    target_mean = shift_P / width_P
                    # Lower-tailed non-rejection test: H0: mean >= target_mean
                    wealth_rev = eval_betting_wealth(
                        1.0 - f_P,
                        m=1.0 - target_mean,
                        alpha=self.delta_P / self.M_lower,
                        population_size=self.pop_size,
                        horizon=self.horizon,
                    )
                    if wealth_rev <= (float(self.M_lower) / self.delta_P):
                        opt_pos_idx = k
                    else:
                        break
                else:
                    opt_pos_idx = k

        tau_pos_opt = float(self.thresholds_upper[opt_pos_idx])
        return tau_pos_opt, tau_neg_opt

    def predict_oracle_rate(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        tau_upper: float | None = None,
        tau_lower: float | None = None,
    ) -> float:
        r"""Predicts the oracle call rate over the full population given thresholds."""
        pop_s = np.asarray(population_scores, dtype=np.float64)
        if tau_upper is None:
            tau_upper = self.current_thresholds.tau_pos
        if tau_lower is None:
            tau_lower = self.current_thresholds.tau_neg

        if tau_lower > tau_upper:
            return 1.0
        else:
            unsure = (pop_s >= tau_lower) & (pop_s < tau_upper)
            return float(np.mean(unsure))

    def should_continue_sampling(
        self,
        population_scores: Sequence[float] | NDArray[np.float64],
        batch_size: int,
        max_sample_size: int | None = None,
        min_expected_savings: float = 0.0,
        min_positives: int = 30,
    ) -> tuple[bool, dict[str, float]]:
        r"""Determines whether continuing to sample is justified based on the
        Uncertainty Gap between currently accepted conservative thresholds
        and the best-case (optimistic) thresholds within the confidence bounds.

        Max Plausible Savings = (r_curr - r_opt) * N_rem

        Sampling continues if and only if the maximum plausible savings
        exceed the marginal acquisition cost of the next batch:
            Plausible Savings > max(batch_size * (1 - r_curr), min_expected_savings)
        or if the minimum positive label count (min_positives) has not yet been reached.
        """
        pop_s = np.asarray(population_scores, dtype=np.float64)
        N = len(pop_s) if self.pop_size is None else self.pop_size
        n_draws = len(self.scores_list)
        n_seen = len(self.seen_indices)
        N_rem = max(0, N - n_seen)

        if max_sample_size is None:
            max_sample_size = N

        num_positives = int(np.sum(self.labels_list))
        warmup_needed = (num_positives < min_positives)

        if N_rem == 0 or n_draws >= max_sample_size:
            return False, {
                "n_draws": float(n_draws),
                "n_seen": float(n_seen),
                "N_rem": float(N_rem),
                "num_positives": float(num_positives),
                "min_positives": float(min_positives),
                "warmup_needed": float(warmup_needed),
                "r_curr": float(self.current_thresholds.predicted_oracle_rate or 0.0),
                "r_opt": float(self.current_thresholds.predicted_oracle_rate or 0.0),
                "marginal_batch_savings": 0.0,
                "marginal_batch_cost": 0.0,
                "threshold_savings": 0.0,
                "tau_pos_curr": float(self.current_thresholds.tau_pos),
                "tau_neg_curr": float(self.current_thresholds.tau_neg),
                "tau_pos_opt": float(self.current_thresholds.tau_pos),
                "tau_neg_opt": float(self.current_thresholds.tau_neg),
            }

        # 1. Guaranteed Conservative Thresholds and Oracle Rate
        tau_pos_curr = self.current_thresholds.tau_pos
        tau_neg_curr = self.current_thresholds.tau_neg
        r_curr = self.predict_oracle_rate(
            pop_s,
            tau_upper=tau_pos_curr,
            tau_lower=tau_neg_curr
        )

        # 2. Best-Case (Optimistic) Thresholds within Confidence Bounds
        tau_pos_opt, tau_neg_opt = self.compute_optimistic_thresholds()
        r_opt = self.predict_oracle_rate(
            pop_s,
            tau_upper=tau_pos_opt,
            tau_lower=tau_neg_opt
        )

        # 3. Marginal Value of Information for the next batch:
        # Expected reduction in deployment oracle calls from contracting the confidence
        # radius by B / (2n):
        # Delta Radius / Radius = B / (2n)
        marginal_batch_savings = (
            (float(batch_size) / (2.0 * max(1.0, float(n_draws))))
            * max(0.0, r_curr - r_opt)
            * float(N_rem)
        )

        # 4. Marginal cost of acquiring the next batch
        marginal_batch_cost = float(batch_size) * (1.0 - r_curr)
        threshold_savings = max(marginal_batch_cost, min_expected_savings)

        # Decision rule:
        # Continue if marginal batch savings > marginal batch cost, or if warm-up is
        # still needed
        should_continue = (
            (marginal_batch_savings > threshold_savings) or warmup_needed
        ) and (n_draws < max_sample_size)

        diagnostics = {
            "n_draws": float(n_draws),
            "n_seen": float(n_seen),
            "N_rem": float(N_rem),
            "num_positives": float(num_positives),
            "min_positives": float(min_positives),
            "warmup_needed": float(warmup_needed),
            "r_curr": float(r_curr),
            "r_opt": float(r_opt),
            "marginal_batch_savings": float(marginal_batch_savings),
            "marginal_batch_cost": float(marginal_batch_cost),
            "threshold_savings": float(threshold_savings),
            "tau_pos_curr": float(tau_pos_curr),
            "tau_neg_curr": float(tau_neg_curr),
            "tau_pos_opt": float(tau_pos_opt),
            "tau_neg_opt": float(tau_neg_opt),
        }
        return should_continue, diagnostics
