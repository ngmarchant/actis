import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class BaseScenario(ABC):
    """Abstract base class for synthetic coverage scenarios."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a synthetic population dataset of proxy scores and oracle labels.

        Args:
            pop_size: Total number of items in the population. If None, uses the
                scenario default.
            rng: NumPy random generator for reproducibility. If None, creates a
                default generator.
            **kwargs: Optional scenario-specific parameters, such as gamma_P
                or gamma_R.

        Returns:
            A tuple (proxy_scores, oracle_outputs):
                - proxy_scores: 1D array of float proxy confidence scores in [0.0, 1.0].
                - oracle_outputs: 1D boolean array of true labels (True for positive,
                  False for negative).
        """
        pass

    def evaluate_true_recall(
        self,
        tau_lower: float,
        proxy_scores: NDArray[np.float64],
        oracle_outputs: NDArray[np.bool_],
    ) -> float:
        """
        Calculates population recall for a given lower threshold.

        Items with proxy_score >= tau_lower are retained (either accepted by the proxy
        or sent to the oracle).

        Args:
            tau_lower: Lower threshold below which items are dropped.
            proxy_scores: 1D array of proxy confidence scores.
            oracle_outputs: 1D boolean array of true labels.
        """
        retained = (proxy_scores >= tau_lower) & oracle_outputs
        total_positives = np.sum(oracle_outputs)
        return float(np.sum(retained) / total_positives) if total_positives > 0 else 1.0

    def evaluate_true_precision(
        self,
        tau_upper: float,
        tau_lower: float,
        proxy_scores: NDArray[np.float64],
        oracle_outputs: NDArray[np.bool_],
    ) -> float:
        """
        Calculates population precision for thresholds.

        Items with score >= tau_upper are accepted by the proxy. Items with
        tau_lower <= score < tau_upper are sent to the oracle.

        Args:
            tau_upper: Upper threshold above which items are accepted by the proxy.
            tau_lower: Lower threshold below which items are dropped.
            proxy_scores: 1D array of proxy confidence scores.
            oracle_outputs: 1D boolean array of true labels.
        """
        helper_accepted = (proxy_scores >= tau_upper)
        oracle_sent = (proxy_scores >= tau_lower) & (proxy_scores < tau_upper)
        oracle_positives = oracle_sent & oracle_outputs

        true_positives = np.sum(helper_accepted & oracle_outputs) + np.sum(oracle_positives)
        predicted_positives = np.sum(helper_accepted) + np.sum(oracle_positives)

        if predicted_positives > 0:
            return float(true_positives / predicted_positives)

        return 1.0

    def evaluate_oracle_call_rate(
        self,
        tau_upper: float,
        tau_lower: float,
        proxy_scores: NDArray[np.float64],
    ) -> float:
        """
        Calculates population oracle call rate for given thresholds.

        Items with tau_lower <= score < tau_upper are routed to the expensive oracle.

        Args:
            tau_upper: Upper threshold above which items are accepted by the proxy.
            tau_lower: Lower threshold below which items are dropped.
            proxy_scores: 1D array of proxy confidence scores.

        Returns:
            Fraction of the population sent to the oracle in [0.0, 1.0].
        """
        if tau_upper <= tau_lower:
            return 0.0
        oracle_sent = (proxy_scores >= tau_lower) & (proxy_scores < tau_upper)
        return float(np.mean(oracle_sent)) if len(proxy_scores) > 0 else 0.0

    def compute_ideal_oracle_call_rate(
        self,
        proxy_scores: NDArray[np.float64],
        oracle_outputs: NDArray[np.bool_],
        gamma_R: float,
        gamma_P: float,
    ) -> float:
        """
        Computes the minimum oracle call rate satisfying precision and recall targets.

        Finds the optimal pair of thresholds (tau_upper, tau_lower) on the full
        population that achieves Recall >= gamma_R and Precision >=
        gamma_P while minimizing oracle queries.

        Args:
            proxy_scores: 1D array of proxy confidence scores.
            oracle_outputs: 1D boolean array of true labels.
            gamma_R: Minimum required population recall.
            gamma_P: Minimum required population precision.

        Returns:
            The theoretical minimum oracle call rate in [0.0, 1.0].
        """
        scores = np.asarray(proxy_scores, dtype=np.float64)
        oracle = np.asarray(oracle_outputs, dtype=bool)

        total_weight = len(scores)
        total_pos_weight = np.sum(oracle)

        if total_pos_weight == 0 or total_weight == 0:
            return 0.0

        # Sort population by score ascending
        sort_idx = np.argsort(scores)
        s_sorted = scores[sort_idx]
        o_sorted = oracle[sort_idx]

        pos_counts = np.where(o_sorted, 1.0, 0.0)
        neg_counts = np.where(~o_sorted, 1.0, 0.0)

        # Group by unique proxy scores
        unq, unq_inv = np.unique(s_sorted, return_inverse=True)
        unq_pos = np.bincount(unq_inv, weights=pos_counts)
        unq_neg = np.bincount(unq_inv, weights=neg_counts)
        unq_tot = np.bincount(unq_inv)

        M = len(unq)

        TP_suffix = np.zeros(M + 1, dtype=np.float64)
        TP_suffix[:M] = np.cumsum(unq_pos[::-1])[::-1]

        FP_suffix = np.zeros(M + 1, dtype=np.float64)
        FP_suffix[:M] = np.cumsum(unq_neg[::-1])[::-1]

        Tot_suffix = np.zeros(M + 1, dtype=np.float64)
        Tot_suffix[:M] = np.cumsum(unq_tot[::-1])[::-1]

        valid_k = np.where(TP_suffix / total_pos_weight >= gamma_R)[0]
        if len(valid_k) == 0:
            return 1.0

        tp = TP_suffix[valid_k]
        if gamma_P > 0:
            max_fp = tp * (1.0 - gamma_P) / gamma_P
        else:
            max_fp = np.full_like(tp, np.inf)

        # Check if helper model alone satisfies precision
        fp = FP_suffix[valid_k]
        if np.any(fp <= max_fp):
            return 0.0

        neg_FP_suffix = -FP_suffix
        idx = np.searchsorted(neg_FP_suffix, -max_fp, side="left")
        k_pos = np.maximum(valid_k + 1, idx)

        valid_pos = k_pos <= M
        if not np.any(valid_pos):
            return 1.0

        oracle_w = Tot_suffix[valid_k[valid_pos]] - Tot_suffix[k_pos[valid_pos]]
        min_oracle_rate = float(np.min(oracle_w / total_weight))
        return min(1.0, max(0.0, min_oracle_rate))


class UninformativeProxyFloor(BaseScenario):
    """
    Uninformative Proxy Floor Stress Test.

    Positives are rare (1.5% of the population) and the proxy scores carry minimal
    ranking signal (AUC ~0.6). Because true precision in [0.35, 0.85] is only ~3%,
    the proxy cannot accept items on its own. This forces a high theoretical oracle
    call floor (>= 55% at 80% recall).

    This scenario tests whether algorithms correctly adhere to the high oracle floor
    instead of prematurely under-calling the oracle and violating guarantees.
    """

    def __init__(self):
        super().__init__(
            name="uninformative_proxy_floor",
            description="Weak proxy (AUC ~0.6) and 1.5% prevalence forcing an oracle "
            "floor >= 55%.",
        )

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population where scores carry minimal ranking signal.

        Positives make up 1.5% of the data and have scores uniformly distributed in
        [0.35, 0.85]. Negatives have scores uniformly distributed in [0.0, 1.0].

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        scores = rng.uniform(0.0, 1.0, size=pop_size)
        oracle = np.zeros(pop_size, dtype=bool)

        n_positives = int(0.015 * pop_size)
        pos_idx = rng.choice(pop_size, size=n_positives, replace=False)
        oracle[pos_idx] = True

        # Positives have proxy scores uniformly in [0.35, 0.85]
        scores[pos_idx] = rng.uniform(0.35, 0.85, size=n_positives)

        return scores, oracle


class PrecisionTailOverfit(BaseScenario):
    """
    Precision Tail Overfitting.

    The upper tail of proxy scores [0.75, 1.0] has a true population precision
    equal to (gamma_P - delta_P), which is strictly below the precision target.

    Because high scores are sparse, a small sample drawn from the tail often contains
    only true positives by chance. When this happens, the observed sample variance
    evaluates to zero. Naive variance estimators collapse their confidence intervals,
    falsely concluding that precision is 100% and accepting thresholds that are too low.
    """

    def __init__(self, gamma_P: float = 0.8, delta_P: float = 0.15):
        super().__init__(
            name="precision_tail_overfit",
            description=f"Sparse upper tail with precision calibrated to gamma_P - "
            f"{delta_P:.2f}, testing against zero sample variance collapse.",
        )
        self.gamma_P = gamma_P
        self.delta_P = delta_P

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        gamma_P: float | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population with an overfit upper tail that scales with target
        precision.

        Positives are distributed in [0.5, 1.0], with half falling in the upper tail
        [0.75, 1.0].
        Negative distractors in [0.75, 1.0] are dynamically scaled so that the true
        precision in the upper tail equals (gamma_P - delta_P).

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            gamma_P: Primary precision target override passed by the evaluation
                runner.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        eff_gamma_P = (
            gamma_P if gamma_P is not None
            else self.gamma_P
        )
        p_tail = min(0.9, max(0.4, eff_gamma_P - self.delta_P))

        # 85% negatives, 15% positives
        n_pos = int(0.15 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.zeros(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives uniformly distributed in [0.5, 1.0] -> half in [0.75, 1.0]
        scores[:n_pos] = rng.uniform(0.5, 1.0, size=n_pos)
        n_pos_high = int(0.5 * n_pos)

        # Negative distractors in [0.75, 1.0] tuned so tail precision is p_tail:
        # n_pos_high / (n_pos_high + n_neg_high) = p_tail
        n_neg_high = int(n_pos_high * (1.0 - p_tail) / p_tail)
        n_neg_high = min(n_neg_high, int(0.4 * n_neg))
        n_neg_low = n_neg - n_neg_high

        scores[n_pos : n_pos + n_neg_low] = rng.uniform(0.0, 0.6, size=n_neg_low)
        scores[n_pos + n_neg_low :] = rng.uniform(0.75, 1.0, size=n_neg_high)

        return scores, oracle


class Benign(BaseScenario):
    """
    Benign Data (Calibrated Polarized Neural Proxy).

    Simulates a high-quality, calibrated classifier where scores are strongly polarized.
    Positives concentrate near 1.0, and negatives concentrate near 0.0 (AUC > 0.95).

    This scenario measures the minimal calibration overhead and fast early stopping
    of algorithms on well-behaved workloads.
    """

    def __init__(self):
        super().__init__(
            name="benign",
            description="Calibrated data with bimodal Beta proxy scores (AUC > 0.95).",
        )

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a clean population using calibrated bimodal Beta distributions.

        15% of items are positive with scores centered near 0.85 (Beta(4.0, 0.8)).
        85% of items are negative with scores centered near 0.15 (Beta(0.8, 4.0)).

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        n_pos = int(0.15 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives centered near 0.85; Negatives centered near 0.15
        scores[:n_pos] = rng.beta(4.0, 0.8, size=n_pos)
        scores[n_pos:] = rng.beta(0.8, 4.0, size=n_neg)

        return scores, oracle


class SemanticJoinNeedle(BaseScenario):
    """
    Extreme Sparsity (Semantic Join Needle-in-a-Haystack).

    Models a cross-product join where matching pairs are exceptionally rare (0.05%
    prevalence).
    The proxy has high discriminative ability: positives have scores in [0.7, 0.95],
    while most negatives have scores near 0.0.

    This scenario tests positive discovery under extreme class imbalance. Uniform
    sampling struggles to find any positives, while importance sampling concentrates
    queries on the tail.
    """

    def __init__(self):
        super().__init__(
            name="semantic_join_needle",
            description="Semantic join with 0.05% prevalence testing positive discovery"
            " under extreme sparsity.",
        )

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates an extremely sparse population modeling a semantic join operator.

        Only 0.05% of items are positive, with proxy scores in [0.7, 0.95].
        Negatives are concentrated near 0.0 (98% have scores < 0.02).

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        n_pos = max(1, int(0.0005 * pop_size))
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives concentrated in [0.7, 0.95]
        scores[:n_pos] = rng.beta(5.0, 1.5, size=n_pos)
        # Negatives overwhelmingly concentrated near 0.0 (98% < 0.02)
        scores[n_pos:] = rng.beta(0.2, 8.0, size=n_neg)

        return scores, oracle


class DiscreteLexical(BaseScenario):
    """
    Discrete Lexical Proxy.

    Models a keyword or quantized proxy (such as BM25 overlap or hash counts).
    85% of items have a score of 0.0, and the remaining 15% are distributed over
    discrete levels {0.2, 0.4, 0.6, 0.8, 1.0}.

    This scenario tests how algorithms handle ties and score collisions. It evaluates
    whether candidate threshold grids handle duplicates cleanly and whether sequential
    searches stall on identical scores.
    """

    def __init__(self):
        super().__init__(
            name="discrete_lexical_gate",
            description="Lexical proxy with 85% at score 0.0 and tied discrete levels "
            "{0.2, 0.4, 0.6, 0.8, 1.0}.",
        )

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population modeling a lightweight lexical proxy with tied discrete
        scores.

        85% of items have a score of 0.0 with minimal positive leakage (0.1%).
        The remaining 15% are distributed across discrete levels {0.2, 0.4, 0.6, 0.8,
        1.0}.

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        scores = np.zeros(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)

        # 85% items have score 0.0 with tiny positive leakage (0.1%)
        n_zero = int(0.85 * pop_size)
        oracle[:n_zero] = rng.random(size=n_zero) < 0.001

        # Remaining 15% items distributed over discrete levels: 0.2, 0.4, 0.6, 0.8, 1.0
        n_active = pop_size - n_zero
        levels = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        level_probs = np.array([0.05, 0.15, 0.4, 0.75, 0.95])
        assigned_levels = rng.choice(levels, size=n_active)
        scores[n_zero:] = assigned_levels

        for lvl, p in zip(levels, level_probs):
            mask = np.zeros(pop_size, dtype=bool)
            mask[n_zero:] = (assigned_levels == lvl)
            oracle[mask] = rng.random(size=np.sum(mask)) < p

        return scores, oracle


class PrecisionBoundaryCriticalMargin(BaseScenario):
    """
    Precision Boundary Critical Margin.

    Creates a population where precision across a broad score band [0.7, 0.9] is
    stubbornly (gamma_P - margin_gap), strictly below the precision target. Above 0.9,
    precision rises comfortably above gamma_P.

    This scenario stress-tests sequential early stopping against false acceptances.
    Algorithms that stop prematurely based on a lucky run in early batches will
    accept a threshold in [0.7, 0.9] and fail the precision guarantee.
    """

    def __init__(self, gamma_P: float = 0.8, margin_gap: float = 0.015):
        super().__init__(
            name="precision_boundary_critical_margin",
            description=f"Near-target precision boundary test with precision in "
            f"[0.7, 0.9] at gamma_P - {margin_gap:.3f} testing early-stopping false "
            f"acceptance.",
        )
        self.gamma_P = gamma_P
        self.margin_gap = margin_gap

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        gamma_P: float | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population with a sub-target precision band tuned to target
        precision.

        Items in [0.7, 0.9] have precision set to (gamma_P - margin_gap).
        Items above 0.9 have precision comfortably above gamma_P.

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            gamma_P: Primary precision target override passed by the evaluation
                runner.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        eff_gamma_P = (
            gamma_P if gamma_P is not None
            else self.gamma_P
        )

        # Precision in [0.7, 0.9] is strictly below target gamma_P by margin_gap
        boundary_prec = max(0.05, eff_gamma_P - self.margin_gap)
        # Above 0.9, precision is comfortably above gamma_P
        high_prec = min(0.99, eff_gamma_P + max(0.02, (1.0 - eff_gamma_P) * 0.25))

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)

        # Region 1: 70% in [0.0, 0.7] with low positive rate (1%)
        n1 = int(0.7 * pop_size)
        scores[:n1] = rng.uniform(0.0, 0.7, size=n1)
        oracle[:n1] = rng.random(size=n1) < 0.01

        # Region 2: 27% in [0.7, 0.9] with precision just below target
        n2 = int(0.27 * pop_size)
        scores[n1 : n1 + n2] = rng.uniform(0.7, 0.9, size=n2)
        oracle[n1 : n1 + n2] = rng.random(size=n2) < boundary_prec

        # Region 3: 3% in [0.9, 1.0] with high precision above target
        n3 = pop_size - n1 - n2
        scores[n1 + n2 :] = rng.uniform(0.9, 1.0, size=n3)
        oracle[n1 + n2 :] = rng.random(size=n3) < high_prec

        return scores, oracle


class RecallBoundaryCriticalMargin(BaseScenario):
    """
    Recall Boundary Critical Margin.

    Partitions positives so that dropping items with scores below candidate cutoffs in
    [0.15, 0.35] retains only (gamma_R - margin_gap) of all positives, falling just
    short of the recall target. To satisfy the target, the cascade must keep items
    with scores down to 0.05.

    This scenario stress-tests sequential recall bounds against under-coverage.
    Algorithms that underestimate variance in the dropped region will set the negative
    threshold too high and fail the recall guarantee.
    """

    def __init__(self, gamma_R: float = 0.8, margin_gap: float = 0.015):
        super().__init__(
            name="recall_boundary_critical_margin",
            description=f"Near-target recall boundary test where tau_lower in "
            f"[0.15, 0.35] achieves gamma_R - {margin_gap:.3f} testing premature lower "
            f"threshold acceptance.",
        )
        self.gamma_R = gamma_R
        self.margin_gap = margin_gap

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        gamma_R: float | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population with a sub-target recall boundary tuned to target recall.

        Positives are partitioned so that dropping scores below 0.15 retains only
        (gamma_R - margin_gap) of all positives. To satisfy the recall target,
        the cascade must choose a lower threshold tau_lower <= 0.05.

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            gamma_R: Primary recall target override passed by the evaluation
                runner.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        eff_gamma_R = (
            gamma_R if gamma_R is not None
            else self.gamma_R
        )

        n_pos = int(0.15 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives: split into safe low-score tail [0.0, 0.15] and upper band
        # [0.15, 1.0]
        # Dropping [0.0, 0.15] loses (1 - gamma_R + margin_gap) of positives,
        # leaving exactly gamma_R - margin_gap of positives in [0.15, 1.0].
        f_low = min(0.5, max(0.04, (1.0 - eff_gamma_R) + self.margin_gap))
        n_pos_low = int(f_low * n_pos)
        n_pos_high = n_pos - n_pos_low

        scores[:n_pos_low] = rng.uniform(0.0, 0.15, size=n_pos_low)
        scores[n_pos_low:n_pos] = rng.beta(4.0, 0.8, size=n_pos_high)

        # Negatives: distributed across low to moderate scores
        scores[n_pos:] = rng.beta(0.8, 4.0, size=n_neg)

        return scores, oracle


class PowerLawTailLeakage(BaseScenario):
    """
    Power-Law Tail Leakage.

    Simulates dense vector retrieval where 98% of data has low scores, while the top 2%
    contains true positives mixed with power-law distractors extending into [0.95, 1.0].

    Achieving high precision requires fine threshold resolution near 1.0. This scenario
    tests whether candidate threshold grids place sufficient resolution in the extreme
    tail.
    """

    def __init__(self):
        super().__init__(
            name="power_law_tail_leakage",
            description="Dense retrieval power-law tail testing fine upper-threshold "
            "quantile grid resolution.",
        )

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population modeling dense retrieval with power-law distractor
        leakage.

        98% of negatives have low scores, while 1% leak into [0.95, 1.0] under a
        power-law tail.
        Positives are concentrated near 1.0 (Beta(15.0, 0.5)), requiring fine
        upper-threshold resolution to isolate them from high-scoring distractors.

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        n_pos = int(0.02 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives concentrated near 1.0 (median ~ 0.985)
        scores[:n_pos] = rng.beta(15.0, 0.5, size=n_pos)

        # Negatives: 99% low scores, 1% power-law distractors in [0.95, 1.0]
        n_neg_distractors = int(0.01 * n_neg)
        n_neg_normal = n_neg - n_neg_distractors

        scores[n_pos : n_pos + n_neg_normal] = rng.beta(0.5, 5.0, size=n_neg_normal)
        u = rng.uniform(0.0, 1.0, size=n_neg_distractors)
        scores[n_pos + n_neg_normal :] = 0.95 + 0.05 * (u ** (1.0 / 4.0))

        return scores, oracle


class ProxyMiscalibrated(BaseScenario):
    """
    Proxy Miscalibration with Sub-population Blindspot.

    The proxy model is accurate for most data, but has a blindspot on a sub-population
    of true positives that receive low scores in [0.0, 0.2]. The blindspot size is
    scaled so that it holds (1 - gamma_R + margin_gap) of all positives.

    If an algorithm drops these low scores, it achieves recall below the target.
    This forces the algorithm to detect the blindspot and send it to the oracle.
    """

    def __init__(self, gamma_R: float = 0.8, margin_gap: float = 0.02):
        super().__init__(
            name="proxy_miscalibrated",
            description=f"Proxy model blindspot scaled to hold (1 - gamma_R + "
            f"{margin_gap:.3f}) of positives in [0.0, 0.2].",
        )
        self.gamma_R = gamma_R
        self.margin_gap = margin_gap

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        gamma_R: float | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population with a low-score blindspot scaled to target recall.

        The blindspot in [0.0, 0.2] contains (1 - gamma_R + margin_gap) of all
        positives.
        If an algorithm ignores the blindspot and drops low scores, it achieves recall
        strictly below the target.

        Args:
            pop_size: Total number of items in the population.
            rng: NumPy random generator.
            gamma_R: Primary recall target override passed by the evaluation
                runner.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if rng is None:
            rng = np.random.default_rng()

        if pop_size is None:
            pop_size = 100000

        eff_gamma_R = (
            gamma_R if gamma_R is not None
            else self.gamma_R
        )

        # Desired fraction of all true positives located in blindspot
        f_blind = min(0.5, max(0.04, (1.0 - eff_gamma_R) + self.margin_gap))

        # Main population has 50% positive rate.
        # Total positives = 0.5 * n_main + n_blindspot
        # n_blindspot / (0.5 * n_main + n_blindspot) = f_blind
        # => n_main = pop_size * (1 - f_blind) / (1 - 0.5 * f_blind)
        n_main = int(pop_size * (1.0 - f_blind) / (1.0 - 0.5 * f_blind))
        n_blindspot = pop_size - n_main

        scores = np.zeros(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)

        # Main population: 50% positive rate
        # Positives in [0.4, 1.0], negatives in [0.0, 0.4]
        main_oracle = rng.random(size=n_main) < 0.5
        oracle[:n_main] = main_oracle
        scores[:n_main] = np.where(
            main_oracle,
            rng.uniform(0.4, 1.0, size=n_main),
            rng.uniform(0.0, 0.4, size=n_main),
        )

        # Blindspot subpopulation: 100% positive, low proxy scores in [0.0, 0.2]
        oracle[n_main:] = True
        scores[n_main:] = rng.uniform(0.0, 0.2, size=n_blindspot)

        return scores, oracle


def _read_dataframe(path: Path) -> pd.DataFrame:
    """
    Reads a tabular dataset from disk based on file extension.

    Supports CSV, TSV, Feather, Arrow/IPC stream, and Parquet formats.

    Args:
        path: Path to the tabular dataset file.

    Returns:
        A pandas DataFrame with the loaded dataset.
    """
    suffix = path.suffix.lower()
    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    elif suffix in [".feather", ".arrow", ".ipc"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                return pd.read_feather(path)
            except Exception:
                try:
                    import pyarrow.ipc as ipc
                    with ipc.open_stream(str(path)) as reader:
                        return reader.read_all().to_pandas()
                except Exception:
                    import pyarrow.feather as feather
                    return feather.read_table(str(path)).to_pandas()
    elif suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        # Fallback: try feather/arrow first, then csv
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return pd.read_feather(path)
        except Exception:
            return pd.read_csv(path)


class TabularDataset(BaseScenario):
    """
    Scenario for file-backed benchmark datasets (CSV, Feather, Arrow, Parquet).

    Loads proxy scores and ground-truth oracle labels directly from a file on disk.
    """

    def __init__(
        self,
        name: str,
        description: str,
        data_path: str | Path,
        score_col: str = "proxy_score",
        label_col: str = "label",
    ):
        super().__init__(name=name, description=description)
        self.data_path = Path(data_path)
        self.score_col = score_col
        self.label_col = label_col
        self._data: tuple[NDArray[np.float64], NDArray[np.bool_]] | None = None

    def _load_data(self) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Loads and caches proxy scores and oracle labels from the underlying dataset.

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        if self._data is None:
            path = self.data_path
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at '{self.data_path}'."
                )
            df = _read_dataframe(path)
            if self.score_col not in df.columns:
                raise ValueError(
                    f"Score column '{self.score_col}' not found in dataset columns: "
                    f"{df.columns.tolist()}"
                )
            if self.label_col not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_col}' not found in dataset columns: "
                    f"{df.columns.tolist()}"
                )

            scores = df[self.score_col].to_numpy(dtype=np.float64)
            raw_label = df[self.label_col]
            if raw_label.dtype == bool:
                oracle = raw_label.to_numpy(dtype=bool)
            elif np.issubdtype(raw_label.dtype, np.number):
                oracle = (raw_label.to_numpy(dtype=np.float64) == 1.0)
            else:
                oracle = (
                    raw_label.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["1", "1.0", "true", "t", "yes", "y"])
                    .to_numpy(dtype=bool)
                )
            self._data = (scores, oracle)
        return self._data

    def generate_population(
        self,
        pop_size: int | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Generates a population by subsampling or loading from a disk-backed dataset.

        Args:
            pop_size: Number of items to sample. If None or larger than the dataset,
                returns all items.
            rng: NumPy random generator for random subsampling.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple (proxy_scores, oracle_outputs).
        """
        scores, oracle = self._load_data()
        if pop_size is None:
            pop_size = len(scores)
        if pop_size < len(scores):
            if rng is None:
                rng = np.random.default_rng()
            idx = rng.choice(len(scores), size=pop_size, replace=False)
            return scores[idx], oracle[idx]
        return scores.copy(), oracle.copy()


FileDataset = TabularDataset


class SUPGOnto(TabularDataset):
    """
    SUPG Onto Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=11,165 tuples with ~2.5% positive prevalence.
    """

    def __init__(
        self,
        data_path: str | Path = "experiments/data/supg/onto/source.csv",
        score_col: str = "proxy_score",
        label_col: str = "label",
    ):
        super().__init__(
            name="supg_onto",
            description="SUPG Onto benchmark dataset",
            data_path=data_path,
            score_col=score_col,
            label_col=label_col,
        )


class SUPGImageNet(TabularDataset):
    """
    SUPG ImageNet Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=50,000 tuples with ~0.1% positive prevalence (50 positives).
    """

    def __init__(
        self,
        data_path: str | Path = "experiments/data/supg/imagenet/source.csv",
        score_col: str = "proxy_score",
        label_col: str = "label",
    ):
        super().__init__(
            name="supg_imagenet",
            description="SUPG ImageNet benchmark dataset",
            data_path=data_path,
            score_col=score_col,
            label_col=label_col,
        )


SUPGImagenet = SUPGImageNet


class SUPGJackson(TabularDataset):
    """
    SUPG Jackson Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=973,085 tuples with ~29.2% positive prevalence.
    """

    def __init__(
        self,
        data_path: str | Path = "experiments/data/supg/jackson/2017-12-17.feather",
        score_col: str = "proxy_score",
        label_col: str = "label",
    ):
        super().__init__(
            name="supg_jackson",
            description="SUPG Jackson benchmark dataset",
            data_path=data_path,
            score_col=score_col,
            label_col=label_col,
        )


class SUPGTACRED(TabularDataset):
    """
    SUPG TACRED Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=22,631 tuples with ~2.4% positive prevalence.
    """

    def __init__(
        self,
        data_path: str | Path = "experiments/data/supg/tacred/source.csv",
        score_col: str = "proxy_score",
        label_col: str = "label",
    ):
        super().__init__(
            name="supg_tacred",
            description="SUPG TACRED benchmark dataset",
            data_path=data_path,
            score_col=score_col,
            label_col=label_col,
        )


SUPGTacred = SUPGTACRED


SCENARIOS = {
    # Adversarial / Floor Stress Tests
    "uninformative_proxy_floor": UninformativeProxyFloor(),
    "precision_tail_overfit": PrecisionTailOverfit(),
    "proxy_miscalibrated": ProxyMiscalibrated(),
    "precision_boundary_critical_margin": PrecisionBoundaryCriticalMargin(),
    "recall_boundary_critical_margin": RecallBoundaryCriticalMargin(),
    # Semantic Database Operators / Workloads
    "benign": Benign(),
    "semantic_join_needle": SemanticJoinNeedle(),
    "discrete_lexical_gate": DiscreteLexical(),
    "power_law_tail_leakage": PowerLawTailLeakage(),
    # Real Benchmarks (SUPG)
    "supg_onto": SUPGOnto(),
    "supg_imagenet": SUPGImageNet(),
    "supg_jackson": SUPGJackson(),
    "supg_tacred": SUPGTACRED(),
}


__all__ = [
    "BaseScenario",
    "TabularDataset",
    "FileDataset",
    "UninformativeProxyFloor",
    "PrecisionTailOverfit",
    "Benign",
    "ProxyMiscalibrated",
    "SemanticJoinNeedle",
    "DiscreteLexical",
    "PrecisionBoundaryCriticalMargin",
    "RecallBoundaryCriticalMargin",
    "PowerLawTailLeakage",
    "SUPGOnto",
    "SUPGImageNet",
    "SUPGImagenet",
    "SUPGJackson",
    "SUPGTACRED",
    "SUPGTacred",
    "SCENARIOS",
]

