import hashlib
import json
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Population:
    """
    Represents a population dataset of proxy scores, oracle labels, and cost metrics.
    """

    scores: NDArray[np.float64]
    labels: NDArray[np.bool_]
    oracle_costs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    proxy_costs: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self):
        """Allows unpacking: scores, labels = population."""
        return iter((self.scores, self.labels))

    def slice(self, idx: NDArray[np.int64]) -> "Population":
        """Returns a subsampled Population using the given indices."""
        return Population(
            scores=self.scores[idx],
            labels=self.labels[idx],
            oracle_costs={k: v[idx] for k, v in self.oracle_costs.items()},
            proxy_costs={k: v[idx] for k, v in self.proxy_costs.items()},
        )

    def compute_trial_costs(
        self,
        oracle_queried_mask: NDArray[np.bool_],
    ) -> dict[str, dict[str, float]]:
        """
        Computes disaggregated trial costs organized hierarchically by model (oracle vs
        proxy).
        """
        pop_size = len(self.scores)
        num_oracle_calls = float(np.sum(oracle_queried_mask))
        oracle_dict: dict[str, float] = {
            "num_calls": num_oracle_calls,
            "call_rate": num_oracle_calls / pop_size if pop_size > 0 else 0.0,
        }
        for k, arr in self.oracle_costs.items():
            oracle_dict[k] = float(np.sum(arr[oracle_queried_mask]))

        proxy_dict: dict[str, float] = {
            "num_calls": float(pop_size),
            "call_rate": 1.0 if pop_size > 0 else 0.0,
        }
        for k, arr in self.proxy_costs.items():
            # Proxy cost is incurred for all items in the population
            proxy_dict[k] = float(np.sum(arr))

        return {"oracle": oracle_dict, "proxy": proxy_dict}


def compute_ideal_oracle_call_rate(
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


@dataclass(kw_only=True)
class BaseScenario(ABC):
    """Abstract base class for coverage scenarios."""

    name: str = ""
    description: str = ""
    pop_size: int | None = None
    seed: int | None = 42

    def to_dict(self) -> dict[str, Any]:
        """Serializes scenario configuration to a JSON-compatible dictionary."""
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
        """Deterministic hash of scenario configuration."""
        d = self.to_dict()
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]

    @abstractmethod
    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population dataset of proxy scores, oracle labels, and cost metrics.

        Returns:
            A Population object containing scores, labels, and cost metrics.
        """
        pass


@dataclass(kw_only=True)
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

    name: str = "uninformative_proxy_floor"
    description: str = (
        "Weak proxy (AUC ~0.6) and 1.5% prevalence forcing an oracle floor >= 55%."
    )
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population where scores carry minimal ranking signal.

        Positives make up 1.5% of the data and have scores uniformly distributed in
        [0.35, 0.85]. Negatives have scores uniformly distributed in [0.0, 1.0].
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

        scores = rng.uniform(0.0, 1.0, size=pop_size)
        oracle = np.zeros(pop_size, dtype=bool)

        n_positives = int(0.015 * pop_size)
        pos_idx = rng.choice(pop_size, size=n_positives, replace=False)
        oracle[pos_idx] = True

        # Positives have proxy scores uniformly in [0.35, 0.85]
        scores[pos_idx] = rng.uniform(0.35, 0.85, size=n_positives)

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
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

    name: str = "precision_tail_overfit"
    description: str = (
        "Sparse upper tail with precision calibrated to gamma_P - delta_P, "
        "testing against zero sample variance collapse."
    )
    pop_size: int | None = 100000
    gamma_P: float = 0.8
    delta_P: float = 0.15

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population with an overfit upper tail that scales with target
        precision.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

        eff_gamma_P = kwargs.get("gamma_P", self.gamma_P)
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

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
class Benign(BaseScenario):
    """
    Benign Data (Calibrated Polarized Neural Proxy).

    Simulates a high-quality, calibrated classifier where scores are strongly polarized.
    Positives concentrate near 1.0, and negatives concentrate near 0.0 (AUC > 0.95).

    This scenario measures the minimal calibration overhead and fast early stopping
    of algorithms on well-behaved workloads.
    """

    name: str = "benign"
    description: str = "Calibrated data with bimodal Beta proxy scores (AUC > 0.95)."
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a clean population using calibrated bimodal Beta distributions.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

        n_pos = int(0.15 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives centered near 0.85; Negatives centered near 0.15
        scores[:n_pos] = rng.beta(4.0, 0.8, size=n_pos)
        scores[n_pos:] = rng.beta(0.8, 4.0, size=n_neg)

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
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

    name: str = "semantic_join_needle"
    description: str = (
        "Semantic join with 0.05% prevalence testing positive discovery under extreme "
        "sparsity."
    )
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates an extremely sparse population modeling a semantic join operator.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

        n_pos = max(1, int(0.0005 * pop_size))
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives concentrated in [0.7, 0.95]
        scores[:n_pos] = rng.beta(5.0, 1.5, size=n_pos)
        # Negatives overwhelmingly concentrated near 0.0 (98% < 0.02)
        scores[n_pos:] = rng.beta(0.2, 8.0, size=n_neg)

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
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

    name: str = "discrete_lexical_gate"
    description: str = (
        "Lexical proxy with 85% at score 0.0 and tied discrete levels "
        "{0.2, 0.4, 0.6, 0.8, 1.0}."
    )
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population modeling a lightweight lexical proxy with tied discrete
        scores.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

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
            mask[n_zero:] = assigned_levels == lvl
            oracle[mask] = rng.random(size=np.sum(mask)) < p

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
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

    name: str = "precision_boundary_critical_margin"
    description: str = (
        "Near-target precision boundary test with precision in [0.7, 0.9] at "
        "gamma_P - margin_gap testing early-stopping false acceptance."
    )
    pop_size: int | None = 100000
    gamma_P: float = 0.8
    margin_gap: float = 0.015

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population with a sub-target precision band tuned to target
        precision.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000
        eff_gamma_P = kwargs.get("gamma_P", self.gamma_P)

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

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
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

    name: str = "recall_boundary_critical_margin"
    description: str = (
        "Near-target recall boundary test where tau_lower in [0.15, 0.35] achieves "
        "gamma_R - margin_gap testing premature lower threshold acceptance."
    )
    pop_size: int | None = 100000
    gamma_R: float = 0.8
    margin_gap: float = 0.015

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population with a sub-target recall boundary tuned to target recall.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000
        eff_gamma_R = kwargs.get("gamma_R", self.gamma_R)

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

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
class TrappedHead(BaseScenario):
    """
    Trapped Head Distribution.

    Features a deceptive upper score profile where an extreme high-scoring 'head'
    in [0.95, 1.0] contains 100% true positives, while the immediately preceding
    upper band in [0.75, 0.95) has precision strictly below target
    (gamma_P - margin_gap).

    Heuristic sliding windows or aggressive early stopping can prematurely accept a
    threshold near 0.75-0.85 after encountering early positive runs, pulling the
    overall accepted region precision below gamma_P.
    """

    name: str = "trapped_head"
    description: str = (
        "Trapped head with pure positive tail in [0.95, 1.0] and "
        "sub-target band in [0.75, 0.95) at gamma_P - margin_gap, "
        "testing sliding-window and early-stopping false acceptance."
    )
    pop_size: int | None = 100000
    gamma_P: float = 0.8
    margin_gap: float = 0.08

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population with a deceptive trapped head score distribution.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000
        eff_gamma_P = kwargs.get("gamma_P", self.gamma_P)

        # Region 1: 90% uninformative body in [0.0, 0.75] with 1% positives
        n_body = int(0.90 * pop_size)

        # Region 2: 9.5% deceptive band in [0.75, 0.95] with sub-target precision
        n_band = int(0.095 * pop_size)
        band_prec = max(0.05, eff_gamma_P - self.margin_gap)

        # Region 3: 0.5% trapped head in [0.95, 1.0] with 100% precision
        n_head = pop_size - n_body - n_band

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)

        scores[:n_body] = rng.uniform(0.0, 0.75, size=n_body)
        oracle[:n_body] = rng.random(size=n_body) < 0.01

        scores[n_body : n_body + n_band] = rng.uniform(0.75, 0.95, size=n_band)
        oracle[n_body : n_body + n_band] = rng.random(size=n_band) < band_prec

        scores[n_body + n_band :] = rng.uniform(0.95, 1.0, size=n_head)
        oracle[n_body + n_band :] = True

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
class PowerLawTailLeakage(BaseScenario):
    """
    Power-Law Tail Leakage.

    Simulates dense vector retrieval where 98% of data has low scores, while the top 2%
    contains true positives mixed with power-law distractors extending into [0.95, 1.0].

    Achieving high precision requires fine threshold resolution near 1.0. This scenario
    tests whether candidate threshold grids place sufficient resolution in the extreme
    tail.
    """

    name: str = "power_law_tail_leakage"
    description: str = (
        "Dense retrieval power-law tail testing fine upper-threshold "
        "quantile grid resolution."
    )
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population modeling dense retrieval with power-law distractor
        leakage.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

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

        return Population(scores=scores, labels=oracle)


@dataclass(kw_only=True)
class ProxyMiscalibrated(BaseScenario):
    """
    Proxy Miscalibration with Sub-population Blindspot.

    The proxy model is accurate for most data, but has a blindspot on a sub-population
    of true positives that receive low scores in [0.0, 0.2]. The blindspot size is
    scaled so that it holds (1 - gamma_R + margin_gap) of all positives.

    If an algorithm drops these low scores, it achieves recall below the target.
    This forces the algorithm to detect the blindspot and send it to the oracle.
    """

    name: str = "proxy_miscalibrated"
    description: str = (
        "Proxy model blindspot scaled to hold (1 - gamma_R + "
        "margin_gap) of positives in [0.0, 0.2]."
    )
    pop_size: int | None = 100000
    gamma_R: float = 0.8
    margin_gap: float = 0.02

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population with a low-score blindspot scaled to target recall.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000
        eff_gamma_R = kwargs.get("gamma_R", self.gamma_R)

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

        # Blindspot: 100% positive rate with very low scores in [0.0, 0.2]
        oracle[n_main:] = True
        scores[n_main:] = rng.uniform(0.0, 0.2, size=n_blindspot)

        # Randomly shuffle items to prevent ordering artifacts
        perm = rng.permutation(pop_size)
        return Population(scores=scores[perm], labels=oracle[perm])


def _read_dataframe(path: Path) -> pd.DataFrame:
    """
    Reads a tabular dataset from disk based on file extension.

    Supports CSV, TSV, Feather, Arrow/IPC stream, and Parquet formats.

    Args:
        path: Path to the tabular dataset file.

    Returns:
        A pandas DataFrame with the loaded dataset.
    """
    if path.is_dir():
        from datasets import Dataset, load_from_disk

        ds = load_from_disk(str(path))
        if not isinstance(ds, Dataset):
            raise ValueError(f"Loaded dataset is not a Dataset: {type(ds)}")
        return ds.to_pandas()  # ty: ignore[invalid-return-type]

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


@dataclass(kw_only=True)
class TabularDataset(BaseScenario):
    """
    Scenario for file-backed benchmark datasets (CSV, Feather, Arrow, Parquet).

    Loads proxy scores and ground-truth oracle labels directly from a file on disk.
    """

    data_path: str | Path = ""
    score_col: str = "proxy_score"
    label_col: str = "label"
    oracle_cost_col: str = "oracle_cost"
    proxy_cost_col: str = "proxy_cost"

    def __post_init__(self):
        if self.data_path != "":
            self.data_path = Path(self.data_path)
        object.__setattr__(self, "_df", None)
        object.__setattr__(self, "_data", None)

    def get_dataframe(self) -> pd.DataFrame:
        """Loads and returns the cached underlying DataFrame."""
        if getattr(self, "_df", None) is None:
            data_path = Path(self.data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found at '{data_path}'.")
            object.__setattr__(self, "_df", _read_dataframe(data_path))
        return getattr(self, "_df")

    def _extract_costs(
        self, df: pd.DataFrame, col_name: str
    ) -> dict[str, NDArray[np.float64]]:
        """Extracts 1D float arrays for each cost metric in a dictionary or scalar
        column."""
        if col_name not in df.columns or len(df) == 0:
            return {}

        series = df[col_name]
        sample = None
        for item in series:
            if item is not None and not (isinstance(item, float) and np.isnan(item)):
                sample = item
                break

        if sample is None:
            return {}

        if isinstance(sample, dict):
            cost_df = pd.DataFrame(series.tolist()).fillna(0.0)
            return {
                str(col): cost_df[col].to_numpy(dtype=np.float64)
                for col in cost_df.columns
            }
        elif np.issubdtype(series.dtype, np.number):
            return {col_name: series.fillna(0.0).to_numpy(dtype=np.float64)}
        else:
            try:
                arr = series.astype(float).fillna(0.0).to_numpy(dtype=np.float64)
                return {col_name: arr}
            except Exception:
                return {}

    def get_costs(
        self,
        cost_key: str = "monetary",
        default_oracle_cost: float = 1.0,
        default_proxy_cost: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extracts 1D arrays of (proxy_costs, oracle_costs) from the dataset.

        Args:
            cost_key: Metric key inside the cost dictionaries (e.g. 'monetary',
                'input_tokens', 'latency_ms').
            default_oracle_cost: Fallback cost if column or cost_key is missing.
            default_proxy_cost: Fallback cost if column or cost_key is missing.

        Returns:
            A tuple (proxy_costs, oracle_costs) of float64 1D arrays.
        """
        pop = self._load_data()
        n = len(pop)
        proxy_arr = pop.proxy_costs.get(
            cost_key, np.full(n, default_proxy_cost, dtype=np.float64)
        )
        oracle_arr = pop.oracle_costs.get(
            cost_key, np.full(n, default_oracle_cost, dtype=np.float64)
        )
        return proxy_arr, oracle_arr

    def _load_data(self) -> Population:
        """
        Loads and caches proxy scores, oracle labels, and cost metrics from the dataset.

        Returns:
            A Population object.
        """
        if getattr(self, "_data", None) is None:
            df = self.get_dataframe()
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
                oracle = raw_label.to_numpy(dtype=np.float64) == 1.0
            else:
                oracle = (
                    raw_label.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["1", "1.0", "true", "t", "yes", "y"])
                    .to_numpy(dtype=bool)
                )

            oracle_costs = self._extract_costs(df, self.oracle_cost_col)
            proxy_costs = self._extract_costs(df, self.proxy_cost_col)

            object.__setattr__(
                self,
                "_data",
                Population(
                    scores=scores,
                    labels=oracle,
                    oracle_costs=oracle_costs,
                    proxy_costs=proxy_costs,
                ),
            )
        return getattr(self, "_data")

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population by subsampling or loading from a disk-backed dataset.

        Returns:
            A Population object.
        """
        pop = self._load_data()
        if self.pop_size is None or self.pop_size >= len(pop):
            return pop
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(pop), size=self.pop_size, replace=False)
        return pop.slice(idx)


FileDataset = TabularDataset


@dataclass(kw_only=True)
class SUPGOnto(TabularDataset):
    """
    SUPG Onto Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=11,165 tuples with ~2.5% positive prevalence.
    """

    name: str = "supg_onto"
    description: str = "SUPG Onto benchmark dataset"
    data_path: str | Path = "experiments/data/supg/onto/source.csv"


@dataclass(kw_only=True)
class SUPGImageNet(TabularDataset):
    """
    SUPG ImageNet Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=50,000 tuples with ~0.1% positive prevalence (50 positives).
    """

    name: str = "supg_imagenet"
    description: str = "SUPG ImageNet benchmark dataset"
    data_path: str | Path = "experiments/data/supg/imagenet/source.csv"


SUPGImagenet = SUPGImageNet


@dataclass(kw_only=True)
class SUPGJackson(TabularDataset):
    """
    SUPG Jackson Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=973,085 tuples with ~29.2% positive prevalence.
    """

    name: str = "supg_jackson"
    description: str = "SUPG Jackson benchmark dataset"
    data_path: str | Path = "experiments/data/supg/jackson/2017-12-17.feather"


@dataclass(kw_only=True)
class SUPGTACRED(TabularDataset):
    """
    SUPG TACRED Benchmark Dataset

    Real-world benchmark dataset from the SUPG paper with proxy scores and binary
    ground-truth labels.
    Contains N=22,631 tuples with ~2.4% positive prevalence.
    """

    name: str = "supg_tacred"
    description: str = "SUPG TACRED benchmark dataset"
    data_path: str | Path = "experiments/data/supg/tacred/source.csv"


SUPGTacred = SUPGTACRED


@dataclass(kw_only=True)
class ScaleDocDataset(TabularDataset):
    """
    Scenario for ScaleDoc benchmark datasets (PubMed, BigPatent, GovReport).

    Loads proxy scores and ground-truth oracle labels for a specific query
    from a file on disk
    (default: experiments/data/scaledoc/{dataset}/q{query_id}.parquet).
    """

    dataset_name: str = ""
    query_id: str = "0"
    data_dir: str | Path = "experiments/data/scaledoc"

    def __post_init__(self):
        clean_name = self.dataset_name.lower().replace("-", "_")
        clean_qid = str(self.query_id).lower()
        if not self.name:
            self.name = f"scaledoc_{clean_name}_q{clean_qid}"
        if not self.description:
            self.description = f"ScaleDoc {clean_name} query {clean_qid}"
        if not self.data_path:
            self.data_path = Path(self.data_dir) / clean_name / f"q{clean_qid}.parquet"
        super().__post_init__()


@dataclass(kw_only=True)
class ScaleDocPubMed(ScaleDocDataset):
    """ScaleDoc PubMed query benchmark scenario."""

    dataset_name: str = "pubmed"


@dataclass(kw_only=True)
class ScaleDocBigPatent(ScaleDocDataset):
    """ScaleDoc BigPatent query benchmark scenario."""

    dataset_name: str = "big_patent"


@dataclass(kw_only=True)
class ScaleDocGovReport(ScaleDocDataset):
    """ScaleDoc GovReport query benchmark scenario."""

    dataset_name: str = "gov_report"


class _ScenarioRegistry(dict):
    """
    Scenario registry supporting static lookups and dynamic instantiation
    for ScaleDoc query scenarios (e.g. 'scaledoc_pubmed_q1' or
    'scaledoc_pubmed_q0_ext').
    """

    def _parse_dynamic_key(self, key: str) -> BaseScenario | None:
        m = re.match(r"^scaledoc_(pubmed|big_patent|gov_report)_q(\d+(?:_ext)?)$", key)
        if m:
            ds_name, qid = m.group(1), m.group(2)
            return ScaleDocDataset(dataset_name=ds_name, query_id=qid)
        return None

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            scenario = self._parse_dynamic_key(key)
            if scenario is not None:
                self[key] = scenario
                return True
        return False

    def __missing__(self, key: object) -> BaseScenario:
        if isinstance(key, str):
            scenario = self._parse_dynamic_key(key)
            if scenario is not None:
                self[key] = scenario
                return scenario
        raise KeyError(f"Unknown scenario '{key}'")

    def get(self, key: object, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


SCENARIOS = _ScenarioRegistry(
    {
        # Adversarial / Floor Stress Tests
        "uninformative_proxy_floor": UninformativeProxyFloor(),
        "precision_tail_overfit": PrecisionTailOverfit(),
        "proxy_miscalibrated": ProxyMiscalibrated(),
        "precision_boundary_critical_margin": PrecisionBoundaryCriticalMargin(),
        "recall_boundary_critical_margin": RecallBoundaryCriticalMargin(),
        "trapped_head": TrappedHead(),
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
        # ScaleDoc Benchmarks (defaults: query 0)
        "scaledoc_pubmed_q0": ScaleDocPubMed(query_id="0"),
        "scaledoc_big_patent_q0": ScaleDocBigPatent(query_id="0"),
        "scaledoc_gov_report_q0": ScaleDocGovReport(query_id="0"),
    }
)


__all__ = [
    "Population",
    "compute_ideal_oracle_call_rate",
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
    "TrappedHead",
    "PowerLawTailLeakage",
    "SUPGOnto",
    "SUPGImageNet",
    "SUPGImagenet",
    "SUPGJackson",
    "SUPGTACRED",
    "SUPGTacred",
    "ScaleDocDataset",
    "ScaleDocPubMed",
    "ScaleDocBigPatent",
    "ScaleDocGovReport",
    "SCENARIOS",
]
