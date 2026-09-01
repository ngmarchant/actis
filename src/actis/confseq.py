import math
from collections.abc import Sequence

import numpy as np
from confseq.betting import diversified_betting_mart
from confseq.betting_strategies import lambda_predmix_eb
from numpy.typing import ArrayLike, NDArray


class AsymptoticSupermartingale:
    r"""Stateful asymptotic Gaussian mixture test supermartingale (Howard et al. 2021).

    For observations $x_1, \dots, x_n \in \mathbb{R}$ against null mean $m$, tracks
    empirical sum $S_n = \sum (x_i - m)$ and variance accumulator
    $V_n = \sum (x_i - \bar{x})^2$.

    The wealth process is:
    $$
        M_n(m) = \sqrt{\frac{v_0}{v_0 + V_n}} \exp\left(
            \frac{(\max(0, S_n))^2}{2(v_0 + V_n)}
        \right)
    $$
    """

    def __init__(self, m: float = 0.0, v0: float = 1.0):
        """
        Args:
            m: Candidate mean under the null hypothesis (default: 0.0).
            v0: Prior variance parameter (default: 1.0).
        """
        self.m = float(m)
        self.v0 = float(v0)
        self.n = 0
        self.S = 0.0
        self.SS = 0.0

    def update(self, x_batch: ArrayLike) -> None:
        """Updates running statistics with a new batch of observations."""
        x = np.asarray(x_batch, dtype=np.float64)
        if x.size == 0:
            return
        self.n += len(x)
        self.S += float(np.sum(x))
        self.SS += float(np.sum(x ** 2))

    def wealth(self, m: float | None = None) -> float:
        """Evaluates current terminal wealth."""
        if self.n == 0:
            return 1.0

        null_m = self.m if m is None else float(m)
        s_n = self.S - self.n * null_m
        if s_n <= 0.0:
            return 0.0

        v_n = max(0.0, self.SS - (self.S ** 2) / self.n)
        denom = self.v0 + v_n
        if denom <= 0.0:
            return 1.0

        log_m = 0.5 * math.log(self.v0 / denom) + (s_n ** 2) / (2.0 * denom)
        try:
            return math.exp(min(log_m, 700.0))
        except OverflowError:
            return float("inf")

    def upper_confidence_bound(self, delta: float) -> float:
        """Returns the Upper Confidence Bound (UCB) on the true mean at level delta."""
        if self.n == 0:
            return float("inf")
        mu_hat = self.S / self.n
        v_n = max(0.0, self.SS - (self.S ** 2) / self.n)
        denom_log = 0.5 * math.log(1.0 + v_n / self.v0)
        rad = (1.0 / self.n) * math.sqrt(
            2.0 * (self.v0 + v_n) * (math.log(1.0 / delta) + denom_log)
        )
        return mu_hat + rad

    def lower_confidence_bound(self, delta: float) -> float:
        """Returns the Lower Confidence Bound (LCB) on the true mean at level delta."""
        if self.n == 0:
            return float("-inf")
        mu_hat = self.S / self.n
        v_n = max(0.0, self.SS - (self.S ** 2) / self.n)
        denom_log = 0.5 * math.log(1.0 + v_n / self.v0)
        rad = (1.0 / self.n) * math.sqrt(
            2.0 * (self.v0 + v_n) * (math.log(1.0 / delta) + denom_log)
        )
        return mu_hat - rad


class BettingSupermartingale:
    r"""Stateful predictable empirical-Bernstein betting supermartingale for $[0, 1]$-bounded
    observations (Waudby-Smith & Ramdas, 2024; based closely on `confseq.betting`).

    Maintains online state ($t$, $S_t$, $D_t$, $K_t$) and advances wealth sample-by-sample
    or batch-by-batch in $O(B)$ time.
    """

    def __init__(
        self,
        m: float,
        alpha: float,
        population_size: int | None = None,
        horizon: int | None = None,
        trunc_scale: float = 0.5,
        m_trunc: bool = True,
        fake_obs: int = 1,
        prior_mean: float = 0.5,
        prior_variance: float = 0.25,
        reverse: bool = False,
    ):
        """
        Args:
            m: Candidate mean / null value in [0, 1].
            alpha: Significance level delta in (0, 1).
            population_size: Finite population size $N$ if sampling without replacement.
            horizon: Fixed sample size horizon. If None, uses anytime-valid 1/sqrt(t log t) scaling.
            trunc_scale: Scale factor for lambda truncation (default: 0.5).
            m_trunc: Whether truncation depends on m (default: True).
            fake_obs: Number of fake observations for regularizing running mean and variance (default: 1).
            prior_mean: Prior mean for regularizing sample mean (default: 0.5).
            prior_variance: Prior variance for regularizing sample variance (default: 0.25).
            reverse: If True, tests $H_0: \mu \ge m$ by running the betting process on $1 - X$
                against null $1 - m$.
        """
        self.m = float(m)
        self.alpha = float(alpha)
        self.population_size = population_size
        self.horizon = horizon
        self.trunc_scale = float(trunc_scale)
        self.m_trunc = bool(m_trunc)
        self.fake_obs = int(fake_obs)
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)
        self.reverse = bool(reverse)

        # Internal state
        self.t = 0
        self.S_t = 0.0
        self.cum_sq_dev = 0.0
        self.last_sigma2 = float(prior_variance)
        self.current_wealth = 1.0

    def update(self, x_batch: ArrayLike) -> None:
        """Advances the betting supermartingale by a new batch of observations."""
        x = np.asarray(x_batch, dtype=np.float64)
        B = x.size
        if B == 0 or self.current_wealth == 0.0:
            return

        if self.reverse:
            x = 1.0 - x
            null_m = 1.0 - self.m
        else:
            null_m = self.m

        t0 = self.t
        t = t0 + np.arange(1, B + 1, dtype=np.float64)
        S_t = self.S_t + np.cumsum(x)

        # Online running regularized mean and variance
        mu_hat_t = np.minimum((self.fake_obs * self.prior_mean + S_t) / (t + self.fake_obs), 1.0)
        sq_dev = (x - mu_hat_t) ** 2
        cum_sq_dev_t = self.cum_sq_dev + np.cumsum(sq_dev)
        sigma2_t = (self.fake_obs * self.prior_variance + cum_sq_dev_t) / (t + self.fake_obs)

        # 1-step predictable variance
        sigma2_prev = np.empty(B, dtype=np.float64)
        sigma2_prev[0] = self.last_sigma2
        if B > 1:
            sigma2_prev[1:] = sigma2_t[:-1]

        # Predictable bets
        if self.horizon is None:
            raw_lambda = np.sqrt(2.0 * np.log(1.0 / self.alpha) / (t * np.log(1.0 + t) * sigma2_prev))
        else:
            raw_lambda = np.sqrt(2.0 * np.log(1.0 / self.alpha) / (self.horizon * sigma2_prev))
        raw_lambda = np.nan_to_num(raw_lambda, nan=0.0, posinf=0.0, neginf=0.0)

        # Conditional null mean (accounting for hypergeometric without-replacement draws if N is set)
        if self.population_size is not None:
            S_prev = np.empty(B, dtype=np.float64)
            S_prev[0] = self.S_t
            if B > 1:
                S_prev[1:] = S_t[:-1]
            mu_t = (self.population_size * null_m - S_prev) / (self.population_size - (t - 1.0))
        else:
            mu_t = np.full(B, null_m, dtype=np.float64)

        # Boundary truncation (matching confseq)
        if self.m_trunc:
            with np.errstate(divide="ignore"):
                upper = np.where(mu_t > 0.0, self.trunc_scale / mu_t, np.inf)
                lower = np.where(mu_t < 1.0, -self.trunc_scale / (1.0 - mu_t), -np.inf)
            lam = np.clip(raw_lambda, lower, upper)
        else:
            lam = np.clip(raw_lambda, -self.trunc_scale, self.trunc_scale)

        mult = 1.0 + lam * (x - mu_t)
        mult = np.maximum(mult, 0.0)
        mult = np.nan_to_num(mult, nan=0.0)
        if self.population_size is not None:
            mult = np.where((mu_t < 0.0) | (mu_t > 1.0), np.inf, mult)

        wealth_batch = float(np.prod(mult))
        self.current_wealth *= wealth_batch

        self.t = int(t[-1])
        self.S_t = float(S_t[-1])
        self.cum_sq_dev = float(cum_sq_dev_t[-1])
        self.last_sigma2 = float(sigma2_t[-1])

    def wealth(self) -> float:
        """Returns the current terminal wealth."""
        return self.current_wealth


def eval_betting_wealth(
    x: NDArray[np.float64],
    m: float,
    alpha: float,
    population_size: int | None = None,
    horizon: int | None = None,
    trunc_scale: float = 0.5,
    m_trunc: bool = True,
) -> float:
    """Evaluates the terminal wealth of the one-sided (positive) betting supermartingale
    against a candidate mean.
    """
    mart = BettingSupermartingale(
        m=m,
        alpha=alpha,
        population_size=population_size,
        horizon=horizon,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )
    mart.update(x)
    return mart.wealth()


def compute_prior_var(
    scores: ArrayLike,
    gamma_R: float,
    q: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    n_init: int = 1000,
) -> float:
    r"""Computes the prior variance scale v0 for the Gaussian mixture confidence
    sequence.
    """
    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N == 0:
        return 0.01

    mean_s = float(np.mean(scores))

    if q is not None or weights is not None:
        v0 = max(0.01, float(n_init) * mean_s * (1.0 - gamma_R))
        return float(np.clip(v0, 0.01, 1.0))

    margin_scale = (1.0 - gamma_R) ** 2
    sigma_sq = margin_scale * max(mean_s * (1.0 - mean_s), 0.01)
    v0 = float(n_init) * sigma_sq
    return float(np.clip(v0, 0.5, 10.0))


def eval_asymptotic_wealth(
    x: ArrayLike,
    m: float = 0.0,
    prior_var: float = 1.0,
) -> float:
    r"""Evaluates the terminal wealth of the asymptotic Gaussian mixture supermartingale
    wealth against a candidate mean.
    """
    mart = AsymptoticSupermartingale(m=m, v0=prior_var)
    mart.update(x)
    return mart.wealth()


def eval_asymptotic_betting_wealth(
    x: ArrayLike,
    m: float,
    alpha: float = 0.05,
    c: float = 0.5,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    fake_obs: int = 1,
) -> float:
    r"""Evaluates the terminal wealth of the asymptotic betting supermartingale
    of Waudby-Smith, Arbour, Sinha & Ramdas (2024)
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N == 0:
        return 1.0

    diff = x - m
    t = np.arange(1, N + 1)

    mu_hat_t = (fake_obs * prior_mean + np.cumsum(diff)) / (t + fake_obs)
    mu_prev = np.append(prior_mean, mu_hat_t[:-1])

    sigma2_t = (fake_obs * prior_var + np.cumsum((diff - mu_hat_t) ** 2)) / (t + fake_obs)
    sigma2_prev = np.append(prior_var, sigma2_t[:-1])

    cum_max = np.maximum.accumulate(np.abs(diff))
    max_prev = np.append(1.0, cum_max[:-1])

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_lambda = np.sqrt(
            2.0 * np.log(1.0 / alpha) / (t * np.log(1.0 + t) * sigma2_prev)
        )
        raw_lambda = np.where(mu_prev > 0.0, raw_lambda, 0.0)
        lambdas = np.minimum(raw_lambda, c / max_prev)
        lambdas = np.nan_to_num(lambdas, nan=0.0, posinf=0.0, neginf=0.0)

        multiplicands = np.maximum(1.0 + lambdas * diff, 0.0)
        wealth_process = np.cumprod(multiplicands)

    return float(wealth_process[-1])

