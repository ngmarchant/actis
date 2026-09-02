import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class TestSupermartingale(ABC):
    r"""Abstract base class for stateful sequential test supermartingales.

    A test supermartingale $(M_t)_{t \ge 0}$ tracks evidence against a null hypothesis
    $H_0: \mu \le m$ (or $H_0: \mu = m$), starting at $M_0 \le 1$ and satisfying
    $$\mathbb{E}[M_t \mid \mathcal{F}_{t-1}] \le M_{t-1}$$
    under $H_0$.

    Evidence against $H_0$ is measured by the accumulated wealth $M_t$, where
    $M_t \ge 1/\alpha$ rejects $H_0$ with anytime-valid Type I error rate $\le \alpha$
    via Ville's inequality:
    $$\mathbb{P}_{H_0}(\exists t \ge 1: M_t \ge 1/\alpha) \le \alpha$$.
    """

    def __init__(self, m: float = 0.0, reverse: bool = False) -> None:
        r"""
        Args:
            m: Mean $m$ in the null hypothesis (default: 0.0).
            reverse: If False (default), tests $H_0: \mu \le m$ vs $H_1: \mu > m$
                (right-tailed). If True, tests $H_0: \mu \ge m$ vs $H_1: \mu < m$
                (left-tailed).
        """
        self.m = float(m)
        self.reverse = bool(reverse)
        self.t = 0
        self.running_sum = 0.0

    def update(self, x: ArrayLike) -> None:
        """Updates internal state with a new batch of observations.

        Args:
            x: New batch of observations (1D array-like).
        """
        x = np.asarray(x, dtype=np.float64)
        self.t += len(x)
        self.running_sum += float(np.sum(x))

    def mean_estimate(self) -> float:
        """Returns the current estimate of the mean."""
        if self.t == 0:
            return float("nan")
        return self.running_sum / self.t

    @abstractmethod
    def log_wealth(self) -> float:
        r"""Returns the log terminal wealth $\log M_t$ against the null hypothesis.

        The null hypothesis is rejected at level $\alpha$ when
        $\log M_t \ge -\log \alpha$.
        """
        pass

    @abstractmethod
    def wealth(self) -> float:
        r"""Returns the current terminal wealth $M_t$ against the null hypothesis.

        The null hypothesis is rejected at level $\alpha$ when $M_t \ge 1/\alpha$.
        """
        pass


class GaussianMixtureSupermartingale(TestSupermartingale):
    r"""Stateful Gaussian mixture test supermartingale.

    Tests the one-sided null hypothesis:
    $$
        H_0: \mu \le m \quad \text{vs.} \quad H_1: \mu > m
    $$
    (or $H_0: \mu \ge m$ vs $H_1: \mu < m$ when `reverse=True`) using the conjugate
    normal mixture supermartingale (Robbins, 1970; Howard et al., 2021, Section 3.2).

    For observations $x_1, \dots, x_t$, centered sum $S_t = \sum_{i = 1}^{t} (x_i - m)$,
    and empirical variance accumulator $V_t = \sum_{i = 1}^{t} (x_i - \bar{x})^2$, the
    one-sided wealth process is given by:
    $$
        M_t = \frac{1}{\sqrt{1 + \frac{V_t}{v_0}}} \exp\left(
            \frac{(\max(0, S_t))^2}{2 (v_0 + V_t)}
        \right)
    $$
    where $v_0 > 0$ is a prior cumulative variance parameter that stabilizes the
    denominator against small-sample zero-variance collapse on rare-event data and tunes
    the horizon.

    References:
        Robbins, H. (1970). Statistical Methods Related to the Law of the Iterated
        Logarithm. The Annals of Mathematical Statistics, 41(5), 1397-1409.
        http://www.jstor.org/stable/2239848

        Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
        nonparametric, nonasymptotic confidence sequences. The Annals of Statistics,
        49(2). https://doi.org/10.1214/20-aos1991

    """

    def __init__(
        self,
        m: float = 0.0,
        v0: float = 1.0,
        reverse: bool = False
    ) -> None:
        r"""
        Args:
            m: Mean $m$ in the null hypothesis.
            v0: Prior cumulative variance parameter $v_0 > 0$.
            reverse: If False (default), tests $H_0: \mu \le m$ vs $H_1: \mu > m$
                (right-tailed). If True, tests $H_0: \mu \ge m$ vs $H_1: \mu < m$
                (left-tailed).
        """
        super().__init__(m=m, reverse=reverse)
        if v0 <= 0.0:
            raise ValueError("Parameter `v0` must be positive.")
        self.v0 = float(v0)
        self.running_sum_sq = 0.0

    def update(self, x: ArrayLike) -> None:
        x = np.asarray(x, dtype=np.float64)
        super().update(x)
        self.running_sum_sq += float(np.sum(x**2))

    def log_wealth(self) -> float:
        if self.t < 2:
            return 0.0

        # Signed deviation from null mean
        s_t = self.running_sum - self.t * self.m
        if self.reverse:
            s_t = -s_t  # Testing H_0: \mu >= m -> evidence when sample mean < m

        if s_t <= 0.0:
            return float("-inf")

        v_t = max(0.0, self.running_sum_sq - (self.running_sum ** 2) / self.t)

        denom = 2.0 * (self.v0 + v_t)
        denom_log = 0.5 * math.log1p(v_t / self.v0)
        return (s_t ** 2) / denom - denom_log

    def wealth(self) -> float:
        log_wealth = self.log_wealth()

        try:
            return math.exp(log_wealth)
        except OverflowError:
            return float("inf")


class BettingSupermartingale(TestSupermartingale):
    r"""Stateful predictable empirical-Bernstein betting supermartingale for
    $[0, 1]$-bounded observations (Waudby-Smith & Ramdas, 2024; based closely on
    `confseq.betting`).

    References:
        Ian Waudby-Smith, Aaditya Ramdas, "Estimating means of bounded random variables
        by betting", Journal of the Royal Statistical Society Series B: Statistical
        Methodology, Volume 86, Issue 1, February 2024, Pages 1-27.
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
        r"""
        Args:
            m: Mean $m$ in the null hypothesis.
            alpha: Significance level in (0, 1).
            population_size: Finite population size $N$ if sampling without replacement.
            horizon: Fixed sample size horizon. If None, uses anytime-valid
                $1/\sqrt{t log t}$ scaling.
            trunc_scale: Scale factor for lambda truncation (default: 0.5).
            m_trunc: Whether truncation depends on m (default: True).
            fake_obs: Number of fake observations for regularizing running mean and
                variance (default: 1).
            prior_mean: Prior mean for regularizing sample mean (default: 0.5).
            prior_variance: Prior variance for regularizing sample variance
                (default: 0.25).
            reverse: If False (default), tests $H_0: \mu \le m$ vs $H_1: \mu > m$
                (right-tailed). If True, tests $H_0: \mu \ge m$ vs $H_1: \mu < m$
                (left-tailed).
        """
        super().__init__(m=m, reverse=reverse)
        self.alpha = float(alpha)
        self.population_size = population_size
        self.horizon = horizon
        self.trunc_scale = float(trunc_scale)
        self.m_trunc = bool(m_trunc)
        self.fake_obs = int(fake_obs)
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)

        # Internal state
        # Note: self.S_t tracks the cumulative sum of the transformed betting
        # observations (i.e. (1 - x) when reverse=True), distinct from
        # self.running_sum in the base class which tracks raw x.
        self.S_t = 0.0
        self.cum_sq_dev = 0.0
        self.last_sigma2 = float(prior_variance)
        self.current_wealth = 1.0

    def update(self, x: ArrayLike) -> None:
        x = np.asarray(x, dtype=np.float64)
        B = x.size
        if B == 0 or self.current_wealth == 0.0:
            super().update(x)
            return
        t0: int = self.t
        super().update(x) # Base class updates self.t and self.running_sum

        if self.reverse:
            x = 1.0 - x
            null_m = 1.0 - self.m
        else:
            null_m = self.m

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
        raw_lambda[~np.isfinite(raw_lambda)] = 0.0

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
            upper = np.full(B, np.inf, dtype=np.float64)
            mask_u = mu_t > 0.0
            upper[mask_u] = self.trunc_scale / mu_t[mask_u]
            lower = np.full(B, -np.inf, dtype=np.float64)
            mask_l = mu_t < 1.0
            lower[mask_l] = -self.trunc_scale / (1.0 - mu_t[mask_l])

            lam = np.clip(raw_lambda, lower, upper)
        else:
            lam = np.clip(raw_lambda, -self.trunc_scale, self.trunc_scale)

        mult = 1.0 + lam * (x - mu_t)
        mult = np.maximum(mult, 0.0)
        mult[np.isnan(mult)] = 0.0
        if self.population_size is not None:
            mult = np.where((mu_t < 0.0) | (mu_t > 1.0), np.inf, mult)

        wealth_batch = float(np.prod(mult))
        self.current_wealth *= wealth_batch

        self.t = int(t[-1])
        self.S_t = float(S_t[-1])
        self.cum_sq_dev = float(cum_sq_dev_t[-1])
        self.last_sigma2 = float(sigma2_t[-1])

    def log_wealth(self) -> float:
        if self.current_wealth <= 0.0:
            return float("-inf")
        return math.log(self.current_wealth)

    def wealth(self) -> float:
        return self.current_wealth


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


# def eval_asymptotic_betting_wealth(
#     x: ArrayLike,
#     m: float,
#     alpha: float = 0.05,
#     c: float = 0.5,
#     prior_mean: float = 0.0,
#     prior_var: float = 1.0,
#     fake_obs: int = 1,
# ) -> float:
#     r"""Evaluates the terminal wealth of the asymptotic betting supermartingale
#     of Waudby-Smith, Arbour, Sinha & Ramdas (2024)
#     """
#     x = np.asarray(x, dtype=np.float64)
#     N = len(x)
#     if N == 0:
#         return 1.0

#     diff = x - m
#     t = np.arange(1, N + 1)

#     mu_hat_t = (fake_obs * prior_mean + np.cumsum(diff)) / (t + fake_obs)
#     mu_prev = np.append(prior_mean, mu_hat_t[:-1])

#     sigma2_t = (fake_obs * prior_var + np.cumsum((diff - mu_hat_t) ** 2)) / (t + fake_obs)
#     sigma2_prev = np.append(prior_var, sigma2_t[:-1])

#     cum_max = np.maximum.accumulate(np.abs(diff))
#     max_prev = np.append(1.0, cum_max[:-1])

#     with np.errstate(divide="ignore", invalid="ignore"):
#         raw_lambda = np.sqrt(
#             2.0 * np.log(1.0 / alpha) / (t * np.log(1.0 + t) * sigma2_prev)
#         )
#         raw_lambda = np.where(mu_prev > 0.0, raw_lambda, 0.0)
#         lambdas = np.minimum(raw_lambda, c / max_prev)
#         lambdas = np.nan_to_num(lambdas, nan=0.0, posinf=0.0, neginf=0.0)

#         multiplicands = np.maximum(1.0 + lambdas * diff, 0.0)
#         wealth_process = np.cumprod(multiplicands)

#     return float(wealth_process[-1])

