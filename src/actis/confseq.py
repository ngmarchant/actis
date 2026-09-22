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

    def __init__(self, m: float = 0.0, v_0: float = 1.0, reverse: bool = False) -> None:
        r"""
        Args:
            m: Mean $m$ in the null hypothesis.
            v_0: Prior cumulative variance parameter $v_0 > 0$.
            reverse: If False (default), tests $H_0: \mu \le m$ vs $H_1: \mu > m$
                (right-tailed). If True, tests $H_0: \mu \ge m$ vs $H_1: \mu < m$
                (left-tailed).
        """
        super().__init__(m=m, reverse=reverse)
        if v_0 <= 0.0:
            raise ValueError("Parameter `v_0` must be positive.")
        self.v_0 = float(v_0)
        self.running_sum_sq = 0.0
        self.x_min = float("inf")
        self.x_max = float("-inf")

    def update(self, x: ArrayLike) -> None:
        x = np.asarray(x, dtype=np.float64)
        if len(x) > 0:
            self.x_min = min(self.x_min, float(np.min(x)))
            self.x_max = max(self.x_max, float(np.max(x)))
        super().update(x)
        self.running_sum_sq += float(np.sum(x**2))

    @property
    def v_t(self) -> float:
        r"""Empirical variance accumulator
        $V_t = \sum_{i=1}^t (x_i - \bar{x}_t)^2$.
        """
        if self.t < 2:
            return 0.0
        return max(0.0, self.running_sum_sq - (self.running_sum**2) / self.t)

    def log_wealth(self) -> float:
        if self.t < 2:
            return 0.0

        # Signed deviation from null mean
        s_t = self.running_sum - self.t * self.m
        if self.reverse:
            s_t = -s_t  # Testing H_0: \mu >= m -> evidence when sample mean < m

        if s_t <= 0.0:
            return float("-inf")

        v_t = self.v_t

        denom = 2.0 * (self.v_0 + v_t)
        denom_log = 0.5 * math.log1p(v_t / self.v_0)
        return (s_t**2) / denom - denom_log

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
        pop_size: int | None = None,
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
            pop_size: Finite population size $N$ if sampling without replacement.
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
        self.pop_size = pop_size
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
        super().update(x)  # Base class updates self.t and self.running_sum

        if self.reverse:
            x = 1.0 - x
            null_m = 1.0 - self.m
        else:
            null_m = self.m

        t = t0 + np.arange(1, B + 1, dtype=np.float64)
        S_t = self.S_t + np.cumsum(x)

        # Online running regularized mean and variance
        mu_hat_t = np.minimum(
            (self.fake_obs * self.prior_mean + S_t) / (t + self.fake_obs), 1.0
        )
        sq_dev = (x - mu_hat_t) ** 2
        cum_sq_dev_t = self.cum_sq_dev + np.cumsum(sq_dev)
        sigma2_t = (self.fake_obs * self.prior_variance + cum_sq_dev_t) / (
            t + self.fake_obs
        )

        # 1-step predictable variance
        sigma2_prev = np.empty(B, dtype=np.float64)
        sigma2_prev[0] = self.last_sigma2
        if B > 1:
            sigma2_prev[1:] = sigma2_t[:-1]

        # Predictable bets
        if self.horizon is None:
            raw_lambda = np.sqrt(
                2.0 * np.log(1.0 / self.alpha) / (t * np.log(1.0 + t) * sigma2_prev)
            )
        else:
            raw_lambda = np.sqrt(
                2.0 * np.log(1.0 / self.alpha) / (self.horizon * sigma2_prev)
            )
        raw_lambda[~np.isfinite(raw_lambda)] = 0.0

        # Conditional null mean (accounting for hypergeometric without-replacement draws
        # if N is set)
        if self.pop_size is not None:
            S_prev = np.empty(B, dtype=np.float64)
            S_prev[0] = self.S_t
            if B > 1:
                S_prev[1:] = S_t[:-1]
            mu_t = (self.pop_size * null_m - S_prev) / (self.pop_size - (t - 1.0))
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
        if self.pop_size is not None:
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
