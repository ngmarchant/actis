import math

import numpy as np
from confseq.betting import diversified_betting_mart
from confseq.betting_strategies import lambda_predmix_eb
from numpy.typing import ArrayLike, NDArray


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

    Args:
        x: Array of observations bounded in [0, 1].
        m: Candidate mean.
        alpha: Significance level.
        population_size: Finite population size if sampling without replacement.
        horizon: Optional fixed sample size horizon for tuning the betting process. If
            None (default), uses anytime-valid 1/sqrt(t log t) scaling for the betting
            process.
        trunc_scale: The factor by which to multiply the upper truncation on bets.
            Setting to 1 performs no additional truncation beyond the
            boundary-preserving truncation defined in `m_trunc`.
        m_trunc: Should truncation of bets depend on `m`? If True, positive bets are
            truncated by `trunc_scale * (1 / m)` (or `trunc_scale / mu_t` when sampling
            without replacement) to guarantee non-negative wealth. If False, bets are
            truncated by `trunc_scale`.

    Returns:
        The terminal wealth.
    """
    lambdas_fns = [
        lambda data, mean: lambda_predmix_eb(data, alpha=alpha, fixed_n=horizon)
    ]

    wealth_process = diversified_betting_mart(
        x,
        m,
        alpha=alpha,
        lambdas_fns_positive=lambdas_fns,
        N=population_size,
        convex_comb=True,
        theta=1,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )

    return float(wealth_process[-1])


def compute_prior_var(
    scores: ArrayLike,
    gamma_R: float,
    q: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    n_init: int = 1000,
) -> float:
    r"""Computes the prior variance scale v0 for the Gaussian mixture confidence
    sequence.

    Calibrates the prior variance scale to match the expected intrinsic variance $V_n$
    at the calibration sample size horizon `n_init`:
    - Under importance sampling: scales with proxy base rate and margin (`1 - gamma_R`).
    - Under uniform sampling: scales with unweighted Bernoulli variance.

    Args:
        scores: Proxy scores across the population in $[0, 1]$.
        gamma_R: Target recall in $[0, 1]$.
        q: Optional proposal distribution array over the population.
        weights: Optional importance weight array $w(x) = 1 / (N * q(x))$.
        n_init: Calibration sample size scale (default: 1000).

    Returns:
        Calibrated positive prior variance v0 > 0.
    """
    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N == 0:
        return 0.01

    mean_s = float(np.mean(scores))

    if q is not None or weights is not None:
        # Importance sampling: scale with proxy base rate, margin, and pilot size
        v0 = max(0.01, float(n_init) * mean_s * (1.0 - gamma_R))
        return float(np.clip(v0, 0.01, 1.0))

    # Uniform sampling
    margin_scale = (1.0 - gamma_R) ** 2
    sigma_sq = margin_scale * max(mean_s * (1.0 - mean_s), 0.01)
    v0 = float(n_init) * sigma_sq
    return float(np.clip(v0, 0.5, 10.0))


def eval_asymptotic_wealth(
    x: ArrayLike,
    m: float,
    prior_var: float = 1.0,
) -> float:
    r"""Evaluates the terminal wealth of the asymptotic Gaussian mixture supermartingale
    wealth against a candidate mean.

    For observations $x_1, ..., x_n$ with empirical variance accumulator
    $V_n = \sum_{i = 1}^{n} (x_i - \bar{x})^2$ and prior variance $v_0$, the wealth of
    the asymptotic Gaussian mixture supermartingale is given by:
    $$
        M_n(m) = \sqrt{\frac{v_0}{v_0 + V_n}} \exp \left(
            \frac{\sum_{i = 1}^{n} (x_i - m)^2}{2(v_0 + V_n)}
        \right)
    $$

    Args:
        x: Array of observations bounded in [0, 1].
        m: Candidate mean.
        prior_var: Prior variance for the Gaussian mixture supermartingale. Default is
            1.0.

    Returns:
        The terminal wealth.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 1.0

    diff = x - m
    s_n = float(diff.sum())
    if s_n <= 0.0:
        return 0.0

    v_n = float(((x - x.mean())**2).sum())
    denom = prior_var + v_n
    if denom <= 0.0:
        return 1.0

    log_m = 0.5 * math.log(prior_var / denom) + s_n**2 / (2.0 * denom)

    try:
        return math.exp(log_m)
    except OverflowError:
        return float("inf")


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

    For observations $x_1, \dots, x_n$ with candidate mean $m$, the wealth process is:
    $$
        K_n = \prod_{t=1}^n (1 + \lambda_t (x_t - m))
    $$
    with predictable empirical-Bernstein bets and adaptive empirical truncation:
    $$
        \lambda_t = \min \left(
            \sqrt{\frac{2 \log(1/\alpha)}{t \ln(1+t) \hat{\sigma}_{t-1}^2}},
            \frac{c}{\max_{1 \le i < t} |x_i - m|}
        \right)
    $$

    Args:
        x: Array of observations bounded in [0, 1].
        m: Candidate mean.
        alpha: Significance level.
        c: Truncation constant for the empirical-Bernstein bet.
        prior_mean: Prior mean for the Gaussian mixture supermartingale. Default is
            0.0.
        prior_var: Prior variance for the Gaussian mixture supermartingale. Default is
            1.0.
        fake_obs: Number of fake observations to use for regularization of the running
            mean and variance. Default is 1.

    Returns:
        The terminal wealth.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N == 0:
        return 1.0

    diff = x - m
    t = np.arange(1, N + 1)

    # Regularized running mean & variance (predictable by 1-step lag)
    mu_hat_t = (fake_obs * prior_mean + np.cumsum(diff)) / (t + fake_obs)
    mu_prev = np.append(prior_mean, mu_hat_t[:-1])

    sigma2_t = (fake_obs * prior_var + np.cumsum((diff - mu_hat_t) ** 2)) / \
        (t + fake_obs)
    sigma2_prev = np.append(prior_var, sigma2_t[:-1])

    # Running maximum absolute deviation (predictable by 1-step lag)
    cum_max = np.maximum.accumulate(np.abs(diff))
    max_prev = np.append(1.0, cum_max[:-1])

    # Predictable empirical-Bernstein bet with adaptive truncation
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
