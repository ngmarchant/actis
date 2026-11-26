# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Proposal distributions for importance sampling in cascade calibration."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def defensive_mixture(
    q: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    r"""Mixes a proposal distribution with a uniform distribution.

    Args:
        q: Proposal distribution over the population dataset for importance sampling.
        alpha: Defensive mixing weight in $[0, 1]$ for the uniform distribution.

    Returns:
        A normalized mixture distribution over the population dataset.
    """
    N = len(q)

    alpha = min(max(alpha, 0.0), 1.0)
    q = alpha * (1.0 / N) + (1.0 - alpha) * q
    q = q / q.sum()

    return q


def compute_pr_proposal(
    scores: ArrayLike,
    gamma_P: float,
    gamma_R: float,
    thresholds: ArrayLike,
    thresholds_upper: ArrayLike | None = None,
    alpha: float | None = None,
    tail_tolerance: float = 0.10,
) -> NDArray[np.float64]:
    r"""Computes a proposal distribution for importance sampling in a cascade.

    The calibration process for a two-threshold cascade jointly evaluates both
    precision and recall targets over a grid of candidate lower thresholds and
    a (potentially separate) grid of candidate upper thresholds. The proposal
    distribution minimizes the joint estimation variance of the precision, recall,
    and prevalence estimators across candidate threshold grids, standardizing each
    constraint margin by its operational slack scale:
    - $\lambda_{\mathrm{TPR}} = 1 + ((1 - \gamma_P) / \gamma_P)^2$
    - $\lambda_{\mathrm{FPR}} = 1$
    - $\lambda_{\pi} = (\gamma_R / (1 - \gamma_R))^2$

    To prevent boundary cancellation caused by proxy overconfidence ($s=0$ causing
    false-negative variance to vanish in recall, or $s=1$ causing false-positive
    variance to vanish in precision), this function applies two-sided
    prevalence-adaptive tail clipping:
    $$\tilde{s}(x) = \mathrm{clip}(s(x), \; c \hat{\pi}, \; 1 - c (1 - \hat{\pi}))$$
    where $\hat{\pi} = \frac{1}{N} \sum_{k=1}^N s(x_k)$ is the unsupervised surrogate
    positive prevalence and $c = \text{tail\_tolerance}$. This is mathematically
    equivalent at the extremes to Bayesian label smoothing toward the prevalence prior.

    Args:
        scores: Array of proxy scores for the dataset in $[0, 1]$.
        gamma_P: Target precision in $[0, 1]$.
        gamma_R: Target recall in $[0, 1]$.
        thresholds: Non-empty 1D grid of candidate lower thresholds in $[0, 1]$.
        thresholds_upper: Optional non-empty 1D grid of candidate upper thresholds in
            $[0, 1]$. If None, defaults to `thresholds`.
        alpha: Defensive mixing weight in $[0, 1]$ for the uniform distribution. If
            None, defaults to 0.0.
        tail_tolerance: Dimensionless tail uncertainty tolerance in $[0, 1)$ for
            prevalence-adaptive clipping (default: 0.10). When 0.0, uses raw proxy
            scores.

    Returns:
        A probability distribution over the population dataset for importance sampling.

    Raises:
        ValueError: If `thresholds` or `thresholds_upper` (when provided) is not a
            non-empty 1D array.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("Parameter `thresholds` must be a non-empty 1D array.")

    if thresholds_upper is not None:
        thresholds_upper = np.asarray(thresholds_upper, dtype=np.float64)
        if thresholds_upper.ndim != 1 or len(thresholds_upper) == 0:
            raise ValueError(
                "Parameter `thresholds_upper` must be a non-empty 1D array."
            )

    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N == 0:
        return np.empty(0, dtype=np.float64)

    pi_hat = float(np.mean(scores))
    # If all proxy scores are 0 or 1, proxy cannot distinguish items; fall back to
    # uniform
    if pi_hat <= 0.0 or pi_hat >= 1.0:
        return np.ones(N, dtype=np.float64) / N

    thresholds = np.unique(thresholds)

    M_lower = np.searchsorted(thresholds, scores, side="right")
    if thresholds_upper is not None:
        thresholds_upper = np.unique(thresholds_upper)
        M_upper = np.searchsorted(thresholds_upper, scores, side="right")
    else:
        thresholds_upper = thresholds
        M_upper = M_lower

    w_lower = 1.0 + ((1.0 - gamma_P) / max(gamma_P, 1e-6)) ** 2
    w_upper = 1.0
    w_base = (gamma_R / max(1.0 - gamma_R, 1e-6)) ** 2

    # Prevalence-adaptive two-sided tail clipping to eliminate boundary blind spots
    if tail_tolerance > 0.0:
        eps_0 = tail_tolerance * pi_hat
        eps_1 = tail_tolerance * (1.0 - pi_hat)
        scores_surrogate = np.clip(scores, eps_0, 1.0 - eps_1)
    else:
        scores_surrogate = scores

    c_s = (
        w_lower * scores_surrogate * M_lower / len(thresholds)
        + w_upper * (1.0 - scores_surrogate) * M_upper / len(thresholds_upper)
        + w_base * scores_surrogate
    )

    c_s = np.maximum(c_s, 0.0)

    q = np.sqrt(c_s)
    sum_q = q.sum()

    if sum_q > 0:
        q = q / sum_q
    else:
        q = np.ones(N, dtype=np.float64) / N

    if alpha is not None and alpha > 0.0:
        q = defensive_mixture(q, alpha)

    return q
