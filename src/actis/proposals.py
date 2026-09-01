from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ProposalMethod = Literal["var_min", "snr_balanced"]


def defensive_mixture(
    q: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    r"""Mixes a proposal distribution with a uniform distribution for defensive
    sampling

    Args:
        q: Proposal distribution over the population dataset for importance sampling.
        alpha: Defensive mixing weight in (0, 1] for the uniform distribution.

    Returns:
        A distribution over the population dataset for importance sampling.
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
    method: ProposalMethod = "snr_balanced"
) -> NDArray[np.float64]:
    r"""Computes a proposal distribution for importance sampling in a
    two-threshold cascade with defensive mixing.

    The calibration process for a two-threshold cascade jointly evaluates both
    precision and recall targets over a grid of candidate lower thresholds and
    a (potentially separate) grid of candidate upper thresholds. This function
    supports two optimization principles for constructing the proposal distribution:

    - ``"snr_balanced"`` (default): Weights each margin variance term by its
      signal scale ($(1 - \gamma_R)^2$ for recall and $\gamma_P^2$ for precision),
      symmetrically balancing true positive discovery and false positive rejection
      based on signal-to-noise ratio.
    - ``"var_min"``: Minimizes the total unweighted sum of estimator variances
      across all candidate threshold pairs (dominated by precision variance).

    Args:
        scores: Array of proxy scores for the dataset. Scores must be in the unit
            interval $[0, 1]$.
        gamma_P: Target precision in $[0, 1]$.
        gamma_R: Target recall in $[0, 1]$.
        thresholds: Grid of candidate thresholds in $[0, 1]$ (for the lower threshold).
        thresholds_upper: Optional separate grid of candidate thresholds in $[0, 1]$ for
            the upper threshold. If None, defaults to `thresholds`.
        alpha: Defensive mixing weight in $(0, 1]$ for the uniform distribution. If
            None, defaults to 0.0.
        method: Optimization principle to use: ``"snr_balanced"`` (default) for
            SNR-standardized weighting or ``"var_min"`` for unweighted total variance
            minimization.

    Returns:
        A distribution over the population dataset for importance sampling.
    """
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(np.asarray(thresholds, dtype=np.float64))
    N = len(scores)

    M_lower = np.searchsorted(thresholds, scores, side="right")
    if thresholds_upper is not None:
        thresholds_upper = np.unique(np.asarray(thresholds_upper, dtype=np.float64))
        M_upper = np.searchsorted(thresholds_upper, scores, side="right")
    else:
        thresholds_upper = thresholds
        M_upper = M_lower

    if method == "snr_balanced":
        w_lower =  1.0 + ((1.0 - gamma_P) / max(gamma_P, 1e-6)) ** 2
        w_upper = 1.0
        w_base = (gamma_R / max(1.0 - gamma_R, 1e-6)) ** 2
    elif method == "var_min":
        w_lower = 2.0 + gamma_P**2 - 2 * gamma_P - 2 * gamma_R
        w_upper = gamma_P**2
        w_base = gamma_R**2
    else:
        raise ValueError(f"Unknown proposal method: {method}")

    c_s = (
        w_lower * scores * M_lower / len(thresholds)
        + w_upper * (1.0 - scores) * M_upper / len(thresholds_upper)
        + w_base * scores
    )

    c_s = np.maximum(c_s, 0.0)

    q = np.sqrt(c_s)
    sum_q = q.sum()

    if sum_q > 0:
        q = q / sum_q
    else:
        q = np.ones(N, dtype=np.float64)
        q = q / q.sum()

    if alpha is not None:
        q = defensive_mixture(q, alpha)

    return q
