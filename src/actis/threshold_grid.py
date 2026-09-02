from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_thresholds(
    thresholds: ArrayLike,
    name: str = "thresholds",
) -> NDArray[np.float64]:
    """Validates that thresholds are a non-empty 1D array sorted in strictly ascending
    order."""
    arr = np.asarray(thresholds, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"`{name}` must be a non-empty 1D array.")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(
            f"`{name}` must be sorted in strictly ascending order without duplicates."
        )
    return arr

def quantile_power_law_grid(
    scores: ArrayLike,
    num_thresholds: int = 500,
    gamma: float | Literal["auto"] = "auto"
) -> NDArray[np.float64]:
    r"""Generates quantile thresholds following a power-law schedule.

    This yield higher resolution in the upper-score tail, where the cascade precision is
    most sensitive.

    Args:
        scores: Array of proxy scores for the dataset. Scores must be in the unit
            interval $[0, 1]$.
        num_thresholds: Number of candidate quantile thresholds to generate.
        gamma: Power-law exponent $>= 1.0$ for tail quantile generation. Higher values
            yield more thresholds in the upper tail. Defaults to "auto", which sets
            $\gamma = \log(N) / \log(M)$, where $N$ is the number of scores and $M$ is
            the number of thresholds.

    Returns:
        Sorted unique threshold values in $[0, 1]$.
    """
    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    M = min(num_thresholds, N)

    if gamma == "auto":
        gamma = max(1.0, np.log(N) / np.log(M))
    elif gamma < 1.0:
        raise ValueError("gamma must be >= 1.0 or 'auto'.")

    p = 1.0 - (1.0 - np.linspace(0.0, 1.0, M)) ** gamma
    return np.unique(np.quantile(scores, p))

