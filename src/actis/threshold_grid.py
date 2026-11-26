# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Candidate threshold grid generation and validation utilities."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_thresholds(
    thresholds: ArrayLike,
    name: str = "thresholds",
) -> NDArray[np.float64]:
    r"""Validates a candidate threshold grid.

    Ensures `thresholds` is a non-empty 1D array of finite values in $[0, 1]$ sorted
    in strictly ascending order without duplicates.

    Args:
        thresholds: Candidate threshold values to validate.
        name: Parameter name used in error messages (default: "thresholds").

    Returns:
        Validated 1D array of threshold values as `float64`.

    Raises:
        ValueError: If `thresholds` is empty, not 1D, contains NaNs or values outside
            $[0, 1]$, or is not strictly ascending.
    """
    arr = np.asarray(thresholds, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"`{name}` must be a non-empty 1D array.")
    if np.any(np.isnan(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"`{name}` must contain only finite values in [0, 1].")
    if len(arr) > 1 and np.any(arr[1:] <= arr[:-1]):
        raise ValueError(
            f"`{name}` must be sorted in strictly ascending order without duplicates."
        )
    return arr


def quantile_power_law_grid(
    scores: ArrayLike,
    num_thresholds: int = 500,
    gamma: float | Literal["auto"] = "auto",
) -> NDArray[np.float64]:
    r"""Generates quantile thresholds following a power-law schedule.

    This yields higher resolution in the upper-score tail, where cascade precision
    is most sensitive.

    Args:
        scores: Array of proxy scores for the dataset in $[0, 1]$.
        num_thresholds: Positive integer specifying the number of candidate quantile
            thresholds to generate (default: 500).
        gamma: Power-law exponent $\ge 1.0$ for tail quantile generation. Higher values
            yield more thresholds in the upper tail. Defaults to "auto", which sets
            $\gamma = \max(1.0, \log(N) / \log(M))$, where $N$ is the number of scores
            and $M = \min(\text{num\_thresholds}, N)$.

    Returns:
        Sorted unique threshold values in $[0, 1]$.

    Raises:
        ValueError: If `num_thresholds` is not a positive integer, or if `gamma` is a
            float less than 1.0.
    """
    if num_thresholds < 1 or num_thresholds != int(num_thresholds):
        raise ValueError("Parameter `num_thresholds` must be a positive integer.")
    num_thresholds = int(num_thresholds)

    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    M = min(num_thresholds, N)

    if gamma != "auto" and gamma < 1.0:
        raise ValueError("gamma must be >= 1.0 or 'auto'.")

    if M == 1:
        return np.array([float(np.min(scores))], dtype=np.float64)

    if gamma == "auto":
        gamma = max(1.0, np.log(N) / np.log(M))

    p = 1.0 - (1.0 - np.linspace(0.0, 1.0, M)) ** gamma
    return np.unique(np.quantile(scores, p))
