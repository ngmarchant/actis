# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Population index sampling utilities for adaptive calibration."""

import numpy as np
from numpy.typing import ArrayLike


class PopulationSampler:
    """Samples item indices from a finite population with or without replacement."""

    def __init__(
        self,
        pop_size: int,
        replace: bool = True,
        p: ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initializes the population sampler.

        Args:
            pop_size: Size of the population dataset (must be a positive integer).
            replace: Whether to sample with replacement (default: True).
            p: Optional 1D array of non-negative sampling probabilities of length
                `pop_size`. If None, uniform sampling is used.
            rng: Optional random number generator. If None, creates a default generator.

        Raises:
            ValueError: If `pop_size` is not a positive integer, if `p` is provided
                with `replace=False`, or if `p` has invalid shape, negative/non-finite
                values, or does not sum to a positive finite value.
        """
        if pop_size <= 0 or pop_size != int(pop_size):
            raise ValueError("Parameter `pop_size` must be a positive integer.")
        if (p is not None) and (not replace):
            raise ValueError(
                "Importance sampling without replacement is not supported."
            )
        self.pop_size = int(pop_size)
        self.replace = replace
        self.p = None
        self._cdf = None
        if p is not None:
            self.p = np.asarray(p, dtype=np.float64)
            if self.p.ndim != 1 or len(self.p) != self.pop_size:
                raise ValueError(
                    "Parameter `p` must be a 1D array of length `pop_size`."
                )
            if np.any(~np.isfinite(self.p)) or np.any(self.p < 0.0):
                raise ValueError(
                    "Probabilities in `p` must be finite and non-negative."
                )
            self._cdf = np.cumsum(self.p)
            if self._cdf[-1] <= 0.0 or not np.isfinite(self._cdf[-1]):
                raise ValueError(
                    "Probabilities in `p` must sum to a positive finite value."
                )
            self._cdf /= self._cdf[-1]
            self._cdf[-1] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()
        self._perm_idx = None
        self._sample_count = 0

    def sample(self, size: int) -> list[int]:
        """Samples indices from the population dataset.

        Args:
            size: Number of samples to draw (non-negative integer).

        Returns:
            A list of sampled integer indices.

        Raises:
            ValueError: If `size` is negative/non-integer, or if `size` exceeds the
                remaining population when `replace=False`.
        """
        if size < 0 or size != int(size):
            raise ValueError("Parameter `size` must be a non-negative integer.")
        size = int(size)
        if not self.replace:
            if self._sample_count + size > self.pop_size:
                raise ValueError(
                    "Sample size exceeds population size when sampling without "
                    "replacement."
                )
            if self._perm_idx is None:
                self._perm_idx = self.rng.permutation(self.pop_size)
            i = self._sample_count
            self._sample_count += size
            return self._perm_idx[i : i + size].tolist()

        self._sample_count += size
        if self._cdf is None:
            idx = self.rng.choice(self.pop_size, size=size, replace=True)
        else:
            u = self.rng.random(size)
            idx = np.searchsorted(self._cdf, u, side="right")
            idx = np.minimum(idx, self.pop_size - 1)
        return idx.tolist()

    @property
    def sample_count(self) -> int:
        """Number of samples drawn from the population dataset."""
        return self._sample_count
