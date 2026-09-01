from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(kw_only=True)
class PopulationSampler:

    def __init__(
            self,
            pop_size: int,
            replace: bool = True,
            p: ArrayLike | None = None,
            rng: np.random.Generator | None = None
        ):
        """
        Args:
            pop_size: Size of the population dataset.
            replace: Whether to sample with replacement. Default is True.
            p: Optional array of probabilities for sampling. If None, uniform
                sampling is used.
            rng: Optional random number generator. If None, a default generator
                is created.
        """
        if (p is not None) and (not replace):
            raise ValueError(
                "Importance sampling without replacement is not supported."
            )
        self.pop_size = pop_size
        self.replace = replace
        self.p = None
        if p is not None:
            self.p = np.asarray(p, dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._perm_idx = None
        self._sample_count = 0

    def sample(self, size: int) -> list[int]:
        """
        Sample indices from the population dataset.

        Args:
            size: Number of samples to draw.

        Returns:
            A list of sampled indices.
        """
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
            return self._perm_idx[i:i+size].tolist()

        self._sample_count += size
        idx = self.rng.choice(self.pop_size, size=size, replace=True, p=self.p)
        return idx.tolist()

    @property
    def sample_count(self) -> int:
        """Number of samples drawn from the population dataset."""
        return self._sample_count
