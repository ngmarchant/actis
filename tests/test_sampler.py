# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for PopulationSampler."""

import numpy as np
import pytest

from actis.sampler import PopulationSampler


def test_population_sampler_identity_and_basic_sampling():
    rng = np.random.default_rng(42)
    s1 = PopulationSampler(pop_size=10, replace=True, rng=rng)
    s2 = PopulationSampler(pop_size=10, replace=True, rng=rng)
    assert s1 != s2

    idx = s1.sample(4)
    assert len(idx) == 4
    assert s1.sample_count == 4
    assert all(0 <= i < 10 for i in idx)


def test_population_sampler_validation():
    with pytest.raises(ValueError, match="pop_size"):
        PopulationSampler(pop_size=0)

    with pytest.raises(ValueError, match="length `pop_size`"):
        PopulationSampler(pop_size=5, p=[0.5, 0.5])

    with pytest.raises(ValueError, match="finite and non-negative"):
        PopulationSampler(pop_size=3, p=[0.5, -0.1, 0.6])

    with pytest.raises(ValueError, match="finite and non-negative"):
        PopulationSampler(pop_size=3, p=[0.5, np.nan, 0.5])

    sampler = PopulationSampler(pop_size=5)
    with pytest.raises(ValueError, match="non-negative integer"):
        sampler.sample(-1)
