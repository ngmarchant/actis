# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

"""ACTIS: Adaptive Cascade Tuning via Importance Sampling."""

from .asymptotic import (
    ACTIS,
    AsymptoticValidityDiagnostics,
    PriorAndTargetVar,
    compute_prior_and_target_var,
)
from .finite import ACTISFinite
from .proposals import compute_pr_proposal, defensive_mixture
from .sampler import PopulationSampler
from .threshold_grid import quantile_power_law_grid, validate_thresholds
from .tuner import (
    CascadeThresholds,
    CascadeTuner,
    StoppingDiagnostics,
)

__all__ = [
    "ACTIS",
    "ACTISFinite",
    "AsymptoticValidityDiagnostics",
    "CascadeThresholds",
    "CascadeTuner",
    "PopulationSampler",
    "PriorAndTargetVar",
    "StoppingDiagnostics",
    "compute_pr_proposal",
    "compute_prior_and_target_var",
    "defensive_mixture",
    "quantile_power_law_grid",
    "validate_thresholds",
]
