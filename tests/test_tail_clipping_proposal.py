# Copyright (C) 2026 The ACTIS Authors
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import pytest

from actis import compute_pr_proposal


def test_strict_support_positivity():
    """Verify that q(x) > 0 on entire support even with extreme boundary scores 0 and
    1."""
    scores = np.array([0.0, 0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0])
    thresholds = [0.0, 0.2, 0.5, 0.8]

    q = compute_pr_proposal(
        scores=scores,
        gamma_P=0.95,
        gamma_R=0.95,
        thresholds=thresholds,
        alpha=0.0,
        tail_tolerance=0.10,
    )

    assert len(q) == len(scores)
    assert np.all(q > 0.0), "Proposal must be strictly positive on the entire support."
    assert np.isclose(np.sum(q), 1.0), "Proposal must sum to 1.0."


def test_all_zero_prevalence_fallback():
    """Verify that proposal gracefully degenerates to uniform when all scores are 0."""
    scores = np.zeros(50)
    thresholds = [0.0, 0.5]

    q = compute_pr_proposal(
        scores=scores,
        gamma_P=0.95,
        gamma_R=0.95,
        thresholds=thresholds,
        tail_tolerance=0.10,
    )

    assert len(q) == 50
    assert np.allclose(q, 1.0 / 50), (
        "All-zero scores must degenerate to uniform distribution."
    )


def test_all_one_prevalence_fallback():
    """Verify that proposal gracefully degenerates to uniform when all scores are 1."""
    scores = np.ones(50)
    thresholds = [0.0, 0.5]

    q = compute_pr_proposal(
        scores=scores,
        gamma_P=0.95,
        gamma_R=0.95,
        thresholds=thresholds,
        tail_tolerance=0.10,
    )

    assert len(q) == 50
    assert np.allclose(q, 1.0 / 50), (
        "All-one scores must degenerate to uniform distribution."
    )


def test_zero_tail_tolerance_unclipped():
    """Verify that tail_tolerance=0.0 bypasses clipping."""
    scores = np.array([0.0, 0.2, 0.8, 1.0])
    thresholds = [0.0, 0.5]

    q_unclipped = compute_pr_proposal(
        scores=scores,
        gamma_P=0.95,
        gamma_R=0.95,
        thresholds=thresholds,
        alpha=0.0,
        tail_tolerance=0.0,
    )

    q_clipped = compute_pr_proposal(
        scores=scores,
        gamma_P=0.95,
        gamma_R=0.95,
        thresholds=thresholds,
        alpha=0.0,
        tail_tolerance=0.10,
    )

    assert not np.allclose(q_unclipped, q_clipped), (
        "tail_tolerance=0.0 should differ from tail_tolerance=0.10 when boundary values"
        " are present."
    )


def test_empty_thresholds_raises():
    scores = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="thresholds.*non-empty 1D array"):
        compute_pr_proposal(scores=scores, gamma_P=0.9, gamma_R=0.9, thresholds=[])

    with pytest.raises(ValueError, match="thresholds_upper.*non-empty 1D array"):
        compute_pr_proposal(
            scores=scores,
            gamma_P=0.9,
            gamma_R=0.9,
            thresholds=[0.2, 0.8],
            thresholds_upper=[],
        )

