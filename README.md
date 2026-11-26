# ACTIS: Adaptive Cascade Tuning for Semantic Filters with Statistical Guarantees

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation and benchmarks for our paper *"ACTIS: Adaptive Cascade Tuning for Semantic Filters with Statistical Guarantees"*.


## Overview

Semantic filters evaluate complex predicates over unstructured data (e.g., text, images). While foundation models (LLMs, VLMs) deliver high accuracy, running them on every item at query scale is expensive. *Model cascades* mitigate this overhead by routing items through a fast, cheap proxy model and invoking the expensive oracle only for ambiguous items.

ACTIS tunes two confidence thresholds $(\tau_{\text{lower}}, \tau_{\text{upper}})$ for a three-way filter cascade:
- **Discard** if proxy score $s(x) < \tau_{\text{lower}}$ (predicted negative).
- **Accept** if proxy score $s(x) \ge \tau_{\text{upper}}$ (predicted positive).
- **Escalate to Oracle** if $\tau_{\text{lower}} \le s(x) < \tau_{\text{upper}}$ (query oracle ground truth).

ACTIS chooses $(\tau_{\text{lower}}, \tau_{\text{upper}})$ to minimize total oracle calls across tuning and deployment, while guaranteeing that precision and recall targets are satisfied with probability at least $1 - \delta$.

Key features:

- **Anytime-Valid Testing with Safeguards**: Evaluates candidate thresholds sequentially across batches using asymptotic Gaussian-mixture confidence sequences (`ACTIS`) paired with finite-sample runtime safeguards.
- **Variance-Minimizing Importance Sampling**: Constructs a variance-minimizing proposal with defensive mixing and prevalence-adaptive tail clipping, focusing oracle calls on informative tuples.
- **Value-of-Information Adaptive Stopping**: Computes optimistic threshold bounds and halts tuning once the expected reduction in deployment oracle calls falls below the cost of labeling the next batch.


## Installation

### Prerequisites
- Python $\ge$ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended)

### Install ACTIS as a Package

Add `actis` as a dependency in another project:
```bash
uv add git+https://github.com/ngmarchant/actis.git
```

### Local Development Setup

Clone the repository and sync the development environment:
```bash
git clone https://github.com/ngmarchant/actis.git
cd actis
uv sync

# Activate the environment (or prefix commands with `uv run`)
source .venv/bin/activate
```

For benchmarking dependencies and external baseline submodules, see [experiments/README.md](experiments/README.md).


## Quickstart: Python API

The `actis` package provides a standalone Python API for tuning two-threshold cascades:

```python
import numpy as np

from actis import (
    ACTIS,
    PopulationSampler,
    compute_pr_proposal,
    compute_prior_and_target_var,
    quantile_power_law_grid,
)

# Simulated population: proxy scores in [0, 1] and ground-truth oracle labels
rng = np.random.default_rng(42)
pop_size = 5000
scores = rng.beta(0.4, 0.8, size=pop_size)
labels = rng.binomial(1, scores).astype(bool)

# Target precision, recall, and joint failure probability
gamma_P = 0.90
gamma_R = 0.90
delta = 0.05

# Construct candidate grids for lower and upper thresholds
thresholds = quantile_power_law_grid(scores, num_thresholds=20, gamma=1.0)
thresholds_upper = quantile_power_law_grid(scores, num_thresholds=200)

# Compute variance-minimizing importance sampling proposal and weights
q = compute_pr_proposal(
    scores=scores,
    gamma_P=gamma_P,
    gamma_R=gamma_R,
    thresholds=thresholds,
    thresholds_upper=thresholds_upper,
    alpha=0.4,
)
weights = 1.0 / (pop_size * q)

# Compute calibrated prior variances for asymptotic confidence sequences
v_0 = compute_prior_and_target_var(
    scores=scores,
    gamma_P=gamma_P,
    gamma_R=gamma_R,
    delta=delta,
    thresholds=thresholds,
    thresholds_upper=thresholds_upper,
    weights=weights,
)

# Initialize ACTIS tuner and importance sampler
tuner = ACTIS(
    gamma_P=gamma_P,
    gamma_R=gamma_R,
    delta=delta,
    thresholds=thresholds,
    thresholds_upper=thresholds_upper,
    v_0=v_0,
)
sampler = PopulationSampler(pop_size=pop_size, replace=True, p=q, rng=rng)

# Sequentially label batches until the stopping condition is satisfied
batch_size = 100
while True:
    idx = sampler.sample(batch_size)
    result = tuner.add_samples(
        indices=idx,
        scores=scores[idx],
        labels=labels[idx],
        weights=weights[idx],
        population_scores=scores,
    )
    should_continue, _ = tuner.should_continue_sampling(
        population_scores=scores,
        batch_size=batch_size,
        max_sample_size=2000,
    )
    if not should_continue:
        break

print(
    f"Tuned Cascade (n={result.num_samples}): "
    f"tau_lower={result.tau_lower:.4f}, tau_upper={result.tau_upper:.4f}"
)
```

For non-asymptotic finite-sample guarantees using betting supermartingales, use `ACTISFinite` (with `pop_size` for uniform without-replacement sampling or `max_weight_ge`/`max_weight_lt`/`max_weight_ge_upper` for importance sampling).


## Repository Structure

```text
├── src/actis/                    # Core ACTIS library package
│   ├── tuner.py                  # Base CascadeTuner, CascadeConfSeqs, and result types
│   ├── asymptotic.py             # Asymptotic tuner, priors, and diagnostics
│   ├── finite.py                 # Finite-sample tuner
│   ├── confseq.py                # Gaussian mixture and betting test supermartingales
│   ├── proposals.py              # Importance-sampling proposals & tail clipping
│   ├── sampler.py                # Population index sampler (with/without replacement)
│   └── threshold_grid.py         # Candidate threshold validation & power-law grids
├── experiments/                  # Benchmark evaluation, soundness auditing, and plots
│   ├── README.md                 # Guide for reproducing paper experiments
│   └── data_prep/                # Dataset loaders and LLM labeling toolkit
│       └── README.md             # Guide for preparing datasets and proxy/oracle labels
└── tests/                        # Core library unit tests
```


## Reproducing Paper Experiments

The benchmarking harness, baseline comparisons (BARGAIN, LOTUS, ScaleDoc), statistical soundness auditing, and figure/table generation scripts live under [`experiments/`](experiments/).

- Running benchmarks, auditing, tables, and plots, see [experiments/README.md](experiments/README.md).
- Preparing datasets and collecting proxy/oracle labels, see [experiments/data_prep/README.md](experiments/data_prep/README.md).


## Development

Run the unit test suite, linter, formatter, and type checker from the repository root:

```bash
# Run core library unit tests
pytest

# Lint and check formatting
ruff check src
ruff format --check src

# Type check
ty check src
```


## License

This project is released under the AGPL-3.0 License.


## Citation

If you use ACTIS in your research, please cite our arXiv preprint:

```bibtex
@misc{marchant2026actis,
  title         = {ACTIS: Adaptive Cascade Tuning for Semantic Filters with Statistical Guarantees},
  author        = {Neil Marchant and Renata Borovica-Gajic and Benjamin Rubinstein and Olga Ohrimenko},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
