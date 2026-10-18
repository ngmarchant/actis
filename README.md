# ACTIS: Adaptive Cascade Tuning for Semantic Filters with Statistical Guarantees

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official reference implementation of **ACTIS** (Adaptive Cascade Tuning via Importance Sampling), as presented in our SIGMOD paper *"ACTIS: Adaptive Cascade Tuning for Semantic Filters with Statistical Guarantees"*.

---

## Overview

Semantic filters are increasingly deployed in data management systems to evaluate complex predicates over unstructured data (e.g., text, images). While foundation models (LLMs, VLMs) deliver high accuracy, their execution cost and latency are prohibitive at query scale. *Model cascades* mitigate this overhead by routing items through a fast, cheap proxy model (e.g., a lightweight classifier or small LLM) and invoking the expensive oracle only for ambiguous predictions.

**ACTIS** tunes two confidence thresholds $(\tau_{\text{neg}}, \tau_{\text{pos}})$ (denoted $\check{\tau}, \hat{\tau}$ in the paper) for a three-way filter cascade:
- **Discard** if proxy score $s(x) < \tau_{\text{neg}}$ (predicted negative).
- **Accept** if proxy score $s(x) \ge \tau_{\text{pos}}$ (predicted positive).
- **Defer to Oracle** if $\tau_{\text{neg}} \le s(x) < \tau_{\text{pos}}$ (query oracle ground truth).

Key features of ACTIS:
1. **Simultaneous Statistical Guarantees**: Guarantees that both precision ($\ge \gamma_P$) and recall ($\ge \gamma_R$) constraints hold jointly with probability at least $1 - \delta$.
2. **Anytime-Valid Confidence Sequences**: Uses stateful Gaussian mixture and betting supermartingales that permit continuous adaptive monitoring and early stopping without inflating error rates.
3. **Variance-Minimizing Importance Sampling**: Samples unlabeled records according to a biased instrumental distribution designed to reduce the required oracle labeling queries.

---

## Repository Structure

```text
├── src/actis/                    # Core ACTIS library package
│   ├── tuner.py                  # Main ACTIS tuner and cascade threshold search
│   ├── confseq.py                # Supermartingales and confidence sequences
│   ├── proposals.py              # Instrumental proposal distributions & tail clipping
│   ├── sampler.py                # Adaptive population sampler
│   └── threshold_grid.py         # Candidate threshold grid generation (power-law quantiles)
├── experiments/                  # Benchmark suite and evaluation tools (see experiments/README.md)
│   ├── compare_tuners.py         # Main benchmarking CLI (ACTIS vs. baselines)
│   ├── analyze_soundness.py      # Statistical soundness auditing CLI
│   └── data_prep/                # Data preparation & LLM labeling pipeline (see data_prep/README.md)
├── externals/ScaleDoc/           # ScaleDoc baseline reference submodule
├── tests/                        # Comprehensive unit test suite
├── config.yaml.example           # Example LiteLLM configuration for proxy/oracle deployments
├── .env.example                  # Environment template for API keys and cache directories
├── pyproject.toml                # Project metadata, build system, and dependencies
└── uv.lock                       # Locked dependency tree for reproducibility
```

---

## Installation and Setup

### Prerequisites
- Python $\ge$ 3.10
- [`uv`](https://docs.astral.sh/uv/)
- Git (with submodule support for baseline evaluations)

### 1. Clone the Repository

Clone this repository and initialize the external baseline submodules:
```bash
git clone <REPO_URL>
cd actis
git submodule update --init --recursive
```

### 2. Environment Setup via `uv`

Install dependencies and set up the virtual environment using `uv sync`:
```bash
# Install core library along with development and experiment dependencies
uv sync --extra experiments --extra dev

# Or for the core library only:
# uv sync
```

Commands and scripts can then be executed in the environment using `uv run`.

---

## Quickstart: Python API

The `actis` package can be used directly as a standalone library:

```python
import numpy as np
from actis import ACTIS, compute_pr_proposal, compute_prior_and_target_var
from actis.sampler import PopulationSampler

# Simulated population: proxy scores in [0, 1] and ground-truth oracle labels
rng = np.random.default_rng(42)
pop_size = 5000
scores = rng.beta(2, 5, size=pop_size)
labels = rng.binomial(1, scores).astype(bool)

# Target quality constraints
gamma_R = 0.90  # Recall constraint
gamma_P = 0.90  # Precision constraint
delta = 0.05    # Joint failure probability (95% guarantee)

# For simplicity search over the same grid for upper and lower thresholds
num_thresholds = 20
thresholds = np.linspace(0.0, 1.0, num_thresholds)

# Variance-minimizing importance sampling distribution
q = compute_pr_proposal(
    scores=scores,
    gamma_P=gamma_P,
    gamma_R=gamma_R,
    thresholds=thresholds,
    tail_tolerance=0.10,
)
weights = 1.0 / (pop_size * q)

# Precompute prior variance for asymptotic confidence sequences
v_0 = compute_prior_and_target_var(
    scores=scores,
    gamma_R=gamma_R,
    gamma_P=gamma_P,
    delta=delta,
    thresholds=thresholds,
    weights=weights,
)

# Initialize ACTIS tuner and adaptive population sampler
tuner = ACTIS(
    gamma_R=gamma_R,
    gamma_P=gamma_P,
    delta=delta,
    thresholds=thresholds,
    conf_seq="asymptotic",
    v_0=v_0,
)
sampler = PopulationSampler(pop_size=pop_size, replace=True, p=q, rng=rng)

# Parameters for sampling oracle-labeled records during tuning
init_sample_size = 100
batch_size = 50
max_sample_size = 1000

# Draw initial batch
batch_idx = sampler.sample(init_sample_size)
calib_res = tuner.add_samples(
    indices=batch_idx,
    scores=scores[batch_idx],
    labels=labels[batch_idx],
    weights=weights[batch_idx],
    population_scores=scores,
)

# Adaptively tune using value-of-information stopping condition
while True:
    should_continue, _ = tuner.should_continue_sampling(
        population_scores=scores,
        batch_size=batch_size,
        max_sample_size=max_sample_size,
    )
    if not should_continue:
        break

    batch_idx = sampler.sample(batch_size)
    calib_res = tuner.add_samples(
        indices=batch_idx,
        scores=scores[batch_idx],
        labels=labels[batch_idx],
        weights=weights[batch_idx],
        population_scores=scores,
    )

print(f"Tuned Cascade: tau_neg={calib_res.tau_neg:.4f}, tau_pos={calib_res.tau_pos:.4f}")
print(f"Oracle calls used: {calib_res.num_samples}")
print(f"Predicted population oracle rate: {calib_res.predicted_oracle_rate:.4f}")
```

---

## Experiments and Reproducibility

Detailed guides, command-line arguments, and reproduction workflows are organized within the `experiments/` directory:
- **Benchmarking & Auditing**: See [experiments/README.md](experiments/README.md) for instructions on running `compare_tuners.py` across benchmark datasets (SUPG, ScaleDoc, BARGAIN) and executing Clopper-Pearson statistical soundness audits with `analyze_soundness.py`.
- **Data Ingestion & Labeling**: See [experiments/data_prep/README.md](experiments/data_prep/README.md) for instructions on ingesting datasets, setting up API endpoints, and generating proxy confidence scores and oracle labels via `prepare_data.py`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

