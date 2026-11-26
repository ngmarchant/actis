# ACTIS Benchmark Experiments

This directory contains the evaluation harness, baseline wrappers, statistical soundness auditing, and table/figure generation scripts used to benchmark **ACTIS** against **BARGAIN**, **LOTUS**, and **ScaleDoc**.


## Setup

### 1. Initialize External Baseline Submodules

The ScaleDoc baseline lives in `externals/ScaleDoc/`:
```bash
git submodule update --init --recursive
```

### 2. Install Experiment Dependencies

From the repository root:
```bash
uv sync --project experiments
```

Run experiment scripts from the repository root with `uv run --project experiments ...`, or activate `experiments/.venv`.

If you plan to run LLM data preparation or labeling, also configure `experiments/.env` and `experiments/config.yaml` as described in [data_prep/README.md](data_prep/README.md).


## Directory Structure

```text
experiments/
├── pyproject.toml                 # Standalone uv environment for experiment dependencies
├── .env.example                   # Template for API credentials and cache directories
├── config.yaml.example            # Template for LiteLLM model and pricing config
├── compare_runners.py             # Main Monte Carlo benchmarking CLI
├── runners.py                     # Unified runner wrappers (ACTIS, BARGAIN, LOTUS, ScaleDoc)
├── scenarios.py                   # Benchmark scenario loaders (SUPG, ScaleDoc, BARGAIN)
├── model_paths.py                 # Canonical model-slugged checkpoint/output paths
├── data_prep/                     # Dataset loaders, LiteLLM wrappers, labeling helpers, and CLI
├── analysis/                      # Post-hoc soundness auditing, stats, tables, and plots
│   ├── results_utils.py           # Hierarchical result discovery and aggregation utilities
│   ├── soundness.py               # Statistical soundness & CI computation helpers
│   ├── analyze_soundness.py       # Exact Clopper-Pearson binomial soundness auditing CLI
│   ├── dataset_stats.py           # Summary stats, PR-AUC, and proxy calibration metrics
│   ├── generate_tables.py         # LaTeX table generator
│   ├── plot_style.py              # Shared figure dimensions, palettes, and save helpers
│   ├── plot_benchmark_summary.py  # Cross-benchmark summary figures (PDF/PGF)
│   └── plot_parameter_sensitivity.py  # Hyperparameter sensitivity plots and tables
└── tests/                         # Experiment and data-preparation test suite
```


## 1. Running Benchmark Comparisons

`compare_runners.py` runs paired Monte Carlo trials comparing ACTIS and baseline cascade tuners across benchmark datasets and writes hierarchical JSON summaries under `experiments/results/` (by default).

### Available Scenarios
- **SUPG Datasets**: `onto`, `imagenet`, `jackson`, `tacred`
- **ScaleDoc Datasets**: `pubmed`, `big_patent`, `gov_report`
- **BARGAIN Datasets**: `court`, `review`, `screenplay`, `wiki`

### Available Runners
- `actis_adaptive_is_asymptotic`: Adaptive ACTIS with importance sampling and asymptotic confidence sequences (with safeguards)
- `actis_adaptive_is_asymptotic_no_safeguards`: Same as above without asymptotic safeguards
- `actis_adaptive_asymptotic`: Adaptive ACTIS with uniform without-replacement sampling and asymptotic confidence sequences
- `actis_adaptive`: Adaptive `ACTISFinite` with uniform without-replacement sampling and betting confidence sequences
- `bargain_pr`: BARGAIN precision/recall cascade baseline
- `lotus_original`: LOTUS cascade calibration baseline
- `scaledoc`: ScaleDoc baseline (with harmonic F1 target conversion for joint P/R)
- `scaledoc_native`: ScaleDoc baseline targeting native F1

### Examples

Run all default runners on a single query:
```bash
python -m experiments.compare_runners \
    --scenario court \
    --query 0 \
    --target-precision 0.90 \
    --target-recall 0.90 \
    --delta 0.05 \
    --num-trials 100
```

Run a subset of runners across all queries of a dataset:
```bash
python -m experiments.compare_runners \
    --scenario pubmed \
    --query all \
    --runners actis_adaptive_is_asymptotic actis_adaptive_asymptotic lotus_original bargain_pr scaledoc
```

Use `--force` to overwrite existing result JSON files, or `--results-dir` to write to a custom directory.


## 2. Statistical Soundness Auditing

`analysis/analyze_soundness.py` audits empirical failure rates across benchmark configurations using exact Clopper-Pearson binomial confidence intervals at significance level `--alpha` (default `0.05`), classifying each configuration as **Certified Sound**, **Inconclusive**, or **Certified Unsound**:

```bash
python -m experiments.analysis.analyze_soundness \
    --results-dir experiments/results \
    --alpha 0.05 \
    --format table \
    --disaggregated
```

Pass `--format all --appendix-tables --output-dir <dir>` to export terminal summaries, CSV files, and LaTeX tables.


## 3. Dataset & Proxy Calibration Statistics

`analysis/dataset_stats.py` reports dataset sizes, missing counts, positive prevalence, PR-AUC, calibration bias, Expected Calibration Error (ECE), and boundary saturation rates across SUPG, BARGAIN, and ScaleDoc benchmarks:

```bash
python -m experiments.analysis.dataset_stats --format table
```


## 4. Generating Paper Tables and Figures

Once benchmark results have been collected in `experiments/results/`:

- **LaTeX Tables** (`booktabs` tables for combined OCR/latency, all ACTIS variants, and constraint slack):
  ```bash
  python -m experiments.analysis.generate_tables --results-dir experiments/results
  ```
- **Benchmark Summary Plots** (head-to-head scatter, IS vs. uniform ablation, teaser unsoundness, and slack distribution):
  ```bash
  python -m experiments.analysis.plot_benchmark_summary --results-dir experiments/results
  ```
- **Parameter Sensitivity Analysis** (e.g., `batch_size`, `alpha`, `num_thresholds`):
  ```bash
  python -m experiments.analysis.plot_parameter_sensitivity --param batch_size
  ```


## 5. Dataset Preparation

- **SUPG benchmarks** (`onto`, `imagenet`, `jackson`, `tacred`) come pre-scored and pre-labeled from `stanford-futuredata/supg` and are downloaded automatically on first use when running `python -m experiments.compare_runners` or `python -m experiments.analysis.dataset_stats` (or pre-fetched via `python -m experiments.data_prep --dataset <name>`). See [data_prep/README.md](data_prep/README.md#pre-labeled-supg-datasets) for details.
- **ScaleDoc and BARGAIN benchmarks** (`pubmed`, `big_patent`, `gov_report`, `court`, `review`, `screenplay`, `wiki`) can be downloaded and labeled using `python -m experiments.data_prep`; see [data_prep/README.md](data_prep/README.md).


## 6. Running Experiment Tests

Sync the experiment environment and run the experiment test suite from the repository root:

```bash
uv sync --project experiments
uv run --project experiments pytest experiments/tests
```
