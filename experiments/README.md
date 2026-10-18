# ACTIS Experimental Evaluation and Auditing

This directory contains the benchmarking and statistical soundness auditing pipeline for **ACTIS** and baseline cascade tuning methods.

---

## Directory Overview

- **[`compare_tuners.py`](compare_tuners.py)**: Primary benchmarking CLI to evaluate ACTIS, BARGAIN, LOTUS, and ScaleDoc cascades across benchmark datasets.
- **[`analyze_soundness.py`](analyze_soundness.py)**: Statistical soundness auditor evaluating empirical constraint satisfaction using exact Clopper-Pearson binomial confidence intervals.
- **[`runners.py`](runners.py)**: Runner adapters wrapping each cascade tuning method (ACTIS, BARGAIN, LOTUS, ScaleDoc).
- **[`scenarios.py`](scenarios.py)**: Population dataset loaders for SUPG, ScaleDoc, and BARGAIN benchmarks.
- **[`soundness.py`](soundness.py)**: Exact binomial coverage and Clopper-Pearson interval calculation utilities.
- **[`results_utils.py`](results_utils.py)**: Hierarchical results discovery, JSON loaders, and formatting metadata.
- **[`model_paths.py`](model_paths.py)**: Naming conventions and slugification for oracle/proxy model pairs.
- **[`data_prep/`](data_prep/)**: Data ingestion, LLM querying, and proxy scoring toolkit (see [data_prep/README.md](data_prep/README.md)).

---

## 1. Benchmarking Cascade Tuners (`compare_tuners.py`)

The `compare_tuners.py` script executes Monte Carlo trials across specified scenarios, evaluating cascade quality (Recall, Precision, F1) and oracle call efficiency against user-specified targets $(\gamma_R, \gamma_P)$ and failure tolerance $\delta$.

### Supported Scenarios

| Scenario Family | Identifier(s) | Description |
| :--- | :--- | :--- |
| **SUPG Datasets** | `onto`, `imagenet`, `jackson`, `tacred` | Image and text classification benchmarks with precomputed proxy scores and labels. |
| **ScaleDoc Datasets** | `pubmed`, `big_patent`, `gov_report` | Document question-answering cascades evaluated per-query (`--query <qid>`). |
| **BARGAIN Datasets** | `court`, `review`, `screenplay`, `wiki` | Long-form semantic filtering cascades evaluated per-query (`--query <qid>`). |

### Available Runners

- `actis_adaptive_is_asymptotic`: ACTIS with adaptive importance sampling, variance-minimizing proposal, and asymptotic confidence sequences with safeguards (default).
- `actis_adaptive_is_asymptotic_no_safeguards`: ACTIS with importance sampling and asymptotic CS, without asymptotic safeguards (variance ratio / binomial checks).
- `actis_adaptive_asymptotic`: ACTIS with uniform sampling without replacement and asymptotic CS.
- `actis_adaptive`: ACTIS with uniform sampling without replacement and finite-sample betting supermartingales.
- `bargain_pr`: BARGAIN with precision/recall trade-off optimization.
- `lotus`: Legacy LOTUS cascade tuner.
- `scaledoc`: ScaleDoc cascade tuner (harmonic target mapping).
- `scaledoc_native`: ScaleDoc targeting native F1 guarantees.

### Usage Examples

#### Run Default Evaluation on a BARGAIN Dataset
```bash
uv run python experiments/compare_tuners.py \
    --scenario court \
    --query 0 \
    --target-recall 0.90 \
    --target-precision 0.90 \
    --delta 0.05 \
    --num-trials 100
```

#### Run Multiple Queries on ScaleDoc
```bash
uv run python experiments/compare_tuners.py \
    --scenario pubmed \
    --query 0 1 2 \
    --target-recall 0.95 \
    --target-precision 0.95 \
    --delta 0.05 \
    --num-trials 100
```

#### Compare Specific Tuners
```bash
uv run python experiments/compare_tuners.py \
    --scenario onto \
    --runners actis_adaptive_is_asymptotic actis_adaptive_asymptotic bargain_pr lotus \
    --target-recall 0.90 \
    --target-precision 0.90 \
    --delta 0.05 \
    --num-trials 100
```

### Key CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--scenario` | *(required)* | Dataset/scenario name (`court`, `pubmed`, `onto`, etc.). |
| `--query`, `--queries` | `'0'` | Query ID(s) for document datasets (e.g. `--query 0 1 2` or `--query all`). |
| `--target-recall` | `0.90` | Target recall constraint $\gamma_R$. |
| `--target-precision` | `0.90` | Target precision constraint $\gamma_P$. |
| `--delta` | `0.05` | Maximum allowable failure probability $\delta$. |
| `--num-trials` | `100` | Number of independent Monte Carlo trials. |
| `--sample-size` | `'auto'` | Initial calibration sample size (count or `'auto'`). |
| `--batch-size` | `100` | Sample size for each adaptive expansion batch. |
| `--max-sample-size` | `0.5` | Maximum calibration budget (fraction in $(0, 1]$ or integer count). |
| `--alpha` | `0.4` | Defensive mixture weight for importance sampling proposal. |
| `--tail-tolerance` | `0.10` | Dimensionless tail uncertainty tolerance for proposal clipping. |
| `--v0` | `'auto'` | Prior variance for asymptotic Gaussian mixture CS. |
| `--results-dir` | `experiments/results` | Root directory for hierarchical results output. |
| `--skip-existing` | `True` | Skip runs whose target result JSON already exists. |
| `--force` | `False` | Overwrite existing result JSONs. |

### Hierarchical Results Structure

Trial results and execution summaries are persisted hierarchically:
```text
experiments/results/
└── <scenario>/
    └── <query_id>/
        └── <oracle_slug>__<proxy_slug>/
            └── trials_<num_trials>/
                └── delta_<delta>/
                    └── gamma_<gamma_R>_<gamma_P>/
                        └── <runner_name>.json
```
Each JSON file contains detailed per-trial metrics (recall, precision, F1, oracle calls, monetary costs, latency) and aggregate summary statistics.

---

## 2. Statistical Soundness Auditing (`analyze_soundness.py`)

The `analyze_soundness.py` script audits experimental results across benchmarks, evaluating whether cascade tuners satisfy their target failure probability guarantee $\delta$.

### Exact Clopper-Pearson Soundness Criteria

For $n$ independent trials with $k$ observed constraint violations (where Recall $< \gamma_R$ or Precision $< \gamma_P$), an exact two-sided Clopper-Pearson confidence interval $[p_{\text{LCB}}, p_{\text{UCB}}]$ is computed at significance level $\alpha$ (default 0.05):
- **Certified Sound**: $p_{\text{UCB}} < \delta$ (guarantee satisfied with 95% statistical confidence).
- **Inconclusive**: $p_{\text{LCB}} \le \delta \le p_{\text{UCB}}$ (empirical trials consistent with the nominal guarantee).
- **Certified Unsound**: $p_{\text{LCB}} > \delta$ (statistically significant violation of user guarantee).

### Usage Examples

#### Run Soundness Audit
```bash
uv run python experiments/analyze_soundness.py \
    --results-dir experiments/results \
    --alpha 0.05 \
    --format table
```

#### Break Down Results Across $(\delta, \gamma)$ Pairs
```bash
uv run python experiments/analyze_soundness.py \
    --results-dir experiments/results \
    --disaggregated \
    --format table
```

#### Export Audit Tables to CSV or LaTeX
```bash
uv run python experiments/analyze_soundness.py \
    --results-dir experiments/results \
    --format all \
    --output-dir experiments/audit_tables
```

### Auditing CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--results-dir` | `experiments/results` | Root directory containing experiment result JSON files. |
| `--alpha` | `0.05` | Significance level for Clopper-Pearson confidence intervals. |
| `--delta` | `None` | Filter analysis to specific $\delta$ values (e.g. `--delta 0.05`). |
| `--gamma` | `None` | Filter analysis to specific $\gamma$ values (e.g. `--gamma 0.90`). |
| `--target` | `'joint'` | Target guarantee: `'joint'`, `'recall'`, `'precision'`, or `'all'`. |
| `--format` | `'table'` | Output format: `'table'` (terminal), `'latex'`, `'csv'`, or `'all'`. |
| `--disaggregated` | `False` | Disaggregate summary across $(\delta, \gamma)$ parameter pairs. |
| `--appendix-tables`| `False` | Generate full disaggregated tables per benchmark/query configuration. |
| `--output-dir` | `experiments/tables` | Destination directory for exported tables. |

