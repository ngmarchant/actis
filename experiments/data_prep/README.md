# Dataset Preparation & LLM Labeling

This package provides dataset ingestion, LiteLLM-backed proxy/oracle model wrappers, cost estimation, and checkpointed labeling utilities exposed via `python -m experiments.data_prep`.


## Module Overview

- `cli.py`: CLI argument parsing, execution planning, dry-run cost estimation, and dataset preparation workflow (invoked via `python -m experiments.data_prep`).
- `loaders.py`: Dataset downloaders and parsers for SUPG (`onto`, `imagenet`, `jackson`, `tacred`), ScaleDoc (`pubmed`, `big_patent`, `gov_report`), and BARGAIN (`court`, `review`, `screenplay`, `wiki`) benchmarks, plus query JSON loading.
- `models.py`: LiteLLM oracle and proxy interfaces (`LiteLLMOracle`, `LiteLLMProxy`), callable wrappers (`CallableOracle`, `CallableProxy`), token/cost estimation (`CostEstimate`), and benchmark prompt defaults.
- `labeling.py`: Checkpointed batch inference (`add_oracle_labels`, `add_proxy_scores`, `get_unprocessed_items`) and Hugging Face `Dataset` serialization (`save_dataset`).


## Pre-labeled SUPG Datasets

The SUPG benchmarks (`onto`, `imagenet`, `jackson`, `tacred`) from [stanford-futuredata/supg](https://github.com/stanford-futuredata/supg/tree/master/data) already provide proxy scores and oracle labels (and omit raw inputs), so they do not require LLM labeling.

`loaders.py` provides `download_supg_dataset`, `load_supg_dataset`, and per-dataset helpers (`load_onto_dataset`, `load_imagenet_dataset`, `load_jackson_dataset`, `load_tacred_dataset`) that automatically fetch and cache the files under `experiments/data/supg/` on first use. When running `python -m experiments.compare_runners` or `python -m experiments.analysis.dataset_stats`, missing SUPG files are downloaded automatically; you can also pre-fetch them explicitly with:

```bash
python -m experiments.data_prep --dataset onto
```


## Configuration

1. **Environment variables**: Copy `experiments/.env.example` to `experiments/.env` and fill in your API credentials and optional cache paths (`HF_HOME`, `KAGGLEHUB_CACHE`, `AZURE_API_KEY`, `AZURE_API_BASE`, `OPENAI_API_KEY`):
   ```bash
   cp experiments/.env.example experiments/.env
   ```
2. **LiteLLM model & pricing config** (optional): Copy `experiments/config.yaml.example` to `experiments/config.yaml` and customize deployments, `top_logprobs`, and token pricing:
   ```bash
   cp experiments/config.yaml.example experiments/config.yaml
   ```


## Supported Datasets

| Group | Dataset (`--dataset`) | Source / Notes | Query File |
|---|---|---|---|
| `supg` | `onto` | Auto-downloaded pre-labeled CSV (`stanford-futuredata/supg`) | — |
| `supg` | `imagenet` | Auto-downloaded pre-labeled CSV (`stanford-futuredata/supg`) | — |
| `supg` | `jackson` | Auto-downloaded pre-labeled Feather (`stanford-futuredata/supg`) | — |
| `supg` | `tacred` | Auto-downloaded pre-labeled CSV (`stanford-futuredata/supg`) | — |
| `scaledoc` | `pubmed` | PubMed 200k RCT abstracts | `experiments/data/scaledoc/query.json` |
| `scaledoc` | `big_patent` | `NortheasternUniversity/big_patent` | `experiments/data/scaledoc/query.json` |
| `scaledoc` | `gov_report` | `ccdv/govreport-summarization` | `experiments/data/scaledoc/query.json` |
| `bargain` | `court` | Requires local file via `--source-path` | `experiments/data/bargain/query.json` |
| `bargain` | `review` | Steam reviews corpus via KaggleHub | `experiments/data/bargain/query.json` |
| `bargain` | `screenplay` | Movie scripts corpus via KaggleHub | `experiments/data/bargain/query.json` |
| `bargain` | `wiki` | Wikipedia talk-page corpus via ConvoKit | `experiments/data/bargain/query.json` |


## Running the CLI (`python -m experiments.data_prep`)

Run a dry run first to download/prepare base documents and estimate token counts and API cost without making LLM calls:

```bash
python -m experiments.data_prep \
    --dataset pubmed \
    --queries 0 \
    --config experiments/config.yaml \
    --dry-run
```

Label oracle ground truth and compute proxy scores (resuming from checkpoints if interrupted):

```bash
python -m experiments.data_prep \
    --dataset pubmed \
    --queries base \
    --oracle-model azure/gpt-5.6-terra \
    --proxy-model azure/gpt-5.6-luna \
    --config experiments/config.yaml \
    --batch-size 50 \
    --concurrency 10
```

Common options:
- `--queries`: One or more query IDs (e.g. `0 1 0_ext`) or presets `all`, `base`, `ext`.
- `--skip-oracle` / `--skip-proxy`: Run only proxy scoring or only oracle labeling.
- `--source-path`: Local source file required for `--dataset court`.
- `--format`: Output format (`parquet` by default, or `hf_disk`).
- `--force`: Recompute existing labels or proxy scores instead of reusing cached outputs.
- `-y` / `--yes`: Skip interactive confirmation prompts.


## Output & Checkpointing Layout

Outputs are stored under `experiments/data/<group>/<dataset>/` (overridable with `--output-dir`):
- `documents.parquet`: Cached base documents sampled with `--num-docs` and `--seed`.
- Oracle and proxy predictions/checkpoints are placed at model-slugged paths generated by `experiments/model_paths.py` (`oracle_output_path`, `proxy_output_path`, `oracle_checkpoint_path`, `proxy_checkpoint_path`), allowing interrupted labeling runs to resume cleanly.
