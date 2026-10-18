# Benchmark Data Preparation and Labeling

This directory provides the ingestion, LLM querying, and proxy scoring pipeline used to prepare benchmark datasets for semantic filtering cascades.

---

## Overview

For document-level query benchmarks (ScaleDoc and BARGAIN), each dataset consists of unstructured documents and a set of queries. To evaluate cascade tuning:
1. **Oracle Labels** are gathered by prompting an expensive, high-accuracy foundation model (e.g., GPT-5.6-terra).
2. **Proxy Confidence Scores** in $[0, 1]$ are generated using a lightweight, cheap model (e.g., GPT-5.6-luna, Llama-3.2-1B) by extracting next-token logprobs for `"Yes"` / `"No"` (or `"True"` / `"False"`).

---

## File Structure

- **[`loaders.py`](loaders.py)**: Raw dataset downloaders and text extractors:
  - **ScaleDoc**: Ingests PubMed RCT, BigPatent, and GovReport documents from HuggingFace and query definitions from GitHub.
  - **BARGAIN**: Ingests Court (SCOTUS via ConvoKit), Movie Review (IMDb via HuggingFace), Screenplay (Kaggle), and Wiki (Wikipedia via HuggingFace) documents and queries.
- **[`models.py`](models.py)**: Model abstraction layers using [LiteLLM](https://github.com/BerriAI/litellm):
  - `LiteLLMOracle`: Prompts models for categorical classification labels with response parsing.
  - `LiteLLMProxy`: Queries next-token logprobs to extract calibrated $[0, 1]$ positive-class probabilities.
- **[`labeling.py`](labeling.py)**: Batching, async execution, checkpointing, and dataset persistence routines.
- **[`prepare_data.py`](../prepare_data.py)**: Unified CLI tool for running the end-to-end data preparation workflow.

---

## Setup and Configuration

### 1. Environment Variables (`.env`)

Copy the template from the repository root:
```bash
cp .env.example .env
```
Populate `.env` with your deployment credentials and desired cache paths:
```bash
# Azure OpenAI endpoints
AZURE_API_KEY=your_api_key_here
AZURE_API_BASE=https://your-resource-name.openai.azure.com/

# Optional custom cache locations for HuggingFace and Kaggle datasets
HF_HOME=/path/to/huggingface/cache
KAGGLEHUB_CACHE=/path/to/kagglehub/cache
```

### 2. LiteLLM Routing and Pricing (`config.yaml`)

To configure custom model deployments, endpoints, or regional pricing:
```bash
cp config.yaml.example config.yaml
```
The configuration defines token prices, context limits, and model parameters for LiteLLM router calls.

---

## Usage: `prepare_data.py`

The CLI tool `experiments/prepare_data.py` manages data downloading, labeling, and scoring.

### 1. Cost and Token Estimation (`--estimate-only`)

Before sending API requests, estimate total token consumption and costs:
```bash
uv run python experiments/prepare_data.py \
    --group bargain \
    --dataset court \
    --oracle-model azure/gpt-5.6-terra \
    --proxy-model azure/gpt-5.6-luna \
    --estimate-only
```

### 2. Gathering Oracle Labels Only
```bash
uv run python experiments/prepare_data.py \
    --group scaledoc \
    --dataset pubmed \
    --oracle-model azure/gpt-5.6-terra \
    --skip-proxy
```

### 3. Computing Proxy Confidence Scores Only
```bash
uv run python experiments/prepare_data.py \
    --group scaledoc \
    --dataset pubmed \
    --proxy-model azure/gpt-5.6-luna \
    --skip-oracle
```

### 4. Full Pipeline (Oracle Labels + Proxy Scores)
```bash
uv run python experiments/prepare_data.py \
    --group bargain \
    --dataset review \
    --oracle-model azure/gpt-5.6-terra \
    --proxy-model azure/gpt-5.6-luna \
    --batch-size 32 \
    --concurrency 10
```

### Key CLI Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--group` | *(required)* | Dataset family: `'scaledoc'` or `'bargain'`. |
| `--dataset` | `None` | Specific dataset name (e.g. `'court'`, `'pubmed'`). If omitted, processes all in the group. |
| `--oracle-model` | `'azure/gpt-5.6-terra'` | LiteLLM model identifier for oracle labels. |
| `--proxy-model` | `'azure/gpt-5.6-luna'` | LiteLLM model identifier for proxy scores. |
| `--estimate-only`| `False` | Estimate token consumption and monetary costs without making API calls. |
| `--skip-oracle` | `False` | Skip gathering oracle labels. |
| `--skip-proxy` | `False` | Skip generating proxy scores. |
| `--batch-size` | `10` | Batch size for concurrent API requests. |
| `--concurrency` | `5` | Maximum number of simultaneous API worker threads. |
| `--max-items` | `None` | Process at most $N$ documents (useful for quick validation). |

---

## Dataset Storage Format

Prepared datasets are serialized to disk using HuggingFace Arrow format:
```text
experiments/data/
├── scaledoc/
│   ├── pubmed/
│   │   ├── oracle/
│   │   │   └── <oracle_slug>/
│   │   │       └── query_<qid>/    # Oracle labels dataset
│   │   └── proxy/
│   │       └── <proxy_slug>/
│   │           └── query_<qid>/    # Proxy scores dataset
│   └── query.json
└── bargain/
    ├── court/
    │   ├── oracle/...
    │   └── proxy/...
    └── query.json
```
Each arrow record contains:
- `doc_id`: Unique document identifier
- `text`: Document content text
- `oracle_label`: Boolean ground-truth label (in oracle dataset)
- `proxy_score`: Float confidence score in $[0, 1]$ (in proxy dataset)
- `costs`: Metadata tracking prompt tokens, completion tokens, and latency

