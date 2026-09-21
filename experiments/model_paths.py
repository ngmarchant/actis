"""Path helpers for model-scoped oracle/proxy cache files.

Oracle and proxy outputs for a given (dataset, query) pair are cached
independently per model, so that different oracle/proxy model combinations can
coexist and be reused without recomputation. Files store only the columns
produced by that stage (e.g. `label`/`oracle_score`/`oracle_cost`), aligned
positionally with the corresponding `documents.parquet` base dataset.

This module is intentionally dependency-free (no `datasets`/`litellm`/etc.) so
that it can be imported cheaply from both `experiments.data_prep` (for data
preparation) and `experiments.scenarios` (for scenario loading) without
pulling in heavy, unrelated dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_model_name(model: str) -> str:
    """Sanitizes a model identifier for use as a filesystem path component."""
    return _SLUG_RE.sub("-", model).strip("-")


def oracle_output_path(output_dir: Path, qid: str, oracle_model: str) -> Path:
    """Returns the cache path for a query's oracle output for a given model."""
    slug = slugify_model_name(oracle_model)
    return output_dir / "oracle" / f"q{qid}" / f"{slug}.parquet"


def proxy_output_path(output_dir: Path, qid: str, proxy_model: str) -> Path:
    """Returns the cache path for a query's proxy output for a given model."""
    slug = slugify_model_name(proxy_model)
    return output_dir / "proxy" / f"q{qid}" / f"{slug}.parquet"


def oracle_checkpoint_path(output_dir: Path, qid: str, oracle_model: str) -> Path:
    """Returns the incremental checkpoint path for oracle labeling."""
    slug = slugify_model_name(oracle_model)
    return output_dir / "checkpoints" / "oracle" / f"q{qid}" / f"{slug}.json"


def proxy_checkpoint_path(output_dir: Path, qid: str, proxy_model: str) -> Path:
    """Returns the incremental checkpoint path for proxy scoring."""
    slug = slugify_model_name(proxy_model)
    return output_dir / "checkpoints" / "proxy" / f"q{qid}" / f"{slug}.json"
