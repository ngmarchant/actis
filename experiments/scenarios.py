import hashlib
import json
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiments.data_prep import DATASET_GROUPS
from experiments.model_paths import (
    oracle_output_path,
    proxy_output_path,
    slugify_model_name,
)


@dataclass
class Population:
    """
    Represents a population dataset of proxy scores, oracle labels, and cost metrics.
    """

    scores: NDArray[np.float64]
    labels: NDArray[np.bool_]
    oracle_costs: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    proxy_costs: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self):
        """Allows unpacking: scores, labels = population."""
        return iter((self.scores, self.labels))

    def slice(self, idx: NDArray[np.int64]) -> "Population":
        """Returns a subsampled Population using the given indices."""
        return Population(
            scores=self.scores[idx],
            labels=self.labels[idx],
            oracle_costs={k: v[idx] for k, v in self.oracle_costs.items()},
            proxy_costs={k: v[idx] for k, v in self.proxy_costs.items()},
        )

    def compute_trial_costs(
        self,
        oracle_queried_mask: NDArray[np.bool_],
    ) -> dict[str, dict[str, float]]:
        """
        Computes disaggregated trial costs organized hierarchically by model (oracle vs
        proxy).
        """
        pop_size = len(self.scores)
        num_oracle_calls = float(np.sum(oracle_queried_mask))
        oracle_dict: dict[str, float] = {
            "num_calls": num_oracle_calls,
            "call_rate": num_oracle_calls / pop_size if pop_size > 0 else 0.0,
        }
        for k, arr in self.oracle_costs.items():
            oracle_dict[k] = float(np.sum(arr[oracle_queried_mask]))

        proxy_dict: dict[str, float] = {
            "num_calls": float(pop_size),
            "call_rate": 1.0 if pop_size > 0 else 0.0,
        }
        for k, arr in self.proxy_costs.items():
            # Proxy cost is incurred for all items in the population
            proxy_dict[k] = float(np.sum(arr))

        return {"oracle": oracle_dict, "proxy": proxy_dict}


def compute_ideal_oracle_call_rate(
    proxy_scores: NDArray[np.float64],
    oracle_outputs: NDArray[np.bool_],
    gamma_R: float,
    gamma_P: float,
) -> float:
    """
    Computes the minimum oracle call rate satisfying precision and recall targets.

    Finds the optimal pair of thresholds (tau_upper, tau_lower) on the full
    population that achieves Recall >= gamma_R and Precision >=
    gamma_P while minimizing oracle queries.

    Args:
        proxy_scores: 1D array of proxy confidence scores.
        oracle_outputs: 1D boolean array of true labels.
        gamma_R: Minimum required population recall.
        gamma_P: Minimum required population precision.

    Returns:
        The theoretical minimum oracle call rate in [0.0, 1.0].
    """
    scores = np.asarray(proxy_scores, dtype=np.float64)
    oracle = np.asarray(oracle_outputs, dtype=bool)

    total_weight = len(scores)
    total_pos_weight = np.sum(oracle)

    if total_pos_weight == 0 or total_weight == 0:
        return 0.0

    # Sort population by score ascending
    sort_idx = np.argsort(scores)
    s_sorted = scores[sort_idx]
    o_sorted = oracle[sort_idx]

    pos_counts = np.where(o_sorted, 1.0, 0.0)
    neg_counts = np.where(~o_sorted, 1.0, 0.0)

    # Group by unique proxy scores
    unq, unq_inv = np.unique(s_sorted, return_inverse=True)
    unq_pos = np.bincount(unq_inv, weights=pos_counts)
    unq_neg = np.bincount(unq_inv, weights=neg_counts)
    unq_tot = np.bincount(unq_inv)

    M = len(unq)

    TP_suffix = np.zeros(M + 1, dtype=np.float64)
    TP_suffix[:M] = np.cumsum(unq_pos[::-1])[::-1]

    FP_suffix = np.zeros(M + 1, dtype=np.float64)
    FP_suffix[:M] = np.cumsum(unq_neg[::-1])[::-1]

    Tot_suffix = np.zeros(M + 1, dtype=np.float64)
    Tot_suffix[:M] = np.cumsum(unq_tot[::-1])[::-1]

    valid_k = np.where(TP_suffix / total_pos_weight >= gamma_R)[0]
    if len(valid_k) == 0:
        return 1.0

    tp = TP_suffix[valid_k]
    if gamma_P > 0:
        max_fp = tp * (1.0 - gamma_P) / gamma_P
    else:
        max_fp = np.full_like(tp, np.inf)

    # Check if helper model alone satisfies precision
    fp = FP_suffix[valid_k]
    if np.any(fp <= max_fp):
        return 0.0

    neg_FP_suffix = -FP_suffix
    idx = np.searchsorted(neg_FP_suffix, -max_fp, side="left")
    k_pos = np.maximum(valid_k + 1, idx)

    valid_pos = k_pos <= M
    if not np.any(valid_pos):
        return 1.0

    oracle_w = Tot_suffix[valid_k[valid_pos]] - Tot_suffix[k_pos[valid_pos]]
    min_oracle_rate = float(np.min(oracle_w / total_weight))
    return min(1.0, max(0.0, min_oracle_rate))


@dataclass(kw_only=True)
class BaseScenario(ABC):
    """Abstract base class for coverage scenarios."""

    name: str = ""
    description: str = ""
    pop_size: int | None = None
    seed: int | None = 42

    def to_dict(self) -> dict[str, Any]:
        """Serializes scenario configuration to a JSON-compatible dictionary."""
        res = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, Path):
                res[f.name] = str(val)
            elif isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                res[f.name] = val
            else:
                res[f.name] = str(val)
        return res

    def config_hash(self, length: int = 8) -> str:
        """Deterministic hash of scenario configuration."""
        d = self.to_dict()
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]

    def results_subpath(self) -> Path:
        """Returns the relative subdirectory path for storing results on disk."""
        return Path(self.name) / f"{self.name}_{self.config_hash()}"

    @abstractmethod
    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population dataset of proxy scores, oracle labels, and cost metrics.

        Returns:
            A Population object containing scores, labels, and cost metrics.
        """
        pass


@dataclass(kw_only=True)
class Benign(BaseScenario):
    """
    Benign Data (Calibrated Polarized Neural Proxy).

    Simulates a high-quality, calibrated classifier where scores are strongly polarized.
    Positives concentrate near 1.0, and negatives concentrate near 0.0 (AUC > 0.95).

    This scenario measures the minimal calibration overhead and fast early stopping
    of algorithms on well-behaved workloads.
    """

    name: str = "benign"
    description: str = "Calibrated data with bimodal Beta proxy scores (AUC > 0.95)."
    pop_size: int | None = 100000

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a clean population using calibrated bimodal Beta distributions.
        """
        rng = np.random.default_rng(self.seed)
        pop_size = self.pop_size if self.pop_size is not None else 100000

        n_pos = int(0.15 * pop_size)
        n_neg = pop_size - n_pos

        scores = np.empty(pop_size, dtype=np.float64)
        oracle = np.zeros(pop_size, dtype=bool)
        oracle[:n_pos] = True

        # Positives centered near 0.85; Negatives centered near 0.15
        scores[:n_pos] = rng.beta(4.0, 0.8, size=n_pos)
        scores[n_pos:] = rng.beta(0.8, 4.0, size=n_neg)

        return Population(scores=scores, labels=oracle)


def _read_dataframe(path: Path) -> pd.DataFrame:
    """
    Reads a tabular dataset from disk based on file extension.

    Supports CSV, TSV, Feather, Arrow/IPC stream, and Parquet formats.

    Args:
        path: Path to the tabular dataset file.

    Returns:
        A pandas DataFrame with the loaded dataset.
    """
    if path.is_dir():
        from datasets import Dataset, load_from_disk

        ds = load_from_disk(str(path))
        if not isinstance(ds, Dataset):
            raise ValueError(f"Loaded dataset is not a Dataset: {type(ds)}")
        return ds.to_pandas()  # ty: ignore[invalid-return-type]

    suffix = path.suffix.lower()
    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    elif suffix in [".feather", ".arrow", ".ipc"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                return pd.read_feather(path)
            except Exception:
                try:
                    import pyarrow.ipc as ipc

                    with ipc.open_stream(str(path)) as reader:
                        return reader.read_all().to_pandas()
                except Exception:
                    import pyarrow.feather as feather

                    return feather.read_table(str(path)).to_pandas()
    elif suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        # Fallback: try feather/arrow first, then csv
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return pd.read_feather(path)
        except Exception:
            return pd.read_csv(path)


@dataclass(kw_only=True)
class TabularDataset(BaseScenario):
    """
    Scenario for file-backed benchmark datasets (CSV, Feather, Arrow, Parquet).

    Loads proxy scores and ground-truth oracle labels directly from a file on disk.
    Optionally, `oracle_path` and/or `proxy_path` may point to separate files
    (e.g. model-scoped oracle/proxy cache files) that are joined positionally
    onto `data_path`, so that document content need not be duplicated across
    oracle/proxy model combinations.
    """

    data_path: str | Path = ""
    oracle_path: str | Path = ""
    proxy_path: str | Path = ""
    score_col: str = "proxy_score"
    label_col: str = "label"
    oracle_cost_col: str = "oracle_cost"
    proxy_cost_col: str = "proxy_cost"

    def __post_init__(self):
        if self.data_path != "":
            self.data_path = Path(self.data_path)
        if self.oracle_path != "":
            self.oracle_path = Path(self.oracle_path)
        if self.proxy_path != "":
            self.proxy_path = Path(self.proxy_path)
        object.__setattr__(self, "_df", None)
        object.__setattr__(self, "_data", None)

    @staticmethod
    def _read_required(path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at '{path}'.")
        return _read_dataframe(path)

    @staticmethod
    def _align_and_concat(
        df: pd.DataFrame, other: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        if len(other) != len(df):
            raise ValueError(
                f"Row count mismatch: '{source_name}' has {len(other)} rows, "
                f"but the base dataset has {len(df)} rows. Files must be "
                f"row-aligned with the base document dataset."
            )
        overlap = set(df.columns) & set(other.columns)
        if overlap:
            raise ValueError(
                f"Column name collision between base dataset and '{source_name}': "
                f"{sorted(overlap)}"
            )
        return pd.concat(
            [df.reset_index(drop=True), other.reset_index(drop=True)], axis=1
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Loads and returns the cached underlying DataFrame."""
        if getattr(self, "_df", None) is None:
            df = self._read_required(Path(self.data_path), "Dataset")
            if self.oracle_path != "":
                oracle_df = self._read_required(Path(self.oracle_path), "Oracle data")
                df = self._align_and_concat(df, oracle_df, str(self.oracle_path))
            if self.proxy_path != "":
                proxy_df = self._read_required(Path(self.proxy_path), "Proxy data")
                df = self._align_and_concat(df, proxy_df, str(self.proxy_path))
            object.__setattr__(self, "_df", df)
        return getattr(self, "_df")

    def _extract_costs(
        self, df: pd.DataFrame, col_name: str
    ) -> dict[str, NDArray[np.float64]]:
        """Extracts 1D float arrays for each cost metric in a dictionary or scalar
        column."""
        if col_name not in df.columns or len(df) == 0:
            return {}

        series = df[col_name]
        sample = None
        for item in series:
            if item is not None and not (isinstance(item, float) and np.isnan(item)):
                sample = item
                break

        if sample is None:
            return {}

        if isinstance(sample, dict):
            cost_df = pd.DataFrame(series.tolist())
            extracted: dict[str, NDArray[np.float64]] = {}
            for col in cost_df.columns:
                num_series = pd.to_numeric(cost_df[col], errors="coerce")
                # Skip columns that contain non-convertible string data
                # (e.g. error messages)
                if cost_df[col].dropna().apply(lambda x: not isinstance(x, str)).all():
                    extracted[str(col)] = num_series.fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
            return extracted
        elif np.issubdtype(series.dtype, np.number):
            return {col_name: series.fillna(0.0).to_numpy(dtype=np.float64)}
        else:
            try:
                arr = (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                return {col_name: arr}
            except Exception:
                return {}

    def get_costs(
        self,
        cost_key: str = "monetary",
        default_oracle_cost: float = 1.0,
        default_proxy_cost: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extracts 1D arrays of (proxy_costs, oracle_costs) from the dataset.

        Args:
            cost_key: Metric key inside the cost dictionaries (e.g. 'monetary',
                'input_tokens', 'latency_ms').
            default_oracle_cost: Fallback cost if column or cost_key is missing.
            default_proxy_cost: Fallback cost if column or cost_key is missing.

        Returns:
            A tuple (proxy_costs, oracle_costs) of float64 1D arrays.
        """
        pop = self._load_data()
        n = len(pop)
        proxy_arr = pop.proxy_costs.get(
            cost_key, np.full(n, default_proxy_cost, dtype=np.float64)
        )
        oracle_arr = pop.oracle_costs.get(
            cost_key, np.full(n, default_oracle_cost, dtype=np.float64)
        )
        return proxy_arr, oracle_arr

    def _load_data(self) -> Population:
        """
        Loads and caches proxy scores, oracle labels, and cost metrics from the dataset.

        Rows where the proxy score or oracle label is missing (e.g. because the
        document triggered Azure's safety filter) are dropped before any subsampling
        is applied. A warning is emitted listing the number of dropped rows.

        Returns:
            A Population object.
        """
        if getattr(self, "_data", None) is None:
            df = self.get_dataframe()
            if self.score_col not in df.columns:
                raise ValueError(
                    f"Score column '{self.score_col}' not found in dataset columns: "
                    f"{df.columns.tolist()}"
                )
            if self.label_col not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_col}' not found in dataset columns: "
                    f"{df.columns.tolist()}"
                )

            missing_mask = df[self.score_col].isna() | df[self.label_col].isna()
            n_missing = int(missing_mask.sum())
            if n_missing > 0:
                warnings.warn(
                    f"Dropping {n_missing} row(s) from '{self.name}' with missing "
                    f"proxy score or oracle label (e.g. due to safety-filter blocks). "
                    f"{len(df) - n_missing} rows remain.",
                    stacklevel=2,
                )
                df = df[~missing_mask].reset_index(drop=True)

            scores = df[self.score_col].to_numpy(dtype=np.float64)
            raw_label = df[self.label_col]
            if raw_label.dtype == bool:
                oracle = raw_label.to_numpy(dtype=bool)
            elif np.issubdtype(raw_label.dtype, np.number):
                oracle = raw_label.to_numpy(dtype=np.float64) == 1.0
            else:
                oracle = (
                    raw_label.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["1", "1.0", "true", "t", "yes", "y"])
                    .to_numpy(dtype=bool)
                )

            oracle_costs = self._extract_costs(df, self.oracle_cost_col)
            proxy_costs = self._extract_costs(df, self.proxy_cost_col)

            object.__setattr__(
                self,
                "_data",
                Population(
                    scores=scores,
                    labels=oracle,
                    oracle_costs=oracle_costs,
                    proxy_costs=proxy_costs,
                ),
            )
        return getattr(self, "_data")

    def generate_population(self, **kwargs) -> Population:
        """
        Generates a population by subsampling or loading from a disk-backed dataset.

        Returns:
            A Population object.
        """
        pop = self._load_data()
        if self.pop_size is None or self.pop_size >= len(pop):
            return pop
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(pop), size=self.pop_size, replace=False)
        return pop.slice(idx)


FileDataset = TabularDataset


SUPG_DATASETS: dict[str, dict[str, Any]] = {
    "onto": {
        "description": "SUPG Onto benchmark dataset",
        "data_path": "experiments/data/supg/onto/source.csv",
    },
    "imagenet": {
        "description": "SUPG ImageNet benchmark dataset",
        "data_path": "experiments/data/supg/imagenet/source.csv",
    },
    "jackson": {
        "description": "SUPG Jackson benchmark dataset",
        "data_path": "experiments/data/supg/jackson/2017-12-17.feather",
    },
    "tacred": {
        "description": "SUPG TACRED benchmark dataset",
        "data_path": "experiments/data/supg/tacred/source.csv",
    },
}


@dataclass(kw_only=True)
class DocumentBenchmarkDataset(TabularDataset):
    """
    Scenario for query-based document benchmark datasets (ScaleDoc, BARGAIN).

    Loads document content from `documents.parquet`, and joins ground-truth
    oracle labels and proxy scores from model-scoped cache files (default:
    experiments/data/{dataset_group}/{dataset}/documents.parquet,
    .../oracle/q{query_id}/{oracle_model_slug}.parquet, and
    .../proxy/q{query_id}/{proxy_model_slug}.parquet).
    """

    query_id: str = "0"
    data_dir: str | Path | None = None
    oracle_model: str | None = "azure/gpt-5.6-terra"
    proxy_model: str | None = "azure/gpt-5.6-luna"

    @property
    def dataset_group(self) -> str:
        clean_name = self.name.lower().replace("-", "_")
        return DATASET_GROUPS.get(clean_name, "")

    def __post_init__(self):
        clean_name = self.name.lower().replace("-", "_")
        if clean_name not in DATASET_GROUPS:
            raise ValueError(
                f"Unknown benchmark dataset name '{self.name}'. "
                f"Must be one of: {sorted(DATASET_GROUPS.keys())}"
            )

        self.name = clean_name
        group = DATASET_GROUPS[clean_name]
        clean_qid = str(self.query_id).lower()

        if not self.description:
            group_display = "BARGAIN" if group == "bargain" else "ScaleDoc"
            self.description = (
                f"{group_display} {clean_name} (query {clean_qid}, "
                f"oracle: {self.oracle_model}, proxy: {self.proxy_model})"
            )

        # Resolve directory paths
        if self.data_dir is None:
            resolved_data_dir = Path("experiments/data") / group
        else:
            resolved_data_dir = Path(self.data_dir)

        dataset_dir = resolved_data_dir / clean_name

        if not self.data_path:
            self.data_path = dataset_dir / "documents.parquet"
        if not self.oracle_path and self.oracle_model:
            self.oracle_path = oracle_output_path(
                dataset_dir, clean_qid, self.oracle_model
            )
        if not self.proxy_path and self.proxy_model:
            self.proxy_path = proxy_output_path(
                dataset_dir, clean_qid, self.proxy_model
            )
        super().__post_init__()

    def results_subpath(self) -> Path:
        """Returns the relative subdirectory path for storing results on disk."""
        oracle_slug = (
            slugify_model_name(self.oracle_model) if self.oracle_model else "none"
        )
        proxy_slug = (
            slugify_model_name(self.proxy_model) if self.proxy_model else "none"
        )
        model_pair = f"{oracle_slug}__{proxy_slug}_{self.config_hash()}"
        return Path(self.name) / f"q{self.query_id}" / model_pair


def get_scenario(
    name: str,
    *,
    query_id: str | int | None = None,
    oracle_model: str | None = None,
    proxy_model: str | None = None,
    **kwargs: Any,
) -> BaseScenario:
    """
    Look up or instantiate a scenario by name/dataset, query, and models.

    Args:
        name: Scenario name or benchmark dataset name.
              - SUPG: 'onto', 'imagenet', 'jackson', 'tacred'
                (or with optional 'supg_' prefix, e.g. 'supg_onto').
              - Document benchmarks: 'court', 'review', 'screenplay', 'wiki' (BARGAIN)
                or 'pubmed', 'big_patent', 'gov_report' (ScaleDoc).
                Prefixes like 'bargain_' or 'scaledoc_' are also accepted.
        query_id: Query identifier for document benchmarks (e.g. '0', '1', '12').
                  Defaults to '0'. Ignored for SUPG scenarios.
        oracle_model: Model identifier for oracle ground truth
                      (default: 'azure/gpt-5.6-terra'). Ignored for SUPG scenarios.
        proxy_model: Model identifier for proxy scores (default: 'azure/gpt-5.6-luna').
                     Ignored for SUPG scenarios.
        **kwargs: Additional keyword arguments forwarded to the scenario dataclass
                  constructor (e.g. pop_size, seed, data_path, data_dir).

    Returns:
        An instantiated BaseScenario object.

    Raises:
        ValueError: If name is not recognized.
    """
    clean_name = name.lower().strip()

    # Extract dataset base name using regex (stripping optional source prefix)
    m = re.match(r"^(?:(?:supg|scaledoc|bargain)_)?([a-z0-9_]+)$", clean_name)
    if m:
        base_name = m.group(1)

        # SUPG real benchmark datasets
        if base_name in SUPG_DATASETS:
            spec = SUPG_DATASETS[base_name]
            params: dict[str, Any] = {
                "name": base_name,
                "description": spec["description"],
                "data_path": spec["data_path"],
            }
            params.update(kwargs)
            return FileDataset(**params)

        # Document benchmark datasets (ScaleDoc & BARGAIN)
        if base_name in DATASET_GROUPS:
            resolved_qid = str(query_id) if query_id is not None else "0"
            resolved_oracle = (
                oracle_model if oracle_model is not None else "azure/gpt-5.6-terra"
            )
            resolved_proxy = (
                proxy_model if proxy_model is not None else "azure/gpt-5.6-luna"
            )
            return DocumentBenchmarkDataset(
                name=base_name,
                query_id=resolved_qid,
                oracle_model=resolved_oracle,
                proxy_model=resolved_proxy,
                **kwargs,
            )

    valid_options = list_scenarios()
    raise ValueError(f"Unknown scenario '{name}'. Valid options: {valid_options}")


def list_scenarios() -> list[str]:
    """Returns a sorted list of all canonical scenario identifiers."""
    supg_names = list(SUPG_DATASETS.keys())
    benchmark_names = sorted(DATASET_GROUPS.keys())
    return sorted(supg_names + benchmark_names)


def get_available_queries(
    name: str,
    data_dir: str | Path | None = None,
) -> list[str]:
    """Returns available query identifiers for a benchmark dataset.

    Checks on-disk oracle directories first; if none are found, inspects
    experiments/data/{group}/query.json. Returns an empty list for SUPG
    scenarios.
    """
    clean_name = name.lower().strip()
    match = re.match(r"^(?:(?:supg|scaledoc|bargain)_)?([a-z0-9_]+)$", clean_name)
    base_name = match.group(1) if match else clean_name

    if base_name not in DATASET_GROUPS:
        return []

    group = DATASET_GROUPS[base_name]
    if data_dir is not None:
        base_dir = Path(data_dir) / base_name
        group_dir = Path(data_dir)
    else:
        base_dir = Path("experiments/data") / group / base_name
        group_dir = Path("experiments/data") / group

    found: set[str] = set()
    oracle_dir = base_dir / "oracle"
    if oracle_dir.is_dir():
        for d in oracle_dir.iterdir():
            if d.is_dir() and d.name.startswith("q"):
                found.add(d.name[1:])

    if not found:
        query_json_path = group_dir / "query.json"
        if query_json_path.is_file():
            try:
                with open(query_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if base_name in data:
                    for item in data[base_name]:
                        found.add(str(item["q_id"]))
            except Exception:
                pass

    if not found:
        return ["0"]

    def _sort_key(s: str):
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p for p in parts]

    return sorted(found, key=_sort_key)


__all__ = [
    "Population",
    "compute_ideal_oracle_call_rate",
    "BaseScenario",
    "TabularDataset",
    "FileDataset",
    "DocumentBenchmarkDataset",
    "Benign",
    "get_scenario",
    "list_scenarios",
    "get_available_queries",
]
