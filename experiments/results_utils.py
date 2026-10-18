"""
Common utilities for discovering, loading, and formatting hierarchical
experiment results across benchmarks.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

from experiments.model_paths import slugify_model_name

# Scenario display and short names for publication artifacts
SCENARIO_DISPLAY_NAMES = {
    # Real benchmarks - SUPG
    "onto": "Onto",
    "imagenet": "ImageNet",
    "jackson": "Jackson",
    "tacred": "TACRED",
    # Real benchmarks - ScaleDoc
    "pubmed": "PubMed",
    "big_patent": "BigPatent",
    "gov_report": "GovReport",
    # Real benchmarks - BARGAIN
    "court": "Court",
    "review": "Review",
    "screenplay": "Screenplay",
    "wiki": "Wiki",
}

SCENARIO_SHORT_NAMES = {
    # Real benchmarks - SUPG
    "onto": "Onto",
    "imagenet": "ImageNet",
    "jackson": "Jackson",
    "tacred": "TACRED",
    # Real benchmarks - ScaleDoc
    "pubmed": "PubMed",
    "big_patent": "BigPatent",
    "gov_report": "GovReport",
    # Real benchmarks - BARGAIN
    "court": "Court",
    "review": "Review",
    "screenplay": "Screenplay",
    "wiki": "Wiki",
}

CANONICAL_SCENARIO_ORDER = [
    # Real benchmarks - SUPG
    "onto",
    "imagenet",
    "jackson",
    "tacred",
    # Real benchmarks - BARGAIN
    "court",
    "review",
    "screenplay",
    "wiki",
    # Real benchmarks - ScaleDoc
    "pubmed",
    "big_patent",
    "gov_report",
]

# Method display names
METHOD_DISPLAY_NAMES = {
    "actis_adaptive": "ACTIS-U",
    "actis_adaptive_asymptotic": "ACTIS-U (Asy)",
    "actis_adaptive_is": "ACTIS (IS Finite)",
    "actis_adaptive_is_asymptotic": "ACTIS",
    "actis_adaptive_is_asymptotic_no_safeguards": "ACTIS (w/o safeguards)",
    "scaledoc": "ScaleDoc",
    "scaledoc_native": "ScaleDoc (Native)",
    "bargain_pr": "BARGAIN",
    "lotus_original": "LOTUS",
}

METHOD_ORDER = [
    "ACTIS",
    "ACTIS (w/o safeguards)",
    "ACTIS-U",
    "ACTIS-U (Asy)",
    "ACTIS (IS Finite)",
    "BARGAIN",
    "ScaleDoc",
    "ScaleDoc (Native)",
    "LOTUS",
]

# Colorblind-safe palette (Paul Tol / Okabe-Ito inspired, high contrast)
METHOD_COLORS = {
    "ACTIS": "#009988",  # Teal
    "ACTIS (w/o safeguards)": "#882255",  # Wine / Deep Berry
    "ACTIS-U": "#117733",  # Dark Green
    "ACTIS-U (Asy)": "#33BBEE",  # Cyan / Sky Blue
    "ACTIS (IS Finite)": "#0077BB",  # Deep Blue
    "ScaleDoc": "#CC3311",  # Rust / Vermilion
    "ScaleDoc (Native)": "#CC3311",
    "BARGAIN": "#EE3377",  # Magenta
    "LOTUS": "#EE7733",  # Orange
}

METHOD_SHAPES = {
    "ACTIS": "^",
    "ACTIS-U": "<",
    "ACTIS-U (Asy)": "v",
    "ACTIS (IS Finite)": "o",
    "ScaleDoc": "P",
    "BARGAIN": "s",
    "LOTUS": "D",
}

DEFAULT_FOCUS_METHODS = [
    ("actis_adaptive_is_asymptotic", "ACTIS"),
    ("actis_adaptive", "ACTIS-U"),
    ("bargain_pr", "BARGAIN"),
    ("scaledoc", "ScaleDoc"),
    ("lotus_original", "LOTUS"),
]

DEFAULT_APPENDIX_METHODS = [
    ("actis_adaptive_is_asymptotic", "ACTIS"),
    ("actis_adaptive", "ACTIS-U"),
    ("actis_adaptive_asymptotic", "ACTIS-U (Asy)"),
    ("bargain_pr", "BARGAIN"),
    ("scaledoc", "ScaleDoc"),
    ("lotus_original", "LOTUS"),
]


# Proxy model display names
PROXY_DISPLAY_NAMES = {
    "azure/gpt-5.6-luna": "GPT-5.6-luna",
    "openai/llama3.2:1b-instruct-fp16": "Llama-3.2-1B",
}


def format_proxy_display_name(proxy_model: str | None) -> str | None:
    """Formats proxy model identifier into publication-ready short name."""
    if not proxy_model:
        return None
    if proxy_model in PROXY_DISPLAY_NAMES:
        return PROXY_DISPLAY_NAMES[proxy_model]
    p_lower = proxy_model.lower()
    if "gpt-5.6-luna" in p_lower:
        return "GPT-5.6-luna"
    if "llama-3.2-1b" in p_lower:
        return "Llama-3.2-1B"
    return slugify_model_name(proxy_model)


def compute_config_hash(config_dict: dict[str, Any], length: int = 8) -> str:
    """Computes a deterministic hash for a configuration dictionary."""
    s = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def format_scenario_display_name(
    sc_name_or_uid: str,
    query_id: str | None = None,
    proxy_model: str | None = None,
    short: bool = False,
) -> str:
    """Formats scenario identifier into publication-ready text.

    Handles canonical scenario names, optional query IDs, and proxy models.
    """
    parts = sc_name_or_uid.rsplit("_", 1)
    if (
        len(parts) == 2
        and len(parts[1]) == 8
        and all(c in "0123456789abcdefABCDEF" for c in parts[1])
    ):
        base_name = parts[0]
    else:
        base_name = sc_name_or_uid

    # Check embedded query in canonical name (e.g. pubmed_q12)
    m_can_qid = re.search(r"^(.*)_q(\d+(?:_ext)?)$", base_name)
    if m_can_qid:
        base_name = m_can_qid.group(1)
        if query_id is None:
            query_id = m_can_qid.group(2)

    name_map = SCENARIO_SHORT_NAMES if short else SCENARIO_DISPLAY_NAMES
    display_base = name_map.get(base_name, base_name.replace("_", " ").title())

    qid_str = None
    if query_id is not None:
        clean_qid = str(query_id).lstrip("q").lstrip("Q")
        if clean_qid.endswith("_ext"):
            qid_str = f"Q{clean_qid[:-4]}_ext"
        else:
            qid_str = f"Q{clean_qid}"

    proxy_str = format_proxy_display_name(proxy_model)

    if qid_str is not None and proxy_str is not None:
        if short:
            return f"{display_base} {qid_str} ({proxy_str})"
        return f"{display_base} ({qid_str}, {proxy_str})"
    elif qid_str is not None:
        if short:
            return f"{display_base} {qid_str}"
        return f"{display_base} ({qid_str})"
    elif proxy_str is not None:
        return f"{display_base} ({proxy_str})"

    return display_base


def discover_hierarchical_records(
    results_dir: Path | str = "experiments/results",
    single_file: Optional[Path | str] = None,
    scenarios: Optional[Sequence[str]] = None,
    runners: Optional[Sequence[str]] = None,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    num_trials: Optional[int | Sequence[int]] = None,
    query_ids: Optional[Sequence[str]] = None,
    oracle_models: Optional[Sequence[str]] = None,
    proxy_models: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """
    Discovers, filters, and parses hierarchical experiment result JSON files.

    Supports both directory trees (`results_dir`) and explicit individual files
    (`single_file`). Validates schema integrity, calculates canonical configuration
    hashes, extracts canonical dataset names, query IDs, oracle/proxy models, and
    deduplicates identical evaluations.
    """
    if single_file is not None:
        p = Path(single_file)
        if not p.exists():
            raise FileNotFoundError(f"Single result file '{single_file}' not found.")
        json_files = [p]
    else:
        root = Path(results_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"Results directory '{results_dir}' does not exist."
            )
        json_files = sorted(root.rglob("*.json"))

    scenarios_set = set(scenarios) if scenarios else None
    runners_set = set(runners) if runners else None
    deltas_set = set(deltas) if deltas else None
    gammas_set = set(gammas) if gammas else None
    if num_trials is not None:
        num_trials_set = (
            {num_trials} if isinstance(num_trials, int) else set(num_trials)
        )
    else:
        num_trials_set = None
    query_ids_set = (
        {str(q).lower().lstrip("q") for q in query_ids} if query_ids else None
    )
    oracle_models_set = set(oracle_models) if oracle_models else None
    proxy_models_set = set(proxy_models) if proxy_models else None

    matching_records: list[dict[str, Any]] = []
    seen_combinations: set[tuple[str, str, float, float, float, Any, Any, Any, int]] = (
        set()
    )

    for p in json_files:
        if p.name.endswith(".tmp") or p.name.startswith("."):
            continue

        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        # Ignore merged comparison files (which contain lists of results)
        if not isinstance(data, dict):
            continue

        # Enforce strict new hierarchical schema
        if "scenario" not in data or "runner" not in data or "cost" not in data:
            continue

        scen_dict = data["scenario"]
        runner_dict = data["runner"]
        cost_dict = data["cost"]

        if (
            not isinstance(scen_dict, dict)
            or not isinstance(runner_dict, dict)
            or not isinstance(cost_dict, dict)
        ):
            continue

        scen_name = scen_dict.get("name")
        runner_name = runner_dict.get("name")
        if not scen_name or not runner_name:
            continue

        if runner_name == "actis_static":
            continue

        scen_hash = compute_config_hash(scen_dict)
        runner_hash = compute_config_hash(runner_dict)
        scen_uid = f"{scen_name}_{scen_hash}"
        runner_uid = f"{runner_name}_{runner_hash}"

        # Extract dataset, query_id, oracle_model, proxy_model
        raw_dataset = scen_dict.get("name") or scen_name
        canonical_dataset = re.sub(r"_q\d+(?:_ext)?$", "", raw_dataset)

        raw_qid = scen_dict.get("query_id")
        if raw_qid is not None:
            query_id = str(raw_qid)
        else:
            m_qid = re.search(r"_q(\d+(?:_ext)?)$", scen_name)
            query_id = m_qid.group(1) if m_qid else None

        oracle_model = scen_dict.get("oracle_model")
        proxy_model = scen_dict.get("proxy_model")

        # Filter by scenario (base name, canonical dataset, or UID)
        if scenarios_set:
            matches_scen = (
                scen_name in scenarios_set
                or scen_uid in scenarios_set
                or canonical_dataset in scenarios_set
            )
            if not matches_scen:
                continue

        # Filter by runner (base name or UID)
        if (
            runners_set
            and runner_name not in runners_set
            and runner_uid not in runners_set
        ):
            continue

        # Filter by query_ids (only filters records that define a query)
        if query_ids_set is not None and query_id is not None:
            clean_qid = str(query_id).lower().lstrip("q")
            if clean_qid not in query_ids_set:
                continue

        # Filter by oracle_models (only filters records that define an oracle)
        if oracle_models_set is not None and oracle_model is not None:
            oracle_slug = slugify_model_name(oracle_model)
            matches_oracle = (
                oracle_model in oracle_models_set or oracle_slug in oracle_models_set
            )
            if not matches_oracle:
                continue

        # Filter by proxy_models (only filters records that define a proxy)
        if proxy_models_set is not None and proxy_model is not None:
            proxy_slug = slugify_model_name(proxy_model)
            matches_proxy = (
                proxy_model in proxy_models_set or proxy_slug in proxy_models_set
            )
            if not matches_proxy:
                continue

        delta = float(data.get("promised_delta", data.get("delta", 0.05)))
        if deltas_set and not any(
            math.isclose(delta, d, abs_tol=1e-5) for d in deltas_set
        ):
            continue

        gamma_R = float(data.get("gamma_R", 0.8))
        gamma_P = float(data.get("gamma_P", 0.8))
        if gammas_set and not (
            any(math.isclose(gamma_R, g, abs_tol=1e-5) for g in gammas_set)
            or any(math.isclose(gamma_P, g, abs_tol=1e-5) for g in gammas_set)
        ):
            continue

        rec_trials = int(data.get("num_trials", 0))
        if num_trials_set is not None and rec_trials not in num_trials_set:
            continue

        # Deduplicate identical evaluations by UID, delta, gammas,
        # query/models, and trials
        comb_key = (
            scen_uid,
            runner_uid,
            delta,
            gamma_R,
            gamma_P,
            query_id,
            oracle_model,
            proxy_model,
            rec_trials,
        )
        if comb_key in seen_combinations:
            continue
        seen_combinations.add(comb_key)

        record_copy = dict(data)
        record_copy["scenario_uid"] = scen_uid
        record_copy["runner_uid"] = runner_uid
        record_copy["scenario_name"] = scen_name
        record_copy["runner_name"] = runner_name
        record_copy["dataset"] = canonical_dataset
        record_copy["query_id"] = query_id
        record_copy["oracle_model"] = oracle_model
        record_copy["proxy_model"] = proxy_model
        matching_records.append(record_copy)

    return matching_records


def resolve_methods(
    df: pd.DataFrame,
    default_methods: Sequence[tuple[str, str]],
    custom_methods: Optional[Sequence[str]] = None,
) -> list[tuple[str, str]]:
    """
    Resolves active methods to present in the table or plot.
    Filters default methods to those present in df, or resolves custom names.
    """
    present_methods = set(df["method"].unique())
    present_uids = (
        set(df["runner_uid"].unique()) if "runner_uid" in df.columns else set()
    )

    if custom_methods:
        resolved = []
        for m in custom_methods:
            matching_default = next(
                (label for raw, label in default_methods if raw == m), None
            )
            label = matching_default or m.replace("_", " ").title()
            resolved.append((m, label))
        return resolved

    # Filter default list to methods present in data
    active = [
        (raw_m, label)
        for raw_m, label in default_methods
        if raw_m in present_methods or raw_m in present_uids
    ]
    if not active:
        active = [(m, m.replace("_", " ").title()) for m in sorted(present_methods)]
    return active


def compute_bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    val_min: Optional[float] = None,
    val_max: Optional[float] = None,
) -> tuple[float, float, float]:
    """Computes sample mean and non-parametric bootstrap (1 - alpha) CI.
    Enforces valid domain bounds [val_min, val_max] if provided.
    Returns (mean, ci_low, ci_high).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if len(arr) == 1 or np.all(arr == arr[0]):
        low, high = mean_val, mean_val
    else:
        if rng is None:
            rng = np.random.default_rng(42)

        n = len(arr)
        boot_idx = rng.choice(n, size=(n_boot, n), replace=True)
        boot_means = np.mean(arr[boot_idx], axis=1)
        low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    if val_min is not None:
        low = max(val_min, low)
        mean_val = max(val_min, mean_val)
    if val_max is not None:
        high = min(val_max, high)
        mean_val = min(val_max, mean_val)

    return mean_val, low, high


def compute_quantile_ci(
    values: np.ndarray,
    q: float,
    alpha: float = 0.05,
    method: str = "order_stats",
    n_boot: int = 2000,
    rng: Optional[np.random.Generator] = None,
    val_min: Optional[float] = None,
    val_max: Optional[float] = None,
) -> tuple[float, float, float]:
    """Computes empirical q-quantile and non-parametric (1 - alpha) CI.

    Methods:
    - 'order_stats': Exact non-parametric order-statistic CI via Binomial quantiles.
      For sample size n, lower and upper bounds are S_{(j)} and S_{(k)} where
      j = binom.ppf(alpha / 2, n, q) and k = binom.ppf(1 - alpha / 2, n, q).
    - 'bootstrap': Non-parametric percentile bootstrap on the q-quantile.

    Returns (q_val, ci_low, ci_high).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    n = len(arr)
    q_val = float(np.percentile(arr, 100.0 * q))
    if len(arr) == 1 or np.all(arr == arr[0]):
        low, high = q_val, q_val
    elif method == "order_stats":
        s = np.sort(arr)
        j = int(binom.ppf(alpha / 2.0, n, q))
        k = int(binom.ppf(1.0 - alpha / 2.0, n, q))
        low = float(s[max(0, j - 1)])
        high = float(s[min(n - 1, k - 1)])
    else:
        if rng is None:
            rng = np.random.default_rng(42)
        boot_idx = rng.choice(n, size=(n_boot, n), replace=True)
        boot_q = np.percentile(arr[boot_idx], 100.0 * q, axis=1)
        low = float(np.percentile(boot_q, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(boot_q, 100.0 * (1.0 - alpha / 2.0)))

    if val_min is not None:
        low = max(val_min, low)
        q_val = max(val_min, q_val)
    if val_max is not None:
        high = min(val_max, high)
        q_val = min(val_max, q_val)

    return q_val, low, high


def get_benchmark_uid(
    dataset: str,
    query_id: str | None = None,
    proxy_model: str | None = None,
) -> str:
    """Creates a filesystem-safe unique benchmark identifier."""
    parts = [dataset]
    if query_id is not None:
        parts.append(f"q{query_id}")
    if proxy_model is not None:
        parts.append(slugify_model_name(proxy_model))
    return "_".join(parts)


def load_tradeoff_data(
    results_dir: Path | str = "experiments/results",
    single_file: Optional[Path | str] = None,
    scenarios: Optional[Sequence[str]] = None,
    runners: Optional[Sequence[str]] = None,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    num_trials: Optional[int | Sequence[int]] = None,
    query_ids: Optional[Sequence[str]] = None,
    oracle_models: Optional[Sequence[str]] = None,
    proxy_models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Loads result JSON files using the shared hierarchical discovery utility and
    computes trial-level bootstrap 95% CIs for oracle call rate, recall,
    precision, and constraint slack.
    """
    from experiments.soundness import is_certified_unsound

    records = discover_hierarchical_records(
        results_dir=results_dir,
        single_file=single_file,
        scenarios=scenarios,
        runners=runners,
        deltas=deltas,
        gammas=gammas,
        num_trials=num_trials,
        query_ids=query_ids,
        oracle_models=oracle_models,
        proxy_models=proxy_models,
    )

    if not records:
        target_desc = str(single_file) if single_file else str(results_dir)
        raise ValueError(
            f"No valid hierarchical JSON records found in '{target_desc}'."
        )

    processed_rows = []
    rng = np.random.default_rng(42)

    for r in records:
        raw_method = r["runner_name"]
        method_clean = METHOD_DISPLAY_NAMES.get(
            raw_method, raw_method.replace("_", " ").title()
        )
        scenario = r["scenario_name"]
        dataset = r.get("dataset", scenario)
        query_id = r.get("query_id")
        oracle_model = r.get("oracle_model")
        proxy_model = r.get("proxy_model")
        scenario_clean = format_scenario_display_name(
            dataset, query_id=query_id, proxy_model=proxy_model, short=False
        )
        benchmark_uid = get_benchmark_uid(dataset, query_id, proxy_model)

        gamma_P = float(r.get("gamma_P", 0.80))
        gamma_R = float(r.get("gamma_R", 0.80))
        delta = float(r.get("promised_delta", r.get("delta", 0.05)))
        ideal_oracle = float(r.get("ideal_oracle_call_rate", 0.0)) * 100.0

        raw_recalls = np.asarray(r.get("recall", {}).get("raw", []), dtype=float)
        raw_precisions = np.asarray(r.get("precision", {}).get("raw", []), dtype=float)
        cost_oracle = r.get("cost", {}).get("oracle", {}).get("call_rate", {})
        raw_oracles = np.asarray(cost_oracle.get("raw", []), dtype=float)

        # Oracle call rate bounded in [0, 100]%
        if len(raw_oracles) == 0:
            mean_oracle = float(cost_oracle.get("mean", 0.0)) * 100.0
            se_oracle = float(cost_oracle.get("se", 0.0)) * 100.0
            oracle_low = max(0.0, mean_oracle - 1.96 * se_oracle)
            oracle_high = min(100.0, mean_oracle + 1.96 * se_oracle)
        else:
            mean_oracle, oracle_low, oracle_high = compute_bootstrap_ci(
                raw_oracles * 100.0, rng=rng, val_min=0.0, val_max=100.0
            )

        # Recall bounded strictly in [0, 1] (delta-quantile and CI)
        if len(raw_recalls) == 0:
            mean_recall = float(r.get("recall", {}).get("mean", 0.0))
            se_recall = float(r.get("recall", {}).get("se", 0.0))
            recall_low = max(0.0, mean_recall - 1.96 * se_recall)
            recall_high = min(1.0, mean_recall + 1.96 * se_recall)
            q_recall = mean_recall
        else:
            mean_recall = float(np.mean(raw_recalls))
            q_recall, recall_low, recall_high = compute_quantile_ci(
                raw_recalls, q=delta, val_min=0.0, val_max=1.0
            )

        # Precision bounded strictly in [0, 1] (delta-quantile and CI)
        if len(raw_precisions) == 0:
            mean_prec = float(r.get("precision", {}).get("mean", 0.0))
            se_prec = float(r.get("precision", {}).get("se", 0.0))
            prec_low = max(0.0, mean_prec - 1.96 * se_prec)
            prec_high = min(1.0, mean_prec + 1.96 * se_prec)
            q_prec = mean_prec
        else:
            mean_prec = float(np.mean(raw_precisions))
            q_prec, prec_low, prec_high = compute_quantile_ci(
                raw_precisions, q=delta, val_min=0.0, val_max=1.0
            )

        # Constraint Slack: delta-quantile of min(precision - gamma_P, recall - gamma_R)
        max_possible_slack = 1.0 - min(gamma_P, gamma_R)
        min_possible_slack = -min(gamma_P, gamma_R)
        if len(raw_precisions) > 0 and len(raw_recalls) > 0:
            trial_slacks = np.minimum(raw_precisions - gamma_P, raw_recalls - gamma_R)
            q_slack, slack_low, slack_high = compute_quantile_ci(
                trial_slacks,
                q=delta,
                val_min=min_possible_slack,
                val_max=max_possible_slack,
            )
            mean_slack = float(np.mean(trial_slacks))
        else:
            mean_slack = min(mean_prec - gamma_P, mean_recall - gamma_R)
            q_slack = mean_slack
            slack_low, slack_high = q_slack, q_slack

        # Joint failure & certified soundness classification
        n_trials_val = int(
            r.get(
                "num_trials",
                len(raw_precisions) if len(raw_precisions) > 0 else 1000,
            )
        )
        joint_dict = r.get("joint_failure", {})
        if "raw" in joint_dict and isinstance(joint_dict["raw"], (list, np.ndarray)):
            k_joint = int(np.sum(np.asarray(joint_dict["raw"], dtype=bool)))
        elif "rate" in joint_dict:
            k_joint = int(round(float(joint_dict["rate"]) * n_trials_val))
        elif "mean" in joint_dict:
            k_joint = int(round(float(joint_dict["mean"]) * n_trials_val))
        elif len(raw_precisions) > 0 and len(raw_recalls) > 0:
            k_joint = int(np.sum((raw_precisions < gamma_P) | (raw_recalls < gamma_R)))
        else:
            k_joint = 0

        joint_fail_val = float(k_joint / n_trials_val) if n_trials_val > 0 else 0.0
        is_unsound = is_certified_unsound(k_joint, n_trials_val, delta)
        sound_label = "Unsound" if is_unsound else "Sound or Inconclusive"

        row: dict[str, Any] = {
            "dataset": dataset,
            "scenario": scenario,
            "scenario_uid": r["scenario_uid"],
            "benchmark_uid": benchmark_uid,
            "query_id": query_id,
            "oracle_model": oracle_model,
            "proxy_model": proxy_model,
            "scenario_clean": scenario_clean,
            "method": raw_method,
            "runner_uid": r["runner_uid"],
            "method_clean": method_clean,
            "gamma_P": gamma_P,
            "gamma_R": gamma_R,
            "gamma": gamma_R,
            "delta": delta,
            "delta_label": f"$\\delta = {delta:.2f}$",
            "ideal_oracle_rate_pct": ideal_oracle,
            "mean_oracle_rate_pct": mean_oracle,
            "oracle_ci_low": oracle_low,
            "oracle_ci_high": oracle_high,
            "mean_true_recall": mean_recall,
            "quantile_recall": q_recall,
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
            "mean_true_precision": mean_prec,
            "quantile_precision": q_prec,
            "prec_ci_low": prec_low,
            "prec_ci_high": prec_high,
            "slack": q_slack,
            "slack_ci_low": slack_low,
            "slack_ci_high": slack_high,
            "quantile_slack": q_slack,
            "mean_slack": mean_slack,
            "num_trials": n_trials_val,
            "k_joint": k_joint,
            "joint_fail": joint_fail_val,
            "unsound": is_unsound,
            "sound": not is_unsound,
            "is_sound": not is_unsound,
            "sound_label": sound_label,
        }
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)
