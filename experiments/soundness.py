"""
Core module for evaluating statistical soundness of model cascade tuning methods.

Provides exact Clopper-Pearson binomial confidence interval calculations,
soundness certification, dataframe annotation, and aggregation utilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from experiments.results_utils import (
    CANONICAL_SCENARIO_ORDER,
    METHOD_DISPLAY_NAMES,
    discover_hierarchical_records,
)

__all__ = [
    "annotate_dataframe_soundness",
    "build_appendix_configuration_table",
    "classify_soundness",
    "compute_exact_binomial_ci",
    "evaluate_record_soundness",
    "is_certified_sound",
    "is_certified_unsound",
    "is_consistent_with_soundness",
    "load_soundness_dataframe",
    "summarize_comprehensive_soundness",
    "summarize_soundness_by_delta",
    "summarize_soundness_by_delta_gamma",
]


def compute_exact_binomial_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Computes two-sided Clopper-Pearson exact binomial confidence interval [LCB, UCB].

    Uses scipy.stats.binomtest.

    Args:
        k: Number of failures / successes (0 <= k <= n).
        n: Total number of trials (n >= 1).
        alpha: Significance level (default 0.05 for 95% confidence).

    Returns:
        (lcb, ucb): Exact lower and upper confidence bounds in [0.0, 1.0].
    """
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(int(k), int(n)))
    res = binomtest(k, n)
    if hasattr(res, "proportion_ci"):
        ci = res.proportion_ci(confidence_level=1.0 - alpha, method="exact")
    else:
        ci = res.confidence_interval(confidence_level=1.0 - alpha)
    return (float(ci.low), float(ci.high))


def classify_soundness(
    k: int,
    n: int,
    delta: float,
    alpha: float = 0.05,
) -> tuple[str, float, float]:
    """Classifies empirical outcome against target delta using exact CI [LCB, UCB].

    Categories:
    - 'sound': UCB < delta (certified sound / conservative)
    - 'unsound': LCB > delta (certified unsound)
    - 'inconclusive': LCB <= delta <= UCB (delta is covered by CI)

    Args:
        k: Number of failures.
        n: Total number of trials.
        delta: Target error budget in (0, 1).
        alpha: Significance level (default 0.05).

    Returns:
        (status, lcb, ucb): Status string and exact confidence bounds.
    """
    lcb, ucb = compute_exact_binomial_ci(k, n, alpha=alpha)
    if ucb < delta:
        status = "sound"
    elif lcb > delta:
        status = "unsound"
    else:
        status = "inconclusive"
    return status, lcb, ucb


def is_certified_unsound(
    k: int,
    n: int,
    delta: float,
    alpha: float = 0.05,
) -> bool:
    """Returns True iff the lower confidence bound strictly exceeds delta."""
    lcb, _ = compute_exact_binomial_ci(k, n, alpha=alpha)
    return lcb > delta


def is_certified_sound(
    k: int,
    n: int,
    delta: float,
    alpha: float = 0.05,
) -> bool:
    """Returns True iff the upper confidence bound is strictly below delta."""
    _, ucb = compute_exact_binomial_ci(k, n, alpha=alpha)
    return ucb < delta


def is_consistent_with_soundness(
    k: int,
    n: int,
    delta: float,
    alpha: float = 0.05,
) -> bool:
    """Returns True iff LCB <= delta (cannot reject null hypothesis p <= delta)."""
    lcb, _ = compute_exact_binomial_ci(k, n, alpha=alpha)
    return lcb <= delta


def _extract_metric_failures(
    metric_dict: dict[str, Any],
    n: int,
) -> tuple[int, float]:
    """Extracts failure count k and empirical rate p_hat from a metric dict."""
    if "raw" in metric_dict and isinstance(metric_dict["raw"], (list, np.ndarray)):
        raw = metric_dict["raw"]
        k = int(np.sum(np.asarray(raw, dtype=bool)))
    else:
        mean_val = float(metric_dict.get("mean", 0.0))
        k = int(round(mean_val * n))
    p_hat = float(k / n) if n > 0 else 0.0
    return k, p_hat


def evaluate_record_soundness(
    record: dict[str, Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluates Clopper-Pearson CIs and soundness classifications for a result record.

    Computes metrics across joint, recall, and precision failures.
    """
    n = int(record.get("num_trials", 1000))
    delta = float(record.get("promised_delta", record.get("delta", 0.05)))

    # 1. Joint failure
    joint_dict = record.get("joint_failure", {})
    k_joint, p_joint = _extract_metric_failures(joint_dict, n)
    status_joint, lcb_joint, ucb_joint = classify_soundness(
        k_joint, n, delta, alpha=alpha
    )

    # 2. Recall failure
    rec_dict = record.get("recall_failure", {})
    k_rec, p_rec = _extract_metric_failures(rec_dict, n)
    status_rec, lcb_rec, ucb_rec = classify_soundness(k_rec, n, delta, alpha=alpha)

    # 3. Precision failure
    prec_dict = record.get("precision_failure", {})
    k_prec, p_prec = _extract_metric_failures(prec_dict, n)
    status_prec, lcb_prec, ucb_prec = classify_soundness(k_prec, n, delta, alpha=alpha)

    # 4. F1 failure (if explicitly recorded)
    f1_dict = record.get("f1_failure", {})
    if f1_dict:
        k_f1, p_f1 = _extract_metric_failures(f1_dict, n)
        status_f1, lcb_f1, ucb_f1 = classify_soundness(k_f1, n, delta, alpha=alpha)
    else:
        k_f1, p_f1, status_f1, lcb_f1, ucb_f1 = None, None, None, None, None

    res = {
        "n_trials": n,
        "delta": delta,
        # Joint
        "k_joint": k_joint,
        "p_hat_joint": p_joint,
        "lcb_joint": lcb_joint,
        "ucb_joint": ucb_joint,
        "status_joint": status_joint,
        "is_unsound_joint": status_joint == "unsound",
        "is_sound_joint": status_joint == "sound",
        "is_inconclusive_joint": status_joint == "inconclusive",
        # Recall
        "k_recall": k_rec,
        "p_hat_recall": p_rec,
        "lcb_recall": lcb_rec,
        "ucb_recall": ucb_rec,
        "status_recall": status_rec,
        "is_unsound_recall": status_rec == "unsound",
        "is_sound_recall": status_rec == "sound",
        "is_inconclusive_recall": status_rec == "inconclusive",
        # Precision
        "k_precision": k_prec,
        "p_hat_precision": p_prec,
        "lcb_precision": lcb_prec,
        "ucb_precision": ucb_prec,
        "status_precision": status_prec,
        "is_unsound_precision": status_prec == "unsound",
        "is_sound_precision": status_prec == "sound",
        "is_inconclusive_precision": status_prec == "inconclusive",
    }
    if k_f1 is not None:
        res.update(
            {
                "k_f1": k_f1,
                "p_hat_f1": p_f1,
                "lcb_f1": lcb_f1,
                "ucb_f1": ucb_f1,
                "status_f1": status_f1,
                "is_unsound_f1": status_f1 == "unsound",
                "is_sound_f1": status_f1 == "sound",
                "is_inconclusive_f1": status_f1 == "inconclusive",
            }
        )
    return res


def load_soundness_dataframe(
    results_dir: Path | str = "experiments/results",
    num_trials: Optional[int | Sequence[int]] = 1000,
    scenarios: Optional[Sequence[str]] = None,
    runners: Optional[Sequence[str]] = None,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    use_native_scaledoc: bool = True,
) -> pd.DataFrame:
    """Loads records, computes exact CIs, and returns an annotated DataFrame."""
    records = discover_hierarchical_records(
        results_dir=results_dir,
        scenarios=scenarios,
        runners=runners,
        deltas=deltas,
        gammas=gammas,
        num_trials=num_trials,
    )

    rows = []
    for rec in records:
        scen_name = rec["scenario_name"]
        scen_uid = rec["scenario_uid"]
        dataset = rec.get("dataset", scen_name)

        runner_name = rec["runner_name"]
        if runners is None:
            if use_native_scaledoc and runner_name == "scaledoc":
                continue
            if not use_native_scaledoc and runner_name == "scaledoc_native":
                continue

        runner_uid = rec["runner_uid"]
        if runner_name in ("scaledoc", "scaledoc_native"):
            if (
                runners is not None
                and "scaledoc" in runners
                and "scaledoc_native" in runners
            ):
                method_label = (
                    "ScaleDoc (Native)"
                    if runner_name == "scaledoc_native"
                    else "ScaleDoc"
                )
            else:
                method_label = "ScaleDoc"
        else:
            method_label = METHOD_DISPLAY_NAMES.get(
                runner_name, runner_name.replace("_", " ").title()
            )

        query_id = rec.get("query_id")
        oracle_model = rec.get("oracle_model")
        proxy_model = rec.get("proxy_model")

        gamma_R = float(rec.get("gamma_R", 0.8))
        gamma_P = float(rec.get("gamma_P", 0.8))

        cost_dict = rec.get("cost", {})
        oracle_stats = cost_dict.get("oracle", {}).get("call_rate", {})
        mean_oracle = float(oracle_stats.get("mean", 0.0))
        se_oracle = float(oracle_stats.get("se", 0.0))

        eval_res = evaluate_record_soundness(rec, alpha=alpha)

        rows.append(
            {
                "dataset": dataset,
                "scenario": scen_name,
                "scenario_uid": scen_uid,
                "query_id": query_id,
                "oracle_model": oracle_model,
                "proxy_model": proxy_model,
                "method": runner_name,
                "method_name": method_label,
                "runner_uid": runner_uid,
                "gamma": gamma_R,
                "gamma_P": gamma_P,
                "oracle_call_rate": mean_oracle,
                "se_oracle_call_rate": se_oracle,
                **eval_res,
            }
        )

    return pd.DataFrame(rows)


def annotate_dataframe_soundness(
    df: pd.DataFrame,
    alpha: float = 0.05,
    prefix: str = "",
) -> pd.DataFrame:
    """Adds exact Clopper-Pearson CIs and soundness classifications to a DataFrame.

    Requires columns: 'num_trials' (or 'n_trials'), 'delta', and
    failure counts or rates for 'joint_fail' / 'k_joint'.
    """
    df = df.copy()
    n_col = "n_trials" if "n_trials" in df.columns else "num_trials"
    if n_col not in df.columns:
        df[n_col] = 1000

    def _eval_row(row):
        n = int(row[n_col])
        delta = float(row["delta"])
        if "k_joint" in row:
            k = int(row["k_joint"])
        elif "joint_fail" in row:
            k = int(round(float(row["joint_fail"]) * n))
        else:
            k = 0
        status, lcb, ucb = classify_soundness(k, n, delta, alpha=alpha)
        return pd.Series(
            {
                f"{prefix}lcb_joint": lcb,
                f"{prefix}ucb_joint": ucb,
                f"{prefix}status_joint": status,
                f"{prefix}is_unsound": status == "unsound",
                f"{prefix}is_sound": status == "sound",
                f"{prefix}is_inconclusive": status == "inconclusive",
            }
        )

    res = df.apply(_eval_row, axis=1)
    for col in res.columns:
        df[col] = res[col]
    return df


def _aggregate_soundness(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str = "joint",
) -> pd.DataFrame:
    """Aggregates soundness rates and conditional failure rates across groups."""
    status_col = f"status_{target}"
    p_hat_col = f"p_hat_{target}"

    records = []
    for keys, group in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(group_cols, keys))

        total_configs = len(group)
        n_sound = int((group[status_col] == "sound").sum())
        n_inconcl = int((group[status_col] == "inconclusive").sum())
        n_unsound = int((group[status_col] == "unsound").sum())

        pct_sound = (n_sound / total_configs) * 100.0 if total_configs > 0 else 0.0
        pct_inconcl = (n_inconcl / total_configs) * 100.0 if total_configs > 0 else 0.0
        pct_unsound = (n_unsound / total_configs) * 100.0 if total_configs > 0 else 0.0

        unsound_rows = group[group[status_col] == "unsound"]
        if len(unsound_rows) > 0:
            cond_mean_fail = float(unsound_rows[p_hat_col].mean()) * 100.0
        else:
            cond_mean_fail = float("nan")

        max_fail = float(group[p_hat_col].max()) * 100.0 if total_configs > 0 else 0.0

        d.update(
            {
                "N": total_configs,
                "pct_sound": pct_sound,
                "pct_inconclusive": pct_inconcl,
                "pct_unsound": pct_unsound,
                "cond_mean_fail_pct": cond_mean_fail,
                "max_fail_pct": max_fail,
            }
        )
        records.append(d)

    return pd.DataFrame(records)


def summarize_soundness_by_delta(
    df: pd.DataFrame,
    target: str = "joint",
) -> pd.DataFrame:
    """Summarizes soundness by delta and method."""
    return _aggregate_soundness(df, group_cols=["delta", "method_name"], target=target)


def summarize_soundness_by_delta_gamma(
    df: pd.DataFrame,
    target: str = "joint",
) -> pd.DataFrame:
    """Summarizes soundness by delta, gamma, and method."""
    return _aggregate_soundness(
        df, group_cols=["delta", "gamma", "method_name"], target=target
    )


def summarize_comprehensive_soundness(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregates joint soundness guarantees and diagnostic recall/precision failure

    metrics across grouping columns (e.g. ['delta', 'method_name'] or
    ['delta', 'gamma', 'method_name']).
    """
    records = []
    for keys, group in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(group_cols, keys))

        total_configs = len(group)
        d["N"] = total_configs

        # 1. Joint Target Guarantee
        n_sound_j = int((group["status_joint"] == "sound").sum())
        n_inconcl_j = int((group["status_joint"] == "inconclusive").sum())
        n_unsound_j = int((group["status_joint"] == "unsound").sum())

        d["pct_sound"] = (
            (n_sound_j / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_inconclusive"] = (
            (n_inconcl_j / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_unsound"] = (
            (n_unsound_j / total_configs) * 100.0 if total_configs > 0 else 0.0
        )

        unsound_j_rows = group[group["status_joint"] == "unsound"]
        d["cond_mean_fail_pct"] = (
            float(unsound_j_rows["p_hat_joint"].mean()) * 100.0
            if len(unsound_j_rows) > 0
            else float("nan")
        )
        d["max_fail_pct"] = (
            float(group["p_hat_joint"].max()) * 100.0 if total_configs > 0 else 0.0
        )
        d["mean_fail_pct"] = (
            float(group["p_hat_joint"].mean()) * 100.0 if total_configs > 0 else 0.0
        )

        # 2. Recall Target Diagnostics
        n_sound_r = int((group["status_recall"] == "sound").sum())
        n_inconcl_r = int((group["status_recall"] == "inconclusive").sum())
        n_unsound_r = int((group["status_recall"] == "unsound").sum())

        d["pct_sound_recall"] = (
            (n_sound_r / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_inconclusive_recall"] = (
            (n_inconcl_r / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_unsound_recall"] = (
            (n_unsound_r / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["mean_fail_recall_pct"] = (
            float(group["p_hat_recall"].mean()) * 100.0 if total_configs > 0 else 0.0
        )
        unsound_r_rows = group[group["status_recall"] == "unsound"]
        d["cond_mean_fail_recall_pct"] = (
            float(unsound_r_rows["p_hat_recall"].mean()) * 100.0
            if len(unsound_r_rows) > 0
            else float("nan")
        )
        d["max_fail_recall_pct"] = (
            float(group["p_hat_recall"].max()) * 100.0 if total_configs > 0 else 0.0
        )

        # 3. Precision Target Diagnostics
        n_sound_p = int((group["status_precision"] == "sound").sum())
        n_inconcl_p = int((group["status_precision"] == "inconclusive").sum())
        n_unsound_p = int((group["status_precision"] == "unsound").sum())

        d["pct_sound_precision"] = (
            (n_sound_p / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_inconclusive_precision"] = (
            (n_inconcl_p / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["pct_unsound_precision"] = (
            (n_unsound_p / total_configs) * 100.0 if total_configs > 0 else 0.0
        )
        d["mean_fail_precision_pct"] = (
            float(group["p_hat_precision"].mean()) * 100.0 if total_configs > 0 else 0.0
        )
        unsound_p_rows = group[group["status_precision"] == "unsound"]
        d["cond_mean_fail_precision_pct"] = (
            float(unsound_p_rows["p_hat_precision"].mean()) * 100.0
            if len(unsound_p_rows) > 0
            else float("nan")
        )
        d["max_fail_precision_pct"] = (
            float(group["p_hat_precision"].max()) * 100.0 if total_configs > 0 else 0.0
        )

        records.append(d)

    return pd.DataFrame(records)


def build_appendix_configuration_table(
    df: pd.DataFrame,
    target: str = "joint",
    delta: Optional[float] = None,
) -> pd.DataFrame:
    """Builds a configuration-level appendix table reporting empirical failure rate

    and two-sided 95% confidence intervals [LCB, UCB] for each method separately.
    """
    sub = df[df["delta"] == delta] if delta is not None else df.copy()

    p_col = f"p_hat_{target}"
    lcb_col = f"lcb_{target}"
    ucb_col = f"ucb_{target}"
    status_col = f"status_{target}"

    def _fmt_cell(row):
        p_pct = row[p_col] * 100.0
        lcb_pct = row[lcb_col] * 100.0
        ucb_pct = row[ucb_col] * 100.0
        star = "*" if row[status_col] == "unsound" else ""
        return f"{p_pct:.1f} [{lcb_pct:.1f}, {ucb_pct:.1f}]{star}"

    sub = sub.copy()
    sub["formatted_result"] = sub.apply(_fmt_cell, axis=1)
    sub["query_str"] = sub["query_id"].fillna("—")
    sub["proxy_str"] = sub["proxy_model"].fillna("—")

    pivot_cols = ["dataset", "query_str", "proxy_str", "delta", "gamma"]
    pivot = sub.pivot(
        index=pivot_cols,
        columns="method_name",
        values="formatted_result",
    ).reset_index()

    # Sort scenarios canonically
    def _sort_key(row):
        base = str(row["dataset"])
        idx = (
            CANONICAL_SCENARIO_ORDER.index(base)
            if base in CANONICAL_SCENARIO_ORDER
            else 999
        )
        qid = str(row["query_str"])
        m = re.search(r"\d+", qid)
        qid_num = int(m.group(0)) if m else 0
        is_ext = 1 if "_ext" in qid else 0
        proxy = str(row["proxy_str"])
        gamma = float(row["gamma"])
        delta_val = float(row["delta"])
        return (idx, qid_num, is_ext, proxy, delta_val, gamma)

    sorted_indices = sorted(range(len(pivot)), key=lambda i: _sort_key(pivot.iloc[i]))
    return pivot.iloc[sorted_indices].reset_index(drop=True)
