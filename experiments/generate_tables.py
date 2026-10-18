#!/usr/bin/env python3
"""
LaTeX Table Generator for Semantic Cascade Benchmark Experiments.

Generates publication-ready LaTeX tables formatted with booktabs:
1. `table_full_width.tex`: Full-width table (table*) presenting oracle call rates
   across scenarios/benchmarks, target guarantees (gamma), and error budgets
   (delta) for ACTIS, ScaleDoc, BARGAIN, and LOTUS, alongside the Ideal
   oracle lower bound.
2. `table_full_width_mean_only.tex`: Same as above, omitting standard errors for
   compactness.
3. `table_single_col_delta_*.tex`: Single-column tables for each delta value.
4. `table_single_col_delta_*_mean_only.tex`: Single-column tables reporting mean
   only.
5. `table_appendix_all_variants_delta_*.tex`: Comprehensive tables comparing all
   ACTIS variants (Uniform, Uniform Asymptotic, IS Finite, IS Asymptotic),
   ScaleDoc, BARGAIN, and LOTUS against the Ideal baseline.
6. `table_appendix_all_variants_delta_*_mean_only.tex`: Appendix tables
   reporting mean only.

Key features:
- Automatically discovers results from the hierarchical results directory:
  `<results_dir>/<scenario_uid>/<runner_uid>/delta_<delta>_gamma_<gamma>_trials_<trials>.json`
- Preserves `scenario_uid` and `runner_uid` (with 8-char configuration hashes)
  to prevent accidental merging across different parameter settings.
- Ignores legacy and merged comparison JSON files.
- Best (lowest) oracle call rate among valid methods is bolded (\\mathbf{...}).
- Methods violating target guarantees (certified unsound via exact 95%
  Clopper-Pearson CI: LCB > delta) are flagged with an asterisk (*).
  Invalid methods are never bolded.
- Uses \\rlap{$^*$} so asterisks do not disrupt column decimal alignment.
- Dynamic column specifications and headers adapting to active methods and
  deltas.
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.results_utils import (  # noqa: E402
    CANONICAL_SCENARIO_ORDER,
    DEFAULT_APPENDIX_METHODS,
    DEFAULT_FOCUS_METHODS,
    METHOD_DISPLAY_NAMES,
    METHOD_ORDER,
    PROXY_DISPLAY_NAMES,
    SCENARIO_DISPLAY_NAMES,
    SCENARIO_SHORT_NAMES,
    compute_config_hash,
    discover_hierarchical_records,
    format_proxy_display_name,
    format_scenario_display_name,
    resolve_methods,
)
from experiments.soundness import is_certified_unsound  # noqa: E402

__all__ = [
    "CANONICAL_SCENARIO_ORDER",
    "DEFAULT_APPENDIX_METHODS",
    "DEFAULT_FOCUS_METHODS",
    "DEFAULT_REPRESENTATIVE_WORKLOADS",
    "METHOD_DISPLAY_NAMES",
    "METHOD_ORDER",
    "PROXY_DISPLAY_NAMES",
    "SCENARIO_DISPLAY_NAMES",
    "SCENARIO_SHORT_NAMES",
    "compute_config_hash",
    "format_latency_value",
    "format_population_size",
    "format_proxy_display_name",
    "format_scenario_display_name",
    "format_slack_value",
    "format_value",
    "generate_all_benchmarks_joint_table",
    "generate_all_benchmarks_slack_table",
    "generate_appendix_latency_table",
    "generate_appendix_table",
    "generate_benchmark_slack_table",
    "generate_full_width_table",
    "generate_joint_table",
    "generate_latency_table",
    "generate_representative_joint_table",
    "generate_representative_oracle_table",
    "generate_single_column_table",
    "generate_slack_distribution_table",
    "get_audit_dataframe",
    "get_ordered_deltas",
    "get_ordered_gammas",
    "get_ordered_scenarios",
    "load_table_data",
    "resolve_methods",
]

# Default representative workloads for publication table (dataset, query_id, proxy_model)
DEFAULT_REPRESENTATIVE_WORKLOADS = [
    ("onto", None, None),
    ("imagenet", None, None),
    ("jackson", None, None),
    ("tacred", None, None),
    ("court", "0", "azure/gpt-5.6-luna"),
    ("review", "0", "azure/gpt-5.6-luna"),
    ("screenplay", "0", "openai/llama3.2:1b-instruct-fp16"),
    ("wiki", "0", "azure/gpt-5.6-luna"),
    ("pubmed", "0", "azure/gpt-5.6-luna"),
    ("big_patent", "0_ext", "openai/llama3.2:1b-instruct-fp16"),
    ("gov_report", "10", "azure/gpt-5.6-luna"),
]

# Canonical proxy model names for datasets without explicit model configurations
DATASET_DEFAULT_PROXIES: dict[str, str] = {
    "onto": "UFT",
    "imagenet": "ResNet-50",
    "jackson": "ResNet-50",
    "tacred": "SpanBERT",
}


def get_representative_display(
    rep_scen: str, rep_qid: str | None, rep_proxy: str | None
) -> tuple[str, str]:
    """Derives publication-ready display name and proxy for representative workloads."""
    disp_name = format_scenario_display_name(rep_scen, query_id=rep_qid, short=True)
    if "_ext" in disp_name:
        disp_name = disp_name.replace("_ext", r"\textsubscript{ext}")

    if rep_proxy is not None:
        disp_proxy = format_proxy_display_name(rep_proxy) or str(rep_proxy)
    else:
        disp_proxy = DATASET_DEFAULT_PROXIES.get(rep_scen, "--")

    return disp_name, disp_proxy


def load_table_data(
    results_dir: Path | str = "experiments/results",
    scenarios: Optional[Sequence[str]] = None,
    runners: Optional[Sequence[str]] = None,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    num_trials: Optional[int | Sequence[int]] = None,
    query_ids: Optional[Sequence[str]] = None,
    oracle_models: Optional[Sequence[str]] = None,
    proxy_models: Optional[Sequence[str]] = None,
    include_scaledoc_native: bool = True,
) -> pd.DataFrame:
    """
    Discovers and parses hierarchical experiment result files into a DataFrame.
    Strictly parses the new hierarchical schema and tracks configuration hashes
    in `scenario_uid` and `runner_uid`. Merged comparison lists are ignored.
    """
    records = discover_hierarchical_records(
        results_dir=results_dir,
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
        raise ValueError(
            f"No valid hierarchical JSON records found under '{results_dir}'."
        )

    rows = []
    for data in records:
        scen_name = data["scenario_name"]
        scen_uid = data["scenario_uid"]
        runner_name = data["runner_name"]
        if runners is None and not include_scaledoc_native and runner_name == "scaledoc_native":
            continue
        runner_uid = data["runner_uid"]
        dataset = data.get("dataset", scen_name)
        query_id = data.get("query_id")
        oracle_model = data.get("oracle_model")
        proxy_model = data.get("proxy_model")

        delta = float(data.get("promised_delta", data.get("delta", 0.05)))
        gamma_R = float(data.get("gamma_R", 0.8))
        gamma_P = float(data.get("gamma_P", 0.8))

        cost_dict = data.get("cost", {})
        oracle_call_stats = cost_dict.get("oracle", {}).get("call_rate", {})
        mean_oracle = float(oracle_call_stats.get("mean", 0.0)) * 100.0
        se_oracle = float(oracle_call_stats.get("se", 0.0)) * 100.0

        ideal_oracle = float(data.get("ideal_oracle_call_rate", 0.0)) * 100.0

        mean_rec = float(data.get("recall", {}).get("mean", 0.0))
        mean_prec = float(data.get("precision", {}).get("mean", 0.0))
        joint_fail = float(data.get("joint_failure", {}).get("mean", 0.0))

        n_trials = int(data.get("num_trials", 1000))
        if n_trials <= 0:
            n_trials = 1000
        k = int(round(joint_fail * n_trials))
        violated = is_certified_unsound(k, n_trials, delta)

        # Constraint Slack calculation matching plot_slack_distribution_vs_gamma
        # For methods on joint P/R (ACTIS, BARGAIN, LOTUS): min(P - gamma_P, R - gamma_R)
        # For ScaleDoc Native targeting F1: F1 - gamma
        raw_recalls = np.asarray(data.get("recall", {}).get("raw", []), dtype=float)
        raw_precisions = np.asarray(data.get("precision", {}).get("raw", []), dtype=float)
        raw_f1 = np.asarray(data.get("f1", {}).get("raw", []), dtype=float)
        target_gamma = max(gamma_P, gamma_R)

        if runner_name == "scaledoc_native" and len(raw_f1) > 0:
            trial_slacks = raw_f1 - target_gamma
            max_possible_slack = 1.0 - target_gamma
            min_possible_slack = -target_gamma
            quantile_slack = float(np.percentile(trial_slacks, 100.0 * delta))
            quantile_slack = max(min_possible_slack, min(max_possible_slack, quantile_slack))
            mean_slack = float(np.mean(trial_slacks))
        elif len(raw_precisions) > 0 and len(raw_recalls) > 0:
            trial_slacks = np.minimum(raw_precisions - gamma_P, raw_recalls - gamma_R)
            max_possible_slack = 1.0 - min(gamma_P, gamma_R)
            min_possible_slack = -min(gamma_P, gamma_R)
            quantile_slack = float(np.percentile(trial_slacks, 100.0 * delta))
            quantile_slack = max(min_possible_slack, min(max_possible_slack, quantile_slack))
            mean_slack = float(np.mean(trial_slacks))
        else:
            if runner_name == "scaledoc_native" and "f1" in data and "mean" in data["f1"]:
                mean_slack = float(data["f1"]["mean"]) - target_gamma
            else:
                mean_slack = min(mean_prec - gamma_P, mean_rec - gamma_R)
            quantile_slack = mean_slack

        method_clean = METHOD_DISPLAY_NAMES.get(
            runner_name, runner_name.replace("_", " ").title()
        )
        scenario_clean = format_scenario_display_name(
            dataset, query_id=query_id, proxy_model=proxy_model, short=False
        )

        runtime_dict = data.get("runtime", {})
        if isinstance(runtime_dict, dict):
            mean_rt = float(runtime_dict.get("mean", 0.0))
            se_rt = float(runtime_dict.get("se", 0.0))
            median_rt = float(runtime_dict.get("median", 0.0))
            std_rt = float(runtime_dict.get("std", 0.0))
        else:
            mean_rt = 0.0
            se_rt = 0.0
            median_rt = 0.0
            std_rt = 0.0
        pop_size = data.get("pop_size")

        rows.append(
            {
                "dataset": dataset,
                "scenario": scen_name,
                "scenario_uid": scen_uid,
                "query_id": query_id,
                "oracle_model": oracle_model,
                "proxy_model": proxy_model,
                "method": runner_name,
                "runner_uid": runner_uid,
                "method_clean": method_clean,
                "scenario_clean": scenario_clean,
                "gamma": gamma_R,
                "gamma_P": gamma_P,
                "delta": delta,
                "num_trials": n_trials,
                "ideal_oracle": ideal_oracle,
                "mean_oracle": mean_oracle,
                "se_oracle": se_oracle,
                "mean_rec": mean_rec,
                "mean_prec": mean_prec,
                "joint_fail": joint_fail,
                "violated": violated,
                "quantile_slack": quantile_slack,
                "mean_slack": mean_slack,
                "mean_runtime": mean_rt,
                "se_runtime": se_rt,
                "median_runtime": median_rt,
                "std_runtime": std_rt,
                "pop_size": pop_size,
            }
        )

    return pd.DataFrame(rows)


def get_ordered_scenarios(df: pd.DataFrame) -> list[str]:
    """Returns ordered unique scenario UIDs from df according to canonical

    dataset order, query ID, proxy model, and UID.
    """
    unique_uids = list(df["scenario_uid"].unique())

    uid_keys = {}
    for uid in unique_uids:
        sub = df[df["scenario_uid"] == uid]
        row = sub.iloc[0]
        dataset = str(row.get("dataset", row.get("scenario", "")))
        base = uid.rsplit("_", 1)[0]
        if dataset in CANONICAL_SCENARIO_ORDER:
            base_idx = CANONICAL_SCENARIO_ORDER.index(dataset)
        elif base in CANONICAL_SCENARIO_ORDER:
            base_idx = CANONICAL_SCENARIO_ORDER.index(base)
        else:
            base_idx = 999

        qid = str(row.get("query_id") or "")
        m = re.search(r"\d+", qid)
        qid_num = int(m.group(0)) if m else 0
        is_ext = 1 if "_ext" in qid else 0

        proxy = str(row.get("proxy_model") or "")
        proxy_key = (
            0 if "luna" in proxy.lower() else (1 if "llama" in proxy.lower() else 2)
        )

        uid_keys[uid] = (base_idx, qid_num, is_ext, proxy_key, proxy, uid)

    return sorted(unique_uids, key=lambda u: uid_keys[u])


def get_ordered_gammas(df: pd.DataFrame) -> list[float]:
    """Returns sorted unique gamma values in df."""
    return sorted(df["gamma"].unique())


def get_ordered_deltas(df: pd.DataFrame) -> list[float]:
    """Returns sorted unique delta values in df."""
    return sorted(df["delta"].unique())


def format_population_size(n: int | float | None) -> str:
    """Formats population size N into publication-ready string (e.g. 11k, 50k, 973k, 1M)."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "--"
    n_int = int(round(n))
    if n_int >= 1_000_000:
        val = n_int / 1_000_000
        return f"{val:.1f}M" if (val % 1 >= 0.05 and val < 10) else f"{round(val)}M"
    elif n_int >= 500:
        val = round(n_int / 1000)
        return f"{val}k"
    return str(n_int)


def format_value(
    mean_val: float,
    se_val: Optional[float],
    violated: bool,
    is_best: bool,
    include_se: bool = True,
    precision: int = 1,
    use_badvalue: bool = False,
) -> str:
    """Formats a single table cell value with bolding, SE, and asterisk."""
    mean_str = f"{mean_val:.{precision}f}"
    se_str = f"{se_val:.{precision}f}" if se_val is not None else ""

    if is_best and not violated:
        formatted_mean = f"\\mathbf{{{mean_str}}}"
    else:
        formatted_mean = mean_str

    if include_se and se_val is not None:
        cell_math = f"${formatted_mean} \\pm {se_str}$"
    else:
        cell_math = f"${formatted_mean}$"

    if violated:
        if use_badvalue:
            return f"\\badvalue{{{cell_math}\\rlap{{*}}}}"
        else:
            return f"{cell_math}\\rlap{{$^*$}}"

    return cell_math


def format_latency_value(
    mean_sec: float,
    se_sec: Optional[float],
    violated: bool,
    is_best: bool,
    include_se: bool = True,
    precision: int = 1,
    use_badvalue: bool = False,
) -> str:
    """Formats a single table cell latency value in milliseconds with bolding, SE, and asterisk."""
    if mean_sec is None or np.isnan(mean_sec):
        return "--"

    mean_ms = mean_sec * 1000.0
    se_ms = se_sec * 1000.0 if se_sec is not None else None

    mean_str = f"{mean_ms:.{precision}f}"
    se_str = f"{se_ms:.{precision}f}" if se_ms is not None else ""

    if is_best and not violated:
        formatted_mean = f"\\mathbf{{{mean_str}}}"
    else:
        formatted_mean = mean_str

    if include_se and se_ms is not None:
        cell_math = f"${formatted_mean} \\pm {se_str}$"
    else:
        cell_math = f"${formatted_mean}$"

    if violated:
        if use_badvalue:
            return f"\\badvalue{{{cell_math}\\rlap{{*}}}}"
        else:
            return f"{cell_math}\\rlap{{$^*$}}"

    return cell_math


def format_slack_value(
    val: float | None,
    precision: int = 3,
    use_badvalue: bool = True,
    show_sign: bool = True,
) -> str:
    """Formats a constraint slack value with optional sign and badvalue highlight for violations."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    fmt = f"{val:+.{precision}f}" if show_sign else f"{val:.{precision}f}"
    if val < 0.0:
        if use_badvalue:
            return rf"\badvalue{{${fmt}$}}"
        else:
            return f"${fmt}$"
    return f"${fmt}$"


def get_audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe for audit figures and slack tables, using ScaleDoc Native as 'ScaleDoc'."""
    sub = df.copy()
    if "scaledoc_native" in sub["method"].values:
        sub = sub[sub["method"] != "scaledoc"].copy()
        if "method_clean" in sub.columns:
            sub.loc[sub["method"] == "scaledoc_native", "method_clean"] = "ScaleDoc"
        else:
            sub.loc[sub["method"] == "scaledoc_native", "method"] = "scaledoc"
    elif "method_clean" not in sub.columns and "method" in sub.columns:
        sub["method_clean"] = sub["method"].map(METHOD_DISPLAY_NAMES).fillna(sub["method"])
    return sub


def generate_full_width_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    include_se: bool = True,
    precision: int = 1,
) -> str:
    """
    Generates a full-width LaTeX table (table*) containing all deltas dynamically
    side-by-side.
    """
    scenario_uids = get_ordered_scenarios(df)
    deltas = get_ordered_deltas(df)

    header_label, dataset_desc = "Benchmark", "benchmarks"

    # Dynamic column specification
    col_spec = "ll c " + " ".join(["c" * len(focus_methods) for _ in deltas])

    delta_header_cells = []
    for d in deltas:
        d_math = f"$\\delta = {d:.2f}$"
        delta_header_cells.append(
            f"\\multicolumn{{{len(focus_methods)}}}{{c}}{{{d_math}}}"
        )
    delta_headers = " & ".join(delta_header_cells)

    cmidrule_parts = []
    start_col = 4
    for _ in deltas:
        end_col = start_col + len(focus_methods) - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    cmidrules = " ".join(cmidrule_parts)

    sub_method_headers = " & ".join(
        [label_m for _ in deltas for _, label_m in focus_methods]
    )

    label_suffix = "_mean_only" if not include_se else ""
    tabcol_cmd = (
        r"\setlength{\tabcolsep}{3.0pt}"
        if include_se
        else r"\setlength{\tabcolsep}{4.5pt}"
    )
    font_cmd = (
        r"\footnotesize"
        if len(focus_methods) * len(deltas) > 8 and include_se
        else r"\small"
    )

    caption_text = (
        f"Average Oracle Call Rate (\\%) across {dataset_desc} under varying "
        r"target guarantees $\gamma$ and error budgets $\delta$. "
        "For ACTIS, we report the adaptive importance sampling variant with "
        "asymptotic confidence sequences (ACTIS-IS-Asymptotic). "
        "'Ideal' represents the theoretical oracle lower bound. "
        r"Asterisks ($^*$) indicate that the algorithm is certified unsound "
        r"($LCB > \delta$ via exact 95\% Clopper-Pearson CI). "
        "Bold values denote the lowest oracle call rate among methods "
        "satisfying the target guarantee (including ties)."
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        font_cmd,
        tabcol_cmd,
        f"\\caption{{{caption_text}}}",
        f"\\label{{tab:oracle_call_rates_full{label_suffix}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & & & {delta_headers} \\\\",
        f"{cmidrules}",
        f"{header_label} & $\\gamma$ & Ideal & {sub_method_headers} \\\\",
        r"\midrule",
    ]

    for sc_idx, sc_uid in enumerate(scenario_uids):
        df_sc = df[df["scenario_uid"] == sc_uid]
        if df_sc.empty:
            continue

        base_sc = df_sc["scenario"].iloc[0]
        multiple_configs = (
            len(df[df["scenario"] == base_sc]["scenario_uid"].unique()) > 1
        )
        row_qid = (
            df_sc["query_id"].iloc[0]
            if "query_id" in df_sc.columns and pd.notna(df_sc["query_id"].iloc[0])
            else None
        )
        row_proxy = (
            df_sc["proxy_model"].iloc[0]
            if "proxy_model" in df_sc.columns and pd.notna(df_sc["proxy_model"].iloc[0])
            else None
        )
        sc_display = format_scenario_display_name(
            sc_uid if multiple_configs else base_sc,
            query_id=row_qid,
            proxy_model=row_proxy,
            short=False,
        )
        sc_display = sc_display.replace("%", r"\%")

        sc_gammas = sorted(df_sc["gamma"].unique())
        for g_idx, g in enumerate(sc_gammas):
            df_g = df_sc[np.isclose(df_sc["gamma"], g)]
            if df_g.empty:
                continue

            ideal_val = df_g["ideal_oracle"].iloc[0]
            ideal_str = f"${ideal_val:.{precision}f}$"

            benchmark_col = sc_display if g_idx == 0 else ""
            gamma_col = f"${g:.2f}$"

            cell_strs = []
            for d in deltas:
                df_d = df_g[np.isclose(df_g["delta"], d)]

                valid_means: dict[str, float] = {}
                for raw_m, _ in focus_methods:
                    sub = df_d[
                        (df_d["method"] == raw_m) | (df_d["runner_uid"] == raw_m)
                    ]
                    if not sub.empty:
                        row = sub.iloc[0]
                        if not row["violated"]:
                            valid_means[raw_m] = float(
                                f"{row['mean_oracle']:.{precision}f}"
                            )

                if valid_means:
                    min_val = min(valid_means.values())
                    best_methods = {
                        m
                        for m, val in valid_means.items()
                        if math.isclose(val, min_val, abs_tol=1e-9)
                    }
                else:
                    best_methods = set()

                for raw_m, _ in focus_methods:
                    sub = df_d[
                        (df_d["method"] == raw_m) | (df_d["runner_uid"] == raw_m)
                    ]
                    if sub.empty:
                        cell_strs.append("--")
                    else:
                        row = sub.iloc[0]
                        is_best = raw_m in best_methods
                        cell_strs.append(
                            format_value(
                                row["mean_oracle"],
                                row["se_oracle"] if include_se else None,
                                row["violated"],
                                is_best,
                                include_se=include_se,
                                precision=precision,
                            )
                        )

            row_line = (
                f"{benchmark_col:<24} & {gamma_col:<8} & {ideal_str:<8} & "
                + " & ".join(cell_strs)
                + r" \\"
            )
            lines.append(row_line)

        if sc_idx < len(scenario_uids) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    return "\n".join(lines)


def generate_single_column_table(
    df: pd.DataFrame,
    delta_val: float,
    focus_methods: Sequence[tuple[str, str]],
    include_se: bool = True,
    precision: int = 1,
) -> str:
    """
    Generates a single-column LaTeX table (table) for a specific delta value.
    """
    scenario_uids = get_ordered_scenarios(df)

    header_label, dataset_desc = "Benchmark", "benchmarks"

    delta_label = f"{delta_val:.2f}"
    font_cmd = r"\footnotesize" if len(focus_methods) > 3 or include_se else r"\small"
    tabcol_cmd = (
        r"\setlength{\tabcolsep}{2.5pt}"
        if include_se
        else r"\setlength{\tabcolsep}{4.0pt}"
    )
    label_suffix = "_mean_only" if not include_se else ""

    col_spec = "ll c " + "c" * len(focus_methods)
    method_headers = " & ".join([label_m for _, label_m in focus_methods])

    caption_text = (
        f"Average Oracle Call Rate (\\%) across {dataset_desc} for "
        f"$\\delta = {delta_label}$. "
        "For ACTIS, we report the adaptive importance sampling asymptotic "
        "variant (ACTIS-IS-Asymptotic). "
        r"Asterisks ($^*$) denote certified unsoundness ($LCB > \delta$ "
        r"via exact 95\% Clopper-Pearson CI). "
        "Bold font indicates the lowest oracle call rate among valid methods "
        "satisfying guarantees (including ties)."
    )
    table_label = (
        f"tab:oracle_call_rates_delta_{delta_label.replace('.', '')}{label_suffix}"
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        font_cmd,
        tabcol_cmd,
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header_label} & $\\gamma$ & Ideal & {method_headers} \\\\",
        r"\midrule",
    ]

    for sc_idx, sc_uid in enumerate(scenario_uids):
        df_sc = df[(df["scenario_uid"] == sc_uid) & np.isclose(df["delta"], delta_val)]
        if df_sc.empty:
            continue

        base_sc = df_sc["scenario"].iloc[0]
        multiple_configs = (
            len(df[df["scenario"] == base_sc]["scenario_uid"].unique()) > 1
        )
        row_qid = (
            df_sc["query_id"].iloc[0]
            if "query_id" in df_sc.columns and pd.notna(df_sc["query_id"].iloc[0])
            else None
        )
        row_proxy = (
            df_sc["proxy_model"].iloc[0]
            if "proxy_model" in df_sc.columns and pd.notna(df_sc["proxy_model"].iloc[0])
            else None
        )
        sc_display = format_scenario_display_name(
            sc_uid if multiple_configs else base_sc,
            query_id=row_qid,
            proxy_model=row_proxy,
            short=True,
        )
        sc_display = sc_display.replace("%", r"\%")

        sc_gammas = sorted(df_sc["gamma"].unique())
        for g_idx, g in enumerate(sc_gammas):
            df_g = df_sc[np.isclose(df_sc["gamma"], g)]
            if df_g.empty:
                continue

            ideal_val = df_g["ideal_oracle"].iloc[0]
            ideal_str = f"${ideal_val:.{precision}f}$"

            benchmark_col = sc_display if g_idx == 0 else ""
            gamma_col = f"${g:.2f}$"

            valid_means: dict[str, float] = {}
            for raw_m, _ in focus_methods:
                sub = df_g[(df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)]
                if not sub.empty:
                    row = sub.iloc[0]
                    if not row["violated"]:
                        valid_means[raw_m] = float(
                            f"{row['mean_oracle']:.{precision}f}"
                        )

            if valid_means:
                min_val = min(valid_means.values())
                best_methods = {
                    m
                    for m, val in valid_means.items()
                    if math.isclose(val, min_val, abs_tol=1e-9)
                }
            else:
                best_methods = set()

            cell_strs = []
            for raw_m, _ in focus_methods:
                sub = df_g[(df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)]
                if sub.empty:
                    cell_strs.append("--")
                else:
                    row = sub.iloc[0]
                    is_best = raw_m in best_methods
                    cell_strs.append(
                        format_value(
                            row["mean_oracle"],
                            row["se_oracle"] if include_se else None,
                            row["violated"],
                            is_best,
                            include_se=include_se,
                            precision=precision,
                        )
                    )

            row_line = (
                f"{benchmark_col:<16} & {gamma_col:<8} & {ideal_str:<8} & "
                + " & ".join(cell_strs)
                + r" \\"
            )
            lines.append(row_line)

        if sc_idx < len(scenario_uids) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_appendix_table(
    df: pd.DataFrame,
    delta_val: float,
    appendix_methods: Sequence[tuple[str, str]],
    include_se: bool = True,
    precision: int = 1,
) -> str:
    """
    Generates an extended full-width table comparing all ACTIS variants and
    baselines for a specific delta.
    """
    scenario_uids = get_ordered_scenarios(df)

    header_label, dataset_desc = "Benchmark", "benchmarks"

    delta_label = f"{delta_val:.2f}"
    tabcol_cmd = (
        r"\setlength{\tabcolsep}{2.8pt}"
        if include_se
        else r"\setlength{\tabcolsep}{4.0pt}"
    )
    font_cmd = (
        r"\footnotesize" if len(appendix_methods) > 5 and include_se else r"\small"
    )
    label_suffix = "_mean_only" if not include_se else ""

    col_spec = "ll c " + "c" * len(appendix_methods)
    method_headers = " & ".join([label_m for _, label_m in appendix_methods])

    caption_text = (
        "Detailed comparison of all ACTIS variants and baselines "
        f"(Average Oracle Call Rate \\%) across {dataset_desc} for "
        f"$\\delta = {delta_label}$. "
        r"Asterisks ($^*$) denote certified unsoundness ($LCB > \delta$ "
        r"via exact 95\% Clopper-Pearson CI). "
        "Bold font indicates the lowest oracle call rate among valid methods "
        "satisfying guarantees (including ties)."
    )
    table_label = (
        f"tab:appendix_all_variants_delta_{delta_label.replace('.', '')}{label_suffix}"
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        font_cmd,
        tabcol_cmd,
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header_label} & $\\gamma$ & Ideal & {method_headers} \\\\",
        r"\midrule",
    ]

    for sc_idx, sc_uid in enumerate(scenario_uids):
        df_sc = df[(df["scenario_uid"] == sc_uid) & np.isclose(df["delta"], delta_val)]
        if df_sc.empty:
            continue

        base_sc = df_sc["scenario"].iloc[0]
        multiple_configs = (
            len(df[df["scenario"] == base_sc]["scenario_uid"].unique()) > 1
        )
        row_qid = (
            df_sc["query_id"].iloc[0]
            if "query_id" in df_sc.columns and pd.notna(df_sc["query_id"].iloc[0])
            else None
        )
        row_proxy = (
            df_sc["proxy_model"].iloc[0]
            if "proxy_model" in df_sc.columns and pd.notna(df_sc["proxy_model"].iloc[0])
            else None
        )
        sc_display = format_scenario_display_name(
            sc_uid if multiple_configs else base_sc,
            query_id=row_qid,
            proxy_model=row_proxy,
            short=False,
        )
        sc_display = sc_display.replace("%", r"\%")

        sc_gammas = sorted(df_sc["gamma"].unique())
        for g_idx, g in enumerate(sc_gammas):
            df_g = df_sc[np.isclose(df_sc["gamma"], g)]
            if df_g.empty:
                continue

            ideal_val = df_g["ideal_oracle"].iloc[0]
            ideal_str = f"${ideal_val:.{precision}f}$"

            benchmark_col = sc_display if g_idx == 0 else ""
            gamma_col = f"${g:.2f}$"

            valid_means: dict[str, float] = {}
            for raw_m, _ in appendix_methods:
                sub = df_g[(df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)]
                if not sub.empty:
                    row = sub.iloc[0]
                    if not row["violated"]:
                        valid_means[raw_m] = float(
                            f"{row['mean_oracle']:.{precision}f}"
                        )

            if valid_means:
                min_val = min(valid_means.values())
                best_methods = {
                    m
                    for m, val in valid_means.items()
                    if math.isclose(val, min_val, abs_tol=1e-9)
                }
            else:
                best_methods = set()

            cell_strs = []
            for raw_m, _ in appendix_methods:
                sub = df_g[(df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)]
                if sub.empty:
                    cell_strs.append("--")
                else:
                    row = sub.iloc[0]
                    is_best = raw_m in best_methods
                    cell_strs.append(
                        format_value(
                            row["mean_oracle"],
                            row["se_oracle"] if include_se else None,
                            row["violated"],
                            is_best,
                            include_se=include_se,
                            precision=precision,
                        )
                    )

            row_line = (
                f"{benchmark_col:<24} & {gamma_col:<8} & {ideal_str:<8} & "
                + " & ".join(cell_strs)
                + r" \\"
            )
            lines.append(row_line)

        if sc_idx < len(scenario_uids) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    return "\n".join(lines)


def generate_representative_oracle_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    representative_workloads: Optional[Sequence[tuple[Any, ...]]] = None,
    delta_val: float = 0.05,
    gamma_val: float = 0.95,
    include_se: bool = True,
    precision: int = 1,
) -> str:
    """
    Generates a publication-ready single-column LaTeX table presenting
    oracle call rates across representative workloads (summary version of single-col table)
    for ACM SIGMOD format.
    """
    if representative_workloads is None:
        workloads = DEFAULT_REPRESENTATIVE_WORKLOADS
    else:
        workloads = representative_workloads

    col_spec = "ll r " + "r" * len(focus_methods)
    method_headers = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )

    caption_text = (
        f"Average Oracle Call Rate (\\%) across representative workloads "
        f"($\\delta = {delta_val:.2f}$, target $\\gamma = {gamma_val:.2f}$). "
        r"Asterisks ($^*$) denote certified unsoundness ($LCB > \delta$ via exact 95\% Clopper-Pearson CI). "
        "Bold font indicates the lowest oracle call rate among valid methods satisfying guarantees (including ties)."
    )
    label_suffix = "_mean_only" if not include_se else ""
    table_label = f"tab:oracle_call_rates_representative{label_suffix}"

    tabcol_cmd = (
        r"\setlength{\tabcolsep}{2.8pt}"
        if include_se
        else r"\setlength{\tabcolsep}{3.5pt}"
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        tabcol_cmd,
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\multicolumn{2}{c}{\textbf{Workload}} & & \multicolumn{4}{c}{\textbf{Oracle Call Rate (\%)}} \\",
        r"\cmidrule(lr){1-2} \cmidrule(lr){4-7}",
        f"\\textbf{{Name}} & \\textbf{{Proxy}} & \\textbf{{Ideal}} & {method_headers} \\\\",
        r"\midrule",
    ]

    for item in workloads:
        rep_scen, rep_qid, rep_proxy = item[0], item[1], item[2]
        if len(item) >= 5:
            disp_name, disp_proxy = item[3], item[4]
        else:
            disp_name, disp_proxy = get_representative_display(
                rep_scen, rep_qid, rep_proxy
            )
        cond = (
            (df["dataset"] == rep_scen)
            & np.isclose(df["delta"], delta_val)
            & np.isclose(df["gamma"], gamma_val)
        )
        if rep_qid is not None:
            clean_rep_qid = str(rep_qid).lower().lstrip("q")
            cond = cond & (
                df["query_id"].astype(str).str.lower().str.lstrip("q")
                == clean_rep_qid
            )
        if rep_proxy is not None:
            cond = cond & (df["proxy_model"] == rep_proxy)

        df_row = df[cond]

        ideal_val = df_row["ideal_oracle"].iloc[0] if not df_row.empty else 0.0
        ideal_str = f"${ideal_val:.{precision}f}$"

        valid_means: dict[str, float] = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_means[raw_m] = float(
                        f"{r0['mean_oracle']:.{precision}f}"
                    )

        best_methods = set()
        if valid_means:
            min_val = min(valid_means.values())
            best_methods = {
                m
                for m, val in valid_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs.append(
                    format_value(
                        r0["mean_oracle"],
                        r0["se_oracle"] if include_se else None,
                        r0["violated"],
                        raw_m in best_methods,
                        include_se=include_se,
                        precision=precision,
                    )
                )

        lines.append(
            f"{disp_name} & {disp_proxy} & {ideal_str} & "
            + " & ".join(cell_strs)
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_latency_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    representative_workloads: Optional[Sequence[tuple[Any, ...]]] = None,
    delta_val: float = 0.05,
    gamma_val: float = 0.95,
    results_batch_size_dir: Optional[Path | str] = "experiments/results_batch_size",
    include_se: bool = False,
    precision: int = 1,
) -> str:
    """
    Generates a publication-ready single-column LaTeX table (table) comparing
    tuner wall-clock execution time (in ms) across representative workloads
    for ACM SIGMOD format.
    """
    if representative_workloads is None:
        workloads = DEFAULT_REPRESENTATIVE_WORKLOADS
    else:
        workloads = representative_workloads

    col_spec = "ll r " + "r" * len(focus_methods)
    method_headers = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )

    caption_text = (
        f"Tuner wall-clock execution time (ms) across representative workloads "
        f"($\\delta = {delta_val:.2f}$, target $\\gamma = {gamma_val:.2f}$ unless noted). "
        "Bold indicates the fastest method satisfying accuracy guarantees. "
        r"Asterisks ($^*$) denote certified unsoundness ($LCB > \delta$ via exact 95\% Clopper-Pearson CI)."
    )
    label_suffix = "_mean_only" if not include_se else ""
    table_label = f"tab:tuning_latency{label_suffix}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\multicolumn{2}{c}{\textbf{Workload}} & & \multicolumn{4}{c}{\textbf{Tuner Latency (ms)}} \\",
        r"\cmidrule(lr){1-2} \cmidrule(lr){4-7}",
        f"\\textbf{{Name}} & \\textbf{{Proxy}} & \\textbf{{$N$}} & {method_headers} \\\\",
        r"\midrule",
    ]

    # Optionally load batch size sweep for Jackson
    jackson_b1000: dict[float, float] = {}
    if results_batch_size_dir:
        b_path = Path(results_batch_size_dir)
        if b_path.exists():
            import glob
            import json

            for p in glob.glob(
                str(
                    b_path
                    / "**"
                    / "jackson"
                    / "**"
                    / f"delta_{delta_val}_gamma_*_trials_1000.json"
                ),
                recursive=True,
            ):
                try:
                    with open(p, "r") as f:
                        d = json.load(f)
                    if d.get("runner", {}).get("batch_size") == 1000:
                        g = float(d.get("gamma_R", 0.8))
                        jackson_b1000[g] = float(
                            d.get("runtime", {}).get("mean", 0.0)
                        )
                except Exception:
                    pass

    def fmt_n(n: int) -> str:
        if n >= 1000000:
            return f"{n/1000000:.1f}M"
        elif n >= 1000:
            return f"{n/1000:.1f}k"
        return str(n)

    for item in workloads:
        rep_scen, rep_qid, rep_proxy = item[0], item[1], item[2]
        if len(item) >= 5:
            disp_name, disp_proxy = item[3], item[4]
        else:
            disp_name, disp_proxy = get_representative_display(
                rep_scen, rep_qid, rep_proxy
            )

        cond = (
            (df["dataset"] == rep_scen)
            & np.isclose(df["delta"], delta_val)
            & np.isclose(df["gamma"], gamma_val)
        )
        if rep_qid is not None:
            clean_rep_qid = str(rep_qid).lower().lstrip("q")
            cond = cond & (
                df["query_id"].astype(str).str.lower().str.lstrip("q")
                == clean_rep_qid
            )
        if rep_proxy is not None:
            cond = cond & (df["proxy_model"] == rep_proxy)

        df_row = df[cond]

        valid_means: dict[str, float] = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_means[raw_m] = float(
                        f"{r0['mean_runtime'] * 1000.0:.{precision}f}"
                    )

        best_methods = set()
        if valid_means:
            min_val = min(valid_means.values())
            best_methods = {
                m
                for m, val in valid_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs.append(
                    format_latency_value(
                        r0["mean_runtime"],
                        r0["se_runtime"] if include_se else None,
                        r0["violated"],
                        raw_m in best_methods,
                        include_se=include_se,
                        precision=precision,
                    )
                )

        if "pop_size" in df_row.columns and df_row["pop_size"].notna().any():
            pop_size_val = int(df_row["pop_size"].dropna().iloc[0])
            n_str = fmt_n(pop_size_val)
        elif len(item) >= 6 and item[5] is not None:
            n_str = fmt_n(int(item[5]))
        else:
            n_str = "--"

        lines.append(
            f"{disp_name} & {disp_proxy} & {n_str} & "
            + " & ".join(cell_strs)
            + r" \\"
        )

        if rep_scen == "jackson":
            df_j90 = df[
                (df["dataset"] == "jackson")
                & np.isclose(df["delta"], delta_val)
                & np.isclose(df["gamma"], 0.90)
            ]
            if not df_j90.empty:
                v_means_90: dict[str, float] = {}
                for raw_m, _ in focus_methods:
                    sub = df_j90[
                        (df_j90["method"] == raw_m)
                        | (df_j90["runner_uid"] == raw_m)
                    ]
                    if not sub.empty:
                        r0 = sub.iloc[0]
                        if not r0["violated"]:
                            v_means_90[raw_m] = float(
                                f"{r0['mean_runtime'] * 1000.0:.{precision}f}"
                            )
                best_90 = set()
                if v_means_90:
                    min_90 = min(v_means_90.values())
                    best_90 = {
                        m
                        for m, val in v_means_90.items()
                        if math.isclose(val, min_90, abs_tol=1e-9)
                    }

                cell_j90 = []
                for raw_m, _ in focus_methods:
                    sub = df_j90[
                        (df_j90["method"] == raw_m)
                        | (df_j90["runner_uid"] == raw_m)
                    ]
                    if sub.empty:
                        cell_j90.append("--")
                    else:
                        r0 = sub.iloc[0]
                        cell_j90.append(
                            format_latency_value(
                                r0["mean_runtime"],
                                r0["se_runtime"] if include_se else None,
                                r0["violated"],
                                raw_m in best_90,
                                include_se=include_se,
                                precision=precision,
                            )
                        )
                lines.append(
                    r"\quad\emph{at $\gamma{=}0.90$} & -- & -- & "
                    + " & ".join(cell_j90)
                    + r" \\"
                )

            if 0.95 in jackson_b1000:
                b_ms = jackson_b1000[0.95] * 1000.0
                b_str = f"\\mathbf{{{b_ms:.{precision}f}}}"
                empty_rest = " & ".join(["--"] * (len(focus_methods) - 1))
                lines.append(
                    f"\\quad\\emph{{w/ $B{{=}}1000$}} & -- & -- & {b_str} & {empty_rest} \\\\"
                )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_appendix_latency_table(
    df: pd.DataFrame,
    delta_val: float,
    appendix_methods: Sequence[tuple[str, str]],
    include_se: bool = True,
    precision: int = 1,
) -> str:
    """
    Generates an extended full-width table comparing all ACTIS variants and
    baselines for tuner latency (in ms) across all benchmarks for a specific delta.
    """
    scenario_uids = get_ordered_scenarios(df)
    header_label, dataset_desc = "Benchmark", "benchmarks"

    delta_label = f"{delta_val:.2f}"
    tabcol_cmd = (
        r"\setlength{\tabcolsep}{3.5pt}"
        if include_se
        else r"\setlength{\tabcolsep}{4.5pt}"
    )
    font_cmd = (
        r"\footnotesize" if len(appendix_methods) > 5 and include_se else r"\small"
    )
    label_suffix = "_mean_only" if not include_se else ""

    col_spec = "ll " + "c" * len(appendix_methods)
    method_headers = " & ".join([label_m for _, label_m in appendix_methods])

    caption_text = (
        "Detailed comparison of all ACTIS variants and baselines "
        f"(Average Tuner Latency in ms) across {dataset_desc} for "
        f"$\\delta = {delta_label}$. "
        r"Asterisks ($^*$) denote certified unsoundness ($LCB > \delta$ "
        r"via exact 95\% Clopper-Pearson CI on target accuracy). "
        "Bold font indicates the fastest tuning time among valid methods "
        "satisfying guarantees (including ties)."
    )
    is_focus = len(appendix_methods) <= 4
    table_type_str = "latency_focus" if is_focus else "latency_all_variants"
    table_label = (
        f"tab:appendix_{table_type_str}_delta_{delta_label.replace('.', '')}{label_suffix}"
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        font_cmd,
        tabcol_cmd,
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header_label} & $\\gamma$ & {method_headers} \\\\",
        r"\midrule",
    ]

    for sc_idx, sc_uid in enumerate(scenario_uids):
        df_sc = df[
            (df["scenario_uid"] == sc_uid) & np.isclose(df["delta"], delta_val)
        ]
        if df_sc.empty:
            continue

        base_sc = df_sc["scenario"].iloc[0]
        multiple_configs = (
            len(df[df["scenario"] == base_sc]["scenario_uid"].unique()) > 1
        )
        row_qid = (
            df_sc["query_id"].iloc[0]
            if "query_id" in df_sc.columns and pd.notna(df_sc["query_id"].iloc[0])
            else None
        )
        row_proxy = (
            df_sc["proxy_model"].iloc[0]
            if "proxy_model" in df_sc.columns
            and pd.notna(df_sc["proxy_model"].iloc[0])
            else None
        )
        sc_display = format_scenario_display_name(
            sc_uid if multiple_configs else base_sc,
            query_id=row_qid,
            proxy_model=row_proxy,
            short=False,
        )
        sc_display = sc_display.replace("%", r"\%")

        sc_gammas = sorted(df_sc["gamma"].unique())
        for g_idx, g in enumerate(sc_gammas):
            df_g = df_sc[np.isclose(df_sc["gamma"], g)]
            if df_g.empty:
                continue

            benchmark_col = sc_display if g_idx == 0 else ""
            gamma_col = f"${g:.2f}$"

            valid_means: dict[str, float] = {}
            for raw_m, _ in appendix_methods:
                sub = df_g[
                    (df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)
                ]
                if not sub.empty:
                    row = sub.iloc[0]
                    if not row["violated"]:
                        valid_means[raw_m] = float(
                            f"{row['mean_runtime'] * 1000.0:.{precision}f}"
                        )

            if valid_means:
                min_val = min(valid_means.values())
                best_methods = {
                    m
                    for m, val in valid_means.items()
                    if math.isclose(val, min_val, abs_tol=1e-9)
                }
            else:
                best_methods = set()

            cell_strs = []
            for raw_m, _ in appendix_methods:
                sub = df_g[
                    (df_g["method"] == raw_m) | (df_g["runner_uid"] == raw_m)
                ]
                if sub.empty:
                    cell_strs.append("--")
                else:
                    row = sub.iloc[0]
                    is_best = raw_m in best_methods
                    cell_strs.append(
                        format_latency_value(
                            row["mean_runtime"],
                            row["se_runtime"] if include_se else None,
                            row["violated"],
                            is_best,
                            include_se=include_se,
                            precision=precision,
                        )
                    )

            row_line = (
                f"{benchmark_col:<24} & {gamma_col:<8} & "
                + " & ".join(cell_strs)
                + r" \\"
            )
            lines.append(row_line)

        if sc_idx < len(scenario_uids) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    return "\n".join(lines)


def generate_representative_joint_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    representative_workloads: Optional[Sequence[tuple[Any, ...]]] = None,
    delta_val: float = 0.05,
    gamma_val: float = 0.95,
    include_se: bool = False,
    precision: int = 1,
    use_badvalue: bool = True,
) -> str:
    """
    Generates a publication-ready full-width LaTeX table (table*) presenting
    Oracle Call Rate (%) and Tuner Latency (ms) side-by-side across representative
    workloads for ACM SIGMOD format (matching main.tex:L1092-L1121).
    """
    if representative_workloads is None:
        workloads = DEFAULT_REPRESENTATIVE_WORKLOADS
    else:
        workloads = representative_workloads

    num_methods = len(focus_methods)
    ocr_cols = 1 + num_methods
    lat_cols = num_methods
    col_spec = f"llr {'r' * ocr_cols} {'r' * lat_cols}"

    ocr_start = 4
    ocr_end = 3 + ocr_cols
    lat_start = ocr_end + 1
    lat_end = lat_start + lat_cols - 1

    method_headers_ocr = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )
    method_headers_lat = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )

    caption_text = (
        f"Oracle call rate (\\%) and tuner latency (ms) on representative benchmarks "
        f"for precision\\slash recall target $\\gamma = {gamma_val:.2f}$ and "
        f"error budget $\\delta = {delta_val:.2f}$. Asterisks indicate certified unsoundness. "
        "The best result among statistically sound methods on each benchmark is highlighted in bold."
    )
    label_suffix = "_mean_only" if not include_se else ""
    table_label = f"tab:representative_performance{label_suffix}"

    lines = [
        r"\begin{table*}",
        r"\centering",
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\label{{tab:oracle_call_rates_representative{label_suffix}}}",
        f"\\label{{tab:tuning_latency_representative{label_suffix}}}",
        r"\vspace{-1em}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.8pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{Benchmark}}}} & & \\multicolumn{{{ocr_cols}}}{{c}}{{\\textbf{{Oracle Call Rate (\\%)}}}} & \\multicolumn{{{lat_cols}}}{{c}}{{\\textbf{{Tuner Latency (ms)}}}} \\\\",
        f"\\cmidrule(lr){{1-2}} \\cmidrule(lr){{{ocr_start}-{ocr_end}}} \\cmidrule(lr){{{lat_start}-{lat_end}}}",
        f"\\textbf{{Workload}} & \\textbf{{Proxy}} & \\textbf{{$N$}} & \\textbf{{Ideal}} & {method_headers_ocr} & {method_headers_lat} \\\\",
        r"\midrule",
    ]

    for item in workloads:
        rep_scen, rep_qid, rep_proxy = item[0], item[1], item[2]
        if len(item) >= 5:
            disp_name, disp_proxy = item[3], item[4]
        else:
            disp_name, disp_proxy = get_representative_display(
                rep_scen, rep_qid, rep_proxy
            )

        cond = (
            (df["dataset"] == rep_scen)
            & np.isclose(df["delta"], delta_val)
            & np.isclose(df["gamma"], gamma_val)
        )
        if rep_qid is not None:
            clean_rep_qid = str(rep_qid).lower().lstrip("q")
            cond = cond & (
                df["query_id"].astype(str).str.lower().str.lstrip("q")
                == clean_rep_qid
            )
        if rep_proxy is not None:
            cond = cond & (df["proxy_model"] == rep_proxy)

        df_row = df[cond]

        if "pop_size" in df_row.columns and df_row["pop_size"].notna().any():
            pop_size_val = df_row["pop_size"].dropna().iloc[0]
            n_str = format_population_size(pop_size_val)
        elif len(item) >= 6 and item[5] is not None:
            n_str = format_population_size(item[5])
        else:
            n_str = "--"

        ideal_val = df_row["ideal_oracle"].iloc[0] if not df_row.empty else 0.0
        ideal_str = f"${ideal_val:.{precision}f}$"

        # OCR: Find best (lowest) among statistically valid methods
        valid_ocr_means: dict[str, float] = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_ocr_means[raw_m] = float(
                        f"{r0['mean_oracle']:.{precision}f}"
                    )

        best_ocr_methods = set()
        if valid_ocr_means:
            min_val = min(valid_ocr_means.values())
            best_ocr_methods = {
                m
                for m, val in valid_ocr_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs_ocr = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs_ocr.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs_ocr.append(
                    format_value(
                        r0["mean_oracle"],
                        r0["se_oracle"] if include_se else None,
                        r0["violated"],
                        raw_m in best_ocr_methods,
                        include_se=include_se,
                        precision=precision,
                        use_badvalue=use_badvalue,
                    )
                )

        # Latency: Find best (fastest) among statistically valid methods
        valid_lat_means: dict[str, float] = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_lat_means[raw_m] = float(
                        f"{r0['mean_runtime'] * 1000.0:.{precision}f}"
                    )

        best_lat_methods = set()
        if valid_lat_means:
            min_val = min(valid_lat_means.values())
            best_lat_methods = {
                m
                for m, val in valid_lat_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs_lat = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs_lat.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs_lat.append(
                    format_latency_value(
                        r0["mean_runtime"],
                        r0["se_runtime"] if include_se else None,
                        r0["violated"],
                        raw_m in best_lat_methods,
                        include_se=include_se,
                        precision=precision,
                        use_badvalue=use_badvalue,
                    )
                )

        lines.append(
            f"{disp_name} & {disp_proxy} & {n_str} & {ideal_str} & "
            + " & ".join(cell_strs_ocr)
            + " & "
            + " & ".join(cell_strs_lat)
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_all_benchmarks_joint_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    delta_val: float = 0.05,
    gamma_val: float = 0.95,
    include_se: bool = False,
    precision: int = 1,
    use_badvalue: bool = True,
) -> str:
    """
    Generates a publication-ready full-width LaTeX table (table*) presenting
    Oracle Call Rate (%) and Tuner Latency (ms) side-by-side across all benchmarks
    for a given gamma and delta, matching the structure in main.tex:L1092-L1121.
    """
    benchmark_uids = get_ordered_scenarios(df)

    num_methods = len(focus_methods)
    ocr_cols = 1 + num_methods
    lat_cols = num_methods
    col_spec = f"llr {'r' * ocr_cols} {'r' * lat_cols}"

    ocr_start = 4
    ocr_end = 3 + ocr_cols
    lat_start = ocr_end + 1
    lat_end = lat_start + lat_cols - 1

    method_headers_ocr = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )
    method_headers_lat = " & ".join(
        [f"\\textbf{{{label_m}}}" for _, label_m in focus_methods]
    )

    caption_text = (
        f"Oracle call rate (\\%) and tuner latency (ms) across all benchmarks "
        f"for precision\\slash recall target $\\gamma = {gamma_val:.2f}$ and "
        f"error budget $\\delta = {delta_val:.2f}$. Asterisks indicate certified unsoundness. "
        "The best result among statistically sound methods on each benchmark is highlighted in bold."
    )
    label_suffix = "_mean_only" if not include_se else ""
    table_label = f"tab:all_benchmarks_performance{label_suffix}"

    lines = [
        r"\begin{table*}",
        r"\centering",
        f"\\caption{{{caption_text}}}",
        f"\\label{{{table_label}}}",
        f"\\label{{tab:oracle_call_rates_all_benchmarks{label_suffix}}}",
        f"\\label{{tab:tuning_latency_all_benchmarks{label_suffix}}}",
        r"\vspace{-1em}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.8pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{Benchmark}}}} & & \\multicolumn{{{ocr_cols}}}{{c}}{{\\textbf{{Oracle Call Rate (\\%)}}}} & \\multicolumn{{{lat_cols}}}{{c}}{{\\textbf{{Tuner Latency (ms)}}}} \\\\",
        f"\\cmidrule(lr){{1-2}} \\cmidrule(lr){{{ocr_start}-{ocr_end}}} \\cmidrule(lr){{{lat_start}-{lat_end}}}",
        f"\\textbf{{Workload}} & \\textbf{{Proxy}} & \\textbf{{$N$}} & \\textbf{{Ideal}} & {method_headers_ocr} & {method_headers_lat} \\\\",
        r"\midrule",
    ]

    prev_group = None
    for uid in benchmark_uids:
        df_row = df[
            (df["scenario_uid"] == uid)
            & np.isclose(df["delta"], delta_val)
            & np.isclose(df["gamma"], gamma_val)
        ]
        if df_row.empty:
            continue

        raw_dataset = str(
            df_row["dataset"].iloc[0]
            if "dataset" in df_row.columns
            else df_row["scenario"].iloc[0]
        )
        raw_qid = (
            df_row["query_id"].iloc[0]
            if "query_id" in df_row.columns and pd.notna(df_row["query_id"].iloc[0])
            else None
        )
        proxy_model = (
            df_row["proxy_model"].iloc[0]
            if "proxy_model" in df_row.columns
            and pd.notna(df_row["proxy_model"].iloc[0])
            else None
        )

        clean_qid = (
            str(raw_qid).lower().lstrip("q") if raw_qid is not None else None
        )
        curr_group = (raw_dataset, clean_qid)

        if prev_group is not None and curr_group != prev_group:
            lines.append(r"\addlinespace")
        prev_group = curr_group

        disp_name = format_scenario_display_name(
            raw_dataset, query_id=raw_qid, short=True
        )
        if "_ext" in disp_name:
            disp_name = disp_name.replace("_ext", r"\textsubscript{ext}")

        if proxy_model is not None:
            disp_proxy = (
                format_proxy_display_name(str(proxy_model)) or str(proxy_model)
            )
        else:
            disp_proxy = DATASET_DEFAULT_PROXIES.get(raw_dataset, "--")

        if "pop_size" in df_row.columns and df_row["pop_size"].notna().any():
            pop_size_val = df_row["pop_size"].dropna().iloc[0]
            n_str = format_population_size(pop_size_val)
        else:
            n_str = "--"

        ideal_val = df_row["ideal_oracle"].iloc[0] if not df_row.empty else 0.0
        ideal_str = f"${ideal_val:.{precision}f}$"

        # OCR: Find best (lowest) among statistically valid methods
        valid_ocr_means = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_ocr_means[raw_m] = float(
                        f"{r0['mean_oracle']:.{precision}f}"
                    )

        best_ocr_methods = set()
        if valid_ocr_means:
            min_val = min(valid_ocr_means.values())
            best_ocr_methods = {
                m
                for m, val in valid_ocr_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs_ocr = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs_ocr.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs_ocr.append(
                    format_value(
                        r0["mean_oracle"],
                        r0["se_oracle"] if include_se else None,
                        r0["violated"],
                        raw_m in best_ocr_methods,
                        include_se=include_se,
                        precision=precision,
                        use_badvalue=use_badvalue,
                    )
                )

        # Latency: Find best (fastest) among statistically valid methods
        valid_lat_means = {}
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if not sub.empty:
                r0 = sub.iloc[0]
                if not r0["violated"]:
                    valid_lat_means[raw_m] = float(
                        f"{r0['mean_runtime'] * 1000.0:.{precision}f}"
                    )

        best_lat_methods = set()
        if valid_lat_means:
            min_val = min(valid_lat_means.values())
            best_lat_methods = {
                m
                for m, val in valid_lat_means.items()
                if math.isclose(val, min_val, abs_tol=1e-9)
            }

        cell_strs_lat = []
        for raw_m, _ in focus_methods:
            sub = df_row[
                (df_row["method"] == raw_m) | (df_row["runner_uid"] == raw_m)
            ]
            if sub.empty:
                cell_strs_lat.append("--")
            else:
                r0 = sub.iloc[0]
                cell_strs_lat.append(
                    format_latency_value(
                        r0["mean_runtime"],
                        r0["se_runtime"] if include_se else None,
                        r0["violated"],
                        raw_m in best_lat_methods,
                        include_se=include_se,
                        precision=precision,
                        use_badvalue=use_badvalue,
                    )
                )

        lines.append(
            f"{disp_name} & {disp_proxy} & {n_str} & {ideal_str} & "
            + " & ".join(cell_strs_ocr)
            + " & "
            + " & ".join(cell_strs_lat)
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_joint_table(
    df: pd.DataFrame,
    focus_methods: Sequence[tuple[str, str]],
    scope: str = "representative",
    representative_workloads: Optional[Sequence[tuple[Any, ...]]] = None,
    delta_val: float = 0.05,
    gamma_val: float = 0.95,
    include_se: bool = False,
    precision: int = 1,
    use_badvalue: bool = True,
) -> str:
    """
    Generates a joint LaTeX table presenting both Oracle Call Rate (%) and
    Tuner Latency (ms) side-by-side for a given gamma and delta.

    Parameters
    ----------
    df : pd.DataFrame
        Hierarchical experiment results.
    focus_methods : Sequence[tuple[str, str]]
        List of (runner_name, display_label) pairs.
    scope : str
        Either 'representative' for key representative workloads or 'all' for all benchmarks.
    representative_workloads : Optional[Sequence[tuple[Any, ...]]]
        Optional custom representative workloads tuple list.
    delta_val : float
        Error budget delta (default: 0.05).
    gamma_val : float
        Target guarantee gamma (default: 0.95).
    include_se : bool
        Whether to report standard errors alongside means (default: False).
    precision : int
        Number of decimal digits (default: 1).
    use_badvalue : bool
        Whether to format unsound results with \\badvalue{...} (default: True).
    """
    if scope == "representative":
        return generate_representative_joint_table(
            df=df,
            focus_methods=focus_methods,
            representative_workloads=representative_workloads,
            delta_val=delta_val,
            gamma_val=gamma_val,
            include_se=include_se,
            precision=precision,
            use_badvalue=use_badvalue,
        )
    elif scope in ("all", "all_benchmarks"):
        return generate_all_benchmarks_joint_table(
            df=df,
            focus_methods=focus_methods,
            delta_val=delta_val,
            gamma_val=gamma_val,
            include_se=include_se,
            precision=precision,
            use_badvalue=use_badvalue,
        )
    else:
        raise ValueError(f"Unknown scope '{scope}'. Must be 'representative' or 'all'.")


def generate_slack_distribution_table(
    df: pd.DataFrame,
    delta_val: float = 0.05,
    methods: Sequence[str | tuple[str, str]] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    gammas: Optional[Sequence[float]] = None,
    layout: str = "detailed",
    use_badvalue: bool = True,
    precision: int = 3,
) -> str:
    """
    Generates a LaTeX table presenting the empirical distribution of constraint slack
    across benchmarks for each method vs accuracy target gamma at a given delta.
    Corresponds to `plot_slack_distribution_vs_gamma`.

    Parameters
    ----------
    df : pd.DataFrame
        Hierarchical experiment results (from `load_table_data` or `load_tradeoff_data`).
    delta_val : float
        Nominal error budget delta (default: 0.05).
    methods : Sequence[str | tuple[str, str]]
        Methods to include. Can be method display names or (raw_runner, display_label) tuples.
    gammas : Optional[Sequence[float]]
        Accuracy targets to include (default: all unique gammas sorted).
    layout : str
        'detailed' (reports Median, IQR [Q1, Q3], Min/Worst, Max, Violations %) or
        'compact' (reports Median [Worst-Case Min] in a wide format).
    use_badvalue : bool
        Whether to highlight negative slack values and violation rates via \\badvalue{...}.
    precision : int
        Number of decimal digits (default: 3).
    """
    df_audit = get_audit_dataframe(df)
    sub = df_audit[np.isclose(df_audit["delta"], delta_val)].copy()

    if gammas is None:
        gammas = get_ordered_gammas(sub)

    resolved_methods: list[tuple[str, str]] = []
    for m in methods:
        if isinstance(m, tuple):
            resolved_methods.append(m)
        else:
            resolved_methods.append((m, m))

    delta_str = f"{delta_val:.2f}"
    d_tag = delta_str.replace(".", "")

    if layout == "compact":
        caption = (
            f"Constraint slack distribution across 42 real-world benchmarks for "
            f"$\\delta = {delta_str}$, reporting $\\mathrm{{Median}}$ [Worst-Case $\\mathrm{{Min}}$]. "
            r"Slack is defined as $\min\{\Precision - \gamma, \Recall - \gamma\}$ for ACTIS, BARGAIN, and LOTUS; "
            r"and $F_1 - \gamma$ for ScaleDoc. Negative values denote target guarantee violations and are "
            r"highlighted in maroon (\badvalue{Maroon})."
        )
        col_spec = "l " + "c" * len(resolved_methods)
        headers = [lbl for _, lbl in resolved_methods]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5.0pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:slack_distribution_compact_delta_{d_tag}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            r"$\gamma$ & " + " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        for g in gammas:
            sub_g = sub[np.isclose(sub["gamma"], g)]
            cell_strs = []
            for raw_m, label_m in resolved_methods:
                df_m = sub_g[
                    (sub_g["method_clean"] == label_m)
                    | (sub_g["method"] == raw_m)
                    | (sub_g["runner_uid"] == raw_m)
                ]
                vals = df_m["quantile_slack"].dropna()
                if vals.empty:
                    cell_strs.append("--")
                else:
                    med = float(vals.median())
                    vmin = float(vals.min())
                    med_str = format_slack_value(med, precision=precision, use_badvalue=use_badvalue)
                    min_str = format_slack_value(vmin, precision=precision, use_badvalue=use_badvalue)
                    cell_strs.append(f"{med_str} [{min_str}]")

            lines.append(f"${g:.2f}$ & " + " & ".join(cell_strs) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    # Detailed layout
    caption = (
        f"Distribution of accuracy constraint slack across 42 real-world benchmarks for "
        f"$\\delta = {delta_str}$. Slack is defined as $\\min\\{{\\Precision - \\gamma, \\Recall - \\gamma\\}}$ "
        r"for ACTIS, BARGAIN, and LOTUS; and $F_1 - \gamma$ for ScaleDoc. Values below zero represent "
        r"accuracy target violations and are highlighted in maroon (\badvalue{Maroon})."
    )
    col_spec = "ll rcccc"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:slack_distribution_delta_{d_tag}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"$\gamma$ & Method & Median & $[Q_1, Q_3]$ & Min (Worst) & Max & Violations (\%) \\",
        r"\midrule",
    ]

    for g_idx, g in enumerate(gammas):
        sub_g = sub[np.isclose(sub["gamma"], g)]
        for m_idx, (raw_m, label_m) in enumerate(resolved_methods):
            df_m = sub_g[
                (sub_g["method_clean"] == label_m)
                | (sub_g["method"] == raw_m)
                | (sub_g["runner_uid"] == raw_m)
            ]
            gamma_col = f"${g:.2f}$" if m_idx == 0 else ""
            vals = df_m["quantile_slack"].dropna()
            if vals.empty:
                lines.append(f"{gamma_col:<8} & {label_m:<10} & -- & -- & -- & -- & -- \\\\")
            else:
                med = float(vals.median())
                q25 = float(vals.quantile(0.25))
                q75 = float(vals.quantile(0.75))
                vmin = float(vals.min())
                vmax = float(vals.max())
                pct_viol = float((vals < 0).mean() * 100.0)

                med_str = format_slack_value(med, precision=precision, use_badvalue=use_badvalue)
                q25_str = format_slack_value(q25, precision=precision, use_badvalue=use_badvalue)
                q75_str = format_slack_value(q75, precision=precision, use_badvalue=use_badvalue)
                iqr_str = f"[{q25_str}, {q75_str}]"
                min_str = format_slack_value(vmin, precision=precision, use_badvalue=use_badvalue)
                max_str = format_slack_value(vmax, precision=precision, use_badvalue=use_badvalue)

                viol_fmt = f"{pct_viol:.1f}\\%"
                if pct_viol > 0.0 and use_badvalue:
                    viol_str = rf"\badvalue{{{viol_fmt}}}"
                else:
                    viol_str = viol_fmt

                lines.append(
                    f"{gamma_col:<8} & {label_m:<10} & {med_str:<18} & {iqr_str:<38} & {min_str:<18} & {max_str:<18} & {viol_str} \\\\"
                )

        if g_idx < len(gammas) - 1:
            lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_all_benchmarks_slack_table(
    df: pd.DataFrame,
    delta_val: float = 0.05,
    gamma_val: Optional[float] = None,
    methods: Sequence[str | tuple[str, str]] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    gammas: Optional[Sequence[float]] = None,
    use_badvalue: bool = True,
    precision: int = 3,
) -> str:
    """
    Generates a LaTeX table presenting constraint slack across all 42 benchmarks.
    If gamma_val is specified, produces a single-gamma table with Workload, Proxy, N columns.
    If gamma_val is None, produces a multi-gamma table across all gammas.
    """
    df_audit = get_audit_dataframe(df)
    sub = df_audit[np.isclose(df_audit["delta"], delta_val)].copy()

    scenario_uids = get_ordered_scenarios(sub)
    delta_str = f"{delta_val:.2f}"
    d_tag = delta_str.replace(".", "")

    resolved_methods: list[tuple[str, str]] = []
    for m in methods:
        if isinstance(m, tuple):
            resolved_methods.append(m)
        else:
            resolved_methods.append((m, m))

    method_headers = " & ".join([lbl for _, lbl in resolved_methods])

    if gamma_val is not None:
        sub_g = sub[np.isclose(sub["gamma"], gamma_val)].copy()
        gamma_str = f"{gamma_val:.2f}"
        g_tag = gamma_str.replace(".", "")

        caption = (
            f"Constraint slack across all 42 real-world benchmarks for "
            f"$\\delta = {delta_str}, \\gamma = {gamma_str}$. "
            r"Slack is defined as $\min\{\Precision - \gamma, \Recall - \gamma\}$ for ACTIS, BARGAIN, and LOTUS; "
            r"and $F_1 - \gamma$ for ScaleDoc. Negative slack values represent target guarantee violations "
            r"and are highlighted in maroon (\badvalue{Maroon})."
        )
        col_spec = "ll c " + "c" * len(resolved_methods)

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4.0pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:all_benchmarks_slack_delta_{d_tag}_gamma_{g_tag}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            r"\textbf{Workload} & \textbf{Proxy} & \textbf{$N$} & " + method_headers + r" \\",
            r"\midrule",
        ]

        prev_dataset = None
        for sc_idx, sc_uid in enumerate(scenario_uids):
            df_row = sub_g[sub_g["scenario_uid"] == sc_uid]
            if df_row.empty:
                continue

            r0 = df_row.iloc[0]
            dataset = r0["dataset"]
            qid = r0.get("query_id")
            proxy_raw = r0.get("proxy_model")
            pop_size = r0.get("pop_size")
            n_str = format_population_size(pop_size)

            disp_name = format_scenario_display_name(dataset, query_id=qid, short=True)
            if "_ext" in disp_name:
                disp_name = disp_name.replace("_ext", r"\textsubscript{ext}")

            if proxy_raw is not None and not (isinstance(proxy_raw, float) and math.isnan(proxy_raw)):
                disp_proxy = format_proxy_display_name(proxy_raw) or str(proxy_raw)
            else:
                disp_proxy = DATASET_DEFAULT_PROXIES.get(dataset, "--")

            cell_strs = []
            for raw_m, label_m in resolved_methods:
                df_m = df_row[
                    (df_row["method_clean"] == label_m)
                    | (df_row["method"] == raw_m)
                    | (df_row["runner_uid"] == raw_m)
                ]
                if df_m.empty:
                    cell_strs.append("--")
                else:
                    val = df_m.iloc[0].get("quantile_slack")
                    cell_strs.append(format_slack_value(val, precision=precision, use_badvalue=use_badvalue))

            if prev_dataset is not None and dataset != prev_dataset:
                lines.append(r"\addlinespace")
            prev_dataset = dataset

            lines.append(
                f"{disp_name:<28} & {disp_proxy:<18} & {n_str:<6} & "
                + " & ".join(cell_strs)
                + r" \\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        return "\n".join(lines)

    # Multi-gamma variant
    if gammas is None:
        gammas = get_ordered_gammas(sub)

    caption = (
        f"Constraint slack across all 42 real-world benchmarks for $\\delta = {delta_str}$ "
        r"across quality targets $\gamma \in \{0.70, 0.80, 0.90, 0.95\}$. "
        r"Negative values indicate target guarantee violations and are highlighted in maroon (\badvalue{Maroon})."
    )
    col_spec = "ll " + "c" * len(resolved_methods)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:all_benchmarks_slack_delta_{d_tag}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Benchmark & $\gamma$ & " + method_headers + r" \\",
        r"\midrule",
    ]

    for sc_idx, sc_uid in enumerate(scenario_uids):
        df_sc = sub[sub["scenario_uid"] == sc_uid]
        if df_sc.empty:
            continue

        r0 = df_sc.iloc[0]
        disp_name = format_scenario_display_name(
            r0["dataset"],
            query_id=r0.get("query_id"),
            proxy_model=r0.get("proxy_model"),
            short=True,
        )
        if "_ext" in disp_name:
            disp_name = disp_name.replace("_ext", r"\textsubscript{ext}")

        for g_idx, g in enumerate(gammas):
            df_g = df_sc[np.isclose(df_sc["gamma"], g)]
            bench_col = disp_name if g_idx == 0 else ""
            gamma_col = f"${g:.2f}$"

            cell_strs = []
            for raw_m, label_m in resolved_methods:
                df_m = df_g[
                    (df_g["method_clean"] == label_m)
                    | (df_g["method"] == raw_m)
                    | (df_g["runner_uid"] == raw_m)
                ]
                if df_m.empty:
                    cell_strs.append("--")
                else:
                    val = df_m.iloc[0].get("quantile_slack")
                    cell_strs.append(format_slack_value(val, precision=precision, use_badvalue=use_badvalue))

            lines.append(
                f"{bench_col:<28} & {gamma_col:<8} & "
                + " & ".join(cell_strs)
                + r" \\"
            )

        if sc_idx < len(scenario_uids) - 1:
            lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_benchmark_slack_table(
    df: pd.DataFrame,
    delta_val: float = 0.05,
    gamma_val: Optional[float] = None,
    methods: Sequence[str | tuple[str, str]] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    gammas: Optional[Sequence[float]] = None,
    use_badvalue: bool = True,
    precision: int = 3,
) -> str:
    """Alias for generate_all_benchmarks_slack_table."""
    return generate_all_benchmarks_slack_table(
        df=df,
        delta_val=delta_val,
        gamma_val=gamma_val,
        methods=methods,
        gammas=gammas,
        use_badvalue=use_badvalue,
        precision=precision,
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready LaTeX tables for hierarchical "
            "experiment results."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory containing hierarchical experiment results "
            "(default: experiments/results)"
        ),
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Optional experiment subdirectory name under experiments/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Custom output directory path for tables "
            "(default: experiments/tables/<exp_name>)"
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional list of scenario names or UIDs to include",
    )
    parser.add_argument(
        "--runners",
        "--methods",
        dest="runners",
        nargs="+",
        default=None,
        help="Optional list of runner names or UIDs to include in primary tables",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of deltas to include",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of gammas to include",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Number of trials to include (default: 1000, use 0 or negative for all)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=(
            "Query identifier to filter for document benchmarks "
            "(default: None, includes all queries)"
        ),
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default=None,
        help=(
            "Oracle model identifier to filter for document benchmarks "
            "(default: None, includes all oracles)"
        ),
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default=None,
        help=(
            "Proxy model identifier to filter for document benchmarks "
            "(default: None, includes all proxies)"
        ),
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        default=1,
        help="Number of decimal places for values (default: 1)",
    )
    parser.add_argument(
        "--no-se",
        action="store_true",
        help="Omit standard errors (mean only)",
    )
    parser.add_argument(
        "--joint-gamma",
        type=float,
        default=None,
        help="Target gamma for joint performance tables (default: 0.95 or first of --gammas)",
    )
    parser.add_argument(
        "--joint-delta",
        type=float,
        default=None,
        help="Error budget delta for joint performance tables (default: 0.05 or first of --deltas)",
    )
    parser.add_argument(
        "--no-badvalue",
        action="store_true",
        help="Disable \\badvalue{...} formatting for unsound results in joint and slack tables",
    )
    parser.add_argument(
        "--slack-precision",
        type=int,
        default=3,
        help="Number of decimal places for slack values (default: 3)",
    )
    parser.add_argument(
        "--table-type",
        choices=["oracle", "latency", "joint", "slack", "all"],
        default="all",
        help=(
            "Type of tables to generate: 'oracle' (oracle call rates), "
            "'latency' (tuner latency), 'joint' (joint OCR and latency), "
            "'slack' (constraint slack distribution), or 'all' (default: 'all')"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    elif args.exp_name:
        results_dir = Path("experiments/results") / args.exp_name
    else:
        results_dir = Path("experiments/results")

    if args.output_dir:
        output_dir = args.output_dir
    elif args.exp_name:
        output_dir = Path("experiments/tables") / args.exp_name
    else:
        output_dir = Path("experiments/tables")

    output_dir.mkdir(parents=True, exist_ok=True)

    num_trials = args.num_trials if args.num_trials and args.num_trials > 0 else None
    query_ids = (
        None
        if not args.query or args.query.lower() in ("all", "none", "*", "")
        else [args.query]
    )
    oracle_models = (
        None
        if not args.oracle_model
        or args.oracle_model.lower() in ("all", "none", "*", "")
        else [args.oracle_model]
    )
    proxy_models = (
        None
        if not args.proxy_model or args.proxy_model.lower() in ("all", "none", "*", "")
        else [args.proxy_model]
    )

    print(f"Loading hierarchical results from: {results_dir}")
    df = load_table_data(
        results_dir=results_dir,
        scenarios=args.scenarios,
        runners=args.runners,
        deltas=args.deltas,
        gammas=args.gammas,
        num_trials=num_trials,
        query_ids=query_ids,
        oracle_models=oracle_models,
        proxy_models=proxy_models,
    )

    scenario_uids = get_ordered_scenarios(df)
    deltas = get_ordered_deltas(df)
    gammas = get_ordered_gammas(df)
    methods = sorted(df["method"].unique())

    print(
        f"Loaded {len(df)} records across {len(scenario_uids)} scenario configs, "
        f"{len(methods)} methods, {len(gammas)} gammas, and {len(deltas)} deltas."
    )

    include_se = not args.no_se
    focus_methods = resolve_methods(df, DEFAULT_FOCUS_METHODS, args.runners)
    appendix_methods = resolve_methods(df, DEFAULT_APPENDIX_METHODS)

    do_oracle = args.table_type in ("oracle", "all")
    do_latency = args.table_type in ("latency", "all")
    do_joint = args.table_type in ("joint", "all")
    do_slack = args.table_type in ("slack", "all")

    if do_oracle:
        print("\n--- Generating Oracle Call Rate Tables ---")
        # 1. Full-width tables
        tex_full = generate_full_width_table(
            df, focus_methods, include_se=include_se, precision=args.precision
        )
        path_full = output_dir / "table_full_width.tex"
        path_full.write_text(tex_full)
        print(f"Saved: {path_full}")

        tex_full_m = generate_full_width_table(
            df, focus_methods, include_se=False, precision=args.precision
        )
        path_full_m = output_dir / "table_full_width_mean_only.tex"
        path_full_m.write_text(tex_full_m)
        print(f"Saved: {path_full_m}")

        # 2. Single-column tables per delta
        for d in deltas:
            d_tag = f"{d:.2f}".replace(".", "")
            tex_single = generate_single_column_table(
                df, d, focus_methods, include_se=include_se, precision=args.precision
            )
            path_single = output_dir / f"table_single_col_delta_{d_tag}.tex"
            path_single.write_text(tex_single)
            print(f"Saved: {path_single}")

            tex_single_m = generate_single_column_table(
                df, d, focus_methods, include_se=False, precision=args.precision
            )
            path_single_m = output_dir / f"table_single_col_delta_{d_tag}_mean_only.tex"
            path_single_m.write_text(tex_single_m)
            print(f"Saved: {path_single_m}")

        # 3. Appendix tables per delta
        for d in deltas:
            d_tag = f"{d:.2f}".replace(".", "")
            tex_app = generate_appendix_table(
                df, d, appendix_methods, include_se=include_se, precision=args.precision
            )
            path_app = output_dir / f"table_appendix_all_variants_delta_{d_tag}.tex"
            path_app.write_text(tex_app)
            print(f"Saved: {path_app}")

            tex_app_m = generate_appendix_table(
                df, d, appendix_methods, include_se=False, precision=args.precision
            )
            path_app_m = (
                output_dir / f"table_appendix_all_variants_delta_{d_tag}_mean_only.tex"
            )
            path_app_m.write_text(tex_app_m)
            print(f"Saved: {path_app_m}")

        # 4. Representative summary table (delta = 0.05, gamma = 0.95)
        tex_rep = generate_representative_oracle_table(
            df, focus_methods, include_se=include_se, precision=args.precision
        )
        for p in [
            output_dir / "table_oracle_call_rates_representative.tex",
            output_dir / "table_representative_summary.tex",
        ]:
            p.write_text(tex_rep)
            print(f"Saved: {p}")

        tex_rep_m = generate_representative_oracle_table(
            df, focus_methods, include_se=False, precision=args.precision
        )
        for p_m in [
            output_dir / "table_oracle_call_rates_representative_mean_only.tex",
            output_dir / "table_representative_summary_mean_only.tex",
        ]:
            p_m.write_text(tex_rep_m)
            print(f"Saved: {p_m}")

    if do_latency:
        print("\n--- Generating Tuner Latency Tables ---")
        # 1. Main-text representative workloads latency table
        tex_lat = generate_latency_table(
            df, focus_methods, include_se=include_se, precision=args.precision
        )
        path_lat = output_dir / "table_tuning_latency.tex"
        path_lat.write_text(tex_lat)
        print(f"Saved: {path_lat}")

        tex_lat_m = generate_latency_table(
            df, focus_methods, include_se=False, precision=args.precision
        )
        path_lat_m = output_dir / "table_tuning_latency_mean_only.tex"
        path_lat_m.write_text(tex_lat_m)
        print(f"Saved: {path_lat_m}")

        # 2. Comprehensive appendix latency tables per delta (all variants)
        for d in deltas:
            d_tag = f"{d:.2f}".replace(".", "")
            tex_app_lat = generate_appendix_latency_table(
                df, d, appendix_methods, include_se=include_se, precision=args.precision
            )
            path_app_lat = (
                output_dir / f"table_appendix_latency_all_variants_delta_{d_tag}.tex"
            )
            path_app_lat.write_text(tex_app_lat)
            print(f"Saved: {path_app_lat}")

            tex_app_lat_m = generate_appendix_latency_table(
                df, d, appendix_methods, include_se=False, precision=args.precision
            )
            path_app_lat_m = (
                output_dir
                / f"table_appendix_latency_all_variants_delta_{d_tag}_mean_only.tex"
            )
            path_app_lat_m.write_text(tex_app_lat_m)
            print(f"Saved: {path_app_lat_m}")

            # Also generate focus methods appendix table
            tex_app_lat_f = generate_appendix_latency_table(
                df, d, focus_methods, include_se=include_se, precision=args.precision
            )
            path_app_lat_f = output_dir / f"table_appendix_latency_delta_{d_tag}.tex"
            path_app_lat_f.write_text(tex_app_lat_f)
            print(f"Saved: {path_app_lat_f}")

            tex_app_lat_f_m = generate_appendix_latency_table(
                df, d, focus_methods, include_se=False, precision=args.precision
            )
            path_app_lat_f_m = (
                output_dir / f"table_appendix_latency_delta_{d_tag}_mean_only.tex"
            )
            path_app_lat_f_m.write_text(tex_app_lat_f_m)
            print(f"Saved: {path_app_lat_f_m}")

    if do_joint:
        print("\n--- Generating Joint Performance Tables ---")
        joint_gamma = (
            args.joint_gamma
            if args.joint_gamma is not None
            else (args.gammas[0] if args.gammas and len(args.gammas) == 1 else 0.95)
        )
        joint_delta = (
            args.joint_delta
            if args.joint_delta is not None
            else (args.deltas[0] if args.deltas and len(args.deltas) == 1 else 0.05)
        )
        use_badvalue = not args.no_badvalue

        # 1. Primary representative workloads joint table
        tex_joint_rep = generate_representative_joint_table(
            df,
            focus_methods,
            delta_val=joint_delta,
            gamma_val=joint_gamma,
            include_se=include_se,
            precision=args.precision,
            use_badvalue=use_badvalue,
        )
        for p in [
            output_dir / "table_joint_representative.tex",
            output_dir / "table_representative_performance.tex",
        ]:
            p.write_text(tex_joint_rep)
            print(f"Saved: {p}")

        tex_joint_rep_m = generate_representative_joint_table(
            df,
            focus_methods,
            delta_val=joint_delta,
            gamma_val=joint_gamma,
            include_se=False,
            precision=args.precision,
            use_badvalue=use_badvalue,
        )
        for p_m in [
            output_dir / "table_joint_representative_mean_only.tex",
            output_dir / "table_representative_performance_mean_only.tex",
        ]:
            p_m.write_text(tex_joint_rep_m)
            print(f"Saved: {p_m}")

        # 2. Primary all benchmarks joint table
        tex_joint_all = generate_all_benchmarks_joint_table(
            df,
            focus_methods,
            delta_val=joint_delta,
            gamma_val=joint_gamma,
            include_se=include_se,
            precision=args.precision,
            use_badvalue=use_badvalue,
        )
        for p in [
            output_dir / "table_joint_all_benchmarks.tex",
            output_dir / "table_all_benchmarks_performance.tex",
        ]:
            p.write_text(tex_joint_all)
            print(f"Saved: {p}")

        tex_joint_all_m = generate_all_benchmarks_joint_table(
            df,
            focus_methods,
            delta_val=joint_delta,
            gamma_val=joint_gamma,
            include_se=False,
            precision=args.precision,
            use_badvalue=use_badvalue,
        )
        for p_m in [
            output_dir / "table_joint_all_benchmarks_mean_only.tex",
            output_dir / "table_all_benchmarks_performance_mean_only.tex",
        ]:
            p_m.write_text(tex_joint_all_m)
            print(f"Saved: {p_m}")

        # 3. Parameterized versions across deltas and gammas
        for d in deltas:
            d_tag = f"{d:.2f}".replace(".", "")
            for g in gammas:
                g_tag = f"{g:.2f}".replace(".", "")
                tex_jr_m = generate_representative_joint_table(
                    df,
                    focus_methods,
                    delta_val=d,
                    gamma_val=g,
                    include_se=False,
                    precision=args.precision,
                    use_badvalue=use_badvalue,
                )
                p_jr_m = (
                    output_dir
                    / f"table_joint_representative_delta_{d_tag}_gamma_{g_tag}_mean_only.tex"
                )
                p_jr_m.write_text(tex_jr_m)

                tex_ja_m = generate_all_benchmarks_joint_table(
                    df,
                    focus_methods,
                    delta_val=d,
                    gamma_val=g,
                    include_se=False,
                    precision=args.precision,
                    use_badvalue=use_badvalue,
                )
                p_ja_m = (
                    output_dir
                    / f"table_joint_all_benchmarks_delta_{d_tag}_gamma_{g_tag}_mean_only.tex"
                )
                p_ja_m.write_text(tex_ja_m)

                if include_se:
                    tex_jr = generate_representative_joint_table(
                        df,
                        focus_methods,
                        delta_val=d,
                        gamma_val=g,
                        include_se=True,
                        precision=args.precision,
                        use_badvalue=use_badvalue,
                    )
                    p_jr = (
                        output_dir
                        / f"table_joint_representative_delta_{d_tag}_gamma_{g_tag}.tex"
                    )
                    p_jr.write_text(tex_jr)

                    tex_ja = generate_all_benchmarks_joint_table(
                        df,
                        focus_methods,
                        delta_val=d,
                        gamma_val=g,
                        include_se=True,
                        precision=args.precision,
                        use_badvalue=use_badvalue,
                    )
                    p_ja = (
                        output_dir
                        / f"table_joint_all_benchmarks_delta_{d_tag}_gamma_{g_tag}.tex"
                    )
                    p_ja.write_text(tex_ja)

    if do_slack:
        print("\n--- Generating Constraint Slack Tables ---")
        slack_delta = (
            args.joint_delta
            if args.joint_delta is not None
            else (args.deltas[0] if args.deltas and len(args.deltas) == 1 else 0.05)
        )
        slack_gamma = (
            args.joint_gamma
            if args.joint_gamma is not None
            else (args.gammas[0] if args.gammas and len(args.gammas) == 1 else 0.95)
        )
        use_badvalue = not args.no_badvalue
        slack_precision = getattr(args, "slack_precision", 3)
        slack_methods = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS")
        if args.runners:
            resolved = resolve_methods(df, DEFAULT_FOCUS_METHODS, args.runners)
            slack_methods = [disp for _, disp in resolved]

        # 1. Primary slack distribution summary tables (for primary delta, e.g. 0.05)
        d_tag_pri = f"{slack_delta:.2f}".replace(".", "")
        tex_sd_det = generate_slack_distribution_table(
            df,
            delta_val=slack_delta,
            methods=slack_methods,
            layout="detailed",
            use_badvalue=use_badvalue,
            precision=slack_precision,
        )
        for p in [
            output_dir / "table_slack_distribution.tex",
            output_dir / f"table_slack_distribution_delta_{d_tag_pri}.tex",
        ]:
            p.write_text(tex_sd_det)
            print(f"Saved: {p}")

        tex_sd_comp = generate_slack_distribution_table(
            df,
            delta_val=slack_delta,
            methods=slack_methods,
            layout="compact",
            use_badvalue=use_badvalue,
            precision=slack_precision,
        )
        for p in [
            output_dir / "table_slack_distribution_compact.tex",
            output_dir / f"table_slack_distribution_compact_delta_{d_tag_pri}.tex",
        ]:
            p.write_text(tex_sd_comp)
            print(f"Saved: {p}")

        # 2. Primary all-benchmarks slack table (for primary delta=0.05, gamma=0.95)
        g_tag_pri = f"{slack_gamma:.2f}".replace(".", "")
        tex_bench_slack = generate_all_benchmarks_slack_table(
            df,
            delta_val=slack_delta,
            gamma_val=slack_gamma,
            methods=slack_methods,
            use_badvalue=use_badvalue,
            precision=slack_precision,
        )
        for p in [
            output_dir / "table_benchmark_slack.tex",
            output_dir / f"table_benchmark_slack_delta_{d_tag_pri}_gamma_{g_tag_pri}.tex",
        ]:
            p.write_text(tex_bench_slack)
            print(f"Saved: {p}")

        # Multi-gamma benchmark slack table for primary delta
        tex_bench_slack_mg = generate_all_benchmarks_slack_table(
            df,
            delta_val=slack_delta,
            gamma_val=None,
            methods=slack_methods,
            use_badvalue=use_badvalue,
            precision=slack_precision,
        )
        p_mg = output_dir / f"table_benchmark_slack_delta_{d_tag_pri}.tex"
        p_mg.write_text(tex_bench_slack_mg)
        print(f"Saved: {p_mg}")

        # 3. Across all tested deltas
        for d in deltas:
            d_tag = f"{d:.2f}".replace(".", "")
            if not math.isclose(d, slack_delta):
                tex_d_det = generate_slack_distribution_table(
                    df,
                    delta_val=d,
                    methods=slack_methods,
                    layout="detailed",
                    use_badvalue=use_badvalue,
                    precision=slack_precision,
                )
                p_d = output_dir / f"table_slack_distribution_delta_{d_tag}.tex"
                p_d.write_text(tex_d_det)
                print(f"Saved: {p_d}")

                tex_d_comp = generate_slack_distribution_table(
                    df,
                    delta_val=d,
                    methods=slack_methods,
                    layout="compact",
                    use_badvalue=use_badvalue,
                    precision=slack_precision,
                )
                p_dc = output_dir / f"table_slack_distribution_compact_delta_{d_tag}.tex"
                p_dc.write_text(tex_d_comp)
                print(f"Saved: {p_dc}")

                tex_d_mg = generate_all_benchmarks_slack_table(
                    df,
                    delta_val=d,
                    gamma_val=None,
                    methods=slack_methods,
                    use_badvalue=use_badvalue,
                    precision=slack_precision,
                )
                p_dmg = output_dir / f"table_benchmark_slack_delta_{d_tag}.tex"
                p_dmg.write_text(tex_d_mg)
                print(f"Saved: {p_dmg}")

    print("\n=======================================================")
    print(f"All LaTeX tables successfully generated in:\n  {output_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
