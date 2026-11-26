#!/usr/bin/env python3
"""
Cross-benchmark summary plots for model cascades across the 42 benchmarks.

Generates four paper figures:

1. Pairwise Head-to-Head Comparison (ACTIS vs. Baselines)
   - 3-panel comparison: ACTIS vs. BARGAIN, ACTIS vs. ScaleDoc, ACTIS vs. LOTUS.
2. IS Relative Cost by Proxy Quality (PR-AUC) and Query Selectivity
   - 2-panel side-by-side figure comparing importance sampling vs. uniform sampling.
3. Teaser: Certified Unsoundness Rate by Accuracy Target gamma
   - Single-column figure (delta = 0.05) for paper first page.
4. Constraint Slack Distribution by Accuracy Target gamma
   - Single-column boxplot across benchmarks at fixed delta = 0.05.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_abline,
    geom_boxplot,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    theme,
    theme_bw,
)

from experiments.analysis.dataset_stats import collect_all_stats
from experiments.analysis.plot_style import (
    COL_WIDTH,
    SCALE_FACTOR,
    save_plot,
)
from experiments.analysis.results_utils import (
    METHOD_DISPLAY_NAMES,
    compute_bootstrap_ci,
    compute_quantile_ci,
    discover_hierarchical_records,
    format_scenario_display_name,
    get_benchmark_uid,
)
from experiments.analysis.soundness import is_certified_unsound
from experiments.model_paths import slugify_model_name

# Color-blind friendly palette (Okabe-Ito high contrast)
SOUND_COLORS = {
    "Sound/Inconcl.": "#0072B2",  # Okabe-Ito Blue
    "Unsound": "#D55E00",  # Okabe-Ito Vermilion
}

# Markers: Circle for Sound or Inconclusive, Cross for Unsound
SOUND_SHAPES = {
    "Sound/Inconcl.": "o",  # Circle
    "Unsound": "x",  # Cross
}

# Method palettes for multi-method line and comparison plots
METHOD_COLORS = {
    "ACTIS": "#0072B2",  # Okabe-Ito Blue
    "BARGAIN": "#009E73",  # Okabe-Ito Bluish Green
    "ScaleDoc": "#E69F00",  # Okabe-Ito Amber
    "LOTUS": "#D55E00",  # Okabe-Ito Vermilion
}

METHOD_FILLS = {
    method: mcolors.to_hex(mcolors.to_rgba(color, alpha=0.25), keep_alpha=True)
    for method, color in METHOD_COLORS.items()
}


METHOD_SHAPES = {
    "ACTIS": "^",
    "BARGAIN": "v",
    "ScaleDoc": "o",
    "LOTUS": "D",
}

# Shared plotnine theme defaults (unbolded axis titles)
THEME_PAIRWISE = theme_bw() + theme(
    panel_grid_minor=element_blank(),
    legend_position="top",
    # legend_box_spacing=5
)

THEME_GAMMA_DELTA = theme_bw() + theme(
    panel_grid_minor=element_blank(),
    legend_position="right",
)

THEME_ABLATION = theme_bw() + theme(
    panel_grid_minor=element_blank(),
)


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


def load_enriched_benchmark_data(
    results_dir: Path | str = "experiments/results",
    num_trials: Optional[int] = 1000,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Loads experimental results and enriches them with dataset statistics
    (selectivity and PR-AUC) from dataset_stats.
    """
    print("Loading tradeoff results...", file=sys.stderr)
    df_results = load_tradeoff_data(
        results_dir=results_dir,
        num_trials=num_trials,
        deltas=deltas,
        gammas=gammas,
    )

    print("Collecting dataset statistics...", file=sys.stderr)
    df_stats = collect_all_stats(
        include_supg=True,
        include_benchmarks=True,
        data_dir=data_dir,
    )

    # Match each result record with its corresponding dataset statistics
    def _match_row(row: pd.Series) -> tuple[float, float, float, float]:
        ds = str(row["dataset"])
        qid = str(row["query_id"]) if pd.notna(row["query_id"]) else "—"
        p_raw = row.get("proxy_model")
        p_slug = slugify_model_name(p_raw) if pd.notna(p_raw) and p_raw else "—"

        sub = df_stats[
            (df_stats["scenario"] == ds)
            & (df_stats["query"] == qid)
            & (df_stats["proxy_model"] == p_slug)
        ]
        if sub.empty and p_slug == "—":
            sub = df_stats[(df_stats["scenario"] == ds) & (df_stats["query"] == qid)]
        if sub.empty and qid == "—":
            sub = df_stats[df_stats["scenario"] == ds]

        if not sub.empty:
            r0 = sub.iloc[0]
            return (
                float(r0["pos_rate"]),
                float(r0["pr_auc"]),
                float(r0["cal_bias"]),
                float(r0["ece"]),
            )
        return float("nan"), float("nan"), float("nan"), float("nan")

    stats_tuples = [_match_row(row) for _, row in df_results.iterrows()]
    df_results["pos_rate"] = [t[0] for t in stats_tuples]
    df_results["pr_auc"] = [t[1] for t in stats_tuples]
    df_results["cal_bias"] = [t[2] for t in stats_tuples]
    df_results["ece"] = [t[3] for t in stats_tuples]

    # Derived metrics
    df_results["cost_ratio_to_ideal"] = np.where(
        df_results["ideal_oracle_rate_pct"] > 0,
        df_results["mean_oracle_rate_pct"] / df_results["ideal_oracle_rate_pct"],
        np.nan,
    )

    return df_results


def get_audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe for audit figures, using ScaleDoc Native as 'ScaleDoc'."""
    sub = df.copy()
    if "scaledoc_native" in sub["method"].values:
        sub = sub[sub["method"] != "scaledoc"].copy()
        sub.loc[sub["method"] == "scaledoc_native", "method_clean"] = "ScaleDoc"
    return sub


def get_head_to_head_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe for efficiency and head-to-head figures, excluding
    ScaleDoc Native.
    """
    sub = df.copy()
    sub = sub[sub["method"] != "scaledoc_native"].copy()
    if "scaledoc" in sub["method"].values:
        sub.loc[sub["method"] == "scaledoc", "method_clean"] = "ScaleDoc"
    return sub


SOUND_LABEL_MAP = {
    "Certified Sound": "Sound/Inconcl.",
    "Certified Unsound": "Unsound",
    "Sound or Inconclusive": "Sound/Inconcl.",
    "Unsound": "Unsound",
}


def plot_head_to_head_pairwise(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    actis_method: str = "ACTIS",
    baselines: Sequence[str] = ("BARGAIN", "ScaleDoc", "LOTUS"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Generates pairwise head-to-head scatter comparisons of ACTIS against
    baselines across all benchmarks on identical workloads. Points above y = x
    indicate ACTIS requires fewer oracle calls. Points are styled with equal
    visual weight by baseline soundness guarantee, and panel strips display the
    baseline name and unsound count.
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[np.isclose(df["gamma"], gamma) & np.isclose(df["delta"], delta)].copy()
    actis_sub = sub[sub["method_clean"] == actis_method]
    if actis_sub.empty:
        print(
            f"No ACTIS data for head-to-head comparison at gamma={gamma}",
            file=sys.stderr,
        )
        return

    actis_map = actis_sub.set_index("benchmark_uid")["mean_oracle_rate_pct"].to_dict()

    rows = []
    unsound_counts = {}

    for b_name in baselines:
        base_sub = sub[sub["method_clean"] == b_name]
        n_total = 0
        n_actis_wins = 0
        n_base_unsound = 0

        for _, row in base_sub.iterrows():
            b_uid = row["benchmark_uid"]
            if b_uid in actis_map:
                actis_cost = actis_map[b_uid]
                base_cost = row["mean_oracle_rate_pct"]
                raw_sound = row.get("sound_label", "Sound or Inconclusive")
                base_sound = SOUND_LABEL_MAP.get(
                    str(raw_sound), "Sound or Inconclusive"
                )

                n_total += 1
                if actis_cost <= base_cost:
                    n_actis_wins += 1
                if base_sound == "Unsound":
                    n_base_unsound += 1

                rows.append(
                    {
                        "benchmark_uid": b_uid,
                        "scenario_clean": row["scenario_clean"],
                        "baseline": b_name,
                        "actis_cost": actis_cost,
                        "base_cost": base_cost,
                        "base_sound": base_sound,
                    }
                )

        unsound_counts[b_name] = (n_base_unsound, n_total, n_actis_wins)

    if not rows:
        print("No matching benchmarks found for head-to-head comparison.")
        return

    plot_df = pd.DataFrame(rows)

    # Format panel strips with baseline name and unsound count
    strip_map = {
        b: f"{b}\n({unsound_counts[b][0]}/{unsound_counts[b][1]} Unsound)"
        for b in baselines
        if b in unsound_counts
    }
    plot_df["panel_strip"] = plot_df["baseline"].map(strip_map)
    ordered_panels = [strip_map[b] for b in baselines if b in strip_map]
    plot_df["panel_strip"] = pd.Categorical(
        plot_df["panel_strip"], categories=ordered_panels, ordered=True
    )

    p = (
        ggplot(plot_df, aes(x="actis_cost", y="base_cost"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#555555")
        + geom_point(aes(color="base_sound", shape="base_sound"))
        + facet_wrap("~panel_strip", nrow=1)
        + scale_x_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
        + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
        + scale_color_manual(values=SOUND_COLORS)
        + scale_shape_manual(values=SOUND_SHAPES)
        + labs(
            x="ACTIS OCR (%)",
            y="Baseline OCR (%)",
            color="Baseline Guarantee",
            shape="Baseline Guarantee",
        )
        + THEME_PAIRWISE
        + guides(
            color=guide_legend(ncol=2),
            shape=guide_legend(ncol=2),
        )
    )

    base_name = f"pairwise_head_to_head_gamma_{gamma:.2f}_delta_{delta:.2f}".replace(
        ".", ""
    )
    save_plot(
        p,
        out_path,
        base_name,
        width=COL_WIDTH,
        height=SCALE_FACTOR * 1.6,
        skip_pgf=skip_pgf,
    )


def _prepare_ablation_data(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
) -> Optional[pd.DataFrame]:
    """Extracts merged trial records comparing ACTIS Adaptive IS vs. ACTIS Uniform."""
    sub = df[np.isclose(df["gamma"], gamma) & np.isclose(df["delta"], delta)].copy()
    is_sub = sub[sub["method"] == "actis_adaptive_is_asymptotic"]
    unif_sub = sub[sub["method"].isin(["actis_adaptive_asymptotic", "actis_adaptive"])]

    if is_sub.empty or unif_sub.empty:
        return None

    if "actis_adaptive_asymptotic" in unif_sub["method"].values:
        unif_sub = unif_sub[unif_sub["method"] == "actis_adaptive_asymptotic"]
    else:
        unif_sub = unif_sub[unif_sub["method"] == "actis_adaptive"]

    merged = pd.merge(
        is_sub[
            [
                "benchmark_uid",
                "scenario_clean",
                "pr_auc",
                "pos_rate",
                "mean_oracle_rate_pct",
            ]
        ].rename(columns={"mean_oracle_rate_pct": "cost_is"}),
        unif_sub[["benchmark_uid", "mean_oracle_rate_pct"]].rename(
            columns={"mean_oracle_rate_pct": "cost_unif"}
        ),
        on="benchmark_uid",
    )

    merged["cost_ratio"] = np.where(
        merged["cost_unif"] > 0,
        merged["cost_is"] / merged["cost_unif"],
        np.nan,
    )
    merged["cost_reduction_pct"] = (1.0 - merged["cost_ratio"]) * 100.0
    return merged.dropna(subset=["cost_reduction_pct"])


def plot_ablation_is_vs_uniform_pr_auc_selectivity(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Ablation: IS vs. Uniform relative cost reduction (%) side-by-side
    by proxy quality (PR-AUC) and query positive rate (selectivity) using composition.
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged = _prepare_ablation_data(df, gamma=gamma, delta=delta)
    if merged is None or merged.empty:
        return

    df_pr = merged.dropna(subset=["pr_auc"]).copy()
    df_pos = merged.dropna(subset=["pos_rate"]).copy()

    if df_pr.empty or df_pos.empty:
        return

    # Determine common vertical axis limits aligned with 0.0 reference line
    y_vals = pd.concat(
        [df_pr["cost_reduction_pct"], df_pos["cost_reduction_pct"], pd.Series([0.0])]
    )
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    y_span = max(y_max - y_min, 1.0)
    y_limits = (y_min - 0.05 * y_span, y_max + 0.05 * y_span)

    theme_x_margin = theme(
        axis_title_x=element_text(margin={"t": 10, "unit": "pt"}),
    )

    p1 = (
        ggplot(df_pr, aes(x="pr_auc", y="cost_reduction_pct"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_point(color="#0072B2")
        + scale_y_continuous(limits=y_limits)
        + labs(
            x="Proxy PR-AUC",
            y="Relative Savings (%)",
        )
        + THEME_ABLATION
        + theme_x_margin
    )

    no_y = theme(
        axis_title_y=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks_y=element_blank(),
    )

    p2 = (
        ggplot(df_pos, aes(x="pos_rate", y="cost_reduction_pct"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_point(color="#0072B2")
        + scale_x_log10()
        + scale_y_continuous(limits=y_limits)
        + labs(
            x="Query Selectivity (log)",
        )
        + THEME_ABLATION
        + no_y
        + theme_x_margin
    )

    p_combined = p1 | p2

    tag = f"gamma_{gamma:.2f}_delta_{delta:.2f}".replace(".", "")
    base_name = f"ablation_is_vs_uniform_pr_auc_selectivity_{tag}"
    save_plot(
        p_combined,
        out_path,
        base_name,
        width=COL_WIDTH,
        height=SCALE_FACTOR * 1.2,
        skip_pgf=skip_pgf,
    )


def plot_unsoundness_teaser(
    df: pd.DataFrame,
    delta: float,
    methods: Sequence[str] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Plots certified unsound benchmark rate (%) across accuracy target gamma
    for a fixed failure budget delta (single-column teaser plot).
    """
    df = get_audit_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[df["method_clean"].isin(methods) & np.isclose(df["delta"], delta)].copy()
    if sub.empty:
        return

    agg = (
        sub.groupby(["gamma", "method_clean"])["unsound"]
        .agg(unsound_count="sum", total_count="count")
        .reset_index()
    )
    agg["unsound_pct"] = agg["unsound_count"] / agg["total_count"] * 100.0

    agg["method_clean"] = pd.Categorical(
        agg["method_clean"], categories=list(methods), ordered=True
    )

    unique_gammas = sorted(agg["gamma"].unique())

    p = (
        ggplot(
            agg,
            aes(
                x="gamma",
                y="unsound_pct",
                color="method_clean",
                shape="method_clean",
                group="method_clean",
            ),
        )
        + geom_line()
        + geom_point(fill="none", size=3.0)
        + scale_x_continuous(
            breaks=unique_gammas,
            labels=[f"{g:.2f}" for g in unique_gammas],
        )
        + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
        + scale_color_manual(values=METHOD_COLORS)
        + scale_shape_manual(values=METHOD_SHAPES)
        + labs(
            x="Accuracy Target $\\gamma$",
            y="Unsound (%)",
            color="Method",
            shape="Method",
        )
        + THEME_GAMMA_DELTA
        + guides(
            color=guide_legend(ncol=1),
            shape=guide_legend(ncol=1),
        )
    )

    tag = f"delta_{delta:.2f}".replace(".", "")
    base_name = f"teaser_unsoundness_vs_gamma_{tag}"
    save_plot(
        p,
        out_path,
        base_name,
        width=COL_WIDTH,
        height=SCALE_FACTOR * 1.2,
        skip_pgf=skip_pgf,
    )


def plot_slack_distribution_vs_gamma(
    df: pd.DataFrame,
    delta: float,
    methods: Sequence[str] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    gammas: Optional[Sequence[float]] = None,
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Plots the empirical distribution of constraint slack (quantile slack) across
    methods and accuracy target gamma for a fixed failure budget delta.
    """
    df = get_audit_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[df["method_clean"].isin(methods) & np.isclose(df["delta"], delta)].copy()
    if gammas is not None:
        sub = sub[sub["gamma"].isin(gammas)].copy()
    if sub.empty:
        return

    sub["gamma_cat"] = sub["gamma"].apply(lambda g: f"{g:.2f}")
    ordered_gammas = [f"{g:.2f}" for g in sorted(sub["gamma"].unique())]
    sub["gamma_cat"] = pd.Categorical(
        sub["gamma_cat"], categories=ordered_gammas, ordered=True
    )

    sub["method_clean"] = pd.Categorical(
        sub["method_clean"], categories=list(methods), ordered=True
    )

    p = (
        ggplot(sub, aes(x="gamma_cat", y="quantile_slack", fill="method_clean", color="method_clean"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_boxplot(position=position_dodge(width=0.8))
        + scale_color_manual(values=METHOD_COLORS)
        + scale_fill_manual(values=METHOD_FILLS)
        + labs(
            x="Accuracy Target $\\gamma$",
            y="Constraint Slack",
            color="Method",
            fill="Method"
        )
        + THEME_GAMMA_DELTA
        + guides(
            color=guide_legend(ncol=1),
            fill=guide_legend(ncol=1)
        )
    )

    tag = f"delta_{delta:.2f}".replace(".", "")
    base_name = f"slack_distribution_vs_gamma_{tag}"
    save_plot(
        p,
        out_path,
        base_name,
        width=COL_WIDTH,
        height=SCALE_FACTOR * 1.35,
        skip_pgf=skip_pgf,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cross-benchmark summary plots."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory containing hierarchical experiment results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots/summary"),
        help="Directory to save summary figures (PDF and PGF).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Filter results by number of trials (default: 1000).",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="List of delta values to plot (default: [0.01, 0.05, 0.10]).",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90, 0.95],
        help="List of gamma values to plot (default: [0.70, 0.80, 0.90, 0.95]).",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "all",
            "pairwise",
            "ablation_pr_auc_selectivity",
            "teaser",
            "slack_distribution_vs_gamma",
        ],
        default=["all"],
        help="Types of summary plots to generate.",
    )
    parser.add_argument(
        "--skip-pgf",
        "--no-pgf",
        dest="skip_pgf",
        action="store_true",
        help="Skip generating PGF format plots (generate only PDF).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_plots = set(args.plots)
    if "all" in raw_plots:
        selected_plots = {
            "pairwise",
            "ablation_pr_auc_selectivity",
            "teaser",
            "slack_distribution_vs_gamma",
        }
    else:
        selected_plots = set(raw_plots)

    df = load_enriched_benchmark_data(
        results_dir=args.results_dir,
        num_trials=args.num_trials,
        deltas=args.deltas,
        gammas=args.gammas,
    )

    n_benches = len(df["benchmark_uid"].unique())
    n_methods = len(df["method_clean"].unique())
    print(
        f"\nLoaded {len(df)} total evaluation records across {n_benches} "
        f"benchmarks and {n_methods} methods."
    )

    df_audit = get_audit_dataframe(df)
    df_h2h = get_head_to_head_dataframe(df)

    if "teaser" in selected_plots:
        for delta in args.deltas:
            plot_unsoundness_teaser(
                df_audit, delta=delta, output_dir=args.output_dir, skip_pgf=args.skip_pgf
            )

    if "slack_distribution_vs_gamma" in selected_plots:
        for delta in args.deltas:
            plot_slack_distribution_vs_gamma(
                df_audit, delta=delta, output_dir=args.output_dir, skip_pgf=args.skip_pgf
            )

    for gamma in args.gammas:
        for delta in args.deltas:
            if "pairwise" in selected_plots:
                plot_head_to_head_pairwise(
                    df_h2h,
                    gamma=gamma,
                    delta=delta,
                    output_dir=args.output_dir,
                    skip_pgf=args.skip_pgf,
                )

            if "ablation_pr_auc_selectivity" in selected_plots:
                plot_ablation_is_vs_uniform_pr_auc_selectivity(
                    df_h2h,
                    gamma=gamma,
                    delta=delta,
                    output_dir=args.output_dir,
                    skip_pgf=args.skip_pgf,
                )

    print("\n=======================================================")
    print(f"All summary plots successfully generated in:\n  {args.output_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
