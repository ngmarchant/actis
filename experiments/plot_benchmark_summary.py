#!/usr/bin/env python3
"""
Publication-Ready Summary Visualizations for Model Cascades across All 42 Benchmarks.

Distills results across 42 real-world benchmarks into focused publication figures:

1. Pairwise Head-to-Head Comparison (ACTIS vs. Baselines)
   - 3-panel comparison: ACTIS vs. BARGAIN, ACTIS vs. ScaleDoc, ACTIS vs. LOTUS.
   - X-axis: ACTIS OCR (%)
   - Y-axis: Baseline OCR (%)
   - Points above the diagonal y = x indicate ACTIS requires fewer oracle calls.
   - Points colored and shaped by Baseline Guarantee (Sound or Inconclusive vs. Unsound)
     using color-blind safe palettes (circle for sound/inconclusive, cross for unsound).
   - Panel strip headers report baseline name and unsound configuration count.

2. IS Relative Cost by Proxy Quality (Impact of PR-AUC)
   - X-axis: Proxy PR-AUC
   - Y-axis: Cost Ratio (IS / Uniform)
   - Evaluates adaptive importance sampling relative cost compared to uniform sampling.

3. IS Relative Cost by Dataset Selectivity (Impact of Positive Rate pi)
   - X-axis: Dataset Selectivity (Log Scale)
   - Y-axis: Cost Ratio (IS / Uniform)
   - Highlights importance sampling efficiency gains in class-imbalanced regimes.
   - Also available as a 2-panel side-by-side faceted figure combining PR-AUC and selectivity.

4. Certified Unsoundness Rate by Accuracy Target gamma and Failure Budget delta
   - 3-panel facet by delta in [0.01, 0.05, 0.10].
   - X-axis: Quality target gamma in [0.70, 0.80, 0.90, 0.95].
   - Y-axis: Percentage of benchmarks certified unsound (0% to 100%).

5. Teaser: Certified Unsoundness Rate by Accuracy Target gamma
   - Single-column figure (delta = 0.05) for paper first page.
   - X-axis: Accuracy Target gamma.
   - Y-axis: Percentage of benchmarks certified unsound (0% to 100%).

6. Sensitivity to Failure Budget delta (Paired Benchmark Traces)
   - X-axis: Failure budget delta in [0.01, 0.05, 0.10].
   - Y-axis: ACTIS OCR (%).
   - Faint paired benchmark lines overlaid with bold median trend line.

7. The Quadrant Tradeoff (Constraint Slack vs. Relative Cost)
   - 3-panel comparison (BARGAIN, ScaleDoc, LOTUS).
   - X-axis: Relative Oracle Cost (Baseline / ACTIS) [log scale centered at 1.0].
   - Y-axis: Constraint Slack min(Precision - gamma, Recall - gamma).
   - Reference lines at Slack = 0 and Cost Ratio = 1.0.

8. Proxy Sensitivity (LLaMA-3.2-1B vs. GPT-5.6-Luna Slopegraphs)
   - 4-panel slopegraph across methods on 7 configurable datasets (19 queries).
   - X-axis: Proxy model (LLaMA-3.2-1B -> GPT-5.6-Luna).
   - Y-axis: Oracle Call Rate (%).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
    geom_vline,
    ggplot,
    ggsave,
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

# Ensure repository root is on sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.dataset_stats import collect_all_stats  # noqa: E402
from experiments.model_paths import slugify_model_name  # noqa: E402
from experiments.results_utils import load_tradeoff_data  # noqa: E402

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

# Standard publication themes using plotnine defaults (unbolded axis titles)
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

# Publication dimensions for ACM/SIGMOD (in inches)
SCALE_FACTOR = 11 / 8.5  # Adjust for figure scaling if needed
SIGMOD_TEXT_WIDTH = SCALE_FACTOR * 7.0319  # Two-column text width
SIGMOD_COL_WIDTH = SCALE_FACTOR * 3.34  # Single-column text width

PGF_RC_CONTEXT = {
    "text.usetex": True,
    "pgf.rcfonts": False,         # Block Matplotlib from overriding metrics
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",   # Use pdflatex for fast, predictable sizing
    "pgf.preamble": "\n".join([
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[tt=false]{libertine}",
        r"\usepackage[libertine]{newtxmath}",
    ])
}

def _save_plot(
    p: Any,
    out_path: Path,
    base_name: str,
    width: float,
    height: float,
    skip_pgf: bool = False,
) -> None:
    """Saves a plot (ggplot or composition) to PDF and conditionally to PGF."""
    pdf_path = out_path / f"{base_name}.pdf"
    if hasattr(p, "save_helper"):
        ggsave(p, str(pdf_path), width=width, height=height)
        if not skip_pgf:
            pgf_path = out_path / f"{base_name}.pgf"
            with mpl.rc_context(PGF_RC_CONTEXT):
                ggsave(p, str(pgf_path), width=width, height=height)
                print(f"Saved: {pdf_path} and {pgf_path}")
        else:
            print(f"Saved: {pdf_path}")
    else:
        fig = p.draw()
        fig.set_size_inches(width, height)
        fig.savefig(str(pdf_path))
        if not skip_pgf:
            pgf_path = out_path / f"{base_name}.pgf"
            with mpl.rc_context(PGF_RC_CONTEXT):
                fig.savefig(str(pgf_path))
                print(f"Saved: {pdf_path} and {pgf_path}")
        else:
            print(f"Saved: {pdf_path}")
        plt.close(fig)


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
    """Generates pairwise Head-to-Head scatter comparisons of ACTIS against

    baselines across all benchmarks on identical workloads. Points above y = x
    indicate ACTIS requires fewer oracle calls. Points are styled with equal
    visual weight by baseline soundness guarantee. Panel strips display baseline
    name and unsound count.
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
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
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


def plot_ablation_is_vs_uniform_pr_auc(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Ablation: IS vs. Uniform relative cost reduction (%) by proxy quality
    (PR-AUC).
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged = _prepare_ablation_data(df, gamma=gamma, delta=delta)
    if merged is None or merged.empty:
        return

    merged = merged.dropna(subset=["pr_auc"])

    p = (
        ggplot(merged, aes(x="pr_auc", y="cost_reduction_pct"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_point(color="#0072B2")
        + labs(
            x="Proxy PR-AUC",
            y="Relative Savings (%)",
        )
        + THEME_ABLATION
    )

    tag = f"gamma_{gamma:.2f}_delta_{delta:.2f}".replace(".", "")
    base_name = f"ablation_is_vs_uniform_pr_auc_{tag}"
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.35,
        skip_pgf=skip_pgf,
    )


def plot_ablation_is_vs_uniform_selectivity(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Ablation: IS vs. Uniform relative cost reduction (%) by query positive rate
    (prevalence).
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged = _prepare_ablation_data(df, gamma=gamma, delta=delta)
    if merged is None or merged.empty:
        return

    merged = merged.dropna(subset=["pos_rate"])

    p = (
        ggplot(merged, aes(x="pos_rate", y="cost_reduction_pct"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_point(color="#0072B2")
        + scale_x_log10()
        + labs(
            x="Query Selectivity (log)",
            y="Relative Savings (%)",
        )
        + THEME_ABLATION
    )

    tag = f"gamma_{gamma:.2f}_delta_{delta:.2f}".replace(".", "")
    base_name = f"ablation_is_vs_uniform_selectivity_{tag}"
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.35,
        skip_pgf=skip_pgf,
    )


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
    _save_plot(
        p_combined,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.2,
        skip_pgf=skip_pgf,
    )


def plot_unsoundness_vs_gamma_delta(
    df: pd.DataFrame,
    methods: Sequence[str] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Plots certified unsound benchmark rate (%) across quality target gamma

    and failure budget delta in a faceted publication figure.
    """
    df = get_audit_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[df["method_clean"].isin(methods)].copy()
    if sub.empty:
        return

    agg = (
        sub.groupby(["delta", "gamma", "method_clean"])["unsound"]
        .agg(unsound_count="sum", total_count="count")
        .reset_index()
    )
    agg["unsound_pct"] = agg["unsound_count"] / agg["total_count"] * 100.0

    agg["delta_strip"] = agg["delta"].apply(lambda d: f"$\\delta = {d:.2f}$")
    ordered_deltas = [f"$\\delta = {d:.2f}$" for d in sorted(agg["delta"].unique())]
    agg["delta_strip"] = pd.Categorical(
        agg["delta_strip"], categories=ordered_deltas, ordered=True
    )
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
        + facet_wrap("~delta_strip", nrow=1)
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

    _save_plot(
        p,
        out_path,
        "unsoundness_vs_gamma_delta",
        width=SIGMOD_TEXT_WIDTH,
        height=SCALE_FACTOR * 1.5,
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

    for a fixed failure budget delta (single-column teaser plot for paper first page).
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
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.2,
        skip_pgf=skip_pgf,
    )


plot_teaser_unsoundness_vs_gamma = plot_unsoundness_teaser


def plot_slack_distribution_vs_gamma_delta(
    df: pd.DataFrame,
    methods: Sequence[str] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    gammas: Optional[Sequence[float]] = None,
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Plots the empirical distribution of constraint slack (quantile slack) across

    methods, quality target gamma, and failure budget delta.
    """
    df = get_audit_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[df["method_clean"].isin(methods)].copy()
    if gammas is not None:
        sub = sub[sub["gamma"].isin(gammas)].copy()
    if sub.empty:
        return

    sub["gamma_cat"] = sub["gamma"].apply(lambda g: f"{g:.2f}")
    ordered_gammas = [f"{g:.2f}" for g in sorted(sub["gamma"].unique())]
    sub["gamma_cat"] = pd.Categorical(
        sub["gamma_cat"], categories=ordered_gammas, ordered=True
    )

    sub["delta_strip"] = sub["delta"].apply(lambda d: f"$\\delta = {d:.2f}$")
    ordered_deltas = [f"$\\delta = {d:.2f}$" for d in sorted(sub["delta"].unique())]
    agg_deltas = [d for d in ordered_deltas if d in sub["delta_strip"].unique()]
    sub["delta_strip"] = pd.Categorical(
        sub["delta_strip"], categories=ordered_deltas, ordered=True
    )

    sub["method_clean"] = pd.Categorical(
        sub["method_clean"], categories=list(methods), ordered=True
    )

    p = (
        ggplot(sub, aes(x="gamma_cat", y="quantile_slack", fill="method_clean", color="method_clean"))
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_boxplot(position=position_dodge(width=0.8))
        + facet_wrap("~delta_strip", nrow=1)
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

    _save_plot(
        p,
        out_path,
        "slack_distribution_vs_gamma_delta",
        width=SIGMOD_TEXT_WIDTH,
        height=SCALE_FACTOR * 1.35,
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

    methods and accuracy target gamma for a fixed failure budget delta
    (single-column variant).
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
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.35,
        skip_pgf=skip_pgf,
    )


def plot_delta_sensitivity_paired(
    df: pd.DataFrame,
    gamma: float,
    methods: Sequence[str] = ("ACTIS", "BARGAIN"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Visualizes sensitivity to failure budget delta for ACTIS and BARGAIN

    using faint paired lines per benchmark and bold median trend lines.
    """
    df = get_audit_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[df["method_clean"].isin(methods) & np.isclose(df["gamma"], gamma)].copy()
    if sub.empty:
        return

    delta_order = sorted(sub["delta"].unique())
    sub["delta_cat"] = sub["delta"].apply(lambda d: f"{d:.2f}")
    cat_order = [f"{d:.2f}" for d in delta_order]
    sub["delta_cat"] = pd.Categorical(
        sub["delta_cat"], categories=cat_order, ordered=True
    )
    sub["method_clean"] = pd.Categorical(
        sub["method_clean"], categories=list(methods), ordered=True
    )

    summary_df = (
        sub.groupby(["method_clean", "delta_cat"], observed=False)[
            "mean_oracle_rate_pct"
        ]
        .median()
        .reset_index(name="median_cost")
    )
    summary_df["group"] = summary_df["method_clean"].astype(str)

    p = (
        ggplot(sub, aes(x="delta_cat", y="mean_oracle_rate_pct"))
        + geom_line(
            aes(group="benchmark_uid", color="method_clean"),
            alpha=0.20,
        )
        + geom_point(
            aes(color="method_clean"),
            alpha=0.25,
        )
        + geom_line(
            aes(x="delta_cat", y="median_cost", group="group", color="method_clean"),
            data=summary_df,
            size=1.3,
        )
        + geom_point(
            aes(
                x="delta_cat",
                y="median_cost",
                color="method_clean",
                shape="method_clean",
            ),
            data=summary_df,
            size=2.8,
        )
        + facet_wrap("~method_clean", nrow=1)
        + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
        + scale_color_manual(values=METHOD_COLORS)
        + scale_shape_manual(values=METHOD_SHAPES)
        + labs(
            x="Failure Budget $\\delta$",
            y="Oracle Call Rate (%)",
            color="Method",
            shape="Method",
        )
        + THEME_PAIRWISE
        + guides(
            color=guide_legend(nrow=1),
            shape=guide_legend(nrow=1),
        )
    )

    tag = f"gamma_{gamma:.2f}".replace(".", "")
    base_name = f"delta_sensitivity_paired_{tag}"
    _save_plot(
        p,
        out_path,
        base_name,
        width=7.0,
        height=SCALE_FACTOR * 3.6,
        skip_pgf=skip_pgf,
    )


def plot_quadrant_tradeoff(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    actis_method: str = "ACTIS",
    baselines: Sequence[str] = ("BARGAIN", "ScaleDoc", "LOTUS"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """The Quadrant Tradeoff Plot: Constraint Slack vs. Relative Cost
    (Baseline / ACTIS).

    Divides benchmark outcomes into 4 quadrants via Slack = 0 and Cost Ratio = 1.0.
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[np.isclose(df["gamma"], gamma) & np.isclose(df["delta"], delta)].copy()
    actis_sub = sub[sub["method_clean"] == actis_method]
    if actis_sub.empty:
        return

    actis_map = actis_sub.set_index("benchmark_uid")["mean_oracle_rate_pct"].to_dict()

    rows = []
    for b_name in baselines:
        base_sub = sub[sub["method_clean"] == b_name]
        for _, row in base_sub.iterrows():
            b_uid = row["benchmark_uid"]
            if b_uid in actis_map:
                actis_cost = actis_map[b_uid]
                base_cost = row["mean_oracle_rate_pct"]
                raw_sound = row.get("sound_label", "Sound or Inconclusive")
                base_sound = SOUND_LABEL_MAP.get(
                    str(raw_sound), "Sound or Inconclusive"
                )
                cost_ratio = base_cost / max(actis_cost, 0.01)
                slack = row.get("quantile_slack", row.get("slack", 0.0))

                rows.append(
                    {
                        "benchmark_uid": b_uid,
                        "baseline": b_name,
                        "cost_ratio": cost_ratio,
                        "slack": slack,
                        "base_sound": base_sound,
                    }
                )

    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plot_df["panel_strip"] = pd.Categorical(
        plot_df["baseline"], categories=list(baselines), ordered=True
    )

    p = (
        ggplot(plot_df, aes(x="cost_ratio", y="slack"))
        + geom_vline(xintercept=1.0, linetype="dashed", color="#555555")
        + geom_hline(yintercept=0.0, linetype="dashed", color="#555555")
        + geom_point(aes(color="base_sound", shape="base_sound"))
        + facet_wrap("~panel_strip", nrow=1, scales="free_y")
        + scale_x_log10(
            breaks=[0.25, 0.5, 1.0, 2.0, 4.0],
            labels=[
                "$0.25\\times$",
                "$0.5\\times$",
                "$1.0\\times$",
                "$2.0\\times$",
                "$4.0\\times$",
            ],
        )
        + scale_color_manual(values=SOUND_COLORS)
        + scale_shape_manual(values=SOUND_SHAPES)
        + labs(
            x="Relative Oracle Cost (Baseline / ACTIS)",
            y="Constraint Slack",
            color="Baseline Guarantee",
            shape="Baseline Guarantee",
        )
        + THEME_PAIRWISE
        + guides(
            color=guide_legend(ncol=1),
            shape=guide_legend(ncol=1),
        )
    )

    tag = f"gamma_{gamma:.2f}_delta_{delta:.2f}".replace(".", "")
    base_name = f"quadrant_tradeoff_{tag}"
    _save_plot(
        p,
        out_path,
        base_name,
        width=SIGMOD_TEXT_WIDTH,
        height=SCALE_FACTOR * 2.8,
        skip_pgf=skip_pgf,
    )


def plot_proxy_sensitivity(
    df: pd.DataFrame,
    gamma: float,
    delta: float,
    methods: Sequence[str] = ("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    output_dir: Path | str = "experiments/plots/summary",
    skip_pgf: bool = False,
) -> None:
    """Slopegraph comparing Oracle Call Rate (%) when upgrading from LLaMA-3.2-1B

    to GPT-5.6-Luna across all 7 configurable proxy datasets.
    """
    df = get_head_to_head_dataframe(df)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sub = df[
        df["proxy_model"].notna()
        & df["method_clean"].isin(methods)
        & np.isclose(df["gamma"], gamma)
        & np.isclose(df["delta"], delta)
    ].copy()

    if sub.empty:
        return

    def _clean_proxy(p_name: str) -> str:
        p_str = str(p_name).lower()
        if "luna" in p_str:
            return "GPT-5.6-Luna"
        if "llama" in p_str:
            return "LLaMA-3.2-1B"
        return str(p_name)

    sub["proxy_clean"] = sub["proxy_model"].apply(_clean_proxy)
    sub = sub[sub["proxy_clean"].isin(["LLaMA-3.2-1B", "GPT-5.6-Luna"])].copy()

    proxy_order = ["LLaMA-3.2-1B", "GPT-5.6-Luna"]
    sub["proxy_clean"] = pd.Categorical(
        sub["proxy_clean"], categories=proxy_order, ordered=True
    )
    sub["method_clean"] = pd.Categorical(
        sub["method_clean"], categories=list(methods), ordered=True
    )

    sub["scenario_pair"] = (
        sub["dataset"].astype(str) + "_q" + sub["query_id"].astype(str)
    )

    valid_pairs = sub.groupby(["method_clean", "scenario_pair"], observed=False)[
        "proxy_clean"
    ].nunique()
    valid_idx = valid_pairs[valid_pairs == 2].index
    sub = sub.set_index(["method_clean", "scenario_pair"]).loc[valid_idx].reset_index()

    summary_df = (
        sub.groupby(["method_clean", "proxy_clean"], observed=False)[
            "mean_oracle_rate_pct"
        ]
        .median()
        .reset_index(name="median_cost")
    )
    summary_df["group"] = summary_df["method_clean"]

    p = (
        ggplot(sub, aes(x="proxy_clean", y="mean_oracle_rate_pct"))
        + geom_line(
            aes(group="scenario_pair"),
            color="#888888",
            alpha=0.35,
        )
        + geom_point(
            color="#888888",
            alpha=0.35,
        )
        + geom_line(
            aes(x="proxy_clean", y="median_cost", group="group", color="method_clean"),
            data=summary_df,
            size=1.2,
        )
        + geom_point(
            aes(
                x="proxy_clean",
                y="median_cost",
                color="method_clean",
                shape="method_clean",
            ),
            data=summary_df,
            size=2.5,
        )
        + facet_wrap("~method_clean", nrow=1)
        + scale_y_continuous(limits=(0, 100), breaks=[0, 25, 50, 75, 100])
        + scale_color_manual(values=METHOD_COLORS)
        + scale_shape_manual(values=METHOD_SHAPES)
        + labs(
            x="Proxy Model",
            y="Oracle Call Rate (%)",
            color="Method",
            shape="Method",
        )
        + THEME_PAIRWISE
        + guides(
            color=guide_legend(nrow=1),
            shape=guide_legend(nrow=1),
        )
    )

    tag = f"gamma_{gamma:.2f}_delta_{delta:.2f}".replace(".", "")
    base_name = f"proxy_sensitivity_{tag}"
    _save_plot(
        p,
        out_path,
        base_name,
        width=8.5,
        height=SCALE_FACTOR * 3.6,
        skip_pgf=skip_pgf,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready summary plots across all 42 benchmarks."
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
            "ablation",
            "ablation_prauc",
            "ablation_selectivity",
            "ablation_pr_auc_selectivity",
            "unsoundness",
            "teaser",
            "slack_distribution",
            "slack_distribution_vs_gamma",
            "delta_sensitivity",
            "quadrant",
            "proxy_sensitivity",
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
            "ablation_prauc",
            "ablation_selectivity",
            "ablation_pr_auc_selectivity",
            "unsoundness",
            "teaser",
            "slack_distribution",
            "slack_distribution_vs_gamma",
            "quadrant",
        }
    else:
        selected_plots = set()
        if "pairwise" in raw_plots:
            selected_plots.add("pairwise")
        if "ablation" in raw_plots:
            selected_plots.add("ablation_prauc")
            selected_plots.add("ablation_selectivity")
            selected_plots.add("ablation_pr_auc_selectivity")
        if "ablation_prauc" in raw_plots:
            selected_plots.add("ablation_prauc")
        if "ablation_selectivity" in raw_plots:
            selected_plots.add("ablation_selectivity")
        if "ablation_pr_auc_selectivity" in raw_plots:
            selected_plots.add("ablation_pr_auc_selectivity")
        if "unsoundness" in raw_plots:
            selected_plots.add("unsoundness")
        if "teaser" in raw_plots:
            selected_plots.add("teaser")
        if "slack_distribution" in raw_plots:
            selected_plots.add("slack_distribution")
        if "slack_distribution_vs_gamma" in raw_plots:
            selected_plots.add("slack_distribution_vs_gamma")
        if "delta_sensitivity" in raw_plots:
            selected_plots.add("delta_sensitivity")
        if "quadrant" in raw_plots:
            selected_plots.add("quadrant")
        if "proxy_sensitivity" in raw_plots:
            selected_plots.add("proxy_sensitivity")

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

    if "unsoundness" in selected_plots:
        plot_unsoundness_vs_gamma_delta(
            df_audit, output_dir=args.output_dir, skip_pgf=args.skip_pgf
        )

    if "teaser" in selected_plots:
        for delta in args.deltas:
            plot_unsoundness_teaser(
                df_audit, delta=delta, output_dir=args.output_dir, skip_pgf=args.skip_pgf
            )

    if "slack_distribution" in selected_plots:
        plot_slack_distribution_vs_gamma_delta(
            df_audit, output_dir=args.output_dir, skip_pgf=args.skip_pgf
        )

    if "slack_distribution_vs_gamma" in selected_plots:
        for delta in args.deltas:
            plot_slack_distribution_vs_gamma(
                df_audit, delta=delta, output_dir=args.output_dir, skip_pgf=args.skip_pgf
            )

    for gamma in args.gammas:
        if "delta_sensitivity" in selected_plots:
            plot_delta_sensitivity_paired(
                df_audit, gamma=gamma, output_dir=args.output_dir, skip_pgf=args.skip_pgf
            )

        for delta in args.deltas:
            if "pairwise" in selected_plots:
                plot_head_to_head_pairwise(
                    df_h2h,
                    gamma=gamma,
                    delta=delta,
                    output_dir=args.output_dir,
                    skip_pgf=args.skip_pgf,
                )

            if "ablation_prauc" in selected_plots:
                plot_ablation_is_vs_uniform_pr_auc(
                    df_h2h,
                    gamma=gamma,
                    delta=delta,
                    output_dir=args.output_dir,
                    skip_pgf=args.skip_pgf,
                )

            if "ablation_selectivity" in selected_plots:
                plot_ablation_is_vs_uniform_selectivity(
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

            if "quadrant" in selected_plots:
                plot_quadrant_tradeoff(
                    df_h2h,
                    gamma=gamma,
                    delta=delta,
                    output_dir=args.output_dir,
                    skip_pgf=args.skip_pgf,
                )

            if "proxy_sensitivity" in selected_plots:
                plot_proxy_sensitivity(
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
