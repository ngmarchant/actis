#!/usr/bin/env bash
"""
Publication-Ready Parameter Sensitivity Analysis for ACTIS.

Modular script for analyzing, plotting, and tabulating the sensitivity of ACTIS
with respect to tunable hyperparameters (e.g. batch_size, alpha, num_thresholds).

Features:
1. Flexible loading of hierarchical results across scenarios and parameter grids.
2. Publication-quality vector plots (PDF and PGF) using plotnine / matplotlib
   tailored to ACM/SIGMOD column dimensions:
   - Tuner wall-clock latency vs. parameter (log/linear scales).
   - Oracle Call Rate (%) vs. parameter.
   - Dual-panel composite figures combining latency and efficiency.
3. Publication-ready LaTeX tables formatted with booktabs:
   - Full multi-column table (table_*.tex) and compact single-column table.
   - CSV summary export.

Usage:
  # Analyze batch size sensitivity (default):
  uv run python -m experiments.plot_parameter_sensitivity --param batch_size

  # Analyze alpha (defensive mixture weight) sensitivity:
  uv run python -m experiments.plot_parameter_sensitivity --param alpha --results-dir experiments/results_alpha
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_rect,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    ggsave,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_bw,
)

# Ensure repository root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.plot_benchmark_summary import (  # noqa: E402
    SCALE_FACTOR,
    SIGMOD_COL_WIDTH,
    SIGMOD_TEXT_WIDTH,
    _save_plot,
)
from experiments.results_utils import SCENARIO_DISPLAY_NAMES  # noqa: E402

# Color-blind safe palette for target quality levels (increasing stringency)
GAMMA_COLORS = {
    "0.70": "#117733",  # Green
    "0.80": "#33BBEE",  # Cyan
    "0.90": "#EE7733",  # Orange
    "0.95": "#CC3311",  # Red / Vermilion
}

GAMMA_SHAPES = {
    "0.70": "o",  # Circle
    "0.80": "s",  # Square
    "0.90": "^",  # Triangle
    "0.95": "D",  # Diamond
}

PARAM_META: dict[str, dict[str, Any]] = {
    "batch_size": {
        "display_name": "Batch Size ($B$)",
        "short_name": "B",
        "tex_symbol": "$B$",
        "log_x": True,
        "is_int": True,
        "default_dir": "experiments/results_batch_size",
        "description": "Evaluation of batch size scaling on tuner latency and oracle call rate",
    },
    "alpha": {
        "display_name": r"Defensive Mixture Weight ($\alpha$)",
        "short_name": "alpha",
        "tex_symbol": r"$\alpha$",
        "log_x": False,
        "is_int": False,
        "default_dir": "experiments/results_alpha",
        "description": "Sensitivity to defensive mixture weight alpha",
    },
    "num_thresholds": {
        "display_name": "Candidate Threshold Count ($M$)",
        "short_name": "num_thresholds",
        "tex_symbol": "$M$",
        "log_x": False,
        "is_int": True,
        "default_dir": "experiments/results_num_thresholds",
        "description": "Sensitivity to candidate threshold grid resolution",
    },
    "variance_ratio_bound": {
        "display_name": r"Variance Accumulation Threshold ($\rho$)",
        "short_name": "variance_ratio_bound",
        "tex_symbol": r"$\rho$",
        "log_x": False,
        "is_int": False,
        "default_dir": "experiments/results_variance_ratio",
        "description": "Sensitivity to asymptotic safeguard variance ratio bound",
    },
}


def get_param_meta(param_name: str) -> dict[str, Any]:
    """Retrieves metadata configuration for a given parameter."""
    if param_name in PARAM_META:
        return PARAM_META[param_name]
    clean_name = param_name.replace("_", " ").title()
    return {
        "display_name": clean_name,
        "short_name": param_name,
        "tex_symbol": f"${param_name}$",
        "log_x": False,
        "is_int": False,
        "default_dir": f"experiments/results_{param_name}",
        "description": f"Sensitivity analysis for {clean_name}",
    }


def load_sensitivity_data(
    results_dir: Path | str,
    param_name: str = "batch_size",
    scenarios: Optional[Sequence[str]] = None,
    deltas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Loads and standardizes experimental results varying a specific parameter.

    Traverses the hierarchical directory and extracts parameter values,
    oracle call rates, runtimes, and confidence intervals.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_path}")

    records: list[dict[str, Any]] = []
    json_files = sorted(results_path.rglob("*.json"))

    for f in json_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to read {f}: {e}", file=sys.stderr)
            continue

        runner = data.get("runner", {})
        param_val = runner.get(param_name)
        if param_val is None:
            continue

        scen_dict = data.get("scenario", {})
        scen_name = scen_dict.get("name", "unknown")
        if scenarios is not None and scen_name not in scenarios:
            continue

        delta = float(data.get("promised_delta", data.get("delta", 0.05)))
        if deltas is not None and not any(math.isclose(delta, d, abs_tol=1e-4) for d in deltas):
            continue

        gamma = float(data.get("gamma_R", 0.90))
        if gammas is not None and not any(math.isclose(gamma, g, abs_tol=1e-4) for g in gammas):
            continue

        cost_dict = data.get("cost", {})
        oracle_dict = cost_dict.get("oracle", {}).get("call_rate", {})
        ocr_mean = float(oracle_dict.get("mean", 0.0)) * 100.0
        ocr_se = float(oracle_dict.get("se", 0.0)) * 100.0

        runtime_dict = data.get("runtime", {})
        rt_mean = float(runtime_dict.get("mean", 0.0))
        rt_se = float(runtime_dict.get("se", 0.0))

        ideal_ocr = data.get("ideal_oracle_call_rate")
        if ideal_ocr is not None:
            ideal_ocr = float(ideal_ocr) * 100.0

        joint_fail_dict = data.get("joint_failure", {})
        joint_fail_rate = float(joint_fail_dict.get("rate", 0.0))

        qid = scen_dict.get("query_id")
        proxy = scen_dict.get("proxy_model")
        scen_disp = SCENARIO_DISPLAY_NAMES.get(scen_name, scen_name.replace("_", " ").title())
        if scen_name == "pubmed" and qid:
            scen_disp = f"{scen_disp} Q{qid}"
        elif qid is not None and scen_name not in ["jackson", "imagenet", "onto", "tacred"]:
            scen_disp = f"{scen_disp} Q{qid}"

        records.append({
            "file": str(f),
            "scenario": scen_name,
            "scenario_display": scen_disp,
            "query_id": qid,
            "proxy_model": proxy,
            "param_value": param_val,
            "param_num": float(param_val),
            "delta": delta,
            "gamma": gamma,
            "gamma_str": f"{gamma:.2f}",
            "num_trials": int(data.get("num_trials", 1000)),
            "ocr_mean": ocr_mean,
            "ocr_se": ocr_se,
            "ocr_lower": max(0.0, ocr_mean - 1.96 * ocr_se),
            "ocr_upper": min(100.0, ocr_mean + 1.96 * ocr_se),
            "runtime_mean": rt_mean,
            "runtime_se": rt_se,
            "runtime_lower": max(1e-6, rt_mean - 1.96 * rt_se),
            "runtime_upper": rt_mean + 1.96 * rt_se,
            "ideal_ocr": ideal_ocr,
            "joint_fail_rate": joint_fail_rate,
        })

    if not records:
        print(f"Warning: No valid records found for parameter '{param_name}' in {results_dir}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Sort deterministically
    df = df.sort_values(["scenario", "param_num", "gamma", "delta"]).reset_index(drop=True)
    return df


def generate_sensitivity_latex_tables(
    df: pd.DataFrame,
    param_name: str = "batch_size",
    output_dir: Path | str = "experiments/tables",
) -> dict[str, str]:
    """Generates publication-ready booktabs LaTeX tables and CSV exports."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta = get_param_meta(param_name)
    tex_sym = meta["tex_symbol"]

    # Filter to default delta = 0.05 if multiple exist
    df_plot = df[df["delta"] == 0.05].copy() if 0.05 in df["delta"].values else df.copy()

    unique_params = sorted(df_plot["param_num"].unique())
    unique_gammas = sorted(df_plot["gamma"].unique())
    scenarios = sorted(df_plot["scenario"].unique())

    # 1. Export CSV summary
    csv_file = out_path / f"sensitivity_{meta['short_name']}.csv"
    summary_cols = ["scenario", "param_value", "gamma", "delta", "ocr_mean", "ocr_se", "runtime_mean", "runtime_se"]
    df_plot[summary_cols].to_csv(csv_file, index=False)

    # 2. Comprehensive Wide Table
    # Columns: Scenario & Param & OCR(gamma1..gamma4) & Runtime(gamma1..gamma4)
    lines_wide: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        f"\\caption{{Sensitivity of ACTIS to {meta['display_name']} across quality targets "
        r"$\gamma \in \{0.70, 0.80, 0.90, 0.95\}$ at nominal error budget $\delta = 0.05$ "
        r"($N = 1000$ simulation trials). "
        r"Oracle Call Rate (OCR) represents the percentage of records invoking the oracle; "
        r"Tuner Latency denotes mean wall-clock tuning time.}",
        f"\\label{{tab:sensitivity_{meta['short_name']}}}",
        r"\begin{tabular}{l r rrrr rrrr}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Oracle Call Rate (\%)}} & \multicolumn{4}{c}{\textbf{Tuner Latency (s)}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        f"\\textbf{{Workload}} & \\textbf{{{tex_sym}}} & "
        + " & ".join([f"$\\gamma{{=}}{g:.2f}$" for g in unique_gammas])
        + " & "
        + " & ".join([f"$\\gamma{{=}}{g:.2f}$" for g in unique_gammas])
        + r" \\",
        r"\midrule",
    ]

    for s_idx, scen in enumerate(scenarios):
        scen_sub = df_plot[df_plot["scenario"] == scen]
        scen_disp = scen_sub["scenario_display"].iloc[0]

        for p_idx, p_val in enumerate(unique_params):
            p_sub = scen_sub[scen_sub["param_num"] == p_val]
            p_disp = f"{int(p_val)}" if meta["is_int"] else f"{p_val:.2g}"

            ocr_strs = []
            rt_strs = []
            for g in unique_gammas:
                row = p_sub[p_sub["gamma"] == g]
                if not row.empty:
                    ocr = row["ocr_mean"].iloc[0]
                    rt = row["runtime_mean"].iloc[0]
                    ocr_strs.append(f"{ocr:.2f}")
                    if rt < 0.01:
                        rt_strs.append(f"{rt*1000:.1f}\\,ms")
                    else:
                        rt_strs.append(f"{rt:.2f}")
                else:
                    ocr_strs.append(r"\text{---}")
                    rt_strs.append(r"\text{---}")

            prefix = f"\\textbf{{{scen_disp}}}" if p_idx == 0 else ""
            line = f"{prefix} & {p_disp} & " + " & ".join(ocr_strs) + " & " + " & ".join(rt_strs) + r" \\"
            lines_wide.append(line)

        if s_idx < len(scenarios) - 1:
            lines_wide.append(r"\midrule")

    lines_wide.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    tex_wide = "\n".join(lines_wide)
    wide_file = out_path / f"table_{meta['short_name']}_sensitivity_wide.tex"
    wide_file.write_text(tex_wide)

    # 3. Compact Single-Column Table (Stacked Metrics)
    lines_compact: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.8pt}",
        f"\\caption{{Sensitivity of ACTIS to {meta['display_name']} ($\delta = 0.05$, $N=1000$ trials).}}",
        f"\\label{{tab:compact_sensitivity_{meta['short_name']}}}",
        r"\begin{tabular}{ll rrrr}",
        r"\toprule",
        f"\\textbf{{Workload}} & \\textbf{{{tex_sym}}} & "
        + " & ".join([f"$\\gamma{{=}}{g:.2f}$" for g in unique_gammas])
        + r" \\",
        r"\midrule",
    ]

    for s_idx, scen in enumerate(scenarios):
        scen_sub = df_plot[df_plot["scenario"] == scen]
        scen_disp = scen_sub["scenario_display"].iloc[0]

        lines_compact.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{scen_disp}}}}} \\\\")
        for p_idx, p_val in enumerate(unique_params):
            p_sub = scen_sub[scen_sub["param_num"] == p_val]
            p_disp = f"{int(p_val)}" if meta["is_int"] else f"{p_val:.2g}"

            ocr_strs = []
            for g in unique_gammas:
                row = p_sub[p_sub["gamma"] == g]
                if not row.empty:
                    ocr = row["ocr_mean"].iloc[0]
                    ocr_strs.append(f"{ocr:.2f}")
                else:
                    ocr_strs.append(r"\text{---}")

            lines_compact.append(f" & {p_disp} & " + " & ".join(ocr_strs) + r" \\")

        if s_idx < len(scenarios) - 1:
            lines_compact.append(r"\addlinespace[3pt]")

    lines_compact.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_compact = "\n".join(lines_compact)
    compact_file = out_path / f"table_{meta['short_name']}_sensitivity.tex"
    compact_file.write_text(tex_compact)

    print(f"Generated tables:\n  - {wide_file}\n  - {compact_file}\n  - {csv_file}")
    return {"wide": tex_wide, "compact": tex_compact}


def plot_sensitivity_dual_panel(
    df: pd.DataFrame,
    param_name: str = "batch_size",
    output_dir: Path | str = "experiments/plots/sensitivity",
    skip_pgf: bool = False,
) -> None:
    """Creates publication-ready single-column sensitivity figures.

    1. Tuner wall-clock latency (seconds, log scale) vs. parameter across scenarios.
    2. Oracle Call Rate (%) vs. parameter across scenarios.
    3. Jackson dual-panel figure (latency on left, OCR on right) using plotnine composition.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta = get_param_meta(param_name)
    df_plot = df[df["delta"] == 0.05].copy() if 0.05 in df["delta"].values else df.copy()

    # 1. Plot Latency
    p_latency = (
        ggplot(
            df_plot,
            aes(
                x="param_num",
                y="runtime_mean",
                color="gamma_str",
                shape="gamma_str",
                group="gamma_str",
            ),
        )
        + geom_line()
        + geom_point()
        + facet_wrap("~scenario_display", scales="free_y", ncol=2)
        + scale_color_manual(values=GAMMA_COLORS, name=r"Target $\gamma$")
        + scale_shape_manual(values=GAMMA_SHAPES, name=r"Target $\gamma$")
        + labs(
            x=meta["display_name"],
            y="Latency (s)",
        )
        + theme_bw()
        + theme(
            panel_grid_minor=element_blank(),
            legend_position="right",
            #axis_title_x=element_text(margin={"t": 10, "unit": "pt"}),
        )
    )

    if meta["log_x"]:
        p_latency += scale_x_log10()

    p_latency += scale_y_log10()

    _save_plot(
        p_latency,
        out_path,
        f"sensitivity_{meta['short_name']}_latency",
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.5,
        skip_pgf=skip_pgf,
    )

    # 2. Plot Oracle Call Rate
    p_ocr = (
        ggplot(
            df_plot,
            aes(
                x="param_num",
                y="ocr_mean",
                color="gamma_str",
                shape="gamma_str",
                group="gamma_str",
            ),
        )
        + geom_line()
        + geom_point()
        + facet_wrap("~scenario_display", scales="free_y", ncol=2)
        + scale_color_manual(values=GAMMA_COLORS, name=r"Target $\gamma$")
        + scale_shape_manual(values=GAMMA_SHAPES, name=r"Target $\gamma$")
        + labs(
            x=meta["display_name"],
            y="OCR (%)",
        )
        + theme_bw()
        + theme(
            panel_grid_minor=element_blank(),
            #strip_background=element_rect(fill="#f0f0f0", color="#cccccc"),
            legend_position="right",
            #axis_title_x=element_text(margin={"t": 10, "unit": "pt"}),
        )
    )

    if meta["log_x"]:
        p_ocr += scale_x_log10()

    _save_plot(
        p_ocr,
        out_path,
        f"sensitivity_{meta['short_name']}_ocr",
        width=SIGMOD_COL_WIDTH,
        height=SCALE_FACTOR * 1.5,
        skip_pgf=skip_pgf,
    )

    # 3. Combined Side-by-Side Figure for Jackson (Latency on Left, OCR on Right)
    df_jackson = df_plot[df_plot["scenario"] == "jackson"].copy()
    if not df_jackson.empty:
        theme_common = theme_bw() + theme(
            panel_grid_minor=element_blank(),
            #axis_title_x=element_text(margin={"t": 10, "unit": "pt"}),
        )

        p_jk_lat = (
            ggplot(
                df_jackson,
                aes(
                    x="param_num",
                    y="runtime_mean",
                    color="gamma_str",
                    shape="gamma_str",
                    group="gamma_str",
                ),
            )
            + geom_line()
            + geom_point()
            + scale_color_manual(values=GAMMA_COLORS, name=r"Target $\gamma$")
            + scale_shape_manual(values=GAMMA_SHAPES, name=r"Target $\gamma$")
            + scale_y_log10()
            + labs(
                x=meta["display_name"],
                y="Latency (s)",
            )
            + theme_common
            + theme(legend_position="none")
        )

        p_jk_ocr = (
            ggplot(
                df_jackson,
                aes(
                    x="param_num",
                    y="ocr_mean",
                    color="gamma_str",
                    shape="gamma_str",
                    group="gamma_str",
                ),
            )
            + geom_line()
            + geom_point()
            + scale_color_manual(values=GAMMA_COLORS, name=r"Target $\gamma$")
            + scale_shape_manual(values=GAMMA_SHAPES, name=r"Target $\gamma$")
            + labs(
                x=meta["display_name"],
                y="OCR (%)",
            )
            + theme_common
            + theme(
                legend_position="right",
                #legend_position=(1.1, 1.0),
                #legend_justification=("left", "top"),
                #legend_justification_top=0.0
            )
            + guides(
                color=guide_legend(ncol=1),
                shape=guide_legend(ncol=1),
            )
        )

        if meta["log_x"]:
            p_jk_lat += scale_x_log10()
            p_jk_ocr += scale_x_log10()

        p_jackson = p_jk_lat | p_jk_ocr

        _save_plot(
            p_jackson,
            out_path,
            f"sensitivity_{meta['short_name']}_jackson_dual",
            width=SIGMOD_COL_WIDTH,
            height=SCALE_FACTOR * 1.2,
            skip_pgf=skip_pgf,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-ready parameter sensitivity analysis for ACTIS."
    )
    parser.add_argument(
        "--param",
        type=str,
        default="batch_size",
        help="Hyperparameter name to evaluate (default: 'batch_size'). "
        "Can also be 'alpha', 'num_thresholds', etc.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing hierarchical JSON results. "
        "Defaults to 'experiments/results_<param>' or 'experiments/results_batch_size'.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="experiments/plots/sensitivity",
        help="Output directory for generated PDF and PGF figures (default: experiments/plots/sensitivity).",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="experiments/tables",
        help="Output directory for generated LaTeX tables and CSVs (default: experiments/tables).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional filter for specific scenario names (e.g. 'jackson' 'pubmed').",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=None,
        help="Optional filter for specific gamma targets (e.g. 0.70 0.80 0.90 0.95).",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=None,
        help="Optional filter for specific delta budgets (default: all available).",
    )
    parser.add_argument(
        "--skip-pgf",
        "--no-pgf",
        dest="skip_pgf",
        action="store_true",
        help="Skip generating PGF format plots (generate only PDF).",
    )

    args = parser.parse_args()

    param_meta = get_param_meta(args.param)
    results_dir = args.results_dir if args.results_dir is not None else param_meta["default_dir"]

    print(f"=== Parameter Sensitivity Analysis for '{args.param}' ===")
    print(f"Loading results from: {results_dir}")

    df = load_sensitivity_data(
        results_dir=results_dir,
        param_name=args.param,
        scenarios=args.scenarios,
        deltas=args.deltas,
        gammas=args.gammas,
    )

    if df.empty:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} configurations across scenarios: {df['scenario'].unique().tolist()}")
    print(f"Varied values for '{args.param}': {sorted(df['param_num'].unique().tolist())}")

    # Generate LaTeX tables and CSV
    print(f"\nGenerating LaTeX tables in: {args.tables_dir}")
    generate_sensitivity_latex_tables(df, param_name=args.param, output_dir=args.tables_dir)

    # Generate plots
    print(f"\nGenerating publication plots in: {args.plots_dir}")
    plot_sensitivity_dual_panel(
        df,
        param_name=args.param,
        output_dir=args.plots_dir,
        skip_pgf=args.skip_pgf,
    )

    print("\nParameter sensitivity analysis complete!")


if __name__ == "__main__":
    main()

