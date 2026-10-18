#!/usr/bin/env python3
"""Statistical Soundness Analysis for Semantic Cascades and Baselines.

Analyzes experimental results across real benchmark datasets, evaluating whether
methods satisfy their target failure probability guarantee (delta) using exact
Clopper-Pearson binomial confidence intervals at significance level alpha (e.g. 0.05).

Classifications per configuration:
- Certified Sound: UCB < delta (guarantee satisfied with high confidence)
- Inconclusive: LCB <= delta <= UCB (confidence interval covers delta)
- Certified Unsound: LCB > delta (statistically significant violation of guarantee)

Generates:
1. Summary table: Evaluates guarantees averaged across quality targets gamma,
   reporting % Sound, % Inconclusive, % Unsound, and empirical failure rates
   (conditional mean and max) expressed as decimals to facilitate direct
   comparison with delta.
2. Disaggregated table: Reports the same soundness classifications and decimal
   failure rates broken down across (delta, gamma) parameter pairs.
3. Detailed appendix tables reporting empirical failure rates and 95% CIs for each
   dataset / query / proxy / parameter configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tabulate import tabulate

# Ensure repository root is on sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.results_utils import (  # noqa: E402
    METHOD_ORDER,
    SCENARIO_DISPLAY_NAMES,
)
from experiments.soundness import (  # noqa: E402
    build_appendix_configuration_table,
    load_soundness_dataframe,
    summarize_soundness_by_delta,
    summarize_soundness_by_delta_gamma,
)


def _sort_summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts DataFrame rows by delta, gamma (if present), and METHOD_ORDER."""
    if "method_name" not in df.columns:
        return df

    def _method_key(name: str) -> int:
        return METHOD_ORDER.index(name) if name in METHOD_ORDER else 999

    df = df.copy()
    df["_method_sort"] = df["method_name"].map(_method_key)
    sort_cols = ["delta"] if "delta" in df.columns else []
    if "gamma" in df.columns:
        sort_cols.append("gamma")
    sort_cols.append("_method_sort")
    return df.sort_values(sort_cols).drop(columns=["_method_sort"])


def format_summary_terminal(
    summary_df: pd.DataFrame,
    target: str = "joint",
    disaggregated: bool = False,
) -> str:
    """Formats summary statistics as an aligned terminal table using tabulate."""
    sorted_df = _sort_summary_by_method(summary_df)

    display = sorted_df.copy()
    display["delta_str"] = display["delta"].apply(lambda d: f"{d:.2f}")
    if disaggregated:
        display["gamma_str"] = display["gamma"].apply(lambda g: f"{g:.2f}")
    display["pct_sound_str"] = display["pct_sound"].apply(lambda x: f"{x:.1f}%")
    display["pct_inconcl_str"] = display["pct_inconclusive"].apply(
        lambda x: f"{x:.1f}%"
    )
    display["pct_unsound_str"] = display["pct_unsound"].apply(lambda x: f"{x:.1f}%")
    display["cond_mean_str"] = display["cond_mean_fail_pct"].apply(
        lambda x: f"{x / 100.0:.3f}" if not pd.isna(x) else "—"
    )
    display["max_fail_str"] = display["max_fail_pct"].apply(
        lambda x: f"{x / 100.0:.3f}"
    )

    cols = ["delta_str"]
    headers = ["delta"]
    if disaggregated:
        cols.append("gamma_str")
        headers.append("gamma")
    cols.extend(
        [
            "method_name",
            "N",
            "pct_sound_str",
            "pct_inconcl_str",
            "pct_unsound_str",
            "cond_mean_str",
            "max_fail_str",
        ]
    )
    headers.extend(
        [
            "Method",
            "N",
            "Sound",
            "Inconcl.",
            "Unsound",
            "Cond. Mean Fail",
            "Max Fail",
        ]
    )

    disagg_tag = "Disaggregated by gamma" if disaggregated else "Averaged across gamma"
    title = f"Statistical Soundness Summary ({target.upper()} Target, {disagg_tag})"
    table_str = tabulate(
        display[cols],
        headers=headers,
        tablefmt="simple",
        showindex=False,
        disable_numparse=True,
    )
    return f"\n{title}\n{'=' * len(title)}\n{table_str}\n"


def format_summary_latex(
    summary_df: pd.DataFrame,
    target: str = "joint",
    disaggregated: bool = False,
) -> str:
    """Formats summary table as publication-ready LaTeX with booktabs.

    Sound, Inconclusive, and Unsound rates are percentages.
    Conditional mean failure rate p_unsound and Max are expressed as decimals
    to facilitate direct comparison with the user-specified delta.
    """
    sorted_df = _sort_summary_by_method(summary_df)

    label_type = "disaggregated" if disaggregated else "summary"
    label_name = f"tab:soundness_{label_type}_{target}"

    disagg_desc = (
        r"disaggregated by error budget $\delta$ and quality target $\gamma$"
        if disaggregated
        else r"averaged across quality targets $\gamma \in \{0.70, 0.80, 0.90, 0.95\}$"
    )
    caption_text = (
        f"\\caption{{Statistical soundness evaluation for {target} guarantee "
        f"violations across real benchmarks {disagg_desc} at significance level "
        r"$\alpha = 0.05$ ($N = 1000$ trials per configuration). "
        r"Sound ($\text{UCB} < \delta$), Inconclusive "
        r"($\text{LCB} \le \delta \le \text{UCB}$), and Unsound "
        r"($\text{LCB} > \delta$) rates computed via exact Clopper-Pearson "
        r"binomial confidence intervals. $\bar{p}_{\text{unsound}}$ denotes the "
        r"mean empirical failure rate conditioned on certified unsound configurations; "
        r"$\bar{p}_{\text{unsound}}$ and \textbf{Max} are expressed as decimals "
        r"for direct comparison with $\delta$.}"
    )

    env_name = "table*" if disaggregated else "table"
    lines = [
        f"\\begin{{{env_name}}}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        caption_text,
        f"\\label{{{label_name}}}",
    ]

    if disaggregated:
        lines.append(r"\begin{tabular}{lll rrr rr}")
        lines.append(r"\toprule")
        lines.append(
            r"$\delta$ & $\gamma$ & Method & \textbf{Sound} (\%) & "
            r"\textbf{Inconcl.} (\%) & \textbf{Unsound} (\%) & "
            r"$\bar{p}_{\text{unsound}}$ & \textbf{Max} \\"
        )
    else:
        lines.append(r"\begin{tabular}{ll rrr rr}")
        lines.append(r"\toprule")
        lines.append(
            r"$\delta$ & Method & \textbf{Sound} (\%) & \textbf{Inconcl.} (\%) & "
            r"\textbf{Unsound} (\%) & $\bar{p}_{\text{unsound}}$ & \textbf{Max} \\"
        )
    lines.append(r"\midrule")

    prev_delta = None
    prev_gamma = None

    for _, row in sorted_df.iterrows():
        d_val = float(row["delta"])
        d_str = f"{d_val:.2f}"
        method_str = str(row["method_name"]).replace("_", " ")

        if disaggregated:
            g_val = float(row["gamma"])
            g_str = f"{g_val:.2f}"
            if prev_delta is not None and (d_val != prev_delta or g_val != prev_gamma):
                lines.append(r"\addlinespace")
            d_cell = d_str if d_val != prev_delta else ""
            g_cell = g_str if (d_val != prev_delta or g_val != prev_gamma) else ""
            prev_delta = d_val
            prev_gamma = g_val
            prefix_cells = [d_cell, g_cell, method_str]
        else:
            if prev_delta is not None and d_val != prev_delta:
                lines.append(r"\midrule")
            d_cell = d_str if d_val != prev_delta else ""
            prev_delta = d_val
            prefix_cells = [d_cell, method_str]

        pct_s = f"{row['pct_sound']:.1f}"
        pct_i = f"{row['pct_inconclusive']:.1f}"
        pct_u = f"{row['pct_unsound']:.1f}"
        cond_m = (
            f"{row['cond_mean_fail_pct'] / 100.0:.3f}"
            if not pd.isna(row["cond_mean_fail_pct"])
            else r"\text{---}"
        )
        max_f = f"{row['max_fail_pct'] / 100.0:.3f}"

        data_cells = [pct_s, pct_i, pct_u, cond_m, max_f]
        lines.append(" & ".join(prefix_cells + data_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\end{{{env_name}}}")
    return "\n".join(lines)


def format_appendix_latex(
    appendix_df: pd.DataFrame,
    delta: Optional[float] = None,
    target: str = "joint",
) -> str:
    """Formats appendix table reporting empirical failure rate and two-sided

    95% Clopper-Pearson CIs [LCB, UCB] per method as publication-ready LaTeX.
    """
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
    ]
    delta_caption = f" for $\\delta = {delta:.2f}$" if delta is not None else ""
    lines.append(
        (
            f"\\caption{{Empirical failure rate (\\%) with two-sided 95\\% Clopper-"
            f"Pearson confidence interval [LCB, UCB] for {target} guarantee violations "
            f"across benchmarks and parameter configurations{delta_caption}. "
            r"Asterisks ($^*$) indicate configurations certified unsound "
            r"($\text{LCB} > \delta$).}"
        )
    )
    label_suffix = f"delta_{int(delta * 100):03d}" if delta is not None else "all"
    lines.append(f"\\label{{tab:appendix_soundness_{target}_{label_suffix}}}")

    method_cols = [
        c
        for c in appendix_df.columns
        if c not in ["dataset", "query_str", "proxy_str", "delta", "gamma"]
    ]
    ordered_methods = sorted(
        method_cols,
        key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999,
    )

    col_spec = (
        "lll " + ("l " if delta is None else "") + "c " + "c" * len(ordered_methods)
    )
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header_parts = ["Benchmark", "Query", "Proxy"]
    if delta is None:
        header_parts.append(r"$\delta$")
    header_parts.append(r"$\gamma$")
    header_parts.extend([f"\\textbf{{{m}}}" for m in ordered_methods])
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    def _fmt_proxy(val: Any) -> str:
        s = str(val).lower()
        if not val or s in ("—", "none", "nan"):
            return r"\text{---}"
        if "luna" in s:
            return "GPT-5.6-luna"
        if "llama" in s:
            return "Llama-3.2-1B"
        return str(val).split("/")[-1]

    def _fmt_query(val: Any) -> str:
        s = str(val)
        if not val or s in ("—", "none", "nan"):
            return r"\text{---}"
        clean = s.lstrip("q").lstrip("Q")
        if clean.endswith("_ext"):
            return f"Q{clean[:-4]} (ext)"
        return f"Q{clean}"

    prev_group = None
    for _, row in appendix_df.iterrows():
        scen = row["dataset"]
        display_name = SCENARIO_DISPLAY_NAMES.get(scen, scen.title())
        qid_str = _fmt_query(row["query_str"])
        proxy_str = _fmt_proxy(row["proxy_str"])
        gamma_str = f"{row['gamma']:.2f}"

        group_key = (scen, qid_str, proxy_str)
        if prev_group is not None and group_key != prev_group:
            lines.append(r"\addlinespace")
        prev_group = group_key

        row_cells = [display_name, qid_str, proxy_str]
        if delta is None:
            row_cells.append(f"{row['delta']:.2f}")
        row_cells.append(gamma_str)

        for m in ordered_methods:
            val = str(row.get(m, r"\text{---}"))
            if "*" in val:
                val = val.replace("*", r"\rlap{$^*$}")
            row_cells.append(val)

        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate statistical soundness of semantic cascade tuners."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Root directory containing hierarchical experiment results.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Filter results by number of trials (default: 1000).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for two-sided exact CI (default: 0.05).",
    )
    parser.add_argument(
        "--target",
        choices=["joint", "recall", "precision", "f1", "all"],
        default="all",
        help="Failure target metric to evaluate (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "latex", "csv", "all"],
        default="all",
        help="Output display format (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/tables",
        help="Directory to write generated LaTeX tables and CSVs.",
    )
    parser.add_argument(
        "--disaggregated",
        action="store_true",
        help="Disaggregate summary tables by both delta and gamma.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of deltas to filter.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of gammas to filter.",
    )
    parser.add_argument(
        "--appendix-tables",
        action="store_true",
        help="Generate detailed per-scenario appendix configuration tables.",
    )
    parser.add_argument(
        "--use-harmonic-scaledoc",
        action="store_true",
        help="Audit ScaleDoc using harmonic target instead of native F1 target.",
    )
    return parser.parse_args()


def run_analysis_for_target(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    fmt: str,
    disaggregated: bool,
    generate_appendix: bool,
) -> None:
    """Runs soundness analysis and exports tables for a specific failure target."""
    print(f"\n>>> Running Soundness Analysis for Target: {target.upper()} <<<")

    # 1. Summary table (always averaged across gamma)
    summary_delta = summarize_soundness_by_delta(df, target=target)
    summary_delta["cond_mean_fail"] = summary_delta["cond_mean_fail_pct"] / 100.0
    summary_delta["max_fail"] = summary_delta["max_fail_pct"] / 100.0

    if fmt in ("table", "all"):
        print(
            format_summary_terminal(summary_delta, target=target, disaggregated=False)
        )

    if fmt in ("latex", "all"):
        tex = format_summary_latex(summary_delta, target=target, disaggregated=False)
        out_tex = output_dir / f"table_soundness_summary_{target}.tex"
        out_tex.write_text(tex, encoding="utf-8")
        print(f"  [SAVED] {out_tex}")

    if fmt in ("csv", "all"):
        out_csv = output_dir / f"soundness_summary_{target}.csv"
        _sort_summary_by_method(summary_delta).to_csv(out_csv, index=False)
        print(f"  [SAVED] {out_csv}")

    # 2. Disaggregated table (broken down by delta and gamma)
    if disaggregated or fmt == "all":
        summary_dg = summarize_soundness_by_delta_gamma(df, target=target)
        summary_dg["cond_mean_fail"] = summary_dg["cond_mean_fail_pct"] / 100.0
        summary_dg["max_fail"] = summary_dg["max_fail_pct"] / 100.0

        if fmt in ("table", "all") and disaggregated:
            print(
                format_summary_terminal(summary_dg, target=target, disaggregated=True)
            )

        if fmt in ("latex", "all"):
            tex_dg = format_summary_latex(summary_dg, target=target, disaggregated=True)
            out_tex_dg = output_dir / f"table_soundness_disaggregated_{target}.tex"
            out_tex_dg.write_text(tex_dg, encoding="utf-8")
            print(f"  [SAVED] {out_tex_dg}")

        if fmt in ("csv", "all"):
            out_csv_dg = output_dir / f"soundness_disaggregated_{target}.csv"
            _sort_summary_by_method(summary_dg).to_csv(out_csv_dg, index=False)
            print(f"  [SAVED] {out_csv_dg}")

    # 3. Appendix Configuration Tables
    if generate_appendix:
        unique_deltas = sorted(df["delta"].unique())
        for d in unique_deltas:
            app_df = build_appendix_configuration_table(df, target=target, delta=d)
            d_tag = f"delta_{int(d * 100):03d}"

            if fmt in ("latex", "all"):
                tex_app = format_appendix_latex(app_df, delta=d, target=target)
                out_tex_app = (
                    output_dir / f"table_soundness_appendix_{target}_{d_tag}.tex"
                )
                out_tex_app.write_text(tex_app, encoding="utf-8")
                print(f"  [SAVED] {out_tex_app}")

            if fmt in ("csv", "all"):
                out_csv_app = output_dir / f"soundness_appendix_{target}_{d_tag}.csv"
                app_df.to_csv(out_csv_app, index=False)
                print(f"  [SAVED] {out_csv_app}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading results from '{args.results_dir}' "
        f"(N={args.num_trials}, alpha={args.alpha})...",
        file=sys.stderr,
    )
    df = load_soundness_dataframe(
        results_dir=args.results_dir,
        num_trials=args.num_trials,
        deltas=args.delta,
        gammas=args.gamma,
        alpha=args.alpha,
        use_native_scaledoc=not args.use_harmonic_scaledoc,
    )

    if df.empty:
        print(
            "No matching result records found. Check directory path and filters.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_benchmarks = len(df[["dataset", "query_id", "proxy_model"]].drop_duplicates())
    print(
        f"Loaded {len(df)} records across {df['method_name'].nunique()} methods, "
        f"{n_benchmarks} benchmarks, and {df['dataset'].nunique()} datasets.",
        file=sys.stderr,
    )

    targets = (
        ["joint", "recall", "precision"] if args.target == "all" else [args.target]
    )
    for tgt in targets:
        run_analysis_for_target(
            df=df,
            target=tgt,
            output_dir=output_dir,
            fmt=args.format,
            disaggregated=args.disaggregated,
            generate_appendix=args.appendix_tables,
        )


if __name__ == "__main__":
    main()
