#!/usr/bin/env python3
"""
Compute and display dataset summary and calibration statistics across experimental
scenarios.

Statistics reported per scenario (and, for document benchmarks, per query and
model combination):
- N_total: Original number of items in the dataset before missing-value filtering.
- N_missing: Number of items with a missing oracle label or proxy score.
- Pos. rate: Fraction of non-missing items whose oracle label is positive (pi).
- PR-AUC: Area under the precision-recall curve (Average Precision).
- Cal. bias: Signed calibration bias (mean proxy score minus true positive rate,
  s_bar - pi). Positive indicates systematic overconfidence, negative indicates
  systematic underconfidence.
- ECE: Expected Calibration Error using 10 equal-width bins on [0, 1].
- Sat. rate: Fraction of proxy scores equal to 0.0 or 1.0 (boundary saturation).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tabulate import tabulate

# Ensure repository root is on sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from experiments.data_prep import DATASET_GROUPS  # noqa: E402
from experiments.results_utils import CANONICAL_SCENARIO_ORDER  # noqa: E402
from experiments.scenarios import (  # noqa: E402
    SUPG_DATASETS,
    get_available_queries,
    get_scenario,
)

__all__ = [
    "CANONICAL_SCENARIO_ORDER",
    "DOCUMENT_BENCHMARK_SCENARIOS",
    "SUPG_SCENARIOS",
    "collect_all_stats",
    "main",
    "parse_args",
]

SUPG_SCENARIOS: list[str] = [s for s in CANONICAL_SCENARIO_ORDER if s in SUPG_DATASETS]

DOCUMENT_BENCHMARK_SCENARIOS: list[str] = [
    s for s in CANONICAL_SCENARIO_ORDER if s in DATASET_GROUPS
]

_GROUP_ORDER: dict[str, int] = {
    "supg": 0,
    "bargain": 1,
    "scaledoc": 2,
}

_DISPLAY_COLUMNS = [
    "Scenario",
    "Query",
    "Oracle model",
    "Proxy model",
    "N",
    "Missing",
    "Pos. rate",
    "PR-AUC",
    "Cal. bias",
    "ECE",
    "Sat. rate",
]


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute PR-AUC (Average Precision) for binary labels and continuous scores.

    Args:
        scores: 1D array of proxy confidence scores.
        labels: 1D boolean array of oracle labels.

    Returns:
        Average precision score in [0.0, 1.0], or NaN if only one class is present.
    """
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _calibration_bias(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute signed calibration bias (mean proxy score minus true positive rate).

    A positive value indicates systematic overconfidence (over-predicting positive
    probabilities), while a negative value indicates systematic underconfidence.

    Args:
        scores: 1D array of proxy confidence scores.
        labels: 1D boolean array of oracle labels.

    Returns:
        Signed mean calibration error in [-1.0, 1.0].
    """
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores) - np.mean(labels))


def _expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins on [0, 1].

    Args:
        scores: 1D array of proxy confidence scores.
        labels: 1D boolean array of oracle labels.
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        Expected calibration error in [0.0, 1.0].
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (scores >= b_low) & (scores <= b_high)
        else:
            mask = (scores >= b_low) & (scores < b_high)
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(scores[mask]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _saturation_rate(scores: np.ndarray) -> float:
    """Compute fraction of saturated/extreme proxy scores equal to 0.0 or 1.0.

    Args:
        scores: 1D array of proxy confidence scores.

    Returns:
        Fraction of scores in {0.0, 1.0} in [0.0, 1.0].
    """
    if len(scores) == 0:
        return float("nan")
    return float(np.mean((scores == 0.0) | (scores == 1.0)))


def _stats_from_population(
    scores: np.ndarray,
    labels: np.ndarray,
    n_total: int,
) -> dict[str, Any]:
    """Compute summary statistics from pre-filtered score/label arrays.

    Args:
        scores: Array of proxy scores for valid items.
        labels: Array of oracle boolean labels for valid items.
        n_total: Original dataset size before missing-value filtering.

    Returns:
        Dictionary with N_total, N_missing, pos_rate, pr_auc, cal_bias, ece,
        and sat_rate.
    """
    n_valid = len(scores)
    n_missing = n_total - n_valid
    pos_rate = float(labels.mean()) if n_valid > 0 else float("nan")
    pr_auc = _pr_auc(scores, labels)
    cal_bias = _calibration_bias(scores, labels)
    ece = _expected_calibration_error(scores, labels)
    sat_rate = _saturation_rate(scores)
    return {
        "N_total": n_total,
        "N_missing": n_missing,
        "pos_rate": pos_rate,
        "pr_auc": pr_auc,
        "cal_bias": cal_bias,
        "ece": ece,
        "sat_rate": sat_rate,
    }


def _discover_model_slugs(base_dir: Path, qid: str) -> tuple[list[str], list[str]]:
    """Scan oracle/q{qid} and proxy/q{qid} subdirectories under base_dir and
    return the list of model slugs (parquet stems) found in each.

    Args:
        base_dir: Base directory path for the dataset.
        qid: Query identifier.

    Returns:
        Tuple of (oracle_slugs, proxy_slugs).
    """
    oracle_dir = base_dir / "oracle" / f"q{qid}"
    proxy_dir = base_dir / "proxy" / f"q{qid}"

    oracle_slugs = (
        sorted(p.stem for p in oracle_dir.glob("*.parquet"))
        if oracle_dir.is_dir()
        else []
    )
    proxy_slugs = (
        sorted(p.stem for p in proxy_dir.glob("*.parquet"))
        if proxy_dir.is_dir()
        else []
    )
    return oracle_slugs, proxy_slugs


def _slug_to_model_name(slug: str) -> str:
    """Convert a filesystem slug (e.g. 'azure-gpt-4o') back to a model name."""
    return re.sub(r"^([a-z]+)-", r"\1/", slug, count=1)


def _collect_supg_stats(name: str) -> list[dict[str, Any]]:
    """Stats for a SUPG benchmark dataset.

    Uses get_scenario() -> TabularDataset. N_total comes from get_dataframe()
    (the raw file), while scores/labels come from generate_population() after
    missing-row filtering.
    """
    scen = get_scenario(name)
    n_total = len(scen.get_dataframe())  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pop = scen.generate_population()
    stats = _stats_from_population(pop.scores, pop.labels, n_total=n_total)
    return [
        {
            "scenario": name,
            "group": "supg",
            "query": "—",
            "oracle_model": "—",
            "proxy_model": "—",
            **stats,
        }
    ]


def _collect_document_benchmark_stats(
    name: str,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Stats for all (query x oracle_model x proxy_model) combinations found on
    disk for a ScaleDoc or BARGAIN dataset.
    """
    group = DATASET_GROUPS.get(name, "")
    base_dir = (
        Path(data_dir) / name
        if data_dir is not None
        else Path("experiments/data") / group / name
    )

    queries = get_available_queries(name, data_dir=data_dir)
    rows: list[dict[str, Any]] = []

    for qid in queries:
        oracle_slugs, proxy_slugs = _discover_model_slugs(base_dir, qid)
        if not oracle_slugs or not proxy_slugs:
            continue

        for oracle_slug in oracle_slugs:
            oracle_model = _slug_to_model_name(oracle_slug)
            for proxy_slug in proxy_slugs:
                proxy_model = _slug_to_model_name(proxy_slug)

                scen = get_scenario(
                    name,
                    query_id=qid,
                    oracle_model=oracle_model,
                    proxy_model=proxy_model,
                    **({"data_dir": data_dir} if data_dir is not None else {}),
                )

                n_total = len(scen.get_dataframe())  # type: ignore[attr-defined]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pop = scen.generate_population()

                stats = _stats_from_population(pop.scores, pop.labels, n_total=n_total)
                rows.append(
                    {
                        "scenario": name,
                        "group": group,
                        "query": qid,
                        "oracle_model": oracle_slug,
                        "proxy_model": proxy_slug,
                        **stats,
                    }
                )

    return rows


def collect_all_stats(
    include_supg: bool = True,
    include_benchmarks: bool = True,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Collect statistics for all available scenarios into a DataFrame."""
    rows: list[dict[str, Any]] = []

    if include_supg:
        for name in SUPG_SCENARIOS:
            try:
                rows.extend(_collect_supg_stats(name))
            except Exception as exc:
                print(f"  [WARN] Skipping SUPG '{name}': {exc}", file=sys.stderr)

    if include_benchmarks:
        for name in DOCUMENT_BENCHMARK_SCENARIOS:
            try:
                rows.extend(_collect_document_benchmark_stats(name, data_dir=data_dir))
            except Exception as exc:
                print(f"  [WARN] Skipping benchmark '{name}': {exc}", file=sys.stderr)

    return pd.DataFrame(rows)


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build a display-friendly DataFrame with formatted columns."""
    if df.empty:
        return pd.DataFrame(columns=_DISPLAY_COLUMNS + ["_group"])

    display = df.copy()

    # Sort: group order -> canonical scenario order -> query (numeric) -> models
    display["_group_order"] = display["group"].map(lambda g: _GROUP_ORDER.get(g, 99))
    display["_scen_order"] = display["scenario"].map(
        lambda s: (
            CANONICAL_SCENARIO_ORDER.index(s) if s in CANONICAL_SCENARIO_ORDER else 999
        )
    )
    display["_query_num"] = pd.to_numeric(
        display["query"].replace("—", "-1"), errors="coerce"
    ).fillna(-1)
    display = display.sort_values(
        ["_group_order", "_scen_order", "_query_num", "oracle_model", "proxy_model"]
    ).drop(columns=["_group_order", "_scen_order", "_query_num"])

    display["N"] = display["N_total"].apply(lambda x: f"{x:,}")
    display["Missing"] = display["N_missing"].apply(lambda x: f"{x:,}")
    display["Pos. rate"] = display["pos_rate"].apply(
        lambda x: f"{x:.3f}" if not np.isnan(x) else "—"
    )
    display["PR-AUC"] = display["pr_auc"].apply(
        lambda x: f"{x:.3f}" if not np.isnan(x) else "—"
    )
    display["Cal. bias"] = display["cal_bias"].apply(
        lambda x: f"{x:+.3f}" if not np.isnan(x) else "—"
    )
    display["ECE"] = display["ece"].apply(
        lambda x: f"{x:.3f}" if not np.isnan(x) else "—"
    )
    display["Sat. rate"] = display["sat_rate"].apply(
        lambda x: f"{x * 100:.1f}%" if not np.isnan(x) else "—"
    )
    display["_group"] = display["group"]

    return display.rename(
        columns={
            "scenario": "Scenario",
            "query": "Query",
            "oracle_model": "Oracle model",
            "proxy_model": "Proxy model",
        }
    )[_DISPLAY_COLUMNS + ["_group"]]


def _format_table(df: pd.DataFrame) -> str:
    """Format DataFrame as an aligned terminal table using tabulate.

    Wraps long model and scenario names to prevent terminal line blowouts.
    """
    display = _build_display_df(df)
    if display.empty:
        return "No data found."

    # Max column widths to wrap wide model identifiers
    # Columns: Scenario, Query, Oracle model, Proxy model, N, Missing,
    # Pos. rate, PR-AUC, Cal. bias, ECE, Sat. rate
    maxcolwidths = [None, None, 20, 22, None, None, None, None, None, None, None]
    return tabulate(
        display[_DISPLAY_COLUMNS],
        headers="keys",
        tablefmt="simple",
        showindex=False,
        maxcolwidths=maxcolwidths,
        disable_numparse=True,
    )


def _format_csv(df: pd.DataFrame) -> str:
    """Format DataFrame as CSV string."""
    return _build_display_df(df)[_DISPLAY_COLUMNS].to_csv(index=False)


def _format_latex(df: pd.DataFrame) -> str:
    """Format DataFrame as a publication-ready LaTeX table."""
    display = _build_display_df(df)
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Dataset statistics for all experimental scenarios.}",
        r"  \label{tab:dataset-stats}",
        r"  \begin{tabular}{llllrrrrrrr}",
        r"    \toprule",
        "    "
        + " & ".join(
            [
                r"\textbf{Scenario}",
                r"\textbf{Query}",
                r"\textbf{Oracle model}",
                r"\textbf{Proxy model}",
                r"\textbf{$N$}",
                r"\textbf{Missing}",
                r"\textbf{Pos. rate}",
                r"\textbf{PR-AUC}",
                r"\textbf{Cal. bias}",
                r"\textbf{ECE}",
                r"\textbf{Sat. rate}",
            ]
        )
        + r" \\",
        r"    \midrule",
    ]

    prev_group = None
    for _, row in display.iterrows():
        grp = row["_group"]
        if prev_group is not None and grp != prev_group:
            lines.append(r"    \midrule")
        prev_group = grp

        def _esc(s: str) -> str:
            return str(s).replace("_", r"\_").replace("%", r"\%")

        cells = [
            _esc(row["Scenario"]),
            _esc(row["Query"]),
            _esc(row["Oracle model"]),
            _esc(row["Proxy model"]),
            str(row["N"]),
            str(row["Missing"]),
            str(row["Pos. rate"]),
            str(row["PR-AUC"]),
            str(row["Cal. bias"]),
            str(row["ECE"]),
            _esc(row["Sat. rate"]),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute and display dataset summary and calibration statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        default=None,
        help="Override default data root directory.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "latex", "csv"],
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--no-supg",
        action="store_true",
        help="Skip SUPG benchmark datasets.",
    )
    parser.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="Skip ScaleDoc/BARGAIN benchmark datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None

    print("Collecting dataset statistics...", file=sys.stderr)
    df = collect_all_stats(
        include_supg=not args.no_supg,
        include_benchmarks=not args.no_benchmarks,
        data_dir=data_dir,
    )

    if df.empty:
        print("No data found. Check that data files are present.", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    if fmt == "table":
        output = _format_table(df)
    elif fmt == "latex":
        output = _format_latex(df)
    else:
        output = _format_csv(df)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
