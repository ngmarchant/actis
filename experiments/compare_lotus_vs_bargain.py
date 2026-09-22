#!/usr/bin/env python3
"""
This script compares ACTIS, BARGAIN_PR and LOTUS across benchmark datasets.

Key evaluation dimensions:
1. Statistical soundness / joint coverage
2. Oracle call efficiency: total oracle call rate across the population dataset.
"""

import re
import sys
from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.data_prep import DATASET_GROUPS
from experiments.runners import (
    ACTISRunner,
    BargainPRRunner,
    BaseFilterRunner,
    LotusRunner,
    ScaleDocRunner,
    print_comparison_table,
    run_evaluation_suite,
)
from experiments.scenarios import get_available_queries, get_scenario


def parse_query_args(raw_queries: list[str] | None) -> list[str]:
    """Flattens comma- or space-delimited query arguments."""
    if not raw_queries:
        return []
    result = []
    for item in raw_queries:
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def parse_sample_size_or_fraction(val: str) -> int | float:
    """Parses sample size as either an integer count (>= 1) or a float fraction in
    (0, 1]."""
    try:
        num = float(val)
    except ValueError:
        raise ArgumentTypeError(f"Invalid numeric value for sample size: '{val}'")
    if 0.0 < num < 1.0:
        return num
    if num >= 1.0:
        if "." in val and num == 1.0:
            return 1.0
        if num.is_integer():
            return int(num)
        return num
    raise ArgumentTypeError(
        f"--max-sample-size must be a positive integer count (>= 1) or a float in "
        f"(0, 1] representing a dataset fraction. Got: '{val}'"
    )


def parse_min_positives(val: str) -> int | str:
    """Parses min_positives as either a positive integer or 'auto'."""
    if val.lower() == "auto":
        return "auto"
    try:
        val_int = int(val)
        if val_int <= 0:
            raise ValueError
        return val_int
    except ValueError:
        raise ArgumentTypeError(
            f"--min-positives must be a positive integer or 'auto'. Got: '{val}'"
        )


def parse_sample_size(val: str) -> int | str:
    """Parses initial sample size as either a positive integer count or 'auto'."""
    if val.lower() == "auto":
        return "auto"
    try:
        val_int = int(val)
        if val_int <= 0:
            raise ValueError
        return val_int
    except ValueError:
        raise ArgumentTypeError(
            f"--sample-size must be a positive integer or 'auto'. Got: '{val}'"
        )


def parse_args():
    parser = ArgumentParser(description="Compare LOTUS vs BARGAIN (unconstrained).")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario identifier or benchmark dataset name to run "
        "(e.g. 'court', 'pubmed', 'onto', 'benign').",
    )
    parser.add_argument(
        "--query",
        "--queries",
        dest="queries",
        nargs="+",
        default=None,
        help=(
            "Query identifier(s) for document benchmark datasets (e.g. '--query 0', "
            "'--query 0 1 2', '--queries 0,1,2', or '--query all'). Default: '0'."
        ),
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="azure/gpt-5.6-terra",
        help="Model identifier for oracle ground truth (default: "
        "'azure/gpt-5.6-terra')",
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default="azure/gpt-5.6-luna",
        help="Model identifier for proxy score (default: 'azure/gpt-5.6-luna')",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of Monte Carlo trials (default: 100)",
    )
    parser.add_argument(
        "--sample-size",
        type=parse_sample_size,
        default="auto",
        help=(
            "ACTIS initial sample size: positive integer count or 'auto' "
            "(default: 'auto')"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="ACTIS batch sample size (default: 100)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=None, help="Population size (default: None)"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Target recall (default: 0.90)",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.90,
        help="Target precision (default: 0.90)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Failure probability delta (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=20,
        help="Number of candidates for thresholds for ACTIS (default: 20)",
    )
    parser.add_argument(
        "--num-thresholds-upper",
        type=int,
        default=500,
        help="Number of candidates for upper threshold for ACTIS (default: 500)",
    )
    parser.add_argument(
        "--max-sample-size",
        type=parse_sample_size_or_fraction,
        default=0.5,
        help="Maximum sample size for ACTIS: integer count (e.g. 10000) or float "
        "fraction in (0, 1] (default: 0.5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Defensive mixing weight for importance sampling (default: 0.4)",
    )
    parser.add_argument(
        "--tail-tolerance",
        type=float,
        default=0.10,
        help=(
            "Dimensionless tail uncertainty tolerance for prevalence-adaptive clipping "
            "of importance sampling proposals (default: 0.10)"
        ),
    )
    parser.add_argument(
        "--proposal-method",
        type=str,
        choices=["snr_balanced", "var_min"],
        default="snr_balanced",
        help="Method to compute the proposal distribution (default: 'snr_balanced')",
    )
    parser.add_argument(
        "--power-law-quantiles",
        action=BooleanOptionalAction,
        default=True,
        help="Whether to use power law quantiles for upper thresholds (default: True)",
    )
    parser.add_argument(
        "--min-positives",
        type=parse_min_positives,
        default="auto",
        help=(
            "Minimum number of positive labels before adaptive stopping "
            "(default: 'auto')"
        ),
    )
    parser.add_argument(
        "--v0",
        type=str,
        default="auto",
        help="Prior variance for Gaussian mixture asymptotic CS (default: 'auto')",
    )
    parser.add_argument(
        "--enable-asymptotic-protection",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Whether to enable heuristic protection for anytime-valid FWER control "
            "when operating in the non-asymptotic regime (default: True)"
        ),
    )
    parser.add_argument(
        "--variance-ratio-bound",
        type=float,
        default=0.05,
        help="Maximum allowable ratio of cumulative process variance to prior variance "
        "(default: 0.05)",
    )
    parser.add_argument(
        "--max-jump-ratio-bound",
        type=float,
        default=0.50,
        help="Maximum allowable Lindeberg jump-to-variance ratio (default: 0.50)",
    )
    parser.add_argument(
        "--include-raw",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Whether to include raw per-trial metrics in output summaries "
            "(default: True)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help=(
            "Root directory for hierarchical results storage "
            "(default: experiments/results)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Whether to skip runner runs whose target result JSON already exists "
            "(default: True)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Force re-running even if target results already exist "
            "(disables --skip-existing)"
        ),
    )
    return parser.parse_args(), parser


def main():
    args, parser = parse_args()

    clean_scenario = args.scenario.lower().strip()
    match = re.match(r"^(?:(?:supg|scaledoc|bargain)_)?([a-z0-9_]+)$", clean_scenario)
    base_name = match.group(1) if match else clean_scenario
    is_benchmark = base_name in DATASET_GROUPS

    parsed_queries = parse_query_args(args.queries)

    if not is_benchmark:
        if parsed_queries:
            parser.error(f"Scenario '{args.scenario}' does not support queries.")
        try:
            scenario = get_scenario(
                args.scenario,
                oracle_model=args.oracle_model,
                proxy_model=args.proxy_model,
            )
        except ValueError as e:
            parser.error(str(e))
        scenarios_to_run = [scenario]
    else:
        if not parsed_queries:
            query_ids = ["0"]
        elif any(q.lower() == "all" for q in parsed_queries):
            query_ids = get_available_queries(args.scenario)
            if not query_ids:
                parser.error(f"No queries found for benchmark '{args.scenario}'.")
        else:
            query_ids = parsed_queries

        scenarios_to_run = []
        for qid in query_ids:
            try:
                scenario = get_scenario(
                    args.scenario,
                    query_id=qid,
                    oracle_model=args.oracle_model,
                    proxy_model=args.proxy_model,
                )
                scenarios_to_run.append(scenario)
            except ValueError as e:
                parser.error(str(e))

    runners: list[BaseFilterRunner] = [
        # ACTISRunner(
        #     name="actis_static",
        #     num_thresholds=args.num_thresholds,
        #     num_thresholds_upper=args.num_thresholds_upper,
        #     sampling_method="wor",
        #     adaptive=False,
        #     initial_sample_size=args.sample_size,
        #     batch_size=args.sample_size,
        #     max_sample_size=args.sample_size,
        #     min_positives=args.min_positives,
        #     power_law_quantiles=args.power_law_quantiles,
        # ),
        # ACTISRunner(
        #     name="actis_adaptive_is",
        #     num_thresholds=args.num_thresholds,
        #     num_thresholds_upper=args.num_thresholds_upper,
        #     sampling_method="is",
        #     alpha=args.alpha,
        #     proposal_method=args.proposal_method,
        #     power_law_quantiles=args.power_law_quantiles,
        #     conf_seq="finite",
        #     adaptive=True,
        #     initial_sample_size=args.sample_size,
        #     batch_size=args.batch_size,
        #     max_sample_size=args.max_sample_size,
        #     min_positives=args.min_positives,
        # ),
        ACTISRunner(
            name="actis_adaptive_is_asymptotic",
            num_thresholds=args.num_thresholds,
            num_thresholds_upper=args.num_thresholds_upper,
            sampling_method="is",
            alpha=args.alpha,
            proposal_method=args.proposal_method,
            power_law_quantiles=args.power_law_quantiles,
            conf_seq="asymptotic",
            v_0=args.v0,
            adaptive=True,
            initial_sample_size=args.sample_size,
            batch_size=args.batch_size,
            max_sample_size=args.max_sample_size,
            min_positives=args.min_positives,
            enable_asymptotic_protection=args.enable_asymptotic_protection,
            variance_ratio_bound=args.variance_ratio_bound,
            tail_tolerance=args.tail_tolerance,
        ),
        ACTISRunner(
            name="actis_adaptive_asymptotic",
            num_thresholds=args.num_thresholds,
            num_thresholds_upper=args.num_thresholds_upper,
            sampling_method="wor",
            power_law_quantiles=args.power_law_quantiles,
            conf_seq="asymptotic",
            v_0=args.v0,
            adaptive=True,
            initial_sample_size=args.sample_size,
            batch_size=args.batch_size,
            max_sample_size=args.max_sample_size,
            min_positives=args.min_positives,
            enable_asymptotic_protection=args.enable_asymptotic_protection,
            variance_ratio_bound=args.variance_ratio_bound,
        ),
        ACTISRunner(
            name="actis_adaptive",
            num_thresholds=args.num_thresholds,
            num_thresholds_upper=args.num_thresholds_upper,
            sampling_method="wor",
            power_law_quantiles=args.power_law_quantiles,
            conf_seq="finite",
            adaptive=True,
            initial_sample_size=args.sample_size,
            batch_size=args.batch_size,
            max_sample_size=args.max_sample_size,
            min_positives=args.min_positives,
        ),
        BargainPRRunner(
            name="bargain_pr",
            window_size=50,
            num_thresholds=20,
            sample_step=100,
        ),
        LotusRunner(name="lotus_original"),
        ScaleDocRunner(name="scaledoc"),
    ]

    for scenario in scenarios_to_run:
        if args.pop_size is not None:
            scenario.pop_size = args.pop_size
        if args.seed is not None:
            scenario.seed = args.seed

        suite_res = run_evaluation_suite(
            scenario=scenario,
            runners=runners,
            num_trials=args.num_trials,
            pop_size=args.pop_size,
            gamma_R=args.target_recall,
            gamma_P=args.target_precision,
            delta=args.delta,
            seed=args.seed,
            include_raw=args.include_raw,
            results_dir=args.results_dir,
            skip_existing=args.skip_existing and not args.force,
        )
        print_comparison_table(suite_res)


if __name__ == "__main__":
    main()
