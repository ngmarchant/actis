#!/usr/bin/env python3
"""
This script compares ACTIS, BARGAIN_PR and LOTUS across benchmark datasets.

Key evaluation dimensions:
1. Statistical soundness / joint coverage
2. Oracle call efficiency: total oracle call rate across the population dataset.
"""

import json
from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

from experiments.runners import (
    ACTISRunner,
    BargainPRRunner,
    BaseFilterRunner,
    LotusRunner,
    ScaleDocRunner,
    print_comparison_table,
    run_evaluation_suite,
)
from experiments.scenarios import SCENARIOS


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
        "--scenarios",
        dest="scenarios",
        nargs="+",
        default=["all"],
        help=f"Scenario(s) to run: 'all' or subset of {list(SCENARIOS.keys())} "
        f"(space or comma-separated, default: all)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of Monte Carlo trials (default: 100)"
    )
    parser.add_argument(
        "--sample-size",
        type=parse_sample_size,
        default="auto",
        help="ACTIS initial sample size: positive integer count or 'auto' (default: 'auto')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="ACTIS batch sample size (default: 100)"
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size (default: None)"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Target recall (default: 0.90)"
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.90,
        help="Target precision (default: 0.90)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Failure probability delta (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=20,
        help="Number of candidates for thresholds for ACTIS (default: 20)"
    )
    parser.add_argument(
        "--num-thresholds-upper",
        type=int,
        default=500,
        help="Number of candidates for upper threshold for ACTIS (default: 500)"
    )
    parser.add_argument(
        "--max-sample-size",
        type=parse_sample_size_or_fraction,
        default=0.5,
        help="Maximum sample size for ACTIS: integer count (e.g. 10000) or float "
        "fraction in (0, 1] (default: 0.5).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="comparison",
        help="Experiment name"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Defensive mixing weight for importance sampling (default: 0.4)"
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
        help="Minimum number of positive labels before adaptive stopping (default: 'auto')",
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
        help="Whether to enable heuristic protection for anytime-valid FWER control when "
        "operating in the non-asymptotic regime (default: True)",
    )
    parser.add_argument(
        "--conservative-correction",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to apply a conservative correction to the binomial null failure "
        "probability for the precision supermartingale (default: False)",
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
    return parser.parse_args(), parser


def main():
    args, parser = parse_args()

    # Normalize scenario names (supports multiple args and comma-separated items)
    scenario_keys: list[str] = []
    for item in args.scenarios:
        for sc in item.split(","):
            sc_clean = sc.strip()
            if not sc_clean:
                continue
            if sc_clean != "all" and sc_clean not in SCENARIOS:
                valid = ", ".join(["all"] + list(SCENARIOS.keys()))
                parser.error(f"invalid scenario '{sc_clean}'. Choose from: {valid}")
            scenario_keys.append(sc_clean)

    if "all" in scenario_keys:
        scenarios_to_run = list(dict.fromkeys(SCENARIOS.values()))
    else:
        scenarios_to_run = [SCENARIOS[k] for k in dict.fromkeys(scenario_keys)]

    all_records = []

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
            conservative_correction=args.conservative_correction,
            variance_ratio_bound=args.variance_ratio_bound,
            max_jump_ratio_bound=args.max_jump_ratio_bound
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
            conservative_correction=args.conservative_correction,
            variance_ratio_bound=args.variance_ratio_bound,
            max_jump_ratio_bound=args.max_jump_ratio_bound
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
        LotusRunner(
            name="lotus_original"
        ),
        ScaleDocRunner(
            name="scaledoc"
        )
    ]

    for scenario in scenarios_to_run:
        suite_res = run_evaluation_suite(
            scenario=scenario,
            runners=runners,
            num_trials=args.num_trials,
            pop_size=args.pop_size,
            gamma_R=args.target_recall,
            gamma_P=args.target_precision,
            delta=args.delta,
            seed=args.seed,
            exp_name=args.exp_name,
        )
        print_comparison_table(suite_res)
        all_records.extend(suite_res)

    output_path = args.output_json
    if not output_path:
        res_dir = Path("experiments/results") / args.exp_name
        json_name = f"comparison_delta_{args.delta}_gamma_{args.target_recall}.json"
        output_path = res_dir / json_name
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved comparison results to: {output_path}")


if __name__ == "__main__":
    main()
