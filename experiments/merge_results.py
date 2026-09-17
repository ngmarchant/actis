#!/usr/bin/env python3
"""
Utility to recursively discover, filter, and merge experiment results from the
hierarchical results directory.
"""

import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence


def _scenario_matches(record: dict[str, Any], allowed_scenarios: set[str]) -> bool:
    scen = record.get("scenario")
    if isinstance(scen, dict):
        name = scen.get("name", "")
    else:
        name = str(scen) if scen is not None else ""
    return name in allowed_scenarios


def _runner_matches(record: dict[str, Any], allowed_runners: set[str]) -> bool:
    runner_info = record.get("runner") or record.get("runner_params")
    if isinstance(runner_info, dict):
        name = runner_info.get("name", "")
    else:
        name = str(runner_info) if runner_info is not None else ""
    return name in allowed_runners


def _float_matches(val: float | None, targets: set[float], tol: float = 1e-5) -> bool:
    if val is None:
        return False
    return any(math.isclose(val, t, abs_tol=tol) for t in targets)


def load_experiment_results(
    results_dir: Path | str = "experiments/results",
    scenarios: Sequence[str] | None = None,
    runners: Sequence[str] | None = None,
    deltas: Sequence[float] | None = None,
    gammas: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Recursively scans `results_dir` for JSON experiment results and applies optional
    filters.

    Args:
        results_dir: Root directory containing hierarchical experiment results.
        scenarios: Optional list of scenario names to include.
        runners: Optional list of runner names to include.
        deltas: Optional list of promised deltas to include.
        gammas: Optional list of target gammas (checks gamma_R / gamma_P) to include.

    Returns:
        A list of matching experiment result summary dictionaries.
    """
    root = Path(results_dir)
    if not root.exists():
        return []

    scenarios_set = set(scenarios) if scenarios else None
    runners_set = set(runners) if runners else None
    deltas_set = set(deltas) if deltas else None
    gammas_set = set(gammas) if gammas else None

    matching_results: list[dict[str, Any]] = []

    for file_path in sorted(root.rglob("*.json")):
        # Skip temporary files
        if file_path.name.endswith(".tmp") or file_path.name.startswith("."):
            continue

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        records = data if isinstance(data, list) else [data]

        for r in records:
            if not isinstance(r, dict):
                continue
            # Validate that this is an evaluation summary record
            if "scenario" not in r or ("runner" not in r and "runner_params" not in r):
                continue

            if scenarios_set and not _scenario_matches(r, scenarios_set):
                continue

            if runners_set and not _runner_matches(r, runners_set):
                continue

            if deltas_set:
                r_delta = r.get("promised_delta", r.get("delta"))
                if not _float_matches(r_delta, deltas_set):
                    continue

            if gammas_set:
                r_gamma_r = r.get("gamma_R")
                r_gamma_p = r.get("gamma_P")
                if not (
                    _float_matches(r_gamma_r, gammas_set)
                    or _float_matches(r_gamma_p, gammas_set)
                ):
                    continue

            matching_results.append(r)

    return matching_results


def parse_args():
    parser = ArgumentParser(
        description=(
            "Recursively merge hierarchical experiment results into a single JSON file."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help=(
            "Root directory containing hierarchical experiment results (default: "
            "experiments/results)"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output merged JSON file",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional list of scenario names to filter by",
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        default=None,
        help="Optional list of runner names to filter by",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of deltas to filter by",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of gammas to filter by",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_experiment_results(
        results_dir=args.results_dir,
        scenarios=args.scenarios,
        runners=args.runners,
        deltas=args.deltas,
        gammas=args.gammas,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Merged {len(results)} experiment result records from '{args.results_dir}' -> "
        f"'{output_path}'"
    )


if __name__ == "__main__":
    main()
