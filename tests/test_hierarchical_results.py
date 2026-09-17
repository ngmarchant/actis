import json
from pathlib import Path

import pytest

from experiments.merge_results import load_experiment_results
from experiments.runners import (
    ACTISRunner,
    BargainPRRunner,
    run_evaluation_suite,
)
from experiments.scenarios import (
    Benign,
    PrecisionTailOverfit,
    UninformativeProxyFloor,
)


def test_scenario_to_dict_and_hashing():
    scen1 = Benign(pop_size=1000, seed=42)
    scen2 = Benign(pop_size=1000, seed=42)
    scen3 = Benign(pop_size=2000, seed=42)
    scen4 = Benign(pop_size=1000, seed=99)
    scen5 = PrecisionTailOverfit(pop_size=1000, seed=42)

    # Deterministic
    assert scen1.config_hash() == scen2.config_hash()
    assert len(scen1.config_hash()) == 8

    # Changing pop_size or seed alters hash
    assert scen1.config_hash() != scen3.config_hash()
    assert scen1.config_hash() != scen4.config_hash()

    # Different scenario class alters hash
    assert scen1.config_hash() != scen5.config_hash()

    # to_dict serializes correctly and is JSON-compatible
    d = scen1.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "benign"
    assert d["pop_size"] == 1000
    assert d["seed"] == 42
    # Verify JSON serialization works without error
    json_str = json.dumps(d)
    assert "benign" in json_str


def test_runner_to_dict_and_hashing():
    runner1 = ACTISRunner(
        name="actis_test", alpha=0.4, enable_asymptotic_protection=True
    )
    runner2 = ACTISRunner(
        name="actis_test", alpha=0.4, enable_asymptotic_protection=True
    )
    runner3 = ACTISRunner(
        name="actis_test", alpha=0.1, enable_asymptotic_protection=True
    )
    runner4 = ACTISRunner(
        name="actis_test", alpha=0.4, enable_asymptotic_protection=False
    )
    runner5 = BargainPRRunner(name="bargain_pr", window_size=50)

    assert runner1.config_hash() == runner2.config_hash()
    assert len(runner1.config_hash()) == 8

    # Parameter changes alter hash
    assert runner1.config_hash() != runner3.config_hash()
    assert runner1.config_hash() != runner4.config_hash()
    assert runner1.config_hash() != runner5.config_hash()

    d = runner1.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "actis_test"
    assert d["alpha"] == 0.4
    assert d["enable_asymptotic_protection"] is True
    json_str = json.dumps(d)
    assert "actis_test" in json_str


def test_run_evaluation_suite_hierarchical_saving_and_skipping(tmp_path: Path):
    results_dir = tmp_path / "results"
    scenario = Benign(pop_size=200, seed=42)
    runners = [
        ACTISRunner(
            name="actis_fast",
            num_thresholds=5,
            max_sample_size=50,
            adaptive=False,
        )
    ]

    # Initial run
    res1 = run_evaluation_suite(
        scenario=scenario,
        runners=runners,
        num_trials=2,
        pop_size=1000,
        gamma_R=0.9,
        gamma_P=0.9,
        delta=0.05,
        seed=42,
        results_dir=results_dir,
        skip_existing=True,
    )

    assert len(res1) == 1
    summary = res1[0]
    assert isinstance(summary["scenario"], dict)
    assert summary["scenario"]["name"] == "benign"
    assert isinstance(summary["runner"], dict)
    assert summary["runner"]["name"] == "actis_fast"
    assert "description" not in summary
    assert "experiment_name" not in summary

    # Verify directory structure:
    # results_dir / <scen_name>_<hash> / <runner_name>_<hash> / <filename>
    expected_scen_dir = results_dir / f"{scenario.name}_{scenario.config_hash()}"
    expected_runner_dir = (
        expected_scen_dir / f"{runners[0].name}_{runners[0].config_hash()}"
    )
    expected_file = expected_runner_dir / "delta_0.05_gamma_0.9_trials_2.json"

    assert expected_file.exists()

    with open(expected_file, "r") as f:
        disk_data = json.load(f)
    assert disk_data["scenario"]["name"] == "benign"
    assert disk_data["runner"]["name"] == "actis_fast"

    # Second run with skip_existing=True: should skip execution and load cached results
    res2 = run_evaluation_suite(
        scenario=scenario,
        runners=runners,
        num_trials=2,
        pop_size=1000,
        gamma_R=0.9,
        gamma_P=0.9,
        delta=0.05,
        seed=42,
        results_dir=results_dir,
        skip_existing=True,
    )
    assert len(res2) == 1
    assert res2[0]["num_trials"] == summary["num_trials"]
    assert res2[0]["joint_failure_rate"] == summary["joint_failure_rate"]


def test_merge_results_utility(tmp_path: Path):
    results_dir = tmp_path / "hierarchical"
    scen1 = Benign(pop_size=200, seed=1)
    scen2 = UninformativeProxyFloor(pop_size=200, seed=2)
    runner1 = ACTISRunner(name="actis_a", num_thresholds=5, adaptive=False)
    runner2 = ACTISRunner(name="actis_b", num_thresholds=5, adaptive=False)

    # Run combination 1: scen1, runner1, delta=0.05, gamma=0.9
    run_evaluation_suite(
        scenario=scen1,
        runners=[runner1],
        num_trials=1,
        pop_size=1000,
        gamma_R=0.9,
        gamma_P=0.9,
        delta=0.05,
        seed=1,
        results_dir=results_dir,
    )

    # Run combination 2: scen2, runner2, delta=0.01, gamma=0.8
    run_evaluation_suite(
        scenario=scen2,
        runners=[runner2],
        num_trials=1,
        pop_size=1000,
        gamma_R=0.8,
        gamma_P=0.8,
        delta=0.01,
        seed=2,
        results_dir=results_dir,
    )

    # Test loading all results
    all_res = load_experiment_results(results_dir)
    assert len(all_res) == 2

    # Filter by scenario
    benign_res = load_experiment_results(results_dir, scenarios=["benign"])
    assert len(benign_res) == 1
    assert benign_res[0]["scenario"]["name"] == "benign"

    # Filter by runner
    runner_b_res = load_experiment_results(results_dir, runners=["actis_b"])
    assert len(runner_b_res) == 1
    assert runner_b_res[0]["runner"]["name"] == "actis_b"

    # Filter by delta
    delta_01_res = load_experiment_results(results_dir, deltas=[0.01])
    assert len(delta_01_res) == 1
    assert delta_01_res[0]["promised_delta"] == pytest.approx(0.01)

    # Filter by gamma
    gamma_9_res = load_experiment_results(results_dir, gammas=[0.9])
    assert len(gamma_9_res) == 1
    assert gamma_9_res[0]["gamma_R"] == pytest.approx(0.9)
