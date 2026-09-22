import json
from pathlib import Path

from experiments.runners import (
    ACTISRunner,
    BargainPRRunner,
    run_evaluation_suite,
)
from experiments.scenarios import (
    Benign,
    DocumentBenchmarkDataset,
    PrecisionTailOverfit,
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

    # results_subpath
    assert scen1.results_subpath() == Path("benign") / f"benign_{scen1.config_hash()}"

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
    # results_dir / scenario.results_subpath() / <runner_name>_<hash> / <filename>
    expected_scen_dir = results_dir / scenario.results_subpath()
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
    assert res2[0]["joint_failure"]["mean"] == summary["joint_failure"]["mean"]


def test_document_benchmark_results_subpath(tmp_path: Path):
    dummy_file = tmp_path / "dummy.parquet"
    dummy_file.touch()

    bench = DocumentBenchmarkDataset(
        name="pubmed",
        query_id="0",
        oracle_model="azure/gpt-5.6-terra",
        proxy_model="azure/gpt-5.6-luna",
        data_path=dummy_file,
    )

    subpath = bench.results_subpath()
    assert subpath.parts[0] == "pubmed"
    assert subpath.parts[1] == "q0"
    assert subpath.parts[2].startswith("azure-gpt-5.6-terra__azure-gpt-5.6-luna_")
    assert bench.config_hash() in subpath.parts[2]
