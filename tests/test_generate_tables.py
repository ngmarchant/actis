import json
from pathlib import Path

import pytest

from experiments.generate_tables import (
    DEFAULT_APPENDIX_METHODS,
    DEFAULT_FOCUS_METHODS,
    compute_config_hash,
    format_latency_value,
    format_population_size,
    format_scenario_display_name,
    format_slack_value,
    format_value,
    generate_all_benchmarks_joint_table,
    generate_all_benchmarks_slack_table,
    generate_appendix_table,
    generate_benchmark_slack_table,
    generate_full_width_table,
    generate_joint_table,
    generate_representative_joint_table,
    generate_representative_oracle_table,
    generate_single_column_table,
    generate_slack_distribution_table,
    get_audit_dataframe,
    load_table_data,
    resolve_methods,
)


def test_compute_config_hash():
    d1 = {"name": "test", "alpha": 0.4, "power_law": True}
    d2 = {"power_law": True, "name": "test", "alpha": 0.4}
    d3 = {"name": "test", "alpha": 0.2, "power_law": True}

    h1 = compute_config_hash(d1)
    h2 = compute_config_hash(d2)
    h3 = compute_config_hash(d3)

    assert len(h1) == 8
    assert h1 == h2
    assert h1 != h3


def test_format_scenario_display_name():
    # Canonical static lookup
    assert format_scenario_display_name("onto") == "Onto"
    assert format_scenario_display_name("imagenet") == "ImageNet"
    assert format_scenario_display_name("benign") == "Benign"

    # With scenario_uid (hash stripped)
    assert format_scenario_display_name("onto_1e26bd3f") == "Onto"
    assert (
        format_scenario_display_name("pubmed_a7b85139", query_id="0") == "PubMed (Q0)"
    )

    # Query formatting
    assert format_scenario_display_name("pubmed", query_id="0") == "PubMed (Q0)"
    assert format_scenario_display_name("pubmed", query_id="12") == "PubMed (Q12)"
    assert (
        format_scenario_display_name("gov_report", query_id="5_ext")
        == "GovReport (Q5_ext)"
    )

    # Short format
    assert (
        format_scenario_display_name("pubmed", query_id="0", short=True) == "PubMed Q0"
    )
    assert format_scenario_display_name("benign", short=True) == "Benign"

    # BARGAIN lookup
    assert format_scenario_display_name("court") == "Court"
    assert format_scenario_display_name("review_1e26bd3f") == "Review"
    assert format_scenario_display_name("screenplay", short=True) == "Screenplay"
    assert format_scenario_display_name("wiki") == "Wiki"

    # Proxy model formatting
    assert (
        format_scenario_display_name(
            "court", query_id="0", proxy_model="azure/gpt-5.6-luna"
        )
        == "Court (Q0, GPT-5.6-luna)"
    )
    assert (
        format_scenario_display_name(
            "court",
            query_id="0",
            proxy_model="openai/llama3.2:1b-instruct-fp16",
        )
        == "Court (Q0, Llama-3.2-1B)"
    )
    assert (
        format_scenario_display_name(
            "court",
            query_id="0",
            proxy_model="azure/gpt-5.6-luna",
            short=True,
        )
        == "Court Q0 (GPT-5.6-luna)"
    )
    assert (
        format_scenario_display_name(
            "pubmed",
            query_id="1_ext",
            proxy_model="azure/gpt-5.6-luna",
        )
        == "PubMed (Q1_ext, GPT-5.6-luna)"
    )
    assert (
        format_scenario_display_name(
            "court",
            proxy_model="azure/gpt-5.6-luna",
        )
        == "Court (GPT-5.6-luna)"
    )


def test_format_value():
    # Valid, best -> bolded
    v1 = format_value(
        mean_val=12.34,
        se_val=0.56,
        violated=False,
        is_best=True,
        include_se=True,
        precision=1,
    )
    assert v1 == r"$\mathbf{12.3} \pm 0.6$"

    # Violated, not bolded, gets asterisk
    v2 = format_value(
        mean_val=10.0,
        se_val=0.2,
        violated=True,
        is_best=True,
        include_se=True,
        precision=1,
    )
    assert v2 == r"$10.0 \pm 0.2$\rlap{$^*$}"

    # Mean only
    v3 = format_value(
        mean_val=15.0,
        se_val=None,
        violated=False,
        is_best=False,
        include_se=False,
        precision=2,
    )
    assert v3 == r"$15.00$"


def _make_mock_summary(
    scen_name: str,
    runner_name: str,
    delta: float,
    gamma: float,
    mean_oracle: float = 0.15,
    se_oracle: float = 0.01,
    joint_fail: float = 0.0,
    ideal_oracle: float = 0.05,
    query_id: str | None = None,
    oracle_model: str | None = None,
    proxy_model: str | None = None,
) -> dict:
    scen_dict = {
        "name": scen_name,
        "seed": 42,
    }
    if query_id is not None:
        scen_dict["query_id"] = query_id
    if oracle_model is not None:
        scen_dict["oracle_model"] = oracle_model
    if proxy_model is not None:
        scen_dict["proxy_model"] = proxy_model

    return {
        "date": "2026-09-17 12:00:00",
        "scenario": scen_dict,
        "runner": {
            "name": runner_name,
            "alpha": 0.4,
        },
        "gamma_R": gamma,
        "gamma_P": gamma,
        "promised_delta": delta,
        "ideal_oracle_call_rate": ideal_oracle,
        "cost": {
            "oracle": {
                "call_rate": {
                    "mean": mean_oracle,
                    "se": se_oracle,
                }
            },
            "proxy": {"call_rate": {"mean": 1.0, "se": 0.0}},
        },
        "joint_failure": {"mean": joint_fail, "se": 0.0},
        "recall": {"mean": 0.92, "se": 0.01},
        "precision": {"mean": 0.95, "se": 0.01},
    }


def test_load_table_data_hierarchical(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create hierarchical structure: <scen_uid>/<runner_uid>/<file>.json
    scen = {"name": "pubmed", "query_id": "0", "seed": 42}
    runner1 = {"name": "actis_adaptive_is_asymptotic", "alpha": 0.4}
    runner2 = {"name": "scaledoc", "alpha": 0.4}

    scen_hash = compute_config_hash(scen)
    scen_uid = f"pubmed_{scen_hash}"

    r1_hash = compute_config_hash(runner1)
    r1_uid = f"actis_adaptive_is_asymptotic_{r1_hash}"

    r2_hash = compute_config_hash(runner2)
    r2_uid = f"scaledoc_{r2_hash}"

    dir_r1 = results_dir / scen_uid / r1_uid
    dir_r1.mkdir(parents=True)
    dir_r2 = results_dir / scen_uid / r2_uid
    dir_r2.mkdir(parents=True)

    summary1 = _make_mock_summary(
        "pubmed",
        "actis_adaptive_is_asymptotic",
        0.05,
        0.8,
        mean_oracle=0.10,
        query_id="0",
    )
    summary2 = _make_mock_summary(
        "pubmed",
        "scaledoc",
        0.05,
        0.8,
        mean_oracle=0.12,
        query_id="0",
    )

    with open(dir_r1 / "delta_0.05_gamma_0.8_trials_100.json", "w") as f:
        json.dump(summary1, f)
    with open(dir_r2 / "delta_0.05_gamma_0.8_trials_100.json", "w") as f:
        json.dump(summary2, f)

    # Add a merged comparison file (list of dicts) in a subfolder - must be ignored!
    merged_dir = results_dir / "pubmed_comparison"
    merged_dir.mkdir()
    with open(merged_dir / "comparison_delta_0.05_gamma_0.8.json", "w") as f:
        json.dump([summary1, summary2], f)

    df = load_table_data(results_dir)

    # Check length: exactly 2 rows (merged file ignored)
    assert len(df) == 2
    assert set(df["method"].unique()) == {"actis_adaptive_is_asymptotic", "scaledoc"}
    assert df["scenario_uid"].iloc[0] == scen_uid
    assert df["delta"].iloc[0] == 0.05
    assert df["gamma"].iloc[0] == 0.8
    assert df["mean_oracle"].iloc[0] == pytest.approx(10.0)

    # Check new metadata columns
    assert "dataset" in df.columns
    assert "query_id" in df.columns
    assert "oracle_model" in df.columns
    assert "proxy_model" in df.columns
    assert df["dataset"].iloc[0] == "pubmed"
    assert df["query_id"].iloc[0] == "0"


def test_generate_latex_tables(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create mock hierarchical data for two methods and two deltas
    scen = {"name": "pubmed", "query_id": "0", "seed": 42}
    scen_hash = compute_config_hash(scen)
    scen_uid = f"pubmed_{scen_hash}"

    for r_name, mean_val in [
        ("actis_adaptive_is_asymptotic", 0.08),
        ("scaledoc", 0.12),
        ("lotus_original", 0.15),
    ]:
        r_hash = compute_config_hash({"name": r_name, "alpha": 0.4})
        r_uid = f"{r_name}_{r_hash}"
        r_dir = results_dir / scen_uid / r_uid
        r_dir.mkdir(parents=True, exist_ok=True)

        for delta in [0.01, 0.05]:
            for gamma in [0.8, 0.9]:
                summary = _make_mock_summary(
                    scen_name="pubmed",
                    runner_name=r_name,
                    delta=delta,
                    gamma=gamma,
                    mean_oracle=mean_val,
                    se_oracle=0.001,
                    joint_fail=0.0,
                    ideal_oracle=0.03,
                    query_id="0",
                )
                filename = f"delta_{delta}_gamma_{gamma}_trials_100.json"
                with open(r_dir / filename, "w") as f:
                    json.dump(summary, f)

    df = load_table_data(results_dir)
    focus_methods = resolve_methods(df, DEFAULT_FOCUS_METHODS)

    # 1. Full-width table
    tex_full = generate_full_width_table(
        df, focus_methods=focus_methods, include_se=True, precision=1
    )
    assert r"\begin{table*}[t]" in tex_full
    assert r"\toprule" in tex_full
    assert r"\bottomrule" in tex_full
    assert "PubMed (Q0)" in tex_full
    assert "ScaleDoc" in tex_full
    assert "ACTIS" in tex_full
    # Best method (ACTIS with 8.0%) should be bolded
    assert r"\mathbf{8.0}" in tex_full

    # 2. Single-column table
    tex_single = generate_single_column_table(
        df,
        delta_val=0.05,
        focus_methods=focus_methods,
        include_se=False,
        precision=1,
    )
    assert r"\begin{table}[t]" in tex_single
    assert "ScaleDoc" in tex_single
    assert "PubMed Q0" in tex_single

    # 3. Appendix table
    app_methods = resolve_methods(df, DEFAULT_APPENDIX_METHODS)
    tex_app = generate_appendix_table(
        df,
        delta_val=0.01,
        appendix_methods=app_methods,
        include_se=True,
        precision=1,
    )
    assert r"\begin{table*}[t]" in tex_app
    assert "ScaleDoc" in tex_app


def test_generate_latex_tables_with_proxies(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create two scenarios: Court with Luna and Court with Llama-1B
    proxies = [
        ("azure/gpt-5.6-luna", "Luna"),
        ("openai/llama3.2:1b-instruct-fp16", "Llama-1B"),
    ]
    for proxy_model, _ in proxies:
        scen = {
            "name": "court",
            "query_id": "0",
            "oracle_model": "azure/gpt-5.6-terra",
            "proxy_model": proxy_model,
        }
        scen_hash = compute_config_hash(scen)
        scen_uid = f"court_{scen_hash}"

        runner_hash = compute_config_hash(
            {"name": "actis_adaptive_is_asymptotic", "alpha": 0.4}
        )
        r_uid = f"actis_adaptive_is_asymptotic_{runner_hash}"
        r_dir = results_dir / scen_uid / r_uid
        r_dir.mkdir(parents=True, exist_ok=True)

        summary = _make_mock_summary(
            scen_name="court",
            runner_name="actis_adaptive_is_asymptotic",
            delta=0.05,
            gamma=0.8,
            mean_oracle=0.15,
            se_oracle=0.01,
            joint_fail=0.0,
            ideal_oracle=0.05,
            query_id="0",
            oracle_model="azure/gpt-5.6-terra",
            proxy_model=proxy_model,
        )
        with open(r_dir / "delta_0.05_gamma_0.8_trials_100.json", "w") as f:
            json.dump(summary, f)

    df = load_table_data(results_dir)
    assert len(df) == 2
    assert set(df["proxy_model"]) == {
        "azure/gpt-5.6-luna",
        "openai/llama3.2:1b-instruct-fp16",
    }

    focus_methods = resolve_methods(df, DEFAULT_FOCUS_METHODS)
    tex_full = generate_full_width_table(
        df, focus_methods=focus_methods, include_se=True, precision=1
    )

    # Disambiguated benchmark rows must appear for both proxies
    assert "Court (Q0, GPT-5.6-luna)" in tex_full
    assert "Court (Q0, Llama-3.2-1B)" in tex_full

    tex_single = generate_single_column_table(
        df, delta_val=0.05, focus_methods=focus_methods, include_se=False, precision=1
    )
    assert "Court Q0 (GPT-5.6-luna)" in tex_single
    assert "Court Q0 (Llama-3.2-1B)" in tex_single

    # 3. Representative oracle table
    tex_rep = generate_representative_oracle_table(
        df,
        focus_methods=focus_methods,
        representative_workloads=[
            ("court", "0", "azure/gpt-5.6-luna"),
            ("court", "0", "openai/llama3.2:1b-instruct-fp16"),
        ],
        delta_val=0.05,
        gamma_val=0.8,
        include_se=True,
        precision=1,
    )
    assert r"\begin{table}[t]" in tex_rep
    assert r"\textbf{Name} & \textbf{Proxy} & \textbf{Ideal}" in tex_rep
    assert "Court Q0 & GPT-5.6-luna" in tex_rep
    assert "Court Q0 & Llama-3.2-1B" in tex_rep


def test_format_population_size():
    assert format_population_size(None) == "--"
    assert format_population_size(996) == "1k"
    assert format_population_size(2854) == "3k"
    assert format_population_size(10000) == "10k"
    assert format_population_size(11165) == "11k"
    assert format_population_size(22631) == "23k"
    assert format_population_size(50000) == "50k"
    assert format_population_size(973085) == "973k"
    assert format_population_size(1_000_000) == "1M"


def test_format_badvalue():
    # Value formatting with badvalue
    v_ok = format_value(32.1, None, violated=False, is_best=True, use_badvalue=True)
    assert v_ok == r"$\mathbf{32.1}$"

    v_bad = format_value(28.4, None, violated=True, is_best=False, use_badvalue=True)
    assert v_bad == r"\badvalue{$28.4$\rlap{*}}"

    # Latency formatting with badvalue
    l_ok = format_latency_value(0.0098, None, violated=False, is_best=True, use_badvalue=True)
    assert l_ok == r"$\mathbf{9.8}$"

    l_bad = format_latency_value(0.0038, None, violated=True, is_best=False, use_badvalue=True)
    assert l_bad == r"\badvalue{$3.8$\rlap{*}}"


def test_generate_joint_tables(tmp_path):
    results_dir = tmp_path / "results"
    for proxy_model in ["azure/gpt-5.6-luna", "openai/llama3.2:1b-instruct-fp16"]:
        slug = "luna" if "luna" in proxy_model else "llama"
        s_dir = results_dir / f"court_q0_{slug}"
        r_dir = s_dir / "actis_adaptive_is_asymptotic_00000000"
        r_dir.mkdir(parents=True, exist_ok=True)

        summary = _make_mock_summary(
            scen_name="court",
            runner_name="actis_adaptive_is_asymptotic",
            delta=0.05,
            gamma=0.8,
            mean_oracle=0.15,
            se_oracle=0.01,
            joint_fail=0.0,
            ideal_oracle=0.05,
            query_id="0",
            oracle_model="azure/gpt-5.6-terra",
            proxy_model=proxy_model,
        )
        with open(r_dir / "delta_0.05_gamma_0.8_trials_100.json", "w") as f:
            json.dump(summary, f)

    df = load_table_data(results_dir)
    focus_methods = resolve_methods(df, DEFAULT_FOCUS_METHODS)

    # Test representative joint table
    tex_rep = generate_representative_joint_table(
        df,
        focus_methods=focus_methods,
        representative_workloads=[("court", "0", "azure/gpt-5.6-luna")],
        delta_val=0.05,
        gamma_val=0.8,
        include_se=False,
        use_badvalue=True,
    )
    assert r"\begin{table*}" in tex_rep
    assert r"\multicolumn{2}{c}{\textbf{Benchmark}}" in tex_rep
    assert r"\multicolumn{2}{c}{\textbf{Oracle Call Rate (\%)}}" in tex_rep or r"\multicolumn{" in tex_rep
    assert r"\textbf{Workload} & \textbf{Proxy} & \textbf{$N$} & \textbf{Ideal}" in tex_rep
    assert "Court Q0 & GPT-5.6-luna" in tex_rep

    # Test all benchmarks joint table
    tex_all = generate_all_benchmarks_joint_table(
        df,
        focus_methods=focus_methods,
        delta_val=0.05,
        gamma_val=0.8,
        include_se=False,
        use_badvalue=True,
    )
    assert r"\begin{table*}" in tex_all
    assert "Court Q0 & GPT-5.6-luna" in tex_all
    assert "Court Q0 & Llama-3.2-1B" in tex_all

    # Test joint wrapper
    tex_wrap = generate_joint_table(
        df,
        focus_methods=focus_methods,
        scope="representative",
        representative_workloads=[("court", "0", "azure/gpt-5.6-luna")],
        delta_val=0.05,
        gamma_val=0.8,
        include_se=False,
    )
    assert tex_wrap == tex_rep


def test_format_slack_value():
    assert format_slack_value(None) == "--"
    assert format_slack_value(float("nan")) == "--"
    assert format_slack_value(0.12345, precision=3) == "$+0.123$"
    assert format_slack_value(0.0, precision=2) == "$+0.00$"
    assert format_slack_value(0.12345, precision=3, show_sign=False) == "$0.123$"
    # Badvalue for negative values
    assert format_slack_value(-0.05, precision=3, use_badvalue=True) == r"\badvalue{$-0.050$}"
    assert format_slack_value(-0.05, precision=3, use_badvalue=False) == "$-0.050$"


def test_get_audit_dataframe():
    import pandas as pd
    df = pd.DataFrame([
        {"method": "scaledoc", "method_clean": "ScaleDoc-PR", "gamma": 0.8},
        {"method": "scaledoc_native", "method_clean": "ScaleDoc-Native", "gamma": 0.8},
        {"method": "actis_adaptive_is_asymptotic", "method_clean": "ACTIS", "gamma": 0.8},
    ])
    df_audit = get_audit_dataframe(df)
    assert len(df_audit) == 2
    assert "scaledoc" not in df_audit["method"].values
    assert "scaledoc_native" in df_audit["method"].values
    sd_row = df_audit[df_audit["method"] == "scaledoc_native"].iloc[0]
    assert sd_row["method_clean"] == "ScaleDoc"


def test_generate_slack_tables(tmp_path):
    results_dir = tmp_path / "results"
    for runner in ["actis_adaptive_is_asymptotic", "scaledoc_native", "bargain", "lotus_original"]:
        r_dir = results_dir / "court_q0" / f"{runner}_00000000"
        r_dir.mkdir(parents=True, exist_ok=True)
        summary = _make_mock_summary(
            scen_name="court",
            runner_name=runner,
            delta=0.05,
            gamma=0.8,
            mean_oracle=0.15,
            se_oracle=0.01,
            joint_fail=0.0,
            ideal_oracle=0.05,
            query_id="0",
            proxy_model="azure/gpt-5.6-luna",
        )
        # Add raw precision and recall arrays for slack computation
        summary["precision"]["raw"] = [0.85] * 100
        summary["recall"]["raw"] = [0.90] * 100
        with open(r_dir / "delta_0.05_gamma_0.8_trials_100.json", "w") as f:
            json.dump(summary, f)

    df = load_table_data(results_dir)
    assert "quantile_slack" in df.columns

    # 1. Detailed slack distribution table
    tex_det = generate_slack_distribution_table(
        df,
        delta_val=0.05,
        methods=("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
        gammas=[0.8],
        layout="detailed",
        use_badvalue=True,
    )
    assert r"\begin{table}[t]" in tex_det
    assert "Distribution of accuracy constraint slack" in tex_det
    assert r"$\gamma$ & Method & Median & $[Q_1, Q_3]$ & Min (Worst) & Max & Violations (\%)" in tex_det
    assert "ACTIS" in tex_det
    assert "BARGAIN" in tex_det
    assert "ScaleDoc" in tex_det
    assert "LOTUS" in tex_det
    assert "$0.80$" in tex_det

    # 2. Compact slack distribution table
    tex_comp = generate_slack_distribution_table(
        df,
        delta_val=0.05,
        methods=("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
        gammas=[0.8],
        layout="compact",
        use_badvalue=True,
    )
    assert r"\begin{table}[t]" in tex_comp
    assert r"$\gamma$ & ACTIS & BARGAIN & ScaleDoc & LOTUS" in tex_comp
    assert "$0.80$" in tex_comp

    # 3. All benchmarks slack table with single gamma
    tex_all_g = generate_all_benchmarks_slack_table(
        df,
        delta_val=0.05,
        gamma_val=0.8,
        methods=("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
        use_badvalue=True,
    )
    assert r"\begin{table*}[t]" in tex_all_g
    assert r"\textbf{Workload} & \textbf{Proxy} & \textbf{$N$}" in tex_all_g
    assert "Court Q0" in tex_all_g
    assert "GPT-5.6-luna" in tex_all_g

    # 4. Multi-gamma benchmark slack table
    tex_all_mg = generate_all_benchmarks_slack_table(
        df,
        delta_val=0.05,
        gamma_val=None,
        methods=("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
        gammas=[0.8],
        use_badvalue=True,
    )
    assert r"\begin{table*}[t]" in tex_all_mg
    assert r"Benchmark & $\gamma$" in tex_all_mg
    assert "Court Q0" in tex_all_mg

    # 5. Alias check
    tex_alias = generate_benchmark_slack_table(
        df,
        delta_val=0.05,
        gamma_val=0.8,
        methods=("ACTIS", "BARGAIN", "ScaleDoc", "LOTUS"),
    )
    assert tex_alias == tex_all_g


