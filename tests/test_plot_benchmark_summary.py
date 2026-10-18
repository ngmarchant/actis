import shutil
from pathlib import Path

import pandas as pd
import pytest

from experiments.plot_benchmark_summary import (
    plot_ablation_is_vs_uniform_pr_auc,
    plot_ablation_is_vs_uniform_pr_auc_selectivity,
    plot_ablation_is_vs_uniform_selectivity,
    plot_head_to_head_pairwise,
    plot_slack_distribution_vs_gamma,
    plot_slack_distribution_vs_gamma_delta,
    plot_unsoundness_teaser,
    plot_unsoundness_vs_gamma_delta,
)

HAS_XELATEX = shutil.which("xelatex") is not None


@pytest.fixture
def mock_summary_df():
    """Generates synthetic enriched dataframe covering multiple benchmarks,

    methods, gammas, deltas, selectivities, and PR-AUC values.
    """
    records = []
    methods = [
        ("actis_adaptive_is_asymptotic", "ACTIS"),
        ("actis_adaptive_asymptotic", "ACTIS (Unif. Asy)"),
        ("scaledoc", "ScaleDoc"),
        ("bargain_pr", "BARGAIN"),
        ("lotus_original", "LOTUS"),
    ]

    benchmarks = [
        ("pubmed_q0_azure", "pubmed", "0", 0.05, 0.85),
        ("pubmed_q1_azure", "pubmed", "1", 0.20, 0.95),
        ("court_q0_luna", "court", "0", 0.65, 0.90),
        ("court_q0_llama", "court", "0", 0.65, 0.60),
        ("review_q0_luna", "review", "0", 0.06, 0.80),
        ("onto", "onto", None, 0.02, 0.83),
    ]

    for b_uid, ds, qid, pos_rate, pr_auc in benchmarks:
        for raw_m, m_clean in methods:
            for g in [0.80, 0.90]:
                for d in [0.05]:
                    # ACTIS is sound with lower cost, ScaleDoc unsound on some
                    if "actis" in raw_m:
                        unsound = False
                        slack = 0.05
                        cost = 15.0 + 10.0 * (1.0 - pr_auc)
                    elif raw_m == "scaledoc":
                        unsound = pos_rate < 0.1
                        slack = -0.08 if unsound else 0.02
                        cost = 12.0
                    elif raw_m == "bargain_pr":
                        unsound = False
                        slack = 0.08
                        cost = 35.0
                    else:  # lotus
                        unsound = pr_auc < 0.7
                        slack = -0.05 if unsound else 0.01
                        cost = 25.0

                    sound_label = "Unsound" if unsound else "Sound or Inconclusive"

                    records.append(
                        {
                            "dataset": ds,
                            "scenario": ds,
                            "scenario_uid": f"{ds}_12345678",
                            "benchmark_uid": b_uid,
                            "query_id": qid,
                            "proxy_model": (
                                "azure/gpt-5.6-luna"
                                if "luna" in b_uid
                                else "openai/llama"
                            ),
                            "scenario_clean": f"{ds.capitalize()} ({b_uid})",
                            "method": raw_m,
                            "runner_uid": f"{raw_m}_87654321",
                            "method_clean": m_clean,
                            "gamma": g,
                            "gamma_P": g,
                            "gamma_R": g,
                            "delta": d,
                            "mean_oracle_rate_pct": cost,
                            "ideal_oracle_rate_pct": 5.0,
                            "quantile_slack": slack,
                            "joint_fail": 0.15 if unsound else 0.01,
                            "num_trials": 1000,
                            "unsound": unsound,
                            "sound": not unsound,
                            "sound_label": sound_label,
                            "pos_rate": pos_rate,
                            "pr_auc": pr_auc,
                            "cal_bias": 0.02,
                            "ece": 0.05,
                        }
                    )

    return pd.DataFrame(records)


def test_plot_head_to_head_pairwise(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_head_to_head_pairwise(
        mock_summary_df,
        gamma=0.80,
        delta=0.05,
        baselines=["BARGAIN", "ScaleDoc", "LOTUS"],
        output_dir=out_dir,
        skip_pgf=not HAS_XELATEX,
    )

    pdfs = list(out_dir.glob("pairwise_head_to_head_gamma_080_delta_005.pdf"))
    pgfs = list(out_dir.glob("pairwise_head_to_head_gamma_080_delta_005.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_ablation_is_vs_uniform_pr_auc(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_ablation_is_vs_uniform_pr_auc(
        mock_summary_df,
        gamma=0.80,
        delta=0.05,
        output_dir=out_dir,
        skip_pgf=not HAS_XELATEX,
    )

    pdfs = list(out_dir.glob("ablation_is_vs_uniform_pr_auc_gamma_080_delta_005.pdf"))
    pgfs = list(out_dir.glob("ablation_is_vs_uniform_pr_auc_gamma_080_delta_005.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_ablation_is_vs_uniform_selectivity(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_ablation_is_vs_uniform_selectivity(
        mock_summary_df,
        gamma=0.80,
        delta=0.05,
        output_dir=out_dir,
        skip_pgf=not HAS_XELATEX,
    )

    pdfs = list(
        out_dir.glob("ablation_is_vs_uniform_selectivity_gamma_080_delta_005.pdf")
    )
    pgfs = list(
        out_dir.glob("ablation_is_vs_uniform_selectivity_gamma_080_delta_005.pgf")
    )
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_unsoundness_vs_gamma_delta(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_unsoundness_vs_gamma_delta(
        mock_summary_df, output_dir=out_dir, skip_pgf=not HAS_XELATEX
    )

    pdfs = list(out_dir.glob("unsoundness_vs_gamma_delta.pdf"))
    pgfs = list(out_dir.glob("unsoundness_vs_gamma_delta.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_unsoundness_teaser(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_unsoundness_teaser(
        mock_summary_df, delta=0.05, output_dir=out_dir, skip_pgf=not HAS_XELATEX
    )

    pdfs = list(out_dir.glob("teaser_unsoundness_vs_gamma*.pdf"))
    pgfs = list(out_dir.glob("teaser_unsoundness_vs_gamma*.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_slack_distribution_vs_gamma_delta(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_slack_distribution_vs_gamma_delta(
        mock_summary_df, output_dir=out_dir, skip_pgf=not HAS_XELATEX
    )

    pdfs = list(out_dir.glob("slack_distribution_vs_gamma_delta.pdf"))
    pgfs = list(out_dir.glob("slack_distribution_vs_gamma_delta.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_slack_distribution_vs_gamma(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_slack_distribution_vs_gamma(
        mock_summary_df, delta=0.05, output_dir=out_dir, skip_pgf=not HAS_XELATEX
    )

    pdfs = list(out_dir.glob("slack_distribution_vs_gamma*.pdf"))
    pgfs = list(out_dir.glob("slack_distribution_vs_gamma*.pgf"))
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_ablation_is_vs_uniform_pr_auc_selectivity(
    tmp_path: Path, mock_summary_df
):
    out_dir = tmp_path / "plots"
    plot_ablation_is_vs_uniform_pr_auc_selectivity(
        mock_summary_df,
        gamma=0.80,
        delta=0.05,
        output_dir=out_dir,
        skip_pgf=not HAS_XELATEX,
    )

    pdfs = list(
        out_dir.glob("ablation_is_vs_uniform_pr_auc_selectivity_gamma_080_delta_005.pdf")
    )
    pgfs = list(
        out_dir.glob("ablation_is_vs_uniform_pr_auc_selectivity_gamma_080_delta_005.pgf")
    )
    assert len(pdfs) == 1
    assert len(pgfs) == (1 if HAS_XELATEX else 0)


def test_plot_skip_pgf(tmp_path: Path, mock_summary_df):
    out_dir = tmp_path / "plots"
    plot_ablation_is_vs_uniform_pr_auc_selectivity(
        mock_summary_df,
        gamma=0.80,
        delta=0.05,
        output_dir=out_dir,
        skip_pgf=True,
    )

    pdfs = list(
        out_dir.glob("ablation_is_vs_uniform_pr_auc_selectivity_gamma_080_delta_005.pdf")
    )
    pgfs = list(
        out_dir.glob("ablation_is_vs_uniform_pr_auc_selectivity_gamma_080_delta_005.pgf")
    )
    assert len(pdfs) == 1
    assert len(pgfs) == 0



