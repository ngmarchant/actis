import shutil
from pathlib import Path

import pandas as pd
import pytest

from experiments.plot_parameter_sensitivity import (
    generate_sensitivity_latex_tables,
    get_param_meta,
    plot_sensitivity_dual_panel,
)

HAS_XELATEX = shutil.which("xelatex") is not None or shutil.which("pdflatex") is not None


@pytest.fixture
def mock_sensitivity_df() -> pd.DataFrame:
    """Generates synthetic parameter sensitivity data covering multiple scenarios,

    gammas, and parameter values.
    """
    records = []
    scenarios = [
        ("jackson", "Jackson"),
        ("pubmed", "PubMed Q0"),
    ]
    batch_sizes = [100.0, 500.0, 1000.0]
    gammas = [0.70, 0.80, 0.90, 0.95]

    for scen, scen_disp in scenarios:
        for b in batch_sizes:
            for g in gammas:
                ocr_mean = 5.0 + 10.0 * g
                ocr_se = 0.2
                rt_mean = (1000.0 / b) * 0.1
                rt_se = rt_mean * 0.05

                records.append(
                    {
                        "file": "mock.json",
                        "scenario": scen,
                        "scenario_display": scen_disp,
                        "query_id": "0" if scen == "pubmed" else None,
                        "proxy_model": "mock_proxy",
                        "param_value": b,
                        "param_num": b,
                        "delta": 0.05,
                        "gamma": g,
                        "gamma_str": f"{g:.2f}",
                        "num_trials": 1000,
                        "ocr_mean": ocr_mean,
                        "ocr_se": ocr_se,
                        "ocr_lower": max(0.0, ocr_mean - 1.96 * ocr_se),
                        "ocr_upper": min(100.0, ocr_mean + 1.96 * ocr_se),
                        "runtime_mean": rt_mean,
                        "runtime_se": rt_se,
                        "runtime_lower": max(1e-6, rt_mean - 1.96 * rt_se),
                        "runtime_upper": rt_mean + 1.96 * rt_se,
                        "ideal_ocr": 5.0,
                        "joint_fail_rate": 0.01,
                    }
                )

    return pd.DataFrame(records)


def test_param_meta():
    meta_batch = get_param_meta("batch_size")
    assert meta_batch["short_name"] == "B"
    assert meta_batch["log_x"] is True

    meta_custom = get_param_meta("my_custom_param")
    assert meta_custom["short_name"] == "my_custom_param"
    assert meta_custom["log_x"] is False


def test_plot_sensitivity_dual_panel_skip_pgf(tmp_path: Path, mock_sensitivity_df: pd.DataFrame):
    out_dir = tmp_path / "plots"
    plot_sensitivity_dual_panel(
        mock_sensitivity_df,
        param_name="batch_size",
        output_dir=out_dir,
        skip_pgf=True,
    )

    assert (out_dir / "sensitivity_B_latency.pdf").exists()
    assert (out_dir / "sensitivity_B_ocr.pdf").exists()
    assert (out_dir / "sensitivity_B_jackson_dual.pdf").exists()

    assert not (out_dir / "sensitivity_B_latency.pgf").exists()
    assert not (out_dir / "sensitivity_B_ocr.pgf").exists()
    assert not (out_dir / "sensitivity_B_jackson_dual.pgf").exists()


def test_plot_sensitivity_dual_panel_with_pgf(tmp_path: Path, mock_sensitivity_df: pd.DataFrame):
    out_dir = tmp_path / "plots"
    plot_sensitivity_dual_panel(
        mock_sensitivity_df,
        param_name="batch_size",
        output_dir=out_dir,
        skip_pgf=not HAS_XELATEX,
    )

    assert (out_dir / "sensitivity_B_latency.pdf").exists()
    assert (out_dir / "sensitivity_B_ocr.pdf").exists()
    assert (out_dir / "sensitivity_B_jackson_dual.pdf").exists()

    if HAS_XELATEX:
        assert (out_dir / "sensitivity_B_latency.pgf").exists()
        assert (out_dir / "sensitivity_B_ocr.pgf").exists()
        assert (out_dir / "sensitivity_B_jackson_dual.pgf").exists()


def test_generate_sensitivity_latex_tables(tmp_path: Path, mock_sensitivity_df: pd.DataFrame):
    out_dir = tmp_path / "tables"
    res = generate_sensitivity_latex_tables(
        mock_sensitivity_df,
        param_name="batch_size",
        output_dir=out_dir,
    )

    assert "wide" in res
    assert "compact" in res
    assert (out_dir / "table_B_sensitivity_wide.tex").exists()
    assert (out_dir / "table_B_sensitivity.tex").exists()
    assert (out_dir / "sensitivity_B.csv").exists()

