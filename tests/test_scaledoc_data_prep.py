"""Unit tests for ScaleDoc data preparation, model interfaces, inference,
and scenarios.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from experiments.data_prep.labeling import (
    add_model_output,
    add_oracle_labels,
    add_proxy_scores,
    save_dataset,
)
from experiments.data_prep.loaders import (
    load_bigpatent_documents,
    load_govreport_documents,
    load_scaledoc_queries,
    parse_pubmed_text,
)
from experiments.data_prep.models import (
    DEFAULT_SCALEDOC_SYSTEM_PROMPT,
    BaseOracle,
    BaseProxy,
    CallableOracle,
    CallableProxy,
    CostEstimate,
    LiteLLMOracle,
    LiteLLMProxy,
    OracleOutput,
    _default_prompt_formatter,
    _extract_litellm_binary_probability,
    load_litellm_config,
)
from experiments.scenarios import (
    SCENARIOS,
    ScaleDocDataset,
    ScaleDocPubMed,
    _read_dataframe,
)


def test_parse_pubmed_text():
    raw_sample = """###1001
BACKGROUND
This is the background line.
METHODS
We tested 100 subjects.
RESULTS
Significant improvement was observed.
CONCLUSIONS
The drug works well.

###1002
OBJECTIVE
To evaluate safety.
METHODS
Double-blind trial.
RESULTS
No adverse effects.
CONCLUSIONS
Safe for use.
"""
    docs = parse_pubmed_text(raw_sample, n=10)
    assert len(docs) == 2
    assert "This is the background line." in docs[0]
    assert "We tested 100 subjects." in docs[0]
    assert "BACKGROUND" not in docs[0]
    assert "METHODS" not in docs[0]
    assert "###1001" not in docs[0]

    assert "To evaluate safety." in docs[1]
    assert "Safe for use." in docs[1]
    assert "OBJECTIVE" not in docs[1]
    assert "###1002" not in docs[1]


def test_load_scaledoc_queries():
    queries = load_scaledoc_queries("pubmed")
    assert len(queries) == 26
    assert queries[0]["q_id"] == '0'
    assert "query" in queries[0]
    assert queries[20]["q_id"] == "0_ext"
    assert queries[25]["q_id"] == "5_ext"

    bp_queries = load_scaledoc_queries("big_patent")
    assert len(bp_queries) == 25
    assert bp_queries[0]["q_id"] == '0'
    assert bp_queries[20]["q_id"] == "0_ext"
    assert bp_queries[24]["q_id"] == "4_ext"

    gr_queries = load_scaledoc_queries("gov_report")
    assert len(gr_queries) == 20
    assert gr_queries[0]["q_id"] == '0'
    assert gr_queries[19]["q_id"] == '19'


def test_cost_estimate_summary():
    est = CostEstimate(
        total_items=100,
        prompt_tokens=40000,
        completion_tokens=100,
        total_tokens=40100,
        estimated_cost=0.10,
        model_name="gpt-4o",
    )
    summary = est.summary()
    assert "gpt-4o" in summary
    assert "100" in summary
    assert "$0.1000" in summary


def test_callable_oracle_and_proxy():
    ds = Dataset.from_dict({
        "id": [0, 1, 2],
        "content": ["apple", "banana", "cherry"],
    })

    def mock_oracle_fn(items, query):
        return ["a" in item for item in items]

    def mock_proxy_fn(items, query):
        return [float(len(item) / 10.0) for item in items]

    ds = add_oracle_labels(
        dataset=ds,
        query="fruit",
        oracle=mock_oracle_fn,
        skip_cost_confirm=True,
    )
    assert "label" in ds.column_names
    assert ds["label"] == [True, True, False]

    ds = add_proxy_scores(
        dataset=ds,
        query="fruit",
        proxy=mock_proxy_fn,
        skip_cost_confirm=True,
    )
    assert "proxy_score" in ds.column_names
    assert ds["proxy_score"] == pytest.approx([0.5, 0.6, 0.6])


def test_add_model_output_generic():
    ds = Dataset.from_dict({
        "id": [0, 1],
        "content": ["foo", "bar"],
    })

    def custom_processor(items, query):
        return [f"{query}:{item}" for item in items]

    ds = add_model_output(
        dataset=ds,
        query="tag",
        process_batch_fn=custom_processor,
        output_col="tagged",
        skip_cost_confirm=True,
    )
    assert "tagged" in ds.column_names
    assert ds["tagged"] == ["tag:foo", "tag:bar"]


def test_litellm_oracle_mocked():
    oracle = LiteLLMOracle(model="test-oracle", max_concurrency=2)

    mock_resp_yes = MagicMock()
    mock_resp_yes.choices = [MagicMock(message=MagicMock(content="Yes"))]
    mock_resp_yes.usage = MagicMock(prompt_tokens=10, completion_tokens=1)

    mock_resp_no = MagicMock()
    mock_resp_no.choices = [MagicMock(message=MagicMock(content="No"))]
    mock_resp_no.usage = MagicMock(prompt_tokens=12, completion_tokens=1)

    mock_side_effect = AsyncMock(side_effect=[mock_resp_yes, mock_resp_no])
    with patch("litellm.acompletion", mock_side_effect):
        out = oracle.predict(["doc 1", "doc 2"], "query?")
        assert isinstance(out, OracleOutput)
        assert out.labels == [True, False]
        assert len(out.costs) == 2
        assert out.costs[0]["input_tokens"] == 10
        assert out.costs[1]["input_tokens"] == 12
        assert out.scores == [1.0, 0.0]


def test_litellm_proxy_mocked():
    proxy = LiteLLMProxy(model="test-proxy", max_concurrency=2)

    # Mock choice with logprobs for Yes (-0.5) and No (-1.5)
    mock_choice = MagicMock()
    top_lp1 = MagicMock(token="Yes", logprob=-0.5)
    top_lp2 = MagicMock(token="No", logprob=-1.5)
    mock_choice.logprobs = MagicMock(
        content=[MagicMock(top_logprobs=[top_lp1, top_lp2])]
    )
    mock_choice.message = MagicMock(content="Yes")
    mock_resp = MagicMock(choices=[mock_choice])
    mock_resp.usage = MagicMock(prompt_tokens=15, completion_tokens=1)

    with patch("litellm.acompletion", AsyncMock(return_value=mock_resp)):
        scores, costs = proxy.score(["doc 1"], "query?")
        assert len(scores) == 1
        # P(Yes) = exp(-0.5) / (exp(-0.5) + exp(-1.5)) ~ 0.731
        assert 0.70 < scores[0] < 0.75
        assert len(costs) == 1
        assert costs[0]["input_tokens"] == 15


def test_checkpoint_resumption(tmp_path):
    cp_file = tmp_path / "checkpoint.json"

    # Pre-populate checkpoint for item 0
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump({"0": True}, f)

    call_count = 0

    def counting_oracle(items, query):
        nonlocal call_count
        call_count += len(items)
        return [False for _ in items]

    ds = Dataset.from_dict({
        "id": [0, 1, 2],
        "content": ["d0", "d1", "d2"],
    })

    ds = add_oracle_labels(
        dataset=ds,
        query="q",
        oracle=counting_oracle,
        checkpoint_path=cp_file,
        batch_size=2,
        skip_cost_confirm=True,
    )

    # Item 0 was loaded from checkpoint, so only items 1 and 2 were called
    assert call_count == 2
    assert ds["label"] == [True, False, False]


def test_scaledoc_scenario_and_tabular_dataset(tmp_path):
    data_file = tmp_path / "q0.parquet"
    ds = Dataset.from_dict({
        "id": [0, 1, 2, 3],
        "content": ["text0", "text1", "text2", "text3"],
        "label": [True, False, True, False],
        "proxy_score": [0.9, 0.1, 0.8, 0.2],
    })
    save_dataset(ds, data_file, format="parquet")

    scenario = ScaleDocPubMed(query_id='0', data_path=data_file)
    scores, labels = scenario.generate_population()

    assert np.allclose(scores, [0.9, 0.1, 0.8, 0.2])
    assert np.array_equal(labels, [True, False, True, False])

    # Dynamic registry lookup test
    dynamic_sc = SCENARIOS["scaledoc_pubmed_q0"]
    assert isinstance(dynamic_sc, ScaleDocDataset)
    assert dynamic_sc.dataset_name == "pubmed"
    assert dynamic_sc.query_id == '0'

    dynamic_sc5 = SCENARIOS["scaledoc_gov_report_q5"]
    assert isinstance(dynamic_sc5, ScaleDocDataset)
    assert dynamic_sc5.dataset_name == "gov_report"
    assert dynamic_sc5.query_id == '5'


def test_read_dataframe_hf_disk(tmp_path):
    hf_dir = tmp_path / "saved_hf_dataset"
    ds = Dataset.from_dict({
        "id": [1, 2],
        "label": [True, False],
        "proxy_score": [0.8, 0.3],
    })
    ds.save_to_disk(str(hf_dir))

    df = _read_dataframe(hf_dir)
    assert list(df["id"]) == [1, 2]
    assert list(df["label"]) == [True, False]
    assert list(df["proxy_score"]) == pytest.approx([0.8, 0.3])


def test_cost_confirmation_non_interactive_safety():
    mock_oracle = MagicMock(spec=BaseOracle)
    mock_oracle.estimate_cost.return_value = CostEstimate(
        total_items=10,
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        estimated_cost=5.00,
        model_name="expensive-oracle",
    )
    ds = Dataset.from_dict({"id": [0], "content": ["doc"]})

    with patch("sys.stdin.isatty", return_value=False):
        # Without skip_cost_confirm=True, should fail in non-interactive session
        with pytest.raises(RuntimeError, match="Cannot prompt for cost confirmation"):
            add_oracle_labels(ds, "query", mock_oracle, skip_cost_confirm=False)


def test_loaders_pass_cache_dir():
    with patch("experiments.data_prep.loaders.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = ["summary1"]
        mock_load.return_value = mock_ds

        load_govreport_documents(n=1, cache_dir="/custom/cache/dir")
        mock_load.assert_called_once_with(
            "ccdv/govreport-summarization",
            split="train",
            cache_dir="/custom/cache/dir",
        )

    with patch("experiments.data_prep.loaders.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = ["abs1"]
        mock_load.return_value = mock_ds

        load_bigpatent_documents(n=1, cache_dir="/custom/cache/dir")
        assert mock_load.call_args[1]["cache_dir"] == "/custom/cache/dir"


def test_dotenv_loading(tmp_path):
    from dotenv import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_HOME=/data/custom/hf_cache\nAZURE_API_KEY=super-secret-key\n",
        encoding="utf-8",
    )

    with patch.dict("os.environ", {}, clear=False):
        load_dotenv(env_file, override=True)
        assert os.environ.get("HF_HOME") == "/data/custom/hf_cache"
        assert os.environ.get("AZURE_API_KEY") == "super-secret-key"


def test_load_scaledoc_queries_url(tmp_path):
    mock_json = json.dumps({
        "custom_ds": [{"q_id": 0, "query": "Custom query?"}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_json
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Test loading from explicit URL
        queries = load_scaledoc_queries(
            "custom_ds",
            query_file="https://example.com/query.json",
        )
        assert len(queries) == 1
        assert queries[0]["query"] == "Custom query?"

        # Test downloading to custom cache_dir when default does not exist
        queries2 = load_scaledoc_queries(
            "custom_ds",
            cache_dir=tmp_path / "cache",
        )
        assert len(queries2) == 1
        assert queries2[0]["q_id"] == '0'
        assert (tmp_path / "cache" / "query.json").exists()


def test_config_yaml_custom_rates_and_oracle(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """model_list:
  - model_name: azure-custom-gpt4o
    litellm_params:
      model: azure/aueast-deployment
      api_base: https://custom-azure.openai.azure.com/
      api_key: os.environ/CUSTOM_KEY
    model_info:
      input_cost_per_token: 0.0000050
      output_cost_per_token: 0.0000200
""",
        encoding="utf-8",
    )

    router = load_litellm_config(cfg_file)
    assert router.model_list[0]["model_name"] == "azure-custom-gpt4o"

    # Test initialization with file path
    oracle = LiteLLMOracle(
        model="azure/aueast-deployment",
        config=cfg_file,
    )
    assert oracle.router is not None

    # Test cost estimation uses custom regional rate from router
    est = oracle.estimate_cost(["test document text"], "query?")
    expected_cost = (
        est.prompt_tokens * 0.0000050
        + est.completion_tokens * 0.0000200
    )
    assert est.estimated_cost == pytest.approx(expected_cost, rel=1e-5)

    # Test initialization with dict
    cfg_dict = {
        "model_list": [
            {
                "model_name": "dict-proxy-model",
                "litellm_params": {
                    "model": "azure/dict-deployment",
                },
                "model_info": {
                    "input_cost_per_token": 0.0000030,
                    "output_cost_per_token": 0.0000150,
                },
            }
        ]
    }
    proxy = LiteLLMProxy(
        model="azure/dict-deployment",
        config=cfg_dict,
    )
    assert proxy.router is not None
    est_proxy = proxy.estimate_cost(["test document text"], "query?")
    expected_proxy_cost = (
        est_proxy.prompt_tokens * 0.0000030
        + est_proxy.completion_tokens * 0.0000150
    )
    assert est_proxy.estimated_cost == pytest.approx(expected_proxy_cost, rel=1e-5)

    # Test initialization with existing Router instance
    oracle_with_router = LiteLLMOracle(
        model="azure/aueast-deployment",
        config=router,
    )
    assert oracle_with_router.router is router


def test_oracle_router_acompletion_call(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """model_list:
  - model_name: azure-routed
    litellm_params:
      model: azure/deployment
      input_cost_per_token: 0.000001
      output_cost_per_token: 0.00002
""",
        encoding="utf-8",
    )

    oracle = LiteLLMOracle(
        model="azure-routed",
        config=cfg_file,
    )

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = MagicMock(content="Yes")
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=1)

    oracle.router.acompletion = AsyncMock(return_value=mock_resp)

    out = oracle.predict(["doc 1"], "is valid?")
    assert isinstance(out, OracleOutput)
    assert out.labels == [True]
    assert len(out.costs) == 1
    assert out.costs[0]["input_tokens"] == 5
    assert oracle.router.acompletion.await_count == 1


def test_multi_phase_workflow_and_reuse(tmp_path):
    from experiments.prepare_scaledoc_data import load_query_dataset

    base_ds = Dataset.from_dict({
        "id": ["doc1", "doc2"],
        "content": ["Text 1", "Text 2"],
    })

    target_file = tmp_path / "q0.parquet"

    # Step 1: Add oracle labels with costs
    mock_oracle = MagicMock(spec=BaseOracle)
    mock_oracle.predict.return_value = OracleOutput(
        labels=[True, False],
        scores=[0.95, 0.05],
        costs=[
            {"monetary": 0.001, "input_tokens": 10},
            {"monetary": 0.002, "input_tokens": 20},
        ],
    )
    mock_oracle.estimate_cost.return_value = CostEstimate(
        2, 20, 2, 22, 0.001, "m"
    )

    ds_with_oracle = add_oracle_labels(
        base_ds,
        "query",
        mock_oracle,
        skip_cost_confirm=True,
    )
    save_dataset(ds_with_oracle, target_file, format="parquet")

    # Verify target file has label, oracle_score, and oracle_cost
    loaded = load_query_dataset(target_file, format="parquet")
    assert loaded is not None
    assert loaded.column_names == [
        "id",
        "content",
        "label",
        "oracle_score",
        "oracle_cost",
    ]
    assert loaded["label"] == [True, False]
    assert loaded["oracle_score"] == pytest.approx([0.95, 0.05])
    assert loaded["oracle_cost"][0]["monetary"] == 0.001

    # Step 2: Add proxy scores to the existing dataset
    mock_proxy = MagicMock(spec=BaseProxy)
    mock_proxy.score.return_value = (
        [0.85, 0.15],
        [
            {"monetary": 0.0001, "input_tokens": 5},
            {"monetary": 0.0001, "input_tokens": 5},
        ],
    )
    mock_proxy.estimate_cost.return_value = CostEstimate(
        2, 20, 2, 22, 0.001, "m"
    )

    ds_with_both = add_proxy_scores(
        loaded,
        "query",
        mock_proxy,
        skip_cost_confirm=True,
    )
    save_dataset(ds_with_both, target_file, format="parquet")

    # Verify target file now contains BOTH label, oracle_score, oracle_cost, proxy_score, proxy_cost
    final_ds = load_query_dataset(target_file, format="parquet")
    assert final_ds is not None
    assert final_ds.column_names == [
        "id",
        "content",
        "label",
        "oracle_score",
        "oracle_cost",
        "proxy_score",
        "proxy_cost",
    ]
    assert final_ds["label"] == [True, False]
    assert final_ds["proxy_score"] == [0.85, 0.15]
    assert final_ds["oracle_cost"][0]["input_tokens"] == 10
    assert final_ds["proxy_cost"][0]["monetary"] == 0.0001


def test_create_execution_plan_and_summary(tmp_path):
    from experiments.prepare_scaledoc_data import create_execution_plan

    base_ds = Dataset.from_dict({
        "id": ["d1", "d2"],
        "content": ["Text 1", "Text 2"],
    })

    # Prepare q0.parquet with label already present
    q0_file = tmp_path / "q0.parquet"
    ds_q0 = Dataset.from_dict({
        "id": ["d1", "d2"],
        "content": ["Text 1", "Text 2"],
        "label": [True, False],
    })
    save_dataset(ds_q0, q0_file, format="parquet")

    # Prepare q1.parquet with both label and proxy_score present
    q1_file = tmp_path / "q1.parquet"
    ds_q1 = Dataset.from_dict({
        "id": ["d1", "d2"],
        "content": ["Text 1", "Text 2"],
        "label": [True, True],
        "proxy_score": [0.9, 0.8],
    })
    save_dataset(ds_q1, q1_file, format="parquet")

    # q2 does not exist yet

    mock_oracle = MagicMock(spec=BaseOracle)
    mock_oracle.model = "test-oracle"
    mock_oracle.estimate_cost.return_value = CostEstimate(
        total_items=2,
        prompt_tokens=20,
        completion_tokens=2,
        total_tokens=22,
        estimated_cost=0.01,
        model_name="test-oracle",
    )

    mock_proxy = MagicMock(spec=BaseProxy)
    mock_proxy.model = "test-proxy"
    mock_proxy.estimate_cost.return_value = CostEstimate(
        total_items=2,
        prompt_tokens=20,
        completion_tokens=2,
        total_tokens=22,
        estimated_cost=0.005,
        model_name="test-proxy",
    )

    query_map = {
        '0': "Query 0 text",
        '1': "Query 1 text",
        '2': "Query 2 text",
    }

    plan = create_execution_plan(
        selected_qids=['0', '1', '2'],
        query_map=query_map,
        base_docs=base_ds,
        output_dir=tmp_path,
        output_format="parquet",
        oracle=mock_oracle,
        proxy=mock_proxy,
        force=False,
    )

    assert len(plan.items) == 3

    # Query 0: has_label=True, has_proxy=False -> needs_oracle=False, needs_proxy=True
    assert plan.items[0].has_label is True
    assert plan.items[0].has_proxy is False
    assert plan.items[0].needs_oracle is False
    assert plan.items[0].needs_proxy is True

    # Query 1: has_label=True, has_proxy=True -> needs_oracle=False, needs_proxy=False
    assert plan.items[1].has_label is True
    assert plan.items[1].has_proxy is True
    assert plan.items[1].needs_oracle is False
    assert plan.items[1].needs_proxy is False

    # Query 2: does not exist -> needs_oracle=True, needs_proxy=True
    assert plan.items[2].has_label is False
    assert plan.items[2].has_proxy is False
    assert plan.items[2].needs_oracle is True
    assert plan.items[2].needs_proxy is True

    # Totals
    assert plan.total_oracle_queries == 1  # Only query 2
    assert plan.total_proxy_queries == 2   # Queries 0 and 2
    assert plan.total_oracle_cost == pytest.approx(0.01)
    assert plan.total_proxy_cost == pytest.approx(0.01)
    assert plan.grand_total_cost == pytest.approx(0.02)

    summary = plan.summary()
    assert "Query  0: Oracle: COMPLETED | Proxy: NEEDED" in summary
    assert "Query  1: Oracle: COMPLETED | Proxy: COMPLETED" in summary
    assert "Query  2: Oracle: NEEDED    | Proxy: NEEDED" in summary
    assert "Total Est. Cost:   $0.0200" in summary


def test_callable_oracle_and_proxy_costs():
    # 1. Callable returning plain labels, default cost is empty dict
    co_default = CallableOracle(lambda items, q: [True for _ in items])
    out1 = co_default.predict(["d1", "d2"], "q")
    assert isinstance(out1, OracleOutput)
    assert out1.labels == [True, True]
    assert out1.costs == [{}, {}]

    # 2. Callable returning plain labels with custom default_cost
    co_custom = CallableOracle(
        lambda items, q: [False for _ in items],
        default_cost={"monetary": 0.05, "constant_calls": 1},
    )
    out2 = co_custom.predict(["d1"], "q")
    assert isinstance(out2, OracleOutput)
    assert out2.labels == [False]
    assert out2.costs == [{"monetary": 0.05, "constant_calls": 1}]

    # 3. Callable returning explicit (scores, costs) tuple
    cp_explicit = CallableProxy(
        lambda items, q: (
            [0.9, 0.1],
            [{"latency_ms": 1.2}, {"latency_ms": 1.5}],
        )
    )
    scores, costs3 = cp_explicit.score(["d1", "d2"], "q")
    assert scores == [0.9, 0.1]
    assert costs3 == [{"latency_ms": 1.2}, {"latency_ms": 1.5}]


def test_tabular_dataset_get_costs_and_dataframe(tmp_path):
    q_file = tmp_path / "q0.parquet"
    ds = Dataset.from_dict({
        "id": [0, 1, 2],
        "content": ["a", "b", "c"],
        "label": [True, False, True],
        "proxy_score": [0.8, 0.2, 0.9],
        "oracle_cost": [
            {"monetary": 0.005, "input_tokens": 100},
            {"monetary": 0.003, "input_tokens": 60},
            {"monetary": 0.008, "input_tokens": 160},
        ],
        "proxy_cost": [
            {"monetary": 0.0001, "latency_ms": 5.0},
            {"monetary": 0.0001, "latency_ms": 4.8},
            {"monetary": 0.0001, "latency_ms": 5.2},
        ],
    })
    save_dataset(ds, q_file, format="parquet")

    sc = ScaleDocPubMed(query_id='0', data_path=q_file)
    df = sc.get_dataframe()
    assert len(df) == 3
    assert "oracle_cost" in df.columns

    # Test extracting monetary costs
    proxy_costs, oracle_costs = sc.get_costs("monetary")
    assert np.allclose(proxy_costs, [0.0001, 0.0001, 0.0001])
    assert np.allclose(oracle_costs, [0.005, 0.003, 0.008])

    # Test extracting custom key (latency_ms, input_tokens)
    p_lat, _ = sc.get_costs("latency_ms")
    assert np.allclose(p_lat, [5.0, 4.8, 5.2])

    _, o_tokens = sc.get_costs("input_tokens")
    assert np.allclose(o_tokens, [100.0, 60.0, 160.0])


def test_create_execution_plan_with_ext_queries(tmp_path):
    from experiments.prepare_scaledoc_data import create_execution_plan

    base_ds = Dataset.from_dict({
        "id": ["d1", "d2"],
        "content": ["Text 1", "Text 2"],
    })

    # Pre-populate q0_ext.parquet with label
    q0_ext_file = tmp_path / "q0_ext.parquet"
    ds_ext = Dataset.from_dict({
        "id": ["d1", "d2"],
        "content": ["Text 1", "Text 2"],
        "label": [True, False],
    })
    save_dataset(ds_ext, q0_ext_file, format="parquet")

    mock_oracle = MagicMock(spec=BaseOracle)
    mock_oracle.model = "test-oracle"
    mock_oracle.estimate_cost.return_value = CostEstimate(
        total_items=2,
        prompt_tokens=20,
        completion_tokens=2,
        total_tokens=22,
        estimated_cost=0.01,
        model_name="test-oracle",
    )

    query_map = {
        "0": "Base query 0",
        "0_ext": "Extended query 0",
        "1_ext": "Extended query 1",
    }

    plan = create_execution_plan(
        selected_qids=["0", "0_ext", "1_ext"],
        query_map=query_map,
        base_docs=base_ds,
        output_dir=tmp_path,
        output_format="parquet",
        oracle=mock_oracle,
        proxy=None,
        force=False,
    )

    assert len(plan.items) == 3
    assert plan.items[0].qid == "0"
    assert plan.items[0].target_file == tmp_path / "q0.parquet"
    assert plan.items[0].needs_oracle is True

    assert plan.items[1].qid == "0_ext"
    assert plan.items[1].target_file == tmp_path / "q0_ext.parquet"
    assert plan.items[1].has_label is True
    assert plan.items[1].needs_oracle is False

    assert plan.items[2].qid == "1_ext"
    assert plan.items[2].target_file == tmp_path / "q1_ext.parquet"
    assert plan.items[2].needs_oracle is True

    summary = plan.summary()
    assert "Query  0: Oracle: NEEDED" in summary
    assert "Query 0_ext: Oracle: COMPLETED" in summary
    assert "Query 1_ext: Oracle: NEEDED" in summary


def test_scenarios_registry_with_ext_queries():
    # Test dynamic registration of extended query scenarios
    sc_pub_ext = SCENARIOS["scaledoc_pubmed_q0_ext"]
    assert sc_pub_ext.dataset_name == "pubmed"
    assert sc_pub_ext.query_id == "0_ext"
    assert sc_pub_ext.name == "scaledoc_pubmed_q0_ext"
    assert str(sc_pub_ext.data_path).endswith("experiments/data/scaledoc/pubmed/q0_ext.parquet")

    sc_bp_ext = SCENARIOS["scaledoc_big_patent_q4_ext"]
    assert sc_bp_ext.dataset_name == "big_patent"
    assert sc_bp_ext.query_id == "4_ext"
    assert sc_bp_ext.name == "scaledoc_big_patent_q4_ext"
    assert str(sc_bp_ext.data_path).endswith("experiments/data/scaledoc/big_patent/q4_ext.parquet")


def test_oracle_output_dataclass():
    # Test construction with defaults
    out_default = OracleOutput(labels=[True, False])
    assert out_default.labels == [True, False]
    assert out_default.scores is None
    assert out_default.costs is None

    # Test construction with explicit scores and costs
    out_full = OracleOutput(
        labels=[True],
        scores=[0.95],
        costs=[{"input_tokens": 10}],
    )
    assert out_full.labels == [True]
    assert out_full.scores == [0.95]
    assert out_full.costs == [{"input_tokens": 10}]


def test_extract_binary_probability_custom_labels_and_logsumexp():
    # Mock choice with logprobs for multiple positive tokens ('True', ' True') and negative ('False')
    # Let lp('True') = -0.5, lp(' True') = -1.2, lp('False') = -0.8
    # pos_lse = log(exp(-0.5) + exp(-1.2)) = log(0.6065 + 0.3012) = log(0.9077) = -0.0968
    # neg_lse = -0.8
    # expected sigmoid(pos_lse - neg_lse) = sigmoid(-0.0968 - (-0.8)) = sigmoid(0.7032) = 1 / (1 + exp(-0.7032)) ≈ 0.6689
    mock_choice = MagicMock()
    mock_choice.message.content = "True"
    top_lp1 = MagicMock(token="True", logprob=-0.5)
    top_lp2 = MagicMock(token="False", logprob=-0.8)
    top_lp3 = MagicMock(token=" True", logprob=-1.2)

    content_lp = MagicMock()
    content_lp.top_logprobs = [top_lp1, top_lp2, top_lp3]
    mock_choice.logprobs.content = [content_lp]
    mock_resp = MagicMock(choices=[mock_choice])

    score = _extract_litellm_binary_probability(
        mock_resp,
        positive_label="True",
        negative_label="False",
    )
    assert score == pytest.approx(0.6689, abs=1e-3)


def test_system_prompt_label_slots():
    assert "{positive_label}" in DEFAULT_SCALEDOC_SYSTEM_PROMPT
    assert "{negative_label}" in DEFAULT_SCALEDOC_SYSTEM_PROMPT

    # Test default ScaleDoc formatting ("Yes" / "No")
    messages_scaledoc = _default_prompt_formatter(
        item="Sample doc",
        query="Is it valid?",
        sys_prompt=DEFAULT_SCALEDOC_SYSTEM_PROMPT,
        user_tmpl="Doc: {doc}\nQuery: {query}",
        positive_label="Yes",
        negative_label="No",
    )
    assert "Just output the 'Yes' or 'No' only." in messages_scaledoc[0]["content"]

    # Test custom formatting for BARGAIN ("True" / "False")
    messages_bargain = _default_prompt_formatter(
        item="Sample doc",
        query="Is it valid?",
        sys_prompt=DEFAULT_SCALEDOC_SYSTEM_PROMPT,
        user_tmpl="Doc: {doc}\nQuery: {query}",
        positive_label="True",
        negative_label="False",
    )
    assert "Just output the 'True' or 'False' only." in messages_bargain[0]["content"]


def test_add_oracle_labels_with_oracle_score():
    ds = Dataset.from_dict({
        "id": [0, 1],
        "content": ["doc A", "doc B"],
    })

    mock_oracle = MagicMock(spec=BaseOracle)
    mock_oracle.predict.return_value = OracleOutput(
        labels=[True, False],
        scores=[0.92, 0.08],
        costs=[{"monetary": 0.001}, {"monetary": 0.001}],
    )
    mock_oracle.estimate_cost.return_value = None

    result_ds = add_oracle_labels(
        dataset=ds,
        query="test query",
        oracle=mock_oracle,
        skip_cost_confirm=True,
    )

    assert "label" in result_ds.column_names
    assert "oracle_score" in result_ds.column_names
    assert "oracle_cost" in result_ds.column_names

    assert result_ds["label"] == [True, False]
    assert result_ds["oracle_score"] == pytest.approx([0.92, 0.08])
    assert result_ds["oracle_cost"] == [{"monetary": 0.001}, {"monetary": 0.001}]
