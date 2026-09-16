"""Data preparation and labeling toolkit for ScaleDoc and semantic
database benchmarks.
"""

from dotenv import find_dotenv, load_dotenv

# Automatically load environment variables from .env if present
load_dotenv(find_dotenv(usecwd=True))

from experiments.data_prep.labeling import (  # noqa: E402
    add_model_output,
    add_oracle_labels,
    add_proxy_scores,
    get_unprocessed_items,
    save_dataset,
)
from experiments.data_prep.loaders import (  # noqa: E402
    SCALEDOC_QUERY_URL,
    load_bigpatent_documents,
    load_govreport_documents,
    load_pubmed_documents,
    load_scaledoc_queries,
    parse_pubmed_text,
)
from experiments.data_prep.models import (  # noqa: E402
    BaseLiteLLMModel,
    BaseOracle,
    BaseProxy,
    CallableOracle,
    CallableProxy,
    CostEstimate,
    LiteLLMOracle,
    LiteLLMProxy,
    load_litellm_config,
)

__all__ = [
    "SCALEDOC_QUERY_URL",
    "BaseLiteLLMModel",
    "BaseOracle",
    "BaseProxy",
    "LiteLLMOracle",
    "LiteLLMProxy",
    "CallableOracle",
    "CallableProxy",
    "CostEstimate",
    "load_litellm_config",
    "load_pubmed_documents",
    "load_bigpatent_documents",
    "load_govreport_documents",
    "load_scaledoc_queries",
    "parse_pubmed_text",
    "add_model_output",
    "add_oracle_labels",
    "add_proxy_scores",
    "get_unprocessed_items",
    "save_dataset",
]
