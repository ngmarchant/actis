"""Data preparation and labeling toolkit for ScaleDoc and semantic
database benchmarks.
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Automatically load environment variables from experiments/.env (or CWD .env) if present
_exp_env = Path(__file__).resolve().parents[1] / ".env"
if _exp_env.exists():
    load_dotenv(_exp_env)
load_dotenv(find_dotenv(usecwd=True))

from experiments.data_prep.labeling import (  # noqa: E402
    add_model_output,
    add_oracle_labels,
    add_proxy_scores,
    get_unprocessed_items,
    save_dataset,
)
from experiments.data_prep.loaders import (  # noqa: E402
    DATASET_GROUPS,
    SCALEDOC_QUERY_URL,
    SUPG_BASE_URL,
    SUPG_DATASET_FILES,
    download_supg_dataset,
    load_bigpatent_documents,
    load_court_documents,
    load_govreport_documents,
    load_imagenet_dataset,
    load_jackson_dataset,
    load_onto_dataset,
    load_pubmed_documents,
    load_queries,
    load_review_documents,
    load_screenplay_documents,
    load_supg_dataset,
    load_tacred_dataset,
    load_wiki_documents,
    parse_pubmed_text,
)
from experiments.data_prep.models import (  # noqa: E402
    BENCHMARK_PROMPT_DEFAULTS,
    DEFAULT_BARGAIN_NEGATIVE_LABEL,
    DEFAULT_BARGAIN_POSITIVE_LABEL,
    DEFAULT_BARGAIN_SYSTEM_PROMPT,
    DEFAULT_BARGAIN_USER_TEMPLATE,
    DEFAULT_SCALEDOC_NEGATIVE_LABEL,
    DEFAULT_SCALEDOC_POSITIVE_LABEL,
    DEFAULT_SCALEDOC_SYSTEM_PROMPT,
    DEFAULT_SCALEDOC_USER_TEMPLATE,
    BaseLiteLLMModel,
    BaseOracle,
    BaseProxy,
    CallableOracle,
    CallableProxy,
    CostEstimate,
    LiteLLMOracle,
    LiteLLMProxy,
    get_benchmark_prompt_defaults,
    load_litellm_config,
)
from experiments.model_paths import (  # noqa: E402
    oracle_checkpoint_path,
    oracle_output_path,
    proxy_checkpoint_path,
    proxy_output_path,
    slugify_model_name,
)

__all__ = [
    "BENCHMARK_PROMPT_DEFAULTS",
    "DATASET_GROUPS",
    "DEFAULT_BARGAIN_NEGATIVE_LABEL",
    "DEFAULT_BARGAIN_POSITIVE_LABEL",
    "DEFAULT_BARGAIN_SYSTEM_PROMPT",
    "DEFAULT_BARGAIN_USER_TEMPLATE",
    "DEFAULT_SCALEDOC_NEGATIVE_LABEL",
    "DEFAULT_SCALEDOC_POSITIVE_LABEL",
    "DEFAULT_SCALEDOC_SYSTEM_PROMPT",
    "DEFAULT_SCALEDOC_USER_TEMPLATE",
    "SCALEDOC_QUERY_URL",
    "SUPG_BASE_URL",
    "SUPG_DATASET_FILES",
    "BaseLiteLLMModel",
    "BaseOracle",
    "BaseProxy",
    "CallableOracle",
    "CallableProxy",
    "CostEstimate",
    "LiteLLMOracle",
    "LiteLLMProxy",
    "add_model_output",
    "add_oracle_labels",
    "add_proxy_scores",
    "download_supg_dataset",
    "get_benchmark_prompt_defaults",
    "get_unprocessed_items",
    "load_bigpatent_documents",
    "load_court_documents",
    "load_govreport_documents",
    "load_imagenet_dataset",
    "load_jackson_dataset",
    "load_litellm_config",
    "load_onto_dataset",
    "load_pubmed_documents",
    "load_queries",
    "load_review_documents",
    "load_screenplay_documents",
    "load_supg_dataset",
    "load_tacred_dataset",
    "load_wiki_documents",
    "oracle_checkpoint_path",
    "oracle_output_path",
    "parse_pubmed_text",
    "proxy_checkpoint_path",
    "proxy_output_path",
    "save_dataset",
    "slugify_model_name",
]
