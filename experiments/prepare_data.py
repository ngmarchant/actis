#!/usr/bin/env python3
"""
CLI tool for preparing benchmark datasets, gathering oracle labels, and
computing proxy scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

# Automatically load environment variables from .env before initializing HF or LiteLLM
load_dotenv(find_dotenv(usecwd=True))

import os  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from datasets import Dataset  # noqa: E402

from experiments.data_prep import (  # noqa: E402
    DATASET_GROUPS,
    BaseOracle,
    BaseProxy,
    CostEstimate,
    LiteLLMOracle,
    LiteLLMProxy,
    add_oracle_labels,
    add_proxy_scores,
    get_benchmark_prompt_defaults,
    get_unprocessed_items,
    load_bigpatent_documents,
    load_court_documents,
    load_govreport_documents,
    load_pubmed_documents,
    load_queries,
    load_review_documents,
    load_screenplay_documents,
    load_wiki_documents,
    oracle_checkpoint_path,
    oracle_output_path,
    proxy_checkpoint_path,
    proxy_output_path,
    save_dataset,
)

QUERY_FILES = {
    "scaledoc": _ROOT / "experiments" / "data" / "scaledoc" / "query.json",
    "bargain": _ROOT / "experiments" / "data" / "bargain" / "query.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare datasets with oracle labels and proxy scores."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed",
        choices=sorted(DATASET_GROUPS),
        help="Dataset name to process (default: pubmed)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: 'train')",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10000,
        help="Number of documents to extract (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document sampling (default: 42)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["0"],
        help=(
            "List of query IDs to process (e.g. '0 1 0_ext'), or presets: "
            "'all' (base + ext), 'base' (original queries), or 'ext' (extended queries)"
            " (default: 0)"
        ),
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Optional path to query.json",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Local source file for datasets that require one (currently: court)",
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="azure/gpt-5.6-terra",
        help="Model identifier for oracle ground truth (default: azure/gpt-5.6-terra)",
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default="azure/gpt-5.6-luna",
        help="Model identifier for proxy scores (e.g. llama-3.2-1b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/data/{group}/{dataset})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Source-download cache directory (default: .cache)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to LiteLLM config.yaml defining model_list, custom rates, and "
        "deployments",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for model inference (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent async API requests (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for models (default: 0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "hf_disk"],
        help="Output format: 'parquet' or 'hf_disk' (default: parquet)",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip oracle ground-truth labeling",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Skip proxy scoring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token count and cost without sending API requests",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of existing label or proxy_score columns",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (defaults to benchmark-specific default)",
    )
    parser.add_argument(
        "--user-template",
        type=str,
        default=None,
        help="Custom user prompt template (defaults to benchmark-specific default)",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default=None,
        help=(
            "Positive label string (defaults to 'Yes' for ScaleDoc, 'True' for BARGAIN)"
        ),
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default=None,
        help=(
            "Negative label string (defaults to 'No' for ScaleDoc, 'False' for BARGAIN)"
        ),
    )
    return parser.parse_args()


def get_base_documents(
    dataset_name: str,
    split: str,
    num_docs: int,
    output_dir: Path,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    source_path: Path | str | None = None,
) -> Dataset:
    cache_path = output_dir / "documents.parquet"
    if cache_path.exists():
        import pandas as pd
        df = pd.read_parquet(cache_path)
        if len(df) > 0:
            if len(df) < num_docs:
                print(
                    f"Cached documents at '{cache_path}' contain only "
                    f"{len(df)} documents; using all available documents "
                    f"instead of downloading again for requested n={num_docs}."
                )
            else:
                print(
                    f"Loading cached base documents from '{cache_path}' "
                    f"({len(df)} available, returning {num_docs})..."
                )
            return Dataset.from_pandas(df.iloc[:num_docs])
        else:
            print(
                f"Cached documents at '{cache_path}' are empty. "
                "Re-preparing base documents..."
            )

    print(
        f"Preparing base documents for '{dataset_name}' "
        f"(split='{split}', n={num_docs}, seed={seed})..."
    )
    if dataset_name == "pubmed":
        ds = load_pubmed_documents(
            n=num_docs,
            split=split,
            cache_dir=cache_dir or output_dir,
        )
    elif dataset_name == "big_patent":
        ds = load_bigpatent_documents(
            n=num_docs,
            split=split,
            seed=seed,
            cache_dir=cache_dir,
        )
    elif dataset_name == "gov_report":
        ds = load_govreport_documents(
            n=num_docs,
            split=split,
            cache_dir=cache_dir,
        )
    elif dataset_name == "screenplay":
        ds = load_screenplay_documents(n=num_docs, seed=seed, cache_dir=cache_dir)
    elif dataset_name == "review":
        ds = load_review_documents(n=num_docs, seed=seed, cache_dir=cache_dir)
    elif dataset_name == "wiki":
        ds = load_wiki_documents(n=num_docs, seed=seed, cache_dir=cache_dir)
    elif dataset_name == "court":
        if source_path is None:
            raise ValueError("The court dataset requires --source-path.")
        ds = load_court_documents(
            path=source_path,
            n=num_docs,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    save_dataset(ds, cache_path, format="parquet")
    return ds


def load_query_dataset(path: Path, format: str = "parquet") -> Dataset | None:
    """Loads an existing query dataset if present, or returns None."""
    p = Path(path)
    if format.lower() == "parquet":
        if p.suffix != ".parquet":
            p = p.with_suffix(".parquet")
        if not p.exists():
            return None
        import pandas as pd

        return Dataset.from_pandas(pd.read_parquet(p))
    elif format.lower() in ("hf_disk", "arrow", "hf"):
        if not p.exists():
            return None
        from datasets import load_from_disk

        loaded = load_from_disk(str(p))
        if isinstance(loaded, Dataset):
            return loaded
        raise TypeError(f"Expected a Dataset at '{p}', got {type(loaded).__name__}")
    return None


@dataclass
class QueryPlanItem:
    qid: str
    query_text: str
    oracle_path: Path
    proxy_path: Path | None
    existing_oracle_ds: Dataset | None
    existing_proxy_ds: Dataset | None
    has_label: bool
    has_proxy: bool
    needs_oracle: bool
    needs_proxy: bool
    oracle_est: CostEstimate | None = None
    proxy_est: CostEstimate | None = None


@dataclass
class ExecutionPlan:
    items: list[QueryPlanItem]
    oracle_model: str | None
    proxy_model: str | None

    @property
    def total_oracle_queries(self) -> int:
        return sum(1 for item in self.items if item.needs_oracle)

    @property
    def total_proxy_queries(self) -> int:
        return sum(1 for item in self.items if item.needs_proxy)

    @property
    def total_oracle_items_remaining(self) -> int:
        return sum(
            item.oracle_est.total_items
            for item in self.items
            if item.oracle_est is not None
        )

    @property
    def total_proxy_items_remaining(self) -> int:
        return sum(
            item.proxy_est.total_items
            for item in self.items
            if item.proxy_est is not None
        )

    @property
    def total_oracle_prompt_tokens(self) -> int:
        return sum(
            item.oracle_est.prompt_tokens
            for item in self.items
            if item.oracle_est is not None
        )

    @property
    def total_oracle_compl_tokens(self) -> int:
        return sum(
            item.oracle_est.completion_tokens
            for item in self.items
            if item.oracle_est is not None
        )

    @property
    def total_oracle_cost(self) -> float:
        return sum(
            item.oracle_est.estimated_cost
            for item in self.items
            if item.oracle_est is not None
        )

    @property
    def total_proxy_prompt_tokens(self) -> int:
        return sum(
            item.proxy_est.prompt_tokens
            for item in self.items
            if item.proxy_est is not None
        )

    @property
    def total_proxy_compl_tokens(self) -> int:
        return sum(
            item.proxy_est.completion_tokens
            for item in self.items
            if item.proxy_est is not None
        )

    @property
    def total_proxy_cost(self) -> float:
        return sum(
            item.proxy_est.estimated_cost
            for item in self.items
            if item.proxy_est is not None
        )

    @property
    def grand_total_tokens(self) -> int:
        return (
            self.total_oracle_prompt_tokens
            + self.total_oracle_compl_tokens
            + self.total_proxy_prompt_tokens
            + self.total_proxy_compl_tokens
        )

    @property
    def grand_total_cost(self) -> float:
        return self.total_oracle_cost + self.total_proxy_cost

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Execution Plan & Upfront Cost Estimate",
            "=" * 70,
            f"Total Queries Selected: {len(self.items)}",
            "",
            "Per-Query Status:",
        ]
        for item in self.items:
            o_status = (
                "NEEDED"
                if item.needs_oracle
                else ("COMPLETED" if item.has_label else "SKIPPED")
            )
            p_status = (
                "NEEDED"
                if item.needs_proxy
                else ("COMPLETED" if item.has_proxy else "SKIPPED")
            )
            q_snippet = (
                f'"{item.query_text[:40]}..."'
                if len(item.query_text) > 40
                else f'"{item.query_text}"'
            )
            lines.append(
                f"  - Query {item.qid:>2}: Oracle: {o_status:<9} | "
                f"Proxy: {p_status:<9} {q_snippet}"
            )

        lines.append("")
        lines.append("--- Cost Breakdown ---")
        if self.oracle_model:
            lines.extend([
                f"Oracle ({self.oracle_model}):",
                f"  Queries to run:       "
                f"{self.total_oracle_queries} of {len(self.items)}",
                f"  Items remaining:      {self.total_oracle_items_remaining:,}",
                f"  Est. Prompt Tokens:   {self.total_oracle_prompt_tokens:,}",
                f"  Est. Compl. Tokens:   {self.total_oracle_compl_tokens:,}",
                f"  Est. Cost:            ${self.total_oracle_cost:,.4f}",
            ])
        else:
            lines.append("Oracle: Not configured / skipped.")

        if self.proxy_model:
            lines.extend([
                f"Proxy ({self.proxy_model}):",
                f"  Queries to run:       "
                f"{self.total_proxy_queries} of {len(self.items)}",
                f"  Items remaining:      {self.total_proxy_items_remaining:,}",
                f"  Est. Prompt Tokens:   {self.total_proxy_prompt_tokens:,}",
                f"  Est. Compl. Tokens:   {self.total_proxy_compl_tokens:,}",
                f"  Est. Cost:            ${self.total_proxy_cost:,.4f}",
            ])
        else:
            lines.append("Proxy:  Not configured / skipped.")

        lines.extend([
            "-" * 70,
            f"Total Est. Tokens: {self.grand_total_tokens:,}",
            f"Total Est. Cost:   ${self.grand_total_cost:,.4f}",
            "=" * 70,
        ])
        return "\n".join(lines)


def create_execution_plan(
    selected_qids: Sequence[str],
    query_map: Mapping[str, str],
    base_docs: Dataset,
    output_dir: Path,
    output_format: str,
    oracle_model: str,
    proxy_model: str | None,
    oracle: BaseOracle | None,
    proxy: BaseProxy | None,
    force: bool = False,
) -> ExecutionPlan:
    plan_items: list[QueryPlanItem] = []
    base_len = len(base_docs)
    doc_items = base_docs["content"]

    for qid in selected_qids:
        if qid not in query_map:
            print(f"Warning: Query ID {qid} not found in query list, skipping.")
            continue

        q_text = query_map[qid]
        oracle_path = oracle_output_path(output_dir, qid, oracle_model)
        proxy_path = (
            proxy_output_path(output_dir, qid, proxy_model)
            if proxy_model is not None
            else None
        )

        existing_oracle_ds = load_query_dataset(oracle_path, format=output_format)
        existing_proxy_ds = (
            load_query_dataset(proxy_path, format=output_format)
            if proxy_path is not None
            else None
        )

        has_label = (
            existing_oracle_ds is not None
            and len(existing_oracle_ds) >= base_len
            and "label" in existing_oracle_ds.column_names
            and not force
        )
        has_proxy = (
            existing_proxy_ds is not None
            and len(existing_proxy_ds) >= base_len
            and "proxy_score" in existing_proxy_ds.column_names
            and not force
        )

        needs_oracle = (oracle is not None) and not has_label
        needs_proxy = (proxy is not None) and not has_proxy

        cp_oracle = oracle_checkpoint_path(output_dir, qid, oracle_model)
        cp_proxy = (
            proxy_checkpoint_path(output_dir, qid, proxy_model)
            if proxy_model is not None
            else None
        )

        oracle_est = None
        if needs_oracle and oracle is not None:
            oracle_items = (
                get_unprocessed_items(doc_items, cp_oracle)
                if not force
                else list(doc_items)
            )
            oracle_est = oracle.estimate_cost(oracle_items, q_text)

        proxy_est = None
        if needs_proxy and proxy is not None:
            proxy_items = (
                get_unprocessed_items(doc_items, cp_proxy)
                if not force
                else list(doc_items)
            )
            proxy_est = proxy.estimate_cost(proxy_items, q_text)

        plan_items.append(
            QueryPlanItem(
                qid=qid,
                query_text=q_text,
                oracle_path=oracle_path,
                proxy_path=proxy_path,
                existing_oracle_ds=existing_oracle_ds,
                existing_proxy_ds=existing_proxy_ds,
                has_label=has_label,
                has_proxy=has_proxy,
                needs_oracle=needs_oracle,
                needs_proxy=needs_proxy,
                oracle_est=oracle_est,
                proxy_est=proxy_est,
            )
        )

    oracle_model_name = getattr(oracle, "model", None) if oracle else None
    proxy_model_name = getattr(proxy, "model", None) if proxy else None
    return ExecutionPlan(
        items=plan_items,
        oracle_model=oracle_model_name,
        proxy_model=proxy_model_name,
    )


def main() -> int:
    args = parse_args()

    cache_dir = Path(args.cache_dir or ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If explicit cache directory provided, set HF_HOME environment variable
    if cache_dir:
        resolved_cache = str(cache_dir.resolve())
        os.environ["HF_HOME"] = resolved_cache
        print(f"Set HF_HOME cache directory to: '{resolved_cache}'")

    # Determine split
    split = args.split or "train"

    # Determine output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments/data") / DATASET_GROUPS[args.dataset] / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    query_file = args.query_file or QUERY_FILES[DATASET_GROUPS[args.dataset]]
    all_queries = load_queries(args.dataset, query_file=query_file)
    query_map: dict[str, str] = {q["q_id"]: q["query"] for q in all_queries}

    def sort_key(k: str) -> tuple[int, int]:
        k = str(k)
        if k.endswith("_ext"):
            prefix = k[:-4]
            num = int(prefix) if prefix.isdigit() else 999
            return (1, num)
        num = int(k) if k.isdigit() else 999
        return (0, num)

    all_keys = sorted(query_map.keys(), key=sort_key)
    base_keys = [k for k in all_keys if not k.endswith("_ext")]
    ext_keys = [k for k in all_keys if k.endswith("_ext")]

    raw_queries = [str(q).strip() for q in args.queries]
    if len(raw_queries) == 1 and raw_queries[0].lower() == "all":
        selected_qids = all_keys
    elif len(raw_queries) == 1 and raw_queries[0].lower() == "base":
        selected_qids = base_keys
    elif len(raw_queries) == 1 and raw_queries[0].lower() == "ext":
        if not ext_keys:
            print(
                f"Warning: No extended queries available for dataset '{args.dataset}'."
            )
            return 0
        selected_qids = ext_keys
    else:
        selected_qids = []
        for q in raw_queries:
            if q in query_map:
                selected_qids.append(q)
            else:
                valid_sample = ", ".join(all_keys[:10])
                raise ValueError(
                    f"Invalid query ID '{q}' for dataset '{args.dataset}'. "
                    f"Valid IDs include: {valid_sample}..."
                )

    print(
        f"Selected {len(selected_qids)} queries for '{args.dataset}': {selected_qids}"
    )

    # Load or create base documents
    base_docs = get_base_documents(
        dataset_name=args.dataset,
        split=split,
        num_docs=args.num_docs,
        output_dir=output_dir,
        seed=args.seed,
        cache_dir=cache_dir,
        source_path=args.source_path,
    )
    print(f"Base documents ready: {len(base_docs)} documents.")

    # Detect config.yaml if present
    config_path = args.config
    if config_path is None and Path("config.yaml").exists():
        config_path = "config.yaml"
        print("Auto-detected and loaded 'config.yaml' from current directory.")

    # Resolve prompt templates and labels
    prompt_defaults = get_benchmark_prompt_defaults(args.dataset)
    sys_prompt = (
        args.system_prompt
        if args.system_prompt is not None
        else prompt_defaults["system_prompt"]
    )
    user_tmpl = (
        args.user_template
        if args.user_template is not None
        else prompt_defaults["user_prompt_template"]
    )
    pos_label = (
        args.positive_label
        if args.positive_label is not None
        else prompt_defaults["positive_label"]
    )
    neg_label = (
        args.negative_label
        if args.negative_label is not None
        else prompt_defaults["negative_label"]
    )

    # Initialize models
    # TODO: ensure model supports non-zero temperature.
    litellm_kwargs = {"temperature": args.temperature}
    oracle = None
    if not args.skip_oracle:
        oracle = LiteLLMOracle(
            model=args.oracle_model,
            system_prompt=sys_prompt,
            user_prompt_template=user_tmpl,
            positive_label=pos_label,
            negative_label=neg_label,
            max_concurrency=args.concurrency,
            litellm_kwargs=litellm_kwargs,
            config=config_path,
        )

    proxy = None
    if not args.skip_proxy and args.proxy_model:
        proxy = LiteLLMProxy(
            model=args.proxy_model,
            system_prompt=sys_prompt,
            user_prompt_template=user_tmpl,
            positive_label=pos_label,
            negative_label=neg_label,
            max_concurrency=args.concurrency,
            litellm_kwargs=litellm_kwargs,
            config=config_path,
        )

    # Build upfront execution plan and cost estimates
    plan = create_execution_plan(
        selected_qids=selected_qids,
        query_map=query_map,
        base_docs=base_docs,
        output_dir=output_dir,
        output_format=args.format,
        oracle_model=args.oracle_model,
        proxy_model=args.proxy_model,
        oracle=oracle,
        proxy=proxy,
        force=args.force,
    )

    print("\n" + plan.summary())

    if plan.total_oracle_queries == 0 and plan.total_proxy_queries == 0:
        print(
            "\nAll selected queries already have requested labels and proxy "
            "scores. Nothing to do (use --force to re-generate)."
        )
        return 0

    if args.dry_run:
        print("\n[Dry Run] Exiting without making API calls.")
        return 0

    # Single upfront confirmation prompt
    if not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Cannot prompt for cost confirmation in a non-interactive session "
                "(sys.stdin is not a TTY). To proceed with external API calls, "
                "pass --yes / -y."
            )
        try:
            resp = (
                input("\nProceed with all planned API calls? [y/N]: ")
                .strip()
                .lower()
            )
            if resp not in ("y", "yes"):
                print("Aborted by user.")
                return 0
        except EOFError:
            print("Aborted by user.")
            return 0

    # Process each query: oracle and proxy stages are cached independently per
    # model, so different oracle/proxy combinations can coexist and reuse work.
    for item in plan.items:
        qid = item.qid
        q_text = item.query_text
        print("\n" + "=" * 50)
        print(f"Processing Query {qid}: \"{q_text}\"")
        print("=" * 50)

        # 1. Oracle labeling
        if item.has_label:
            print(
                f"Query {qid} already has oracle output cached at "
                f"'{item.oracle_path}'. Skipping oracle labeling."
            )
        elif item.needs_oracle and oracle is not None:
            oracle_input = Dataset.from_dict({"content": base_docs["content"]})
            cp_oracle = oracle_checkpoint_path(output_dir, qid, args.oracle_model)
            oracle_ds = add_oracle_labels(
                dataset=oracle_input,
                query=q_text,
                oracle=oracle,
                input_col="content",
                label_col="label",
                batch_size=args.batch_size,
                checkpoint_path=cp_oracle,
                skip_cost_confirm=True,
            ).remove_columns(["content"])
            save_dataset(oracle_ds, item.oracle_path, format=args.format)
            print(f"Query {qid} oracle output saved to '{item.oracle_path}'.")

        # 2. Proxy scoring
        if item.has_proxy:
            print(
                f"Query {qid} already has proxy output cached at "
                f"'{item.proxy_path}'. Skipping proxy scoring."
            )
        elif item.needs_proxy and proxy is not None and item.proxy_path is not None:
            proxy_input = Dataset.from_dict({"content": base_docs["content"]})
            cp_proxy = proxy_checkpoint_path(output_dir, qid, args.proxy_model)
            proxy_ds = add_proxy_scores(
                dataset=proxy_input,
                query=q_text,
                proxy=proxy,
                input_col="content",
                score_col="proxy_score",
                batch_size=args.batch_size,
                checkpoint_path=cp_proxy,
                skip_cost_confirm=True,
            ).remove_columns(["content"])
            save_dataset(proxy_ds, item.proxy_path, format=args.format)
            print(f"Query {qid} proxy output saved to '{item.proxy_path}'.")

    print("\nAll tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
