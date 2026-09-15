#!/usr/bin/env python3
"""
CLI Tool for preparing benchmark datasets (PubMed, BigPatent, GovReport),
gathering ground-truth oracle labels, and computing proxy scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    BaseOracle,
    BaseProxy,
    CostEstimate,
    LiteLLMOracle,
    LiteLLMProxy,
    add_oracle_labels,
    add_proxy_scores,
    load_bigpatent_documents,
    load_govreport_documents,
    load_pubmed_documents,
    load_scaledoc_queries,
    save_dataset,
)


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
        choices=["pubmed", "big_patent", "gov_report"],
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
        "--queries",
        nargs="+",
        default=["0"],
        help="List of query IDs to process, e.g. '0 1 2' or 'all' (default: 0)",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Optional path to query.json",
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="gpt-4o",
        help="Model identifier for oracle ground truth (default: gpt-4o)",
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default=None,
        help="Model identifier for proxy scores (e.g. llama-3.2-1b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/data/scaledoc/{dataset})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for Hugging Face datasets",
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
    return parser.parse_args()


def get_base_documents(
    dataset_name: str,
    split: str,
    num_docs: int,
    output_dir: Path,
    cache_dir: Path | str | None = None,
) -> Dataset:
    cache_path = output_dir / "documents.parquet"
    if cache_path.exists():
        import pandas as pd
        df = pd.read_parquet(cache_path)
        if len(df) >= num_docs:
            print(
                f"Loading cached base documents from '{cache_path}' "
                f"({len(df)} available, returning {num_docs})..."
            )
            return Dataset.from_pandas(df.iloc[:num_docs])
        else:
            print(
                f"Cached documents at '{cache_path}' only contains "
                f"{len(df)} documents, but {num_docs} were requested. "
                f"Re-preparing base documents..."
            )

    print(
        f"Preparing base documents for '{dataset_name}' "
        f"(split='{split}', n={num_docs})..."
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
            cache_dir=cache_dir,
        )
    elif dataset_name == "gov_report":
        ds = load_govreport_documents(
            n=num_docs,
            split=split,
            cache_dir=cache_dir,
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
    qid: int
    query_text: str
    target_file: Path
    existing_ds: Dataset | None
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
    selected_qids: list[int],
    query_map: dict[int, str],
    base_docs: Dataset,
    output_dir: Path,
    output_format: str,
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
        target_file = output_dir / f"q{qid}.{output_format}"
        if output_format.lower() == "parquet" and target_file.suffix != ".parquet":
            target_file = target_file.with_suffix(".parquet")

        existing_ds = load_query_dataset(target_file, format=output_format)
        has_label = False
        has_proxy = False

        if existing_ds is not None and len(existing_ds) >= base_len:
            has_label = ("label" in existing_ds.column_names) and not force
            has_proxy = ("proxy_score" in existing_ds.column_names) and not force

        needs_oracle = (oracle is not None) and not has_label
        needs_proxy = (proxy is not None) and not has_proxy

        oracle_est = None
        if needs_oracle and oracle is not None:
            oracle_est = oracle.estimate_cost(doc_items, q_text)

        proxy_est = None
        if needs_proxy and proxy is not None:
            proxy_est = proxy.estimate_cost(doc_items, q_text)

        plan_items.append(
            QueryPlanItem(
                qid=qid,
                query_text=q_text,
                target_file=target_file,
                existing_ds=existing_ds,
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

    # If explicit cache directory provided, set HF_HOME environment variable
    if args.cache_dir:
        resolved_cache = str(Path(args.cache_dir).resolve())
        os.environ["HF_HOME"] = resolved_cache
        print(f"Set HF_HOME cache directory to: '{resolved_cache}'")

    # Determine split
    split = args.split or "train"

    # Determine output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments/data/scaledoc") / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    all_queries = load_scaledoc_queries(args.dataset, query_file=args.query_file)
    query_map = {int(q["q_id"]): q["query"] for q in all_queries}

    if len(args.queries) == 1 and args.queries[0].lower() == "all":
        selected_qids = sorted(query_map.keys())
    else:
        selected_qids = [int(q) for q in args.queries]

    print(
        f"Selected {len(selected_qids)} queries for '{args.dataset}': {selected_qids}"
    )

    # Load or create base documents
    base_docs = get_base_documents(
        dataset_name=args.dataset,
        split=split,
        num_docs=args.num_docs,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
    )
    print(f"Base documents ready: {len(base_docs)} documents.")

    # Detect config.yaml if present
    config_path = args.config
    if config_path is None and Path("config.yaml").exists():
        config_path = "config.yaml"
        print("Auto-detected and loaded 'config.yaml' from current directory.")

    # Initialize models
    oracle = None
    if not args.skip_oracle:
        oracle = LiteLLMOracle(
            model=args.oracle_model,
            max_concurrency=args.concurrency,
            config=config_path,
        )

    proxy = None
    if not args.skip_proxy and args.proxy_model:
        proxy = LiteLLMProxy(
            model=args.proxy_model,
            max_concurrency=args.concurrency,
            config=config_path,
        )

    # Build upfront execution plan and cost estimates
    plan = create_execution_plan(
        selected_qids=selected_qids,
        query_map=query_map,
        base_docs=base_docs,
        output_dir=output_dir,
        output_format=args.format,
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

    # Process each query
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for item in plan.items:
        qid = item.qid
        q_text = item.query_text
        print("\n" + "=" * 50)
        print(f"Processing Query {qid}: \"{q_text}\"")
        print("=" * 50)

        target_file = item.target_file
        if item.existing_ds is not None and len(item.existing_ds) >= len(base_docs):
            curr_dataset = item.existing_ds.select(range(len(base_docs)))
            print(
                f"Loaded existing query dataset from '{target_file}' with "
                f"{len(curr_dataset)} items (columns: {curr_dataset.column_names})."
            )
        else:
            if item.existing_ds is not None:
                print(
                    f"Existing query dataset '{target_file}' only has "
                    f"{len(item.existing_ds)} items (requested {len(base_docs)}). "
                    f"Starting from base documents."
                )
            curr_dataset = base_docs

        modified = False

        # 1. Oracle labeling
        if item.has_label:
            print(
                f"Query {qid} already has 'label' column in '{target_file}' "
                f"({len(curr_dataset)} items). Skipping oracle labeling."
            )
        elif item.needs_oracle and oracle is not None:
            cp_oracle = checkpoints_dir / f"q{qid}_oracle.json"
            curr_dataset = add_oracle_labels(
                dataset=curr_dataset,
                query=q_text,
                oracle=oracle,
                input_col="content",
                label_col="label",
                batch_size=args.batch_size,
                checkpoint_path=cp_oracle,
                skip_cost_confirm=True,
            )
            modified = True

        # 2. Proxy scoring
        if item.has_proxy:
            print(
                f"Query {qid} already has 'proxy_score' column in '{target_file}' "
                f"({len(curr_dataset)} items). Skipping proxy scoring."
            )
        elif item.needs_proxy and proxy is not None:
            cp_proxy = checkpoints_dir / f"q{qid}_proxy.json"
            curr_dataset = add_proxy_scores(
                dataset=curr_dataset,
                query=q_text,
                proxy=proxy,
                input_col="content",
                score_col="proxy_score",
                batch_size=args.batch_size,
                checkpoint_path=cp_proxy,
                skip_cost_confirm=True,
            )
            modified = True

        # Save query file if modified or not yet existing
        if modified or not target_file.exists():
            save_dataset(curr_dataset, target_file, format=args.format)
            print(f"Query {qid} dataset complete: '{target_file}'")
        else:
            print(f"Query {qid} dataset already complete: '{target_file}'")

    print("\nAll tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
