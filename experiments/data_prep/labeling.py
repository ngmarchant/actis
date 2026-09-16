"""Generic inference pipelines for gathering oracle labels and proxy scores.

Works on any HuggingFace Dataset, supporting:
- Polymorphic models (BaseOracle, BaseProxy, or raw callables)
- Arbitrary item modalities (text, images, audio, dicts) via `input_col`
- Upfront cost estimation and interactive confirmation
- Batching and incremental disk checkpointing for crash/restart recovery
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from datasets import Dataset
from tqdm import tqdm

from experiments.data_prep.models import (
    BaseOracle,
    BaseProxy,
    CallableOracle,
    CallableProxy,
    OracleOutput,
)


def get_unprocessed_items(
    items: Sequence[Any],
    checkpoint_path: Path | str | None,
) -> list[Any]:
    """Returns items whose indices are not yet recorded in the checkpoint file."""
    if checkpoint_path is None:
        return list(items)
    p = Path(checkpoint_path)
    if not p.exists():
        return list(items)
    try:
        with open(p, "r", encoding="utf-8") as f:
            saved = json.load(f)
            processed = {int(k) for k in saved.keys()}
        return [item for i, item in enumerate(items) if i not in processed]
    except Exception:
        return list(items)


def _confirm_cost(estimate, skip_prompt: bool = False) -> bool:
    if estimate is None or estimate.estimated_cost <= 0.0 or skip_prompt:
        return True

    print("\n" + estimate.summary())
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Cannot prompt for cost confirmation in a non-interactive session "
            "(sys.stdin is not a TTY). To proceed with external API calls, "
            "explicitly pass skip_cost_confirm=True (or use --yes / -y in CLI)."
        )

    try:
        response = input("Proceed with external API calls? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except EOFError:
        return False


def add_model_output(
    dataset: Dataset,
    query: Any,
    process_batch_fn: Callable[
        [list[Any], Any],
        OracleOutput | tuple[list[Any], list[dict[str, Any]]] | list[Any],
    ],
    output_col: str,
    cost_col: str | None = None,
    aux_col: str | None = None,
    input_col: str = "content",
    cast_fn: Callable[[Any], Any] = lambda x: x,
    aux_cast_fn: Callable[[Any], Any] = lambda x: x,
    batch_size: int = 50,
    checkpoint_path: str | Path | None = None,
    cost_estimate: Any | None = None,
    skip_cost_confirm: bool = False,
    show_progress: bool = True,
    desc: str | None = None,
) -> Dataset:
    """
    Generic pipeline to process dataset items in batches, checkpoint, and add
    output and cost columns.

    Args:
        dataset: HuggingFace Dataset containing documents/items.
        query: Query object or natural language question.
        process_batch_fn: Callable returning OracleOutput, (outputs, costs), or outputs.
        output_col: Column name to store the generated outputs.
        cost_col: Optional column name to store the per-record cost dictionary.
        aux_col: Optional column name to store auxiliary outputs (e.g. continuous
        scores).
        input_col: Column name containing the input items (text, image, etc.).
        cast_fn: Optional casting/validation function applied to each item output.
        aux_cast_fn: Optional casting function applied to auxiliary output items.
        batch_size: Number of items to send per model invocation.
        checkpoint_path: Optional path to store incremental results.
        cost_estimate: Optional CostEstimate for upfront cost confirmation.
        skip_cost_confirm: If True, bypasses interactive cost confirmation.
        show_progress: If True, displays a tqdm progress bar.
        desc: Progress bar description.

    Returns:
        Updated HuggingFace Dataset with `output_col` (and `cost_col`/`aux_col` if
        specified).
    """
    items = dataset[input_col]
    total_items = len(items)

    # 1. Cost confirmation
    if not _confirm_cost(cost_estimate, skip_prompt=skip_cost_confirm):
        raise RuntimeError("Aborted by user during cost confirmation.")

    # 2. Checkpoint recovery
    results: dict[int, dict[str, Any]] = {}
    cp_file: Path | None = Path(checkpoint_path) if checkpoint_path else None
    if cp_file is not None and cp_file.exists():
        try:
            with open(cp_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if isinstance(v, dict) and "value" in v:
                        raw_val = v["value"]
                        results[int(k)] = {
                            "value": (
                                cast_fn(raw_val)
                                if (cast_fn is not None and raw_val is not None)
                                else raw_val
                            ),
                            "aux": (
                                aux_cast_fn(v["aux"])
                                if aux_cast_fn is not None and v.get("aux") is not None
                                else v.get("aux")
                            ),
                            "cost": dict(v.get("cost", {})),
                        }
                    else:
                        results[int(k)] = {
                            "value": (
                                cast_fn(v)
                                if (cast_fn is not None and v is not None)
                                else v
                            ),
                            "aux": None,
                            "cost": {},
                        }
            print(f"Loaded {len(results)} items from checkpoint '{cp_file}'.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint '{cp_file}': {e}")

    # 3. Process remaining items in batches
    indices_to_process = [i for i in range(total_items) if i not in results]

    if indices_to_process:
        num_batches = (len(indices_to_process) + batch_size - 1) // batch_size
        pbar = tqdm(
            range(num_batches),
            desc=desc or f"Processing ({output_col})",
            disable=not show_progress,
        )

        for b in pbar:
            batch_idx = indices_to_process[b * batch_size : (b + 1) * batch_size]
            batch_items = [items[i] for i in batch_idx]

            batch_res = process_batch_fn(batch_items, query)
            if isinstance(batch_res, OracleOutput):
                batch_outputs = batch_res.labels
                batch_aux = batch_res.scores
                batch_costs = (
                    batch_res.costs
                    if batch_res.costs is not None
                    else [{} for _ in batch_outputs]
                )
            elif (
                isinstance(batch_res, tuple)
                and len(batch_res) == 2
                and isinstance(batch_res[1], (list, tuple))
            ):
                batch_outputs, batch_costs = batch_res
                batch_aux = None
            else:
                batch_outputs = batch_res
                batch_aux = None
                batch_costs = [{} for _ in batch_outputs]

            for i, (idx, out, cost) in enumerate(zip(batch_idx, batch_outputs, batch_costs)):
                aux_val = (
                    batch_aux[i]
                    if batch_aux is not None and i < len(batch_aux)
                    else None
                )
                results[idx] = {
                    "value": (
                        cast_fn(out)
                        if (cast_fn is not None and out is not None)
                        else out
                    ),
                    "aux": (
                        aux_cast_fn(aux_val)
                        if (aux_cast_fn is not None and aux_val is not None)
                        else aux_val
                    ),
                    "cost": dict(cost),
                }

            # Save checkpoint after each batch
            if cp_file is not None:
                cp_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cp_file, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in results.items()}, f)

    # 4. Attach columns to Dataset
    final_outputs = [results[i]["value"] for i in range(total_items)]
    if output_col in dataset.column_names:
        dataset = dataset.remove_columns([output_col])
    dataset = dataset.add_column(output_col, final_outputs)

    if aux_col is not None:
        final_aux = [results[i].get("aux") for i in range(total_items)]
        if any(x is not None for x in final_aux):
            if aux_col in dataset.column_names:
                dataset = dataset.remove_columns([aux_col])
            dataset = dataset.add_column(aux_col, final_aux)

    if cost_col is not None:
        final_costs = [results[i]["cost"] for i in range(total_items)]
        if cost_col in dataset.column_names:
            dataset = dataset.remove_columns([cost_col])
        dataset = dataset.add_column(cost_col, final_costs)

    return dataset


def add_oracle_labels(
    dataset: Dataset,
    query: Any,
    oracle: BaseOracle | Callable,
    input_col: str = "content",
    label_col: str = "label",
    score_col: str | None = "oracle_score",
    cost_col: str | None = "oracle_cost",
    batch_size: int = 50,
    checkpoint_path: str | Path | None = None,
    skip_cost_confirm: bool = False,
    show_progress: bool = True,
) -> Dataset:
    """
    Adds ground-truth boolean oracle labels, continuous scores, and costs to a
    HuggingFace Dataset.

    Args:
        dataset: HuggingFace Dataset containing documents/items.
        query: Query object or natural language question.
        oracle: BaseOracle instance or callable.
        input_col: Column name containing the input items (text, image, etc.).
        label_col: Column name to store the generated binary labels.
        score_col: Column name to store continuous confidence scores (default:
            'oracle_score').
        cost_col: Column name to store the per-item cost dictionary
            (default: 'oracle_cost').
        batch_size: Number of items to send per model invocation.
        checkpoint_path: Optional path to store incremental results.
        skip_cost_confirm: If True, bypasses interactive cost confirmation.
        show_progress: If True, displays a tqdm progress bar.

    Returns:
        Updated HuggingFace Dataset with `label_col`, `score_col` (if scores present),
        and `cost_col`.
    """
    if not isinstance(oracle, BaseOracle):
        oracle = CallableOracle(oracle)

    items = dataset[input_col]
    remaining_items = get_unprocessed_items(items, checkpoint_path)
    estimate = oracle.estimate_cost(remaining_items, query)

    return add_model_output(
        dataset=dataset,
        query=query,
        process_batch_fn=oracle.predict,
        output_col=label_col,
        cost_col=cost_col,
        aux_col=score_col,
        input_col=input_col,
        cast_fn=bool,
        aux_cast_fn=lambda x: float(x) if x is not None else None,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        cost_estimate=estimate,
        skip_cost_confirm=skip_cost_confirm,
        show_progress=show_progress,
        desc=f"Oracle labeling ({label_col})",
    )


def add_proxy_scores(
    dataset: Dataset,
    query: Any,
    proxy: BaseProxy | Callable,
    input_col: str = "content",
    score_col: str = "proxy_score",
    cost_col: str | None = "proxy_cost",
    batch_size: int = 50,
    checkpoint_path: str | Path | None = None,
    skip_cost_confirm: bool = False,
    show_progress: bool = True,
) -> Dataset:
    """
    Adds continuous proxy confidence scores and costs to a HuggingFace Dataset.

    Args:
        dataset: HuggingFace Dataset containing documents/items.
        query: Query object or natural language question.
        proxy: BaseProxy instance or callable.
        input_col: Column name containing the input items.
        score_col: Column name to store the generated confidence scores.
        cost_col: Column name to store the per-item cost dictionary
            (default: 'proxy_cost').
        batch_size: Number of items to send per model invocation.
        checkpoint_path: Optional path to store incremental results.
        skip_cost_confirm: If True, bypasses interactive cost confirmation.
        show_progress: If True, displays a tqdm progress bar.

    Returns:
        Updated HuggingFace Dataset with `score_col` and `cost_col`.
    """
    if not isinstance(proxy, BaseProxy):
        proxy = CallableProxy(proxy)

    items = dataset[input_col]
    remaining_items = get_unprocessed_items(items, checkpoint_path)
    estimate = proxy.estimate_cost(remaining_items, query)

    return add_model_output(
        dataset=dataset,
        query=query,
        process_batch_fn=proxy.score,
        output_col=score_col,
        cost_col=cost_col,
        input_col=input_col,
        cast_fn=float,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        cost_estimate=estimate,
        skip_cost_confirm=skip_cost_confirm,
        show_progress=show_progress,
        desc=f"Proxy scoring ({score_col})",
    )


def save_dataset(
    dataset: Dataset,
    output_path: str | Path,
    format: str = "parquet",
) -> None:
    """
    Saves a HuggingFace Dataset to disk as Parquet or HuggingFace Arrow format.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "parquet":
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        dataset.to_parquet(str(path))
        print(f"Dataset saved to '{path}'.")
    elif format.lower() in ("hf_disk", "arrow", "hf"):
        dataset.save_to_disk(str(path))
        print(f"HuggingFace dataset saved to directory '{path}'.")
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'parquet' or 'hf_disk'.")

