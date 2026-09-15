"""Data ingestion and preprocessing loaders for ScaleDoc datasets.

Supports:
- PubMed (PubMed 200k RCT across dev, test, and train splits)
- BigPatent (NortheasternUniversity/big_patent across CPC classifications a-y)
- GovReport (ccdv/govreport-summarization)
- Query loading from ScaleDoc query.json
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

PUBMED_BASE_URL = (
    "https://github.com/Franck-Dernoncourt/pubmed-rct/raw/refs/heads/master/PubMed_200k_RCT"
)
SCALEDOC_QUERY_URL = (
    "https://raw.githubusercontent.com/Seurgul/ScaleDoc/refs/heads/main/dataset/query.json"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scaledoc"
DEFAULT_QUERY_FILE = DEFAULT_DATA_DIR / "query.json"


def parse_pubmed_text(text_or_path: str | Path, n: int = 10000) -> list[str]:
    """
    Parses abstracts from a PubMed RCT text file, following ScaleDoc preprocessing.

    Strips PMIDs (###<id>) and section labels (OBJECTIVE, BACKGROUND, METHODS,
    RESULTS, CONCLUSIONS), grouping sentences into full abstract documents.
    """
    elim = ["OBJECTIVE", "BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    id_pattern = re.compile(r"^###\d+")

    if isinstance(text_or_path, (str, Path)) and os.path.isfile(text_or_path):
        with open(text_or_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = str(text_or_path).splitlines(keepends=True)

    documents: list[str] = []
    current_doc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_doc:
                documents.append(" ".join(current_doc).strip())
                current_doc = []
                if len(documents) >= n:
                    break
            continue

        if id_pattern.match(stripped):
            continue

        # Check for section header match
        matched_header = None
        for header in elim:
            if stripped.startswith(header):
                matched_header = header
                break

        if matched_header is not None:
            clean_line = stripped[len(matched_header):].lstrip("\t :.-")
        else:
            clean_line = stripped

        if clean_line:
            current_doc.append(clean_line)

    if current_doc and len(documents) < n:
        documents.append(" ".join(current_doc).strip())

    return documents[:n]


def load_pubmed_documents(
    n: int = 10000,
    split: str = "train",
    cache_dir: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dataset:
    """
    Loads PubMed RCT documents as a HuggingFace Dataset.

    Args:
        n: Number of document abstracts to return.
        split: 'train', 'dev', or 'test' (default: 'train').
        cache_dir: Directory to cache downloaded raw files.
        local_path: Optional explicit local file path to raw text.

    Returns:
        Dataset with columns ['id', 'content'].
    """
    if local_path and Path(local_path).exists():
        raw_file = Path(local_path)
    else:
        cache_path = Path(cache_dir or DEFAULT_DATA_DIR / "pubmed")
        cache_path.mkdir(parents=True, exist_ok=True)
        raw_file = cache_path / f"{split}.txt"

        if not raw_file.exists():
            if split in ("dev", "test"):
                url = f"{PUBMED_BASE_URL}/{split}.txt"
                print(f"Downloading PubMed {split} split from {url}...")
                urllib.request.urlretrieve(url, raw_file)
            elif split == "train":
                archive_file = cache_path / "train.7z"
                if not archive_file.exists():
                    url = f"{PUBMED_BASE_URL}/train.7z"
                    print(f"Downloading PubMed train split from {url}...")
                    urllib.request.urlretrieve(url, archive_file)
                import py7zr

                print(f"Extracting {archive_file}...")
                with py7zr.SevenZipFile(archive_file, mode="r") as z:
                    z.extractall(path=cache_path)
            else:
                raise ValueError(
                    f"Unknown split '{split}' for PubMed. "
                    "Choose 'train', 'dev', or 'test'."
                )

    docs = parse_pubmed_text(raw_file, n=n)
    if len(docs) < n:
        print(
            f"Warning: Split '{split}' for PubMed only contains {len(docs)} documents "
            f"(fewer than requested n={n}). Note that 'dev' and 'test' splits only "
            "contain 2,500 documents each; use split='train' for up to "
            "~195,000 documents."
        )

    return Dataset.from_dict({
        "id": list(range(len(docs))),
        "content": docs,
    })


def load_bigpatent_documents(
    n: int = 10000,
    split: str = "train",
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """
    Loads BigPatent abstracts across 9 CPC categories as a HuggingFace Dataset.

    Args:
        n: Total number of documents across all categories.
        split: Dataset split ('train', 'validation', 'test').
        seed: Seed for sampling across codes.
        cache_dir: Optional cache directory for HuggingFace downloads.

    Returns:
        Dataset with columns ['id', 'content'].
    """
    codes = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]
    docs_per_code = (n + len(codes) - 1) // len(codes)
    rng = random.Random(seed)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    all_docs: list[str] = []
    for code in codes:
        try:
            ds = load_dataset(
                "NortheasternUniversity/big_patent",
                code,
                split=split,
                cache_dir=cache_dir_str,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                "big_patent",
                code,
                split=split,
                cache_dir=cache_dir_str,
                trust_remote_code=True,
            )

        abstracts = ds["abstract"]
        k = min(docs_per_code, len(abstracts))
        sampled = rng.sample(abstracts, k)
        all_docs.extend(sampled)

    rng.shuffle(all_docs)
    selected_docs = all_docs[:n]
    return Dataset.from_dict({
        "id": list(range(len(selected_docs))),
        "content": selected_docs,
    })


def load_govreport_documents(
    n: int = 10000,
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> Dataset:
    """
    Loads GovReport document summaries as a HuggingFace Dataset.

    Args:
        n: Number of documents to return.
        split: Dataset split ('train', 'validation', 'test').
        cache_dir: Optional cache directory for HuggingFace downloads.

    Returns:
        Dataset with columns ['id', 'content'].
    """
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    ds = load_dataset(
        "ccdv/govreport-summarization",
        split=split,
        cache_dir=cache_dir_str,
    )
    summaries = ds["summary"][:n]
    return Dataset.from_dict({
        "id": list(range(len(summaries))),
        "content": summaries,
    })


def load_scaledoc_queries(
    dataset_name: str,
    query_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Loads query definitions for a given dataset from ScaleDoc query.json or URL.

    Args:
        dataset_name: Name of dataset ('pubmed', 'big_patent', 'gov_report', etc.).
        query_file: Optional path or URL to query.json. If None, downloads from
            SCALEDOC_QUERY_URL and caches locally.
        cache_dir: Optional directory to cache downloaded query.json.

    Returns:
        List of dicts with 'q_id' and 'query'.
    """
    if query_file is not None:
        query_file_str = str(query_file)
        if query_file_str.startswith(("http://", "https://")):
            req = urllib.request.Request(
                query_file_str, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        else:
            path = Path(query_file)
            if not path.exists():
                raise FileNotFoundError(f"ScaleDoc query file not found at '{path}'.")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    else:
        target_dir = Path(cache_dir or DEFAULT_DATA_DIR)
        cached_file = target_dir / "query.json"

        if cached_file.exists():
            with open(cached_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading ScaleDoc queries from {SCALEDOC_QUERY_URL}...")
            req = urllib.request.Request(
                SCALEDOC_QUERY_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
            with open(cached_file, "wb") as f:
                f.write(raw_bytes)

    # Normalize name keys
    clean_name = dataset_name.lower().replace("-", "_")
    base_key = None
    if clean_name in data:
        base_key = clean_name
    else:
        for k in data:
            if k.replace("_", "") == clean_name.replace("_", ""):
                base_key = k
                break

    if base_key is None:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in query definitions. "
            f"Available keys: {list(data.keys())}"
        )

    # Base queries (q_id: 0, 1, 2, ...)
    results: list[dict[str, Any]] = [
        {"q_id": str(item["q_id"]), "query": item["query"]}
        for item in data[base_key]
    ]

    # Check for extended queries under f"{base_key}_ext" (q_id: 0_ext, 1_ext, ...)
    ext_key = f"{base_key}_ext"
    if ext_key in data:
        for item in data[ext_key]:
            results.append({
                "q_id": f"{item['q_id']}_ext",
                "query": item["query"],
            })

    return results
