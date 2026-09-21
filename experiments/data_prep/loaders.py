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

import pandas as pd
from datasets import Dataset, load_dataset

PUBMED_BASE_URL = (
    "https://github.com/Franck-Dernoncourt/pubmed-rct/raw/refs/heads/master/PubMed_200k_RCT"
)
SCALEDOC_QUERY_URL = (
    "https://raw.githubusercontent.com/Seurgul/ScaleDoc/refs/heads/main/dataset/query.json"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scaledoc"
DEFAULT_QUERY_FILE = DEFAULT_DATA_DIR / "query.json"
BARGAIN_DATASET_HANDLES = {
    "screenplay": "gufukuro/movie-scripts-corpus",
    "review": "najzeko/steam-reviews-2021",
}


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


def _download_kaggle_dataset(
    dataset_name: str,
    cache_dir: str | Path | None,
) -> Path:
    """Downloads a Kaggle dataset into a caller-controlled cache directory."""
    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "Loading this dataset requires kagglehub. Install the 'experiments' "
            "extra first."
        ) from error

    if cache_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = str(Path(cache_dir).resolve())
    return Path(kagglehub.dataset_download(BARGAIN_DATASET_HANDLES[dataset_name]))


def load_screenplay_documents(
    n: int = 10000,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Loads one movie screenplay per row from Kaggle's movie scripts corpus."""
    dataset_dir = _download_kaggle_dataset("screenplay", cache_dir)
    screenplay_dir = (
        dataset_dir / "screenplay_data" / "data" / "raw_texts" / "raw_texts"
    )
    if not screenplay_dir.is_dir():
        raise FileNotFoundError(
            "Screenplay text directory not found at "
            f"'{screenplay_dir}'. The Kaggle dataset layout may have changed."
        )

    files = sorted(screenplay_dir.glob("*.txt"))[:n]
    documents = [file.read_text(encoding="utf-8", errors="replace") for file in files]
    return Dataset.from_dict({
        "id": list(range(len(documents))),
        "content": documents,
    })


def load_review_documents(
    n: int = 10000,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Loads Steam review text from Kaggle's Steam Reviews 2021 dataset."""
    import csv

    dataset_dir = _download_kaggle_dataset("review", cache_dir)
    reviews_file = dataset_dir / "steam_reviews.csv"
    if not reviews_file.is_file():
        raise FileNotFoundError(
            "Steam review CSV not found at "
            f"'{reviews_file}'. The Kaggle dataset layout may have changed."
        )

    documents: list[str] = []
    with reviews_file.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "review" not in reader.fieldnames:
            raise ValueError(f"Expected a 'review' column in '{reviews_file}'.")
        for row in reader:
            review = row["review"]
            if review:
                documents.append(review)
            if len(documents) >= n:
                break

    return Dataset.from_dict({
        "id": list(range(len(documents))),
        "content": documents,
    })


def load_wiki_documents(
    n: int = 10000,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Loads one Wikipedia Talk-page conversation per row from ConvoKit."""
    try:
        from convokit import Corpus, download
    except ImportError as error:
        raise ImportError(
            "Loading the wiki dataset requires convokit. Install the "
            "'experiments' extra first."
        ) from error

    if cache_dir is None:
        corpus_path = download("wiki-corpus")
    else:
        corpus_path = download("wiki-corpus", data_dir=str(cache_dir))
    corpus = Corpus(filename=corpus_path)
    documents: list[str] = []
    for conversation in corpus.iter_conversations():
        text = "\n".join(
            utterance.text
            for utterance in conversation.iter_utterances()
            if utterance.text
        )
        if text:
            documents.append(text)
        if len(documents) >= n:
            break

    return Dataset.from_dict({
        "id": list(range(len(documents))),
        "content": documents,
    })


def load_court_documents(
    path: str | Path,
    n: int = 10000,
    text_column: str = "opinion_text",
) -> Dataset:
    """Loads Supreme Court opinions from a local CourtListener-derived CSV file."""

    opinions_file = Path(path)
    if not opinions_file.is_file():
        raise FileNotFoundError(f"Court opinion CSV not found at '{opinions_file}'.")

    df = pd.read_csv(opinions_file, usecols=[text_column], nrows=n, encoding="utf-8")
    documents: list[str] = df[text_column].dropna().astype(str).tolist()

    return Dataset.from_dict({
        "id": list(range(len(documents))),
        "content": documents,
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


def load_queries(
    dataset_name: str,
    query_file: str | Path,
) -> list[dict[str, Any]]:
    """
    Loads query definitions for a given dataset from a JSON file or URL.

    Args:
        dataset_name: Name of dataset ('pubmed', 'big_patent', 'gov_report', etc.).
        query_file: Path or URL to a JSON query definition file.

    Returns:
        List of dicts with 'q_id' and 'query'.
    """
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
            raise FileNotFoundError(f"Query file not found at '{path}'.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

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
