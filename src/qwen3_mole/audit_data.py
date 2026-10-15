from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .data import prompt_text, read_jsonl


WHITESPACE = re.compile(r"\s+")
NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = NON_WORD.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def full_text(record: dict[str, Any]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in record["messages"])


def digest(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode()).hexdigest()


def exact_overlap(
    reference: list[dict[str, Any]], queries: list[dict[str, Any]], extractor: Any
) -> dict[str, Any]:
    reference_hashes: dict[str, list[int]] = {}
    for index, record in enumerate(reference):
        reference_hashes.setdefault(digest(extractor(record)), []).append(index)
    matches = []
    for index, record in enumerate(queries):
        key = digest(extractor(record))
        if key in reference_hashes:
            matches.append({"query_index": index, "reference_indices": reference_hashes[key]})
    return {"query_examples": len(queries), "matches": len(matches), "details": matches[:100]}


def nearest_matches(
    reference_texts: list[str],
    query_texts: list[str],
    reference_labels: list[str] | None = None,
    query_labels: list[str] | None = None,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    reference_matrix = vectorizer.fit_transform(reference_texts)
    query_matrix = vectorizer.transform(query_texts)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(reference_matrix)
    distances, indices = neighbors.kneighbors(query_matrix)
    similarities = 1.0 - distances[:, 0]
    order = np.argsort(-similarities)
    top = []
    for query_index in order[:50]:
        reference_index = int(indices[query_index, 0])
        top.append(
            {
                "similarity": float(similarities[query_index]),
                "query_index": int(query_index),
                "reference_index": reference_index,
                "query_label": query_labels[query_index] if query_labels else None,
                "reference_label": reference_labels[reference_index] if reference_labels else None,
                "query": query_texts[query_index][:300],
                "reference": reference_texts[reference_index][:300],
            }
        )
    return {
        "reference_examples": len(reference_texts),
        "query_examples": len(query_texts),
        "threshold_counts": {
            "0.90": int((similarities >= 0.90).sum()),
            "0.95": int((similarities >= 0.95).sum()),
            "0.98": int((similarities >= 0.98).sum()),
            "1.00": int((similarities >= 0.999999).sum()),
        },
        "maximum_similarity": float(similarities.max()),
        "mean_nearest_similarity": float(similarities.mean()),
        "top_matches": top,
    }


def load_benchmark_prompts() -> tuple[list[str], list[str], list[dict[str, str]]]:
    from datasets import load_dataset

    specifications = [
        ("humaneval", "openai/openai_humaneval", None, "test", "prompt"),
        ("mbpp", "google-research-datasets/mbpp", "sanitized", "test", "prompt"),
        ("gsm8k", "openai/gsm8k", "main", "test", "question"),
    ]
    prompts: list[str] = []
    labels: list[str] = []
    errors: list[dict[str, str]] = []
    for label, repository, config, split, field in specifications:
        try:
            dataset = load_dataset(repository, config, split=split) if config else load_dataset(repository, split=split)
            for row in dataset:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    prompts.append(value)
                    labels.append(label)
        except Exception as error:
            errors.append({"benchmark": label, "error": f"{type(error).__name__}: {error}"})
    return prompts, labels, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split leakage and common benchmark overlap")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = list(read_jsonl(args.train))
    validation = list(read_jsonl(args.validation))
    test = list(read_jsonl(args.test))
    train_prompts = [normalize_text(prompt_text(record)) for record in train]
    validation_prompts = [normalize_text(prompt_text(record)) for record in validation]
    test_prompts = [normalize_text(prompt_text(record)) for record in test]
    train_domains = [record["domain"] for record in train]
    validation_domains = [record["domain"] for record in validation]
    test_domains = [record["domain"] for record in test]

    benchmark_prompts, benchmark_labels, benchmark_errors = load_benchmark_prompts()
    report = {
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "exact_overlap": {
            "validation_prompt_vs_train": exact_overlap(train, validation, prompt_text),
            "test_prompt_vs_train": exact_overlap(train, test, prompt_text),
            "validation_full_vs_train": exact_overlap(train, validation, full_text),
            "test_full_vs_train": exact_overlap(train, test, full_text),
        },
        "near_overlap": {
            "validation_prompt_vs_train": nearest_matches(
                train_prompts, validation_prompts, train_domains, validation_domains
            ),
            "test_prompt_vs_train": nearest_matches(train_prompts, test_prompts, train_domains, test_domains),
        },
        "common_benchmark_overlap": nearest_matches(
            [normalize_text(value) for value in benchmark_prompts],
            train_prompts,
            benchmark_labels,
            train_domains,
        )
        if benchmark_prompts
        else None,
        "benchmark_load_errors": benchmark_errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    headline = {
        "counts": report["counts"],
        "exact_overlap": {key: value["matches"] for key, value in report["exact_overlap"].items()},
        "near_overlap": {
            key: value["threshold_counts"] for key, value in report["near_overlap"].items()
        },
        "common_benchmark_overlap": report["common_benchmark_overlap"]["threshold_counts"]
        if report["common_benchmark_overlap"]
        else None,
        "benchmark_load_errors": benchmark_errors,
    }
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
