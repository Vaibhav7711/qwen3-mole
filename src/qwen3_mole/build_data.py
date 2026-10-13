from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from .data import validate_record, write_jsonl
from .tokenization import input_ids_from_template


WHITESPACE = re.compile(r"\s+")


def normalized_text(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def content_to_text(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.strip() or None
        if decoded != value:
            return content_to_text(decoded)
        return value.strip() or None
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = content_to_text(item)
            if not text:
                continue
            label = str(key).replace("_", " ").strip().title()
            if key == "code_block":
                parts.append(f"Code:\n```\n{text}\n```")
            else:
                parts.append(f"{label}: {text}")
        return "\n\n".join(parts) or None
    if isinstance(value, list):
        parts = [text for item in value if (text := content_to_text(item))]
        return "\n".join(parts) or None
    if value is None:
        return None
    return str(value)


def first_exchange(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]] | None:
    system: dict[str, str] | None = None
    user: dict[str, str] | None = None
    for raw in messages:
        role = raw.get("role")
        content = content_to_text(raw.get("content"))
        if not content:
            continue
        message = {"role": str(role), "content": content}
        if role == "system" and system is None and user is None:
            system = message
        elif role == "user" and user is None:
            user = message
        elif role == "assistant" and user is not None:
            result = [user, message]
            if system is not None:
                result.insert(0, system)
            return result
    return None


def adapt_row(row: dict[str, Any], adapter: str) -> list[dict[str, str]] | None:
    if adapter == "instruction_response":
        instruction = row.get("instruction")
        response = row.get("response")
        if not isinstance(instruction, str) or not isinstance(response, str):
            return None
        return first_exchange(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        )
    if adapter == "messages":
        messages = row.get("messages")
        return first_exchange(messages) if isinstance(messages, list) else None
    raise ValueError(f"unknown source adapter: {adapter}")


def passes_quality(messages: list[dict[str, str]], quality: dict[str, Any]) -> bool:
    user = next(message["content"] for message in messages if message["role"] == "user")
    assistant = next(message["content"] for message in messages if message["role"] == "assistant")
    return (
        int(quality["min_prompt_chars"]) <= len(user) <= int(quality["max_prompt_chars"])
        and int(quality["min_response_chars"])
        <= len(assistant)
        <= int(quality["max_response_chars"])
    )


def content_fingerprint(messages: list[dict[str, str]]) -> str:
    text = "\n".join(
        f"{message['role']}:{normalized_text(message['content']).casefold()}" for message in messages
    )
    return hashlib.sha256(text.encode()).hexdigest()


def load_source(
    source: dict[str, Any],
    seed: int,
    shuffle_buffer: int,
    quality: dict[str, Any],
    limit_override: int | None,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    try:
        from datasets import load_dataset
        import pyarrow as pa
    except ImportError as exc:
        raise SystemExit("datasets is required; install requirements-kaggle.txt") from exc
    pa.set_io_thread_count(1)

    requested = int(limit_override or source["max_records"])
    if source.get("loader") == "parquet_pyarrow":
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            source["repository"],
            source["data_file"],
            repo_type="dataset",
            revision=source["conversion_revision"],
        )
        rows = pq.read_table(path).to_pylist()
        random.Random(seed).shuffle(rows)
        dataset = rows
    elif "loader" in source:
        dataset = load_dataset(
            source["loader"],
            data_files=source["data_files"],
            split=source["split"],
            streaming=True,
        )
    else:
        dataset = load_dataset(
            source["repository"],
            split=source["split"],
            revision=source["revision"],
            streaming=True,
        )
    if hasattr(dataset, "shuffle") and not isinstance(dataset, list):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    accepted: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    iterator = iter(dataset)
    try:
        for index, row in enumerate(iterator):
            stats["scanned"] += 1
            messages = adapt_row(row, source["adapter"])
            if messages is None:
                stats["invalid_schema"] += 1
                continue
            if not passes_quality(messages, quality):
                stats["quality_filtered"] += 1
                continue
            token_ids = input_ids_from_template(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            if len(token_ids) > int(quality["max_tokens"]):
                stats["token_filtered"] += 1
                continue
            fingerprint = content_fingerprint(messages)
            if fingerprint in seen:
                stats["duplicate"] += 1
                continue
            seen.add(fingerprint)
            record = {
                "domain": source["domain"],
                "messages": messages,
                "provenance": {
                    "source": source["repository"],
                    "revision": source["revision"],
                    "license": source["license"],
                    "stream_index": index,
                    "content_sha256": fingerprint,
                },
            }
            validate_record(record)
            accepted.append(record)
            stats["accepted"] += 1
            if len(accepted) >= requested:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator
        gc.collect()
    del dataset
    gc.collect()
    if len(accepted) < requested:
        raise RuntimeError(f"{source['name']} yielded {len(accepted)} of {requested} requested records")
    return accepted, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pinned, normalized expert training data")
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--output", default="data/raw/curated.jsonl")
    parser.add_argument("--manifest", default="data/raw/curated.manifest.json")
    parser.add_argument("--per-source", type=int, help="override each source size for pilot builds")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.sources).read_text())
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required; install requirements-kaggle.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(config["quality"]["tokenizer"], use_fast=True)
    all_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"seed": config["seed"], "sources": []}
    for source_index, source in enumerate(config["sources"]):
        records, stats = load_source(
            source,
            seed=int(config["seed"]) + source_index,
            shuffle_buffer=int(config["shuffle_buffer"]),
            quality=config["quality"],
            limit_override=args.per_source,
            tokenizer=tokenizer,
        )
        all_records.extend(records)
        manifest["sources"].append({**source, "stats": dict(stats)})
        print(source["name"], dict(stats))

    total = write_jsonl(args.output, all_records)
    manifest["total_records"] = total
    manifest["domain_counts"] = dict(sorted(Counter(row["domain"] for row in all_records).items()))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote", total, "records to", args.output)


if __name__ == "__main__":
    main()
