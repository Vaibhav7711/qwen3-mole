from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


VALID_ROLES = {"system", "user", "assistant"}


def validate_record(record: dict[str, Any], domains: Iterable[str] | None = None) -> None:
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    domain = record.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("record domain must be a non-empty string")
    if domains is not None and domain not in set(domains):
        raise ValueError(f"unknown domain: {domain}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        if message.get("role") not in VALID_ROLES:
            raise ValueError(f"message {index} has an invalid role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"message {index} content must be non-empty text")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("record must contain a user message")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("record must contain an assistant message")


def prompt_text(record: dict[str, Any]) -> str:
    """Return router input without leaking the assistant answer."""
    parts = []
    for message in record["messages"]:
        if message["role"] == "assistant":
            break
        parts.append(f"{message['role']}: {message['content'].strip()}")
    return "\n".join(parts)


def read_jsonl(path: str | Path, domains: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
    allowed = tuple(domains) if domains is not None else None
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, allowed)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield record


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def stratified_split(
    records: Iterable[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        validate_record(record)
        grouped[record["domain"]].append(record)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for domain in sorted(grouped):
        values = grouped[domain]
        rng.shuffle(values)
        n_validation = max(1, round(len(values) * validation_fraction))
        if n_validation >= len(values):
            raise ValueError(f"domain {domain!r} needs at least two records")
        validation.extend(values[:n_validation])
        train.extend(values[n_validation:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
