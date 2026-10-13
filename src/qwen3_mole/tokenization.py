from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def input_ids_from_template(rendered: Any) -> list[int]:
    """Normalize chat-template output across Transformers versions."""
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("expected one tokenized conversation")
        rendered = rendered[0]
    return list(rendered)


def tokenize_chat_with_assistant_labels(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> dict[str, list[int]]:
    """Tokenize a chat and compute loss only on assistant turns."""
    input_ids: list[int] = []
    labels: list[int] = []
    prefix: list[dict[str, str]] = []
    prior_tokens: list[int] = []

    for message in messages:
        prefix.append(message)
        current_tokens = input_ids_from_template(
            tokenizer.apply_chat_template(
                prefix,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if current_tokens[: len(prior_tokens)] != prior_tokens:
            raise ValueError("chat template is not prefix-stable for assistant masking")
        new_tokens = list(current_tokens[len(prior_tokens) :])
        input_ids.extend(new_tokens)
        if message["role"] == "assistant":
            labels.extend(new_tokens)
        else:
            labels.extend([-100] * len(new_tokens))
        prior_tokens = current_tokens

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError("example has no assistant tokens inside max_length")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class CausalLMCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
