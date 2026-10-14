import pytest

import qwen3_mole.train_expert as train_expert
from qwen3_mole.train_expert import build_token_rows, compute_warmup_steps


def test_warmup_steps_for_full_epoch_run():
    assert compute_warmup_steps(
        num_examples=10_000,
        batch_size=1,
        gradient_accumulation_steps=16,
        epochs=2.0,
        warmup_ratio=0.03,
        max_steps=None,
    ) == 38


def test_warmup_steps_respects_smoke_cap():
    assert compute_warmup_steps(
        num_examples=10_000,
        batch_size=1,
        gradient_accumulation_steps=16,
        epochs=2.0,
        warmup_ratio=0.1,
        max_steps=100,
    ) == 10


def test_build_token_rows_skips_only_answers_beyond_limit(monkeypatch):
    def fake_tokenize(tokenizer, messages, max_length):
        if messages[0]["content"] == "too long":
            raise ValueError(train_expert.NO_ASSISTANT_TOKENS)
        return {"input_ids": [1], "attention_mask": [1], "labels": [1]}

    monkeypatch.setattr(train_expert, "tokenize_chat_with_assistant_labels", fake_tokenize)
    records = [
        {"messages": [{"role": "user", "content": "too long"}]},
        {"messages": [{"role": "user", "content": "usable"}]},
    ]

    rows, skipped = build_token_rows(records, object(), 512)

    assert skipped == 1
    assert rows == [{"input_ids": [1], "attention_mask": [1], "labels": [1]}]


def test_build_token_rows_preserves_unexpected_tokenization_errors(monkeypatch):
    def fake_tokenize(tokenizer, messages, max_length):
        raise ValueError("chat template is not prefix-stable for assistant masking")

    monkeypatch.setattr(train_expert, "tokenize_chat_with_assistant_labels", fake_tokenize)

    with pytest.raises(ValueError, match="prefix-stable"):
        build_token_rows([{"messages": []}], object(), 512)
