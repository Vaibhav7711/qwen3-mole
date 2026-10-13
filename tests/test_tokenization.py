from qwen3_mole.tokenization import (
    CausalLMCollator,
    input_ids_from_template,
    tokenize_chat_with_assistant_labels,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and not add_generation_prompt
        tokens = []
        for message in messages:
            role_token = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            tokens.extend([role_token, len(message["content"]), 2])
        return tokens


class FakeBatchEncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "input_ids": super().apply_chat_template(messages, tokenize, add_generation_prompt),
            "attention_mask": [1] * (3 * len(messages)),
        }


def test_only_assistant_delta_is_labeled():
    row = tokenize_chat_with_assistant_labels(
        FakeTokenizer(),
        [
            {"role": "system", "content": "Be useful"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        max_length=32,
    )
    assert row["labels"] == [-100, -100, -100, -100, -100, -100, 30, 6, 2]
    assert row["attention_mask"] == [1] * 9


def test_transformers_batch_encoding_output_is_supported():
    row = tokenize_chat_with_assistant_labels(
        FakeBatchEncodingTokenizer(),
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        max_length=32,
    )
    assert row["labels"] == [-100, -100, -100, 30, 6, 2]


def test_template_output_normalization_handles_one_row_batch():
    assert input_ids_from_template({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_collator_pads_inputs_and_masks_padding():
    import torch

    collator = CausalLMCollator(FakeTokenizer())
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2], [3, -100]]))
