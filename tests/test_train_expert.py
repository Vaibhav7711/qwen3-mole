from qwen3_mole.train_expert import compute_warmup_steps


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
