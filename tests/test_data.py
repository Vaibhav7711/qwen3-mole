import pytest

from qwen3_mole.data import prompt_text, stratified_split, validate_record


def record(domain: str, index: int) -> dict:
    return {
        "domain": domain,
        "messages": [
            {"role": "user", "content": f"question {domain} {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ],
    }


def test_prompt_does_not_leak_assistant_answer() -> None:
    value = record("code", 1)
    text = prompt_text(value)
    assert "question code 1" in text
    assert "answer 1" not in text


def test_stratified_split_keeps_every_domain() -> None:
    values = [record(domain, index) for domain in ("code", "math") for index in range(10)]
    train, validation = stratified_split(values, validation_fraction=0.2, seed=7)
    assert len(train) == 16
    assert len(validation) == 4
    assert {row["domain"] for row in validation} == {"code", "math"}


def test_validation_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        validate_record(record("other", 1), domains=("code", "math"))
