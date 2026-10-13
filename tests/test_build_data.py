from qwen3_mole.build_data import (
    adapt_row,
    content_fingerprint,
    content_to_text,
    first_exchange,
    passes_quality,
)


def test_first_exchange_keeps_optional_system_and_stops_after_answer():
    messages = first_exchange(
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    )
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == "First answer"


def test_instruction_response_adapter():
    messages = adapt_row({"instruction": "Do the task", "response": "Here is the result"}, "instruction_response")
    assert messages == [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": "Here is the result"},
    ]


def test_quality_and_fingerprint_normalize_whitespace_and_case():
    one = [
        {"role": "user", "content": "A sufficiently long prompt"},
        {"role": "assistant", "content": "A sufficiently long answer"},
    ]
    two = [
        {"role": "user", "content": "a  sufficiently LONG prompt"},
        {"role": "assistant", "content": "a sufficiently long ANSWER"},
    ]
    limits = {
        "min_prompt_chars": 10,
        "min_response_chars": 10,
        "max_prompt_chars": 100,
        "max_response_chars": 100,
    }
    assert passes_quality(one, limits)
    assert content_fingerprint(one) == content_fingerprint(two)


def test_json_content_is_normalized_to_training_text():
    assert content_to_text('"plain text"') == "plain text"
    structured = content_to_text(
        '{"description":"Restart the service.","code_block":"systemctl restart api"}'
    )
    assert "Description: Restart the service." in structured
    assert "```\nsystemctl restart api\n```" in structured
