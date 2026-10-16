from qwen3_mole.score_semantic import (
    decode_livecodebench_tests,
    execute_humaneval,
    extract_code,
    extract_final_answer,
    normalize_numeric,
    paired_bootstrap,
    score_gsm8k,
    strip_thinking,
    summarize_livecodebench_result,
)


class FakeIFEval:
    class InputExample:
        def __init__(self, **kwargs):
            self.kwargs = kwargs["kwargs"]

    @staticmethod
    def test_instruction_following_strict(value, prompt_to_response):
        assert value.kwargs == [{"num_words": 20}]
        assert prompt_to_response == {"Prompt": "Answer"}
        return type("Output", (), {"follow_all_instructions": True, "follow_instruction_list": [True]})()


def test_strip_thinking_and_extract_code() -> None:
    generated = "<think>private reasoning</think>\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "private" not in strip_thinking(generated)
    assert extract_code(generated, "def add(a, b):\n", "add").startswith("def add")


def test_execute_humaneval_pass_and_failure() -> None:
    record = {
        "prompt": "Complete the following Python function. Return only the completed code.\n\ndef add(a, b):\n",
        "task": {"entry_point": "add", "tests": "def check(candidate):\n    assert candidate(2, 3) == 5\n"},
        "generated": "    return a + b",
    }
    assert execute_humaneval(record)["passed"] is True
    record["generated"] = "    return a - b"
    assert execute_humaneval(record)["passed"] is False


def test_livecodebench_test_conversion_and_summary() -> None:
    task = {
        "public_test_cases": '[{"input":"1 2","output":"3","testtype":"stdin"}]',
        "private_test_cases": '[{"input":"4 5","output":"9","testtype":"stdin"}]',
        "metadata": "{}",
    }
    sample = decode_livecodebench_tests(task)
    values = __import__("json").loads(sample["input_output"])
    assert values == {"inputs": ["1 2", "4 5"], "outputs": ["3", "9"], "fn_name": None}
    assert summarize_livecodebench_result([True, True], {})["passed"] is True
    failed = summarize_livecodebench_result([True, -1], {"error_code": -1})
    assert failed["passed"] is False
    assert failed["tests_passed"] == 1


def test_gsm8k_answer_normalization() -> None:
    assert extract_final_answer("work here. Therefore \\boxed{1,200}.") == "1,200"
    assert normalize_numeric("18.0") == normalize_numeric("18")
    assert normalize_numeric("50%") == normalize_numeric("1/2")
    record = {"generated": "<think>2+3</think> The answer is 5.", "task": {"final_answer": "5"}}
    assert score_gsm8k(record)["passed"] is True


def test_paired_bootstrap_reports_direction() -> None:
    result = paired_bootstrap([1, 1, 1, 0], [0, 0, 1, 0], samples=1000)
    assert result["mean_delta"] == 0.5
    assert result["wins"] == 2
    assert result["losses"] == 0


def test_ifeval_removes_arrow_null_kwargs() -> None:
    from qwen3_mole.score_semantic import score_ifeval

    record = {
        "id": "ifeval:1000",
        "prompt": "Prompt",
        "generated": "Answer",
        "task": {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 20, "language": None}],
        },
    }
    assert score_ifeval(record, FakeIFEval)["passed"] is True
