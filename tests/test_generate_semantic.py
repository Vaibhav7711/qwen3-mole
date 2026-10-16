from pathlib import Path

import pytest

from qwen3_mole.generate_semantic import condition_expert, read_eval_jsonl, router_text


def test_condition_expert_mapping() -> None:
    wrong = {"code": "math"}
    assert condition_expert("base", "code", "systems", wrong) == "base"
    assert condition_expert("oracle", "code", "systems", wrong) == "code"
    assert condition_expert("routed", "code", "systems", wrong) == "systems"
    assert condition_expert("wrong", "code", "systems", wrong) == "math"


def test_router_text_matches_training_role_format() -> None:
    assert router_text("  hello  ") == "user: hello"


def test_read_eval_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    line = '{"id":"x","domain":"code","prompt":"hello"}\n'
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate id"):
        read_eval_jsonl(path)
