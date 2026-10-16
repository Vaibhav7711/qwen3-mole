from __future__ import annotations

from qwen3_mole.build_eval import (
    adapt_row,
    build_suite,
    canonical_json,
    comparison_prompt,
    html_to_text,
    stable_rank,
)


def train_record(domain: str, prompt: str) -> dict:
    return {
        "domain": domain,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "A sufficiently long training response."},
        ],
    }


def eval_record(identifier: str, domain: str, prompt: str) -> dict:
    record = {
        "id": identifier,
        "domain": domain,
        "prompt": prompt,
        "reference": "A reference answer.",
        "task": {},
        "evaluation_method": "test",
        "provenance": {},
        "content_sha256": identifier.rsplit(":", 1)[-1].zfill(64),
    }
    return record


def test_html_to_text_preserves_blocks_and_code() -> None:
    value = html_to_text("<p>Run this:</p><pre><code>sudo true</code></pre><p>Done.</p>")
    assert "Run this:" in value
    assert "sudo true" in value
    assert "Done." in value
    assert "<p>" not in value


def test_stackexchange_adapter_selects_accepted_answer() -> None:
    source = {
        "name": "serverfault",
        "domain": "systems",
        "kind": "stackexchange",
        "repository": "example/repo",
        "revision": "abc",
        "license": "cc-by-sa-4.0",
        "homepage": "https://serverfault.com/",
        "evaluation_method": "blinded_judge",
    }
    row = {
        "qid": 7,
        "question": "<p>How do I restart this service safely?</p>",
        "answers": [
            {"answer_id": 1, "text": "<p>Popular answer.</p>", "pm_score": 20, "selected": False, "author": "A"},
            {"answer_id": 2, "text": "<p>Accepted answer.</p>", "pm_score": 2, "selected": True, "author": "B"},
        ],
        "metadata": ["https://serverfault.com/questions/7/example"],
    }
    record = adapt_row(row, source)
    assert record is not None
    assert record["reference"] == "Accepted answer."
    assert record["task"]["answer_id"] == 2
    assert record["provenance"]["source_url"].endswith("/example")


def test_humaneval_compares_unwrapped_source_prompt() -> None:
    source = {
        "name": "humaneval",
        "domain": "code",
        "kind": "humaneval",
        "repository": "openai/openai_humaneval",
        "revision": "abc",
        "license": "mit",
        "homepage": "https://example.com",
        "evaluation_method": "execution",
    }
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def identity(value):\n    pass\n",
        "canonical_solution": "    return value\n",
        "test": "def check(candidate): assert candidate(1) == 1",
        "entry_point": "identity",
    }
    record = adapt_row(row, source)
    assert record is not None
    assert record["prompt"].startswith("Complete the following")
    assert comparison_prompt(record) == row["prompt"]


def test_build_suite_rejects_exact_and_near_overlap_deterministically() -> None:
    config = {
        "seed": 9,
        "target_per_domain": 1,
        "candidate_pool_multiplier": 10,
        "domains": {"code": {"max_train_similarity": 0.70}},
        "quality": {
            "min_prompt_chars": 5,
            "max_prompt_chars": 1000,
            "min_reference_chars": 1,
            "max_reference_chars": 1000,
        },
    }
    train = [train_record("code", "Write a function that sorts a list of integers ascending.")]
    candidates = [
        eval_record("x:1", "code", "Write a function that sorts a list of integers ascending!"),
        eval_record("x:2", "code", "Explain why the sky appears blue during daytime."),
        eval_record("x:3", "code", "Design a parser for a small arithmetic expression grammar."),
    ]
    selected, report = build_suite(train, candidates, config)
    assert len(selected) == 1
    assert selected[0]["id"] in {"x:2", "x:3"}
    assert report["stats"]["code"]["exact_train_overlap"] == 1
    assert report["stats"]["code"]["selected"] == 1
    expected = min(candidates[1:], key=lambda record: stable_rank(record, 9))["id"]
    assert selected[0]["id"] == expected


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
