import json
from pathlib import Path

from qwen3_mole.judge_systems import (
    aggregate_bidirectional,
    load_comparisons,
    map_decision,
    parse_decision,
    stable_reverse,
    select_manual_review,
    summarize_judgments,
)


def test_parse_and_map_decisions() -> None:
    assert parse_decision("A") == "A"
    assert parse_decision("Decision: tie") == "TIE"
    assert parse_decision("I refuse") == "INVALID"
    assert parse_decision("A is weaker, so B") == "B"
    assert map_decision("B", "routed", "base") == "base"
    assert map_decision("TIE", "routed", "base") == "tie"


def test_bidirectional_aggregation_is_conservative() -> None:
    assert aggregate_bidirectional("routed", "routed") == "routed"
    assert aggregate_bidirectional("tie", "tie") == "tie"
    assert aggregate_bidirectional("routed", "base") == "inconsistent"
    assert aggregate_bidirectional("invalid", "base") == "invalid"


def test_load_comparisons_blinds_all_three_controls(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    with path.open("w") as stream:
        for condition in ("base", "oracle", "routed", "wrong"):
            stream.write(
                json.dumps(
                    {
                        "id": "serverfault:7",
                        "domain": "systems",
                        "condition": condition,
                        "prompt": "How?",
                        "reference": "Reference",
                        "generated": f"<think>hidden</think>{condition} answer",
                    }
                )
                + "\n"
            )
    comparisons = load_comparisons(path, 42)
    assert [value["other"] for value in comparisons] == ["base", "oracle", "wrong"]
    assert all(len(value["orders"]) == 2 for value in comparisons)
    assert all("hidden" not in " ".join(value["answers"].values()) for value in comparisons)
    assert stable_reverse("serverfault:7", "base", 42) == stable_reverse("serverfault:7", "base", 42)


def test_summarize_judgments_reports_routed_preference() -> None:
    results = []
    for other in ("base", "oracle", "wrong"):
        results.extend(
            [
                {"other": other, "winner": "routed"},
                {"other": other, "winner": "routed"},
                {"other": other, "winner": other},
                {"other": other, "winner": "tie"},
            ]
        )
    summary = summarize_judgments(results, 42)
    assert summary["base"]["routed_preference_score"] == 0.625
    assert summary["wrong"]["outcomes"] == {"routed": 2, "tie": 1, "wrong": 1}


def test_manual_review_sample_covers_strata() -> None:
    results = [
        {"comparison_id": "a", "other": "base", "winner": "routed"},
        {"comparison_id": "b", "other": "base", "winner": "base"},
        {"comparison_id": "c", "other": "wrong", "winner": "tie"},
    ]
    data = {value["comparison_id"]: {"comparison_id": value["comparison_id"]} for value in results}
    sample = select_manual_review(results, data, target=3)
    assert {value["comparison_id"] for value in sample} == {"a", "b", "c"}
