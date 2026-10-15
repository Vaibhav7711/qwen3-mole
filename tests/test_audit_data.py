from qwen3_mole.audit_data import digest, exact_overlap, normalize_text


def record(prompt: str, answer: str = "answer") -> dict:
    return {
        "domain": "code",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def test_normalize_text_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_text("  HéLLo,\nWORLD! ") == "héllo world"
    assert digest("Hello-world") == digest("hello   world")


def test_exact_overlap_reports_query_matches() -> None:
    result = exact_overlap(
        [record("Write a parser."), record("Sort values")],
        [record("write a PARSER!"), record("Different")],
        lambda value: value["messages"][0]["content"],
    )
    assert result["matches"] == 1
    assert result["details"] == [{"query_index": 0, "reference_indices": [0]}]
