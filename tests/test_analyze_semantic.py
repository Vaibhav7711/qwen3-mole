from qwen3_mole.analyze_semantic import analyze


def test_analyze_combines_task_and_systems_pairs() -> None:
    task_records = []
    generations = []
    for domain in ("code", "math", "instruction"):
        for condition, score in (("base", 0), ("oracle", 1), ("routed", 1), ("wrong", 0)):
            task_records.append(
                {"id": f"{domain}:1", "domain": domain, "condition": condition, "score": score}
            )
            generations.append(
                {
                    "id": f"{domain}:1",
                    "domain": domain,
                    "condition": condition,
                    "route": {"expert": domain},
                    "generated_tokens": 10,
                    "finish_reason": "stop",
                }
            )
    for condition in ("base", "oracle", "routed", "wrong"):
        generations.append(
            {
                "id": "systems:1",
                "domain": "systems",
                "condition": condition,
                "route": {"expert": "systems"},
                "generated_tokens": 10,
                "finish_reason": "stop",
            }
        )
    systems = {
        "summary": {},
        "comparisons": [
            {"other": "base", "winner": "routed"},
            {"other": "oracle", "winner": "tie"},
            {"other": "wrong", "winner": "routed"},
        ],
    }
    thresholds = {
        "base_ci_lower": -0.01,
        "wrong_ci_lower": -0.01,
        "oracle_delta_floor": -0.05,
        "domain_delta_floor": -0.05,
        "judge_kappa_min": 0.6,
    }
    result = analyze(
        {"records": task_records, "accuracy": {}},
        systems,
        generations,
        {"cohen_kappa": 0.8},
        thresholds,
        bootstrap_samples=100,
    )
    assert result["paired_comparisons"]["routed_minus_base:overall"]["mean_delta"] == 1.0
    assert result["routing"]["accuracy"] == 1.0
    assert result["stage2_passed"] is True
