from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .score_semantic import paired_bootstrap, read_jsonl


VALID_SYSTEMS_WINNERS = {"routed", "base", "oracle", "wrong", "tie"}


def _preference_pair(winner: str, other: str) -> tuple[float, float] | None:
    if winner == "routed":
        return 1.0, 0.0
    if winner == other:
        return 0.0, 1.0
    if winner == "tie":
        return 0.5, 0.5
    return None


def analyze(
    task_scores: dict[str, Any],
    systems_judgments: dict[str, Any],
    generations: list[dict[str, Any]],
    agreement: dict[str, Any] | None,
    thresholds: dict[str, float],
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    by_task: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for record in task_scores["records"]:
        by_task[(record["domain"], record["id"])][record["condition"]] = float(record["score"])

    systems_by_other: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in systems_judgments["comparisons"]:
        systems_by_other[record["other"]].append(record)

    comparisons: dict[str, Any] = {}
    for other in ("base", "oracle", "wrong"):
        by_domain: dict[str, tuple[list[float], list[float]]] = {}
        combined_left: list[float] = []
        combined_right: list[float] = []
        for domain in ("code", "math", "instruction"):
            pairs = [
                (values["routed"], values[other])
                for (record_domain, _), values in sorted(by_task.items())
                if record_domain == domain and "routed" in values and other in values
            ]
            left = [pair[0] for pair in pairs]
            right = [pair[1] for pair in pairs]
            by_domain[domain] = (left, right)
            combined_left.extend(left)
            combined_right.extend(right)

        systems_pairs = [
            pair
            for record in systems_by_other[other]
            if (pair := _preference_pair(record["winner"], other)) is not None
        ]
        systems_left = [pair[0] for pair in systems_pairs]
        systems_right = [pair[1] for pair in systems_pairs]
        by_domain["systems"] = (systems_left, systems_right)
        combined_left.extend(systems_left)
        combined_right.extend(systems_right)

        for domain, (left, right) in by_domain.items():
            if left:
                comparisons[f"routed_minus_{other}:{domain}"] = paired_bootstrap(
                    left, right, samples=bootstrap_samples
                )
        comparisons[f"routed_minus_{other}:overall"] = paired_bootstrap(
            combined_left, combined_right, samples=bootstrap_samples
        )
        comparisons[f"routed_minus_{other}:overall"]["systems_excluded"] = (
            len(systems_by_other[other]) - len(systems_pairs)
        )

    unique_routes: dict[str, dict[str, Any]] = {}
    finish_reasons = Counter()
    token_totals: dict[str, list[int]] = defaultdict(list)
    for record in generations:
        unique_routes.setdefault(record["id"], record)
        finish_reasons[str(record.get("finish_reason"))] += 1
        token_totals[record["domain"]].append(int(record.get("generated_tokens", 0)))
    route_counts = Counter(record["route"]["expert"] for record in unique_routes.values())
    route_accuracy = sum(
        record["route"]["expert"] == record["domain"] for record in unique_routes.values()
    ) / len(unique_routes)

    base = comparisons["routed_minus_base:overall"]
    wrong = comparisons["routed_minus_wrong:overall"]
    oracle = comparisons["routed_minus_oracle:overall"]
    domain_base = [
        comparisons[f"routed_minus_base:{domain}"]["mean_delta"]
        for domain in ("code", "math", "systems", "instruction")
    ]
    gates = {
        "routed_beats_base_ci": base["ci95"][0] > thresholds["base_ci_lower"],
        "routed_beats_wrong_ci": wrong["ci95"][0] > thresholds["wrong_ci_lower"],
        "routed_close_to_oracle": oracle["mean_delta"] >= thresholds["oracle_delta_floor"],
        "no_material_domain_regression": min(domain_base) >= thresholds["domain_delta_floor"],
        "human_judge_agreement": bool(
            agreement
            and agreement.get("cohen_kappa", -1.0) >= thresholds["judge_kappa_min"]
        ),
    }
    return {
        "task_accuracy": task_scores["accuracy"],
        "systems_summary": systems_judgments.get("summary", {}),
        "paired_comparisons": comparisons,
        "routing": {
            "examples": len(unique_routes),
            "accuracy": route_accuracy,
            "expert_counts": dict(sorted(route_counts.items())),
        },
        "generation": {
            "records": len(generations),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "mean_tokens_by_domain": {
                domain: sum(values) / len(values) for domain, values in sorted(token_totals.items())
            },
        },
        "judge_agreement": agreement,
        "exit_thresholds": thresholds,
        "exit_gates": gates,
        "stage2_passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine final independent semantic benchmark results")
    parser.add_argument("--task-scores", required=True)
    parser.add_argument("--systems-judge", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--agreement")
    parser.add_argument("--config", default="configs/semantic_eval.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    agreement = json.loads(Path(args.agreement).read_text()) if args.agreement else None
    result = analyze(
        json.loads(Path(args.task_scores).read_text()),
        json.loads(Path(args.systems_judge).read_text()),
        read_jsonl(args.generations),
        agreement,
        {key: float(value) for key, value in config["stage2_exit"].items()},
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
