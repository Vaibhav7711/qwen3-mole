from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def agreement_metrics(judge: list[str], human: list[str]) -> dict[str, Any]:
    if len(judge) != len(human) or not judge:
        raise ValueError("judge and human labels must have the same non-zero length")
    labels = sorted(set(judge) | set(human))
    observed = sum(left == right for left, right in zip(judge, human)) / len(judge)
    judge_counts = Counter(judge)
    human_counts = Counter(human)
    expected = sum(
        (judge_counts[label] / len(judge)) * (human_counts[label] / len(human))
        for label in labels
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "examples": len(judge),
        "labels": labels,
        "exact_agreement": observed,
        "cohen_kappa": kappa,
        "judge_counts": dict(sorted(judge_counts.items())),
        "human_counts": dict(sorted(human_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure judge agreement on annotated systems review data")
    parser.add_argument("--review", required=True, help="manual-review JSON with human_winner fields")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.review).read_text())
    missing = [record["comparison_id"] for record in records if not record.get("human_winner")]
    if missing:
        raise SystemExit(f"manual review is incomplete for {len(missing)} items: {missing[:5]}")
    valid = [
        record
        for record in records
        if record["judgment"]["winner"] not in {"invalid", "inconsistent"}
    ]
    result = agreement_metrics(
        [record["judgment"]["winner"] for record in valid],
        [record["human_winner"] for record in valid],
    )
    result["excluded_invalid_or_inconsistent"] = len(records) - len(valid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
