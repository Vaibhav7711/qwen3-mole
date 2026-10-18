from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .score_semantic import paired_bootstrap, strip_thinking


OTHER_CONDITIONS = ("base", "oracle", "wrong")
DECISION = re.compile(r"\b(TIE|A|B)\b", flags=re.IGNORECASE)
RUBRIC = """You are an impartial evaluator of answers to systems-administration questions.
Use the reference answer as evidence, but correct alternatives are allowed. Prefer technical
correctness, operational safety, relevance, and completeness. Ignore verbosity and superficial
style. Choose the better response. Output exactly A, B, or TIE and nothing else."""


def stable_reverse(identifier: str, other: str, seed: int) -> bool:
    value = hashlib.sha256(f"{seed}:{identifier}:{other}".encode()).digest()
    return bool(value[0] & 1)


def truncate_tokens(tokenizer: Any, value: str, limit: int) -> str:
    ids = tokenizer.encode(value, add_special_tokens=False)
    return value if len(ids) <= limit else tokenizer.decode(ids[:limit], skip_special_tokens=True)


def judge_prompt(
    tokenizer: Any,
    question: str,
    reference: str,
    answer_a: str,
    answer_b: str,
) -> str:
    user = "\n\n".join(
        [
            "EVALUATION INSTRUCTIONS:\n" + RUBRIC,
            "QUESTION:\n" + truncate_tokens(tokenizer, question, 1200),
            "REFERENCE:\n" + truncate_tokens(tokenizer, reference, 1600),
            "RESPONSE A:\n" + truncate_tokens(tokenizer, answer_a, 1800),
            "RESPONSE B:\n" + truncate_tokens(tokenizer, answer_b, 1800),
            "DECISION (A, B, or TIE):",
        ]
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_decision(value: str) -> str:
    matches = DECISION.findall(value.strip())
    return matches[-1].upper() if matches else "INVALID"


def map_decision(decision: str, answer_a: str, answer_b: str) -> str:
    if decision == "A":
        return answer_a
    if decision == "B":
        return answer_b
    return decision.casefold()


def aggregate_bidirectional(first: str, second: str) -> str:
    if first == second:
        return first
    if "invalid" in {first, second}:
        return "invalid"
    return "inconsistent"


def load_comparisons(path: str | Path, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["domain"] == "systems":
                grouped[record["id"]][record["condition"]] = record
    comparisons = []
    for identifier, conditions in sorted(grouped.items()):
        if set(conditions) != {"base", "oracle", "routed", "wrong"}:
            raise ValueError(f"{identifier} is missing systems conditions: {sorted(conditions)}")
        routed = conditions["routed"]
        for other in OTHER_CONDITIONS:
            other_record = conditions[other]
            reverse = stable_reverse(identifier, other, seed)
            order = [other, "routed"] if reverse else ["routed", other]
            answers = {
                "routed": strip_thinking(routed["generated"]),
                other: strip_thinking(other_record["generated"]),
            }
            comparisons.append(
                {
                    "comparison_id": f"{identifier}:routed-vs-{other}",
                    "id": identifier,
                    "other": other,
                    "question": routed["prompt"],
                    "reference": routed["reference"],
                    "answers": answers,
                    "orders": [order, list(reversed(order))],
                }
            )
    return comparisons


def summarize_judgments(results: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for other in OTHER_CONDITIONS:
        values = [result for result in results if result["other"] == other]
        counts: dict[str, int] = defaultdict(int)
        preferences = []
        for result in values:
            winner = result["winner"]
            counts[winner] += 1
            preferences.append(1.0 if winner == "routed" else 0.0 if winner == other else 0.5)
        interval = paired_bootstrap(
            preferences,
            [0.5] * len(preferences),
            seed=seed,
        )
        summary[other] = {
            "comparisons": len(values),
            "outcomes": dict(sorted(counts.items())),
            "routed_preference_score": sum(preferences) / len(preferences),
            "routed_preference_minus_chance": interval,
        }
    return summary


def select_manual_review(
    results: list[dict[str, Any]],
    comparison_data: dict[str, dict[str, Any]],
    target: int = 15,
) -> list[dict[str, Any]]:
    """Deterministically sample across control and judge-outcome strata."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        rank = hashlib.sha256(f"review:{result['comparison_id']}".encode()).hexdigest()
        groups[(result["other"], result["winner"])].append(
            {
                "review_rank": rank,
                **comparison_data[result["comparison_id"]],
                "judgment": result,
            }
        )
    for values in groups.values():
        values.sort(key=lambda value: value["review_rank"])
    sample = []
    group_keys = sorted(groups)
    while len(sample) < target:
        added = False
        for key in group_keys:
            if groups[key] and len(sample) < target:
                sample.append(groups[key].pop(0))
                added = True
        if not added:
            break
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind bidirectional systems-answer judging")
    parser.add_argument("--config", default="configs/semantic_eval.yaml")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be within num shards")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    config = yaml.safe_load(Path(args.config).read_text())
    judge = config["systems_judge"]
    tokenizer = AutoTokenizer.from_pretrained(judge["model"], revision=judge["revision"])
    all_comparisons = load_comparisons(args.generations, int(config["seed"]))
    comparisons = [
        comparison
        for index, comparison in enumerate(all_comparisons)
        if index % args.num_shards == args.shard_index
    ]
    prompts = []
    metadata = []
    for comparison in comparisons:
        for pass_index, order in enumerate(comparison["orders"]):
            prompts.append(
                judge_prompt(
                    tokenizer,
                    comparison["question"],
                    comparison["reference"],
                    comparison["answers"][order[0]],
                    comparison["answers"][order[1]],
                )
            )
            metadata.append((comparison, pass_index, order))
    llm = LLM(
        model=judge["model"],
        revision=judge["revision"],
        quantization=judge["quantization"],
        dtype="half",
        tensor_parallel_size=1,
        max_model_len=int(judge["max_model_len"]),
        max_num_seqs=int(judge["max_num_seqs"]),
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        seed=int(config["seed"]),
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=int(judge["max_new_tokens"]), seed=int(config["seed"])),
        use_tqdm=True,
    )
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (comparison, pass_index, order), output in zip(metadata, outputs):
        text = output.outputs[0].text
        decision = parse_decision(text)
        raw[comparison["comparison_id"]].append(
            {
                "pass": pass_index,
                "order": order,
                "raw": text,
                "decision": decision,
                "mapped_winner": map_decision(decision, order[0], order[1]),
            }
        )
    results = []
    by_id = {comparison["comparison_id"]: comparison for comparison in comparisons}
    for comparison_id in sorted(raw):
        passes = sorted(raw[comparison_id], key=lambda value: value["pass"])
        winner = aggregate_bidirectional(passes[0]["mapped_winner"], passes[1]["mapped_winner"])
        comparison = by_id[comparison_id]
        results.append(
            {
                "comparison_id": comparison_id,
                "id": comparison["id"],
                "other": comparison["other"],
                "winner": winner,
                "passes": passes,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"comparisons": results}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"shard": args.shard_index, "comparisons": len(results), "judge_calls": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
