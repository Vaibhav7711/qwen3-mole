from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THINKING = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(?:python)?\s*\n?(.*?)```", flags=re.DOTALL | re.IGNORECASE)
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER = re.compile(r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?(?:/[0-9][0-9,]*)?%?")


def strip_thinking(value: str) -> str:
    return THINKING.sub("", value).strip()


def extract_code(generated: str, starter_code: str, entry_point: str) -> str:
    value = THINKING.sub("", generated).strip("\r\n")
    fenced = FENCE.findall(value)
    if fenced:
        value = next((block for block in fenced if f"def {entry_point}" in block), fenced[0])
    value = value.strip("\r\n")
    if f"def {entry_point}" in value:
        return value.strip()
    return starter_code.rstrip() + "\n" + value + "\n"


def _limit_child() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    except (ImportError, OSError, ValueError):
        pass


def execute_humaneval(record: dict[str, Any], timeout_seconds: float = 6.0) -> dict[str, Any]:
    prefix = "Complete the following Python function. Return only the completed code.\n\n"
    prompt = record["prompt"]
    starter = prompt[len(prefix) :] if prompt.startswith(prefix) else prompt
    task = record["task"]
    code = extract_code(record["generated"], starter, task["entry_point"])
    script = code + "\n" + task["tests"] + f"\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="mole-humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_limit_child if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "passed": False, "error": "timeout"}
    error = (result.stderr or result.stdout).strip()[-1000:]
    return {
        "score": float(result.returncode == 0),
        "passed": result.returncode == 0,
        "error": error if result.returncode else None,
    }


def extract_final_answer(value: str) -> str | None:
    value = strip_thinking(value)
    if "####" in value:
        value = value.rsplit("####", 1)[-1]
    boxes = BOXED.findall(value)
    if boxes:
        return boxes[-1].strip()
    numbers = NUMBER.findall(value)
    return numbers[-1].strip() if numbers else None


def normalize_numeric(value: str | None) -> Fraction | str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        result = Fraction(cleaned)
        return result / 100 if percent else result
    except (ValueError, ZeroDivisionError):
        return cleaned.casefold()


def score_gsm8k(record: dict[str, Any]) -> dict[str, Any]:
    predicted = extract_final_answer(record["generated"])
    expected = str(record["task"]["final_answer"])
    passed = normalize_numeric(predicted) == normalize_numeric(expected)
    return {
        "score": float(passed),
        "passed": passed,
        "predicted_answer": predicted,
        "expected_answer": expected,
    }


def score_ifeval(record: dict[str, Any], evaluation_lib: Any) -> dict[str, Any]:
    source_id = record["id"].split(":", 1)[-1]
    task = record["task"]
    value = evaluation_lib.InputExample(
        key=int(source_id),
        instruction_id_list=task["instruction_id_list"],
        prompt=record["prompt"],
        # Arrow materializes the union of every IFEval kwargs field for each
        # instruction. Google's native JSONL contains only applicable keys.
        kwargs=[
            {key: value for key, value in item.items() if value is not None}
            for item in task["kwargs"]
        ],
    )
    output = evaluation_lib.test_instruction_following_strict(
        value, {record["prompt"]: strip_thinking(record["generated"])}
    )
    return {
        "score": float(output.follow_all_instructions),
        "passed": output.follow_all_instructions,
        "instruction_scores": output.follow_instruction_list,
    }


def paired_bootstrap(
    left: list[float], right: list[float], seed: int = 20261016, samples: int = 10_000
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired score lists must have the same non-zero length")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    sampled = differences[indices].mean(axis=1)
    return {
        "examples": len(differences),
        "mean_delta": float(differences.mean()),
        "ci95": [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))],
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score code, math, and IFEval semantic generations")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ifeval-root", help="parent directory containing instruction_following_eval")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    evaluation_lib = None
    if args.ifeval_root:
        sys.path.insert(0, args.ifeval_root)
        from langdetect import DetectorFactory
        from instruction_following_eval import evaluation_lib as loaded_evaluation_lib

        DetectorFactory.seed = 0
        evaluation_lib = loaded_evaluation_lib

    generations = read_jsonl(args.generations)
    scored = []
    for index, record in enumerate(generations, start=1):
        if record["domain"] == "code":
            score = execute_humaneval(record)
        elif record["domain"] == "math":
            score = score_gsm8k(record)
        elif record["domain"] == "instruction":
            if evaluation_lib is None:
                continue
            score = score_ifeval(record, evaluation_lib)
        else:
            continue
        scored.append({"id": record["id"], "domain": record["domain"], "condition": record["condition"], **score})
        if index % 100 == 0:
            print(f"scored {index}/{len(generations)}", flush=True)

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in scored:
        buckets[row["condition"]][row["domain"]].append(float(row["score"]))
        buckets[row["condition"]]["overall"].append(float(row["score"]))
        by_key[(row["domain"], row["id"])][row["condition"]] = float(row["score"])
    accuracy = {
        condition: {
            domain: {"examples": len(values), "accuracy": sum(values) / len(values)}
            for domain, values in sorted(domains.items())
        }
        for condition, domains in sorted(buckets.items())
    }
    comparisons = {}
    for other in ("base", "oracle", "wrong"):
        for domain in sorted({row["domain"] for row in scored} | {"overall"}):
            pairs = []
            for (row_domain, _), values in sorted(by_key.items()):
                if domain != "overall" and row_domain != domain:
                    continue
                if "routed" in values and other in values:
                    pairs.append((values["routed"], values[other]))
            if pairs:
                comparisons[f"routed_minus_{other}:{domain}"] = paired_bootstrap(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                    samples=args.bootstrap_samples,
                )
    result = {
        "generation_records": len(generations),
        "scored_records": len(scored),
        "score_counts": dict(sorted(Counter(row["domain"] for row in scored).items())),
        "accuracy": accuracy,
        "paired_bootstrap": comparisons,
        "records": scored,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
