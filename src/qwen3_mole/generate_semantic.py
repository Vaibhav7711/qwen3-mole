from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .benchmark_vllm import DOMAINS, resolve_adapter
from .router import PromptRouter


def read_eval_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate id")
            if record.get("domain") not in DOMAINS:
                raise ValueError(f"{path}:{line_number}: invalid domain")
            if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
                raise ValueError(f"{path}:{line_number}: missing prompt")
            seen.add(identifier)
            records.append(record)
    return records


def router_text(prompt: str) -> str:
    return f"user: {prompt.strip()}"


def condition_expert(
    condition: str,
    domain: str,
    routed_expert: str,
    wrong_expert: dict[str, str],
) -> str:
    if condition == "base":
        return "base"
    if condition == "oracle":
        return domain
    if condition == "routed":
        return routed_expert
    if condition == "wrong":
        return wrong_expert[domain]
    raise ValueError(f"unknown condition: {condition}")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent semantic-evaluation answers with vLLM")
    parser.add_argument("--config", default="configs/semantic_eval.yaml")
    parser.add_argument("--domains", nargs="+", required=True, choices=DOMAINS)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    config = yaml.safe_load(Path(args.config).read_text())
    records = [record for record in read_eval_jsonl(args.eval_data) if record["domain"] in args.domains]
    expected = len(args.domains) * 100
    if len(records) != expected:
        raise ValueError(f"expected {expected} evaluation records, found {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["model_revision"], use_fast=True
    )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(config["thinking"]),
        )
        for record in records
    ]
    router = PromptRouter.load(args.router)
    routes = [router.route(router_text(record["prompt"])) for record in records]

    adapter_root = Path(args.adapter_root)
    requests = {
        domain: LoRARequest(domain, index + 1, str(resolve_adapter(adapter_root, domain)))
        for index, domain in enumerate(DOMAINS)
    }
    started = time.perf_counter()
    llm = LLM(
        model=config["model"],
        revision=config["model_revision"],
        quantization="awq",
        dtype="half",
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=4,
        max_cpu_loras=4,
        max_model_len=int(config["max_model_len"]),
        max_num_seqs=int(config["max_num_seqs"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        trust_remote_code=False,
        seed=int(config["seed"]),
    )
    token_budgets = {
        domain: int(config.get("max_new_tokens_by_domain", {}).get(domain, config["max_new_tokens"]))
        for domain in DOMAINS
    }
    results: list[dict[str, Any]] = []
    condition_seconds: dict[str, float] = {}
    output_path = Path(args.output)
    for condition in config["conditions"]:
        condition_started = time.perf_counter()
        assignments = [
            condition_expert(
                condition,
                record["domain"],
                route.expert,
                config["wrong_expert"],
            )
            for record, route in zip(records, routes)
        ]
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, expert in enumerate(assignments):
            grouped[(expert, records[index]["domain"])].append(index)
        decoded: dict[int, tuple[str, int, str | None]] = {}
        for (expert, domain), indices in sorted(grouped.items()):
            sampling = SamplingParams(
                temperature=float(config["temperature"]),
                max_tokens=token_budgets[domain],
                seed=int(config["seed"]),
            )
            outputs = llm.generate(
                [prompts[index] for index in indices],
                sampling,
                lora_request=requests[expert] if expert != "base" else None,
                use_tqdm=True,
            )
            for index, output in zip(indices, outputs):
                generated = output.outputs[0]
                decoded[index] = (
                    generated.text,
                    len(generated.token_ids),
                    str(generated.finish_reason) if generated.finish_reason is not None else None,
                )
        for index, (record, route, expert) in enumerate(zip(records, routes, assignments)):
            text, tokens, finish_reason = decoded[index]
            result = {
                    "id": record["id"],
                    "domain": record["domain"],
                    "condition": condition,
                    "expert": expert,
                    "route": {
                        "expert": route.expert,
                        "predicted_domain": route.predicted_domain,
                        "confidence": route.confidence,
                        "probabilities": route.probabilities,
                    },
                    "prompt": record["prompt"],
                    "evaluation_method": record["evaluation_method"],
                    "generated": text,
                    "generated_tokens": tokens,
                    "finish_reason": finish_reason,
                }
            # Systems answers need their reference in the separate judge
            # kernel. Executable private tests never leave the frozen suite.
            if record["domain"] == "systems":
                result["reference"] = record["reference"]
            results.append(result)
        condition_seconds[condition] = time.perf_counter() - condition_started
        write_jsonl(output_path, results)
        print(f"completed {condition}: {len(records)} answers", flush=True)

    metadata = {
        "model": config["model"],
        "model_revision": config["model_revision"],
        "thinking": config["thinking"],
        "temperature": config["temperature"],
        "max_new_tokens": config["max_new_tokens"],
        "max_new_tokens_by_domain": token_budgets,
        "records": len(records),
        "answers": len(results),
        "domains": dict(sorted(Counter(record["domain"] for record in records).items())),
        "routes": dict(sorted(Counter(route.expert for route in routes).items())),
        "condition_seconds": condition_seconds,
        "wall_seconds": time.perf_counter() - started,
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
