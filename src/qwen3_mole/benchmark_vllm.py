from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import prompt_text, read_jsonl
from .router import PromptRouter


DOMAINS = ("code", "math", "systems", "instruction")
MODEL = "Qwen/Qwen3-8B-AWQ"


def resolve_adapter(root: Path, domain: str) -> Path:
    direct = root / domain / "final" / domain
    if (direct / "adapter_config.json").exists():
        return direct
    matches = list(root.glob(f"{domain}/final/**/adapter_config.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve {domain} adapter below {root}")
    return matches[0].parent


def summarize_run(outputs: list[Any], seconds: float) -> dict[str, float | int]:
    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompts = sum(len(output.prompt_token_ids) for output in outputs)
    return {
        "requests": len(outputs),
        "prompt_tokens": prompts,
        "generated_tokens": generated,
        "seconds": seconds,
        "generated_tokens_per_second": generated / seconds,
        "requests_per_second": len(outputs) / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AWQ vLLM multi-LoRA serving")
    parser.add_argument("--domains", nargs="+", required=True, choices=DOMAINS)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    router = PromptRouter.load(args.router)
    selected_records = []
    seen: dict[str, int] = defaultdict(int)
    for record in read_jsonl(args.test_data, DOMAINS):
        domain = record["domain"]
        if domain in args.domains and seen[domain] < args.samples_per_domain:
            selected_records.append(record)
            seen[domain] += 1
    expected = len(args.domains) * args.samples_per_domain
    if len(selected_records) != expected:
        raise ValueError(f"expected {expected} samples, found {len(selected_records)}")

    prompts = []
    routes = []
    for record in selected_records:
        messages = []
        for message in record["messages"]:
            if message["role"] == "assistant":
                break
            messages.append(message)
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        routes.append(router.route(prompt_text(record)).expert)

    adapter_root = Path(args.adapter_root)
    requests = {
        domain: LoRARequest(domain, index + 1, str(resolve_adapter(adapter_root, domain)))
        for index, domain in enumerate(DOMAINS)
    }
    engine_started = time.perf_counter()
    llm = LLM(
        model=MODEL,
        quantization="awq",
        dtype="half",
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=4,
        max_cpu_loras=4,
        max_model_len=2048,
        max_num_seqs=16,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
    )
    engine_load_seconds = time.perf_counter() - engine_started
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    warmup = SamplingParams(temperature=0.0, max_tokens=4)
    llm.generate([prompts[0]], warmup, use_tqdm=False)
    for domain in DOMAINS:
        llm.generate([prompts[0]], warmup, lora_request=requests[domain], use_tqdm=False)

    def generate(values: list[str], loras: Any = None) -> tuple[list[Any], dict[str, float | int]]:
        started = time.perf_counter()
        outputs = llm.generate(
            values,
            sampling,
            lora_request=loras,
            use_tqdm=False,
        )
        return outputs, summarize_run(outputs, time.perf_counter() - started)

    sequential_base_outputs = []
    sequential_base_seconds = 0.0
    for prompt in prompts:
        outputs, metric = generate([prompt])
        sequential_base_outputs.extend(outputs)
        sequential_base_seconds += float(metric["seconds"])
    sequential_base = summarize_run(sequential_base_outputs, sequential_base_seconds)

    sequential_routed_outputs = []
    sequential_routed_seconds = 0.0
    for prompt, route in zip(prompts, routes):
        outputs, metric = generate([prompt], requests[route] if route != "base" else None)
        sequential_routed_outputs.extend(outputs)
        sequential_routed_seconds += float(metric["seconds"])
    sequential_routed = summarize_run(sequential_routed_outputs, sequential_routed_seconds)

    batch_base_outputs, batch_base = generate(prompts)
    batch_routed_outputs = []
    batch_routed_seconds = 0.0
    grouped: dict[str, list[str]] = defaultdict(list)
    for prompt, route in zip(prompts, routes):
        grouped[route].append(prompt)
    for route, route_prompts in grouped.items():
        outputs, metric = generate(
            route_prompts,
            requests[route] if route != "base" else None,
        )
        batch_routed_outputs.extend(outputs)
        batch_routed_seconds += float(metric["seconds"])
    batch_routed = summarize_run(batch_routed_outputs, batch_routed_seconds)

    mixed_requests = [requests[route] for route in routes if route != "base"]
    if len(mixed_requests) == len(routes):
        _, mixed_batch = generate(prompts, mixed_requests)
    else:
        mixed_batch = {"skipped": "sample contains base fallback"}

    result = {
        "model": MODEL,
        "domains": args.domains,
        "samples": len(prompts),
        "route_counts": {route: routes.count(route) for route in sorted(set(routes))},
        "engine_load_seconds": engine_load_seconds,
        "sequential_base": sequential_base,
        "sequential_routed": sequential_routed,
        "batch_base": batch_base,
        "batch_routed_grouped": batch_routed,
        "batch_routed_mixed": mixed_batch,
        "sample_outputs": [
            {
                "domain": record["domain"],
                "route": route,
                "base": tokenizer.decode(base.outputs[0].token_ids, skip_special_tokens=True),
                "routed": tokenizer.decode(routed.outputs[0].token_ids, skip_special_tokens=True),
            }
            for record, route, base, routed in zip(
                selected_records, routes, sequential_base_outputs, sequential_routed_outputs
            )
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_outputs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
