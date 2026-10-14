from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from .data import prompt_text, read_jsonl
from .router import PromptRouter
from .tokenization import tokenize_chat_with_assistant_labels


DOMAINS = ("code", "math", "systems", "instruction")
WRONG_EXPERT = dict(zip(DOMAINS, DOMAINS[1:] + DOMAINS[:1]))


def adapter_path(root: Path, domain: str) -> Path:
    direct = root / domain / "final" / domain
    if (direct / "adapter_config.json").exists():
        return direct
    matches = list(root.glob(f"{domain}/final/**/adapter_config.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve final {domain} adapter below {root}")
    return matches[0].parent


def assistant_token_count(labels: Any) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def update_metric(bucket: dict[str, float], loss: float, tokens: int) -> None:
    bucket["negative_log_likelihood"] += loss * tokens
    bucket["assistant_tokens"] += tokens
    bucket["examples"] += 1


def finalize_metric(bucket: dict[str, float]) -> dict[str, float | int]:
    tokens = int(bucket["assistant_tokens"])
    mean_nll = bucket["negative_log_likelihood"] / tokens
    return {
        "examples": int(bucket["examples"]),
        "assistant_tokens": tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark base and routed Qwen3-MoLE experts")
    parser.add_argument("--domains", nargs="+", required=True, choices=DOMAINS)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    set_seed(42)
    adapter_root = Path(args.adapter_root)
    router = PromptRouter.load(args.router)
    records = [r for r in read_jsonl(args.test_data, DOMAINS) if r["domain"] in args.domains]
    base_model = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_path(adapter_root, DOMAINS[0]),
        adapter_name=DOMAINS[0],
        is_trainable=False,
    )
    for domain in DOMAINS[1:]:
        model.load_adapter(adapter_path(adapter_root, domain), adapter_name=domain, is_trainable=False)
    model.eval()
    device = torch.device("cuda:0")

    @contextlib.contextmanager
    def selected(expert: str) -> Iterator[None]:
        if expert == "base":
            with model.disable_adapter():
                yield
        else:
            model.set_adapter(expert)
            yield

    raw_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    route_counts: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            row = tokenize_chat_with_assistant_labels(tokenizer, record["messages"], 1024)
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
            labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
            tokens = assistant_token_count(labels)
            route = router.route(prompt_text(record))
            route_counts[route.expert] += 1
            choices = {
                "base": "base",
                "oracle": record["domain"],
                "routed": route.expert,
                "wrong": WRONG_EXPERT[record["domain"]],
            }
            for condition, expert in choices.items():
                with selected(expert):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = functional.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item() / tokens
                update_metric(raw_metrics[condition][record["domain"]], loss, tokens)
                update_metric(raw_metrics[condition]["overall"], loss, tokens)
                del logits
            if index % 25 == 0:
                print(f"evaluated {index}/{len(records)}", flush=True)

    quality = {
        condition: {domain: finalize_metric(bucket) for domain, bucket in domains.items()}
        for condition, domains in raw_metrics.items()
    }

    generation_records: list[dict[str, Any]] = []
    latency: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    per_domain_seen: dict[str, int] = defaultdict(int)
    generation_subset = []
    for record in records:
        if per_domain_seen[record["domain"]] < args.generation_samples:
            generation_subset.append(record)
            per_domain_seen[record["domain"]] += 1

    with torch.inference_mode():
        for record in generation_subset:
            prompt_messages = []
            for message in record["messages"]:
                if message["role"] == "assistant":
                    break
                prompt_messages.append(message)
            encoded = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            route_started = time.perf_counter()
            route = router.route(prompt_text(record))
            route_ms = (time.perf_counter() - route_started) * 1000
            choices = {
                "base": "base",
                "oracle": record["domain"],
                "routed": route.expert,
                "wrong": WRONG_EXPERT[record["domain"]],
            }
            for condition, expert in choices.items():
                with selected(expert):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    prefill_started = time.perf_counter()
                    model(**encoded, use_cache=False)
                    torch.cuda.synchronize(device)
                    prefill_seconds = time.perf_counter() - prefill_started
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    generate_started = time.perf_counter()
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    torch.cuda.synchronize(device)
                    generation_seconds = time.perf_counter() - generate_started
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
                prompt_tokens = int(encoded["input_ids"].shape[1])
                new_ids = output_ids[0, prompt_tokens:]
                new_tokens = int(new_ids.shape[0])
                latency[condition].append(
                    {
                        "domain": record["domain"],
                        "expert": expert,
                        "prompt_tokens": prompt_tokens,
                        "new_tokens": new_tokens,
                        "prefill_seconds": prefill_seconds,
                        "generation_seconds": generation_seconds,
                        "tokens_per_second": new_tokens / generation_seconds,
                        "peak_memory_gib": peak_gib,
                        "router_ms": route_ms if condition == "routed" else 0.0,
                    }
                )
                generation_records.append(
                    {
                        "domain": record["domain"],
                        "condition": condition,
                        "expert": expert,
                        "prompt": prompt_text(record),
                        "generated": tokenizer.decode(new_ids, skip_special_tokens=True),
                    }
                )

    switch_started = time.perf_counter()
    for index in range(100):
        model.set_adapter(DOMAINS[index % len(DOMAINS)])
    switch_seconds = time.perf_counter() - switch_started

    latency_summary = {}
    for condition, values in latency.items():
        latency_summary[condition] = {
            "samples": len(values),
            "mean_prefill_seconds": sum(float(v["prefill_seconds"]) for v in values) / len(values),
            "mean_generation_seconds": sum(float(v["generation_seconds"]) for v in values) / len(values),
            "mean_tokens_per_second": sum(float(v["tokens_per_second"]) for v in values) / len(values),
            "max_peak_memory_gib": max(float(v["peak_memory_gib"]) for v in values),
            "mean_router_ms": sum(float(v["router_ms"]) for v in values) / len(values),
        }

    result = {
        "domains": args.domains,
        "test_examples": len(records),
        "quality": quality,
        "route_counts": dict(sorted(route_counts.items())),
        "latency": latency_summary,
        "adapter_switch_mean_ms": switch_seconds * 1000 / 100,
        "generation_records": generation_records,
        "wall_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "generation_records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
