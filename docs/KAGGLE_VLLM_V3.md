# Kaggle vLLM benchmark v3

Kernel: `bhabhihunter/qwen3-mole-vllm-benchmark`, version 3

Final state: `COMPLETE`  
Model: `Qwen/Qwen3-8B-AWQ` with four rank-16 LoRA adapters  
Hardware: two NVIDIA Tesla T4 GPUs, one independent replica per GPU

## Outcome

The optimized serving benchmark completed on both GPUs. It loaded an AWQ
backbone and all four trained adapters into each vLLM replica, then measured
base and routed generation sequentially, in domain-grouped batches, and in one
mixed-expert batch.

| Mode | Mean per replica | Dual-replica aggregate |
| --- | ---: | ---: |
| Sequential base | 36.08 tokens/s | 72.16 tokens/s |
| Sequential routed | 12.23 tokens/s | 24.46 tokens/s |
| Batched base | 366.24 tokens/s | 732.48 tokens/s |
| Batched routed, grouped | 80.13 tokens/s | 160.25 tokens/s |
| Batched routed, mixed | **103.90 tokens/s** | **207.81 tokens/s** |

Sequential routed vLLM was 66.9% faster than the earlier Transformers/PEFT
baseline of 7.33 tokens/s. Mixed-expert batching improved routed throughput by
29.7% over separate expert groups and delivered 8.50 times the sequential
routed throughput per replica.

The two workers evaluated 32 prompts: eight per domain and at most 64 new
tokens per response. All sampled prompts routed to their expected domain.

## Resource behavior

- AWQ model loading used approximately 6.07 GiB per T4.
- Approximately 5.41 GiB per GPU remained for KV cache.
- Each replica provisioned 39,392 KV-cache tokens at a 2,048-token maximum
  model length.
- Model loading, compilation, CUDA graph capture, and warm-up averaged 297.8
  seconds per replica.

The first two kernel versions exposed reproducible environment issues rather
than model failures. Version 1 mixed CUDA 13 PyTorch with CUDA 12.8 TorchAudio.
Version 2 removed the incompatible optional packages, but vLLM's warm-up path
still imported TorchVision. Version 3 pinned the matching
`torchvision==0.28.0+cu130` wheel and completed.

## Interpretation

The serving objective is met: vLLM removes the slow PEFT request path and
supports efficient mixed-adapter continuous batching on T4 hardware. The large
gap between base and LoRA throughput remains worth profiling, but it no longer
prevents useful serving throughput.

This is a throughput benchmark, not a production service-level benchmark. A
deployment follow-up should record p50/p95 time to first token, inter-token
latency, request throughput at controlled concurrency, cold versus warm adapter
behavior, and quality parity between AWQ and the NF4 evaluation path.

Machine-readable headline metrics are in
`benchmarks/kaggle_vllm_v3/summary.json`.
