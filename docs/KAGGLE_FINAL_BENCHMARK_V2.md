# Kaggle final benchmark v2

Kernel: `bhabhihunter/qwen3-mole-final-benchmark`, version 2

Final state: `COMPLETE`  
Hardware: two NVIDIA Tesla T4 GPUs  
Evaluation set: 408 untouched test records and 113,453 assistant tokens

## Result

The production adapters passed the first complete held-out evaluation gate.
Every test answer was scored under four conditions: untouched Qwen3-8B, the
known correct domain expert, the automatically routed system, and a fixed
wrong-expert control. Lower token-weighted negative log likelihood (NLL) and
perplexity are better.

| Condition | Mean NLL | Perplexity | NLL change from base |
| --- | ---: | ---: | ---: |
| Base | 1.667 | 5.295 | — |
| Oracle expert | **0.883** | **2.417** | **-47.0%** |
| Routed system | 0.897 | 2.453 | -46.2% |
| Wrong expert | 0.988 | 2.687 | -40.7% |

Routing recovered nearly all of the available oracle performance: routed NLL
was only 1.64% above oracle NLL. The correct expert had the lowest NLL in every
domain.

| Domain | Base | Oracle | Routed | Wrong |
| --- | ---: | ---: | ---: | ---: |
| Code | 1.667 | **0.993** | 1.010 | 1.043 |
| Math | 1.323 | **0.565** | 0.584 | 0.727 |
| Systems | 3.116 | **1.070** | **1.070** | 1.331 |
| Instruction | 1.763 | **1.103** | 1.111 | 1.166 |

The wrong experts also beat the untouched base by 40.7%. This means the result
contains two effects: a large general supervised-fine-tuning gain shared by the
adapters, and a smaller but consistent domain-specialization gain from choosing
the correct adapter. It would be inaccurate to attribute the entire 47% oracle
improvement to expert routing.

## Router behavior

Across 408 prompts, the router selected code 99 times, math 102, systems 102,
instruction 98, and the base fallback seven times. Mean routing latency during
the generation sample was approximately 2.30 ms. Its small 1.64% penalty from
oracle selection is consistent with the separately measured 97.3% router test
accuracy.

## Inference baseline

The deterministic generation sample used four prompts per domain and 64 new
tokens per condition.

| Condition | Mean throughput | Mean generation time |
| --- | ---: | ---: |
| Base | **12.11 tokens/s** | 5.32 s |
| Oracle | 7.31 tokens/s | 8.39 s |
| Routed | 7.33 tokens/s | 8.37 s |
| Wrong | 7.31 tokens/s | 8.50 s |

Peak allocated memory was 6.75 GiB, adapter switching averaged 20.55 ms, and
the routed path was 39.5% slower than base in the Transformers/PEFT
implementation. This identifies an engineering problem, not an inherent MoLE
limit: the benchmark loads four PEFT adapters into one 4-bit model and switches
them request by request. The next serving experiment should use an optimized
multi-LoRA engine and compare identical prompts and decoding limits.

## Interpretation boundary

The likelihood result is strong evidence that the adapters better model held-
out reference answers. It is not yet a semantic correctness score. Some base
generations spent the 64-token budget reasoning while adapted generations moved
quickly into an answer, so simple string overlap would also be misleading.

Before a publication-style claim, complete two checks:

1. audit train/test near-duplicates and known benchmark contamination;
2. run blinded human or independent judge evaluation on longer generations.

The raw private Kaggle output retains all deterministic generations and
per-example condition metadata. Machine-readable headline results are in
`benchmarks/kaggle_final_v2/summary.json`.
