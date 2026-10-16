# Independent semantic evaluation v1

Status: **generation and task-scoring harness implemented; execution gated on
the independent-set verification**.

The benchmark evaluates four conditions on the same frozen prompts:

1. untouched Qwen3-8B-AWQ base;
2. the oracle domain LoRA;
3. the TF-IDF router-selected LoRA or base fallback;
4. a fixed wrong-domain LoRA control.

Generation uses the pinned `Qwen/Qwen3-8B-AWQ` revision, vLLM multi-LoRA,
greedy decoding, Qwen thinking mode, and a 512-new-token budget in every
non-code domain. Code receives 1,024 new tokens because complete competitive
programming solutions are materially longer. Every condition within a domain
uses the same budget. Two independent T4 workers each host one AWQ backbone and
all four adapters. Conditions are grouped by selected expert and domain so
base fallbacks do not break mixed-LoRA batching. Outputs are checkpointed after
every condition.

Task scoring is domain-specific:

- LiveCodeBench v6 solutions run through its pinned official evaluator against
  7,000 public and private tests; test payloads are never included in prompts
  or generation files;
- GSM8K uses normalized final-answer exact match;
- IFEval uses Google's official strict checker pinned to the revision in
  `configs/semantic_eval.yaml`;
- Server Fault responses go through a separate blinded pairwise judge pass
  using a pinned Mistral-family GPTQ model. Each routed-versus-control pair is
  judged in both answer orders; disagreements are reported as inconsistent
  rather than forced into a win. A deterministic 15-item package is retained
  for manual review. `mole-review-agreement` refuses incomplete annotations
  and reports exact agreement plus Cohen's kappa.

The scorer reports per-condition accuracy and paired 10,000-sample bootstrap
confidence intervals for routed minus base, oracle, and wrong-expert scores.
The final analyzer combines the three automatically scored domains with
systems pairwise preferences on the same `[-1, 1]` difference scale. Before
seeing results, Stage 2 is defined to pass only if routed-minus-base and
routed-minus-wrong 95% interval lower bounds are above zero, the overall routed
point estimate is no more than 0.05 below oracle, no domain is more than 0.05
below base, and human/judge Cohen's kappa is at least 0.60. Invalid or
order-inconsistent judge decisions are excluded and counted. No outcome claim
will be made until all five gates are resolved.

The private dual-T4 kernel is
`bhabhihunter/qwen3-mole-semantic-generation`. It consumes immutable outputs
from the production training and independent evaluation kernels.
The dependent judge kernel is `bhabhihunter/qwen3-mole-systems-judge`.
