# Independent semantic evaluation v1

Status: **generation and task-scoring harness implemented; execution gated on
the independent-set verification**.

The benchmark evaluates four conditions on the same frozen prompts:

1. untouched Qwen3-8B-AWQ base;
2. the oracle domain LoRA;
3. the TF-IDF router-selected LoRA or base fallback;
4. a fixed wrong-domain LoRA control.

Generation uses the pinned `Qwen/Qwen3-8B-AWQ` revision and vLLM multi-LoRA.
Code and math use Qwen thinking mode with up to 8,192 new tokens; systems and
instruction following use direct non-thinking mode with 2,048 and 1,024 tokens
respectively. The pinned model card's recommended sampling is used: temperature
0.6, top-p 0.95, and top-k 20 for thinking; temperature 0.7, top-p 0.8, and
top-k 20 for non-thinking. Both use a 1.5 presence penalty recommended for
quantized Qwen3. The seed is fixed. Every condition within a domain uses the
same mode, sampling, and budget. Two independent T4 workers each host one AWQ
backbone and all four adapters; code/systems and math/instruction are paired to
balance workloads. Conditions are grouped by selected expert and domain so
base fallbacks do not break mixed-LoRA batching. Outputs are checkpointed after
every condition.

The first semantic execution used thinking mode for all domains with only 512
tokens outside code and 1,024 for code. It was rejected before systems judging:
96% of base code, 100% of base math, and 100% of base systems generations hit
the length ceiling, often before emitting a user-visible answer. The corrected
second execution increased budgets and used direct mode for systems and
instruction, but retained greedy decoding. It was also rejected when 80% of
base code still repeated until the 4,096-token ceiling. The pinned model card
explicitly warns that greedy thinking can cause endless repetition. The final
protocol above follows its recommended sampling and was frozen before systems
judging.

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
