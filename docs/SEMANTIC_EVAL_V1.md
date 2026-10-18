# Independent semantic evaluation v1

Status: **complete; Stage 2 failed**.

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
  using a pinned Mistral-family AWQ model. Each routed-versus-control pair is
  judged in both answer orders; disagreements are reported as inconsistent
  rather than forced into a win. A deterministic 15-item package is retained
  for manual review. `mole-review-agreement` refuses incomplete annotations
  and reports exact agreement plus Cohen's kappa.

The first judge launch used a pinned GPTQ checkpoint whose dynamic activation
ordering is no longer supported by vLLM 0.31. It failed before any judgments
were generated. The replacement is an Apache-2.0 Mistral-7B-Instruct-v0.3 AWQ
checkpoint with a standard static AWQ configuration, pinned by commit.
The first completed AWQ pass allowed only eight output tokens and asked for an
immediate A/B/TIE label. It was rejected by the predeclared order check: 241 of
300 comparisons changed mapped winner when response order was reversed. The
final pass gives the judge 256 tokens to assess both answers independently,
requires an explicit terminal `FINAL:` line, and excludes invalid or
order-inconsistent comparisons from preference statistics.

The scorer reports per-condition accuracy and paired 10,000-sample bootstrap
confidence intervals for routed minus base, oracle, and wrong-expert scores.
The final analyzer combines the three automatically scored domains with
systems pairwise preferences on the same `[-1, 1]` difference scale. Before
seeing results, Stage 2 is defined to pass only if routed-minus-base and
routed-minus-wrong 95% interval lower bounds are above zero, the overall routed
point estimate is no more than 0.05 below oracle, no domain is more than 0.05
below base, and human/judge Cohen's kappa is at least 0.60. Invalid or
order-inconsistent judge decisions are excluded and counted.

The accepted final run produced 1,600 generations and 1,200 automatic task
scores. Automatic accuracy was 68.7% for base, 37.3% for oracle, 41.0% for
routed, and 49.3% for the wrong-expert control. Routed minus base was -0.277
with a 95% paired interval of [-0.330, -0.223]. The corrected systems pass
completed all 600 order-swapped judge calls; among 48 eligible routed/base
comparisons, base won 35, 13 tied, and routed won none. The combined
routed-minus-base difference was -0.339 with interval [-0.391, -0.287].

The human/judge agreement gate is deliberately reported as unresolved: the
15-item package has not been annotated by a human. This does not make the
Stage 2 outcome ambiguous, because three independent required gates already
failed: routed did not beat base, did not beat the wrong control, and had
material domain regressions. The full result and limitations are in
`docs/FINAL_INDEPENDENT_BENCHMARK.md`.

The private dual-T4 kernel is
`bhabhihunter/qwen3-mole-semantic-generation`. It consumes immutable outputs
from the production training and independent evaluation kernels.
The dependent judge kernel is `bhabhihunter/qwen3-mole-systems-judge`.
