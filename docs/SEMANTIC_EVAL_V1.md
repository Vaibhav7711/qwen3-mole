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
condition. Two independent T4 workers each host one AWQ backbone and all four
adapters. Conditions are grouped by selected expert so base fallbacks do not
break mixed-LoRA batching. Outputs are checkpointed after every condition.

Task scoring is domain-specific:

- HumanEval completions run in isolated, resource-limited subprocesses against
  tests that were not included in the generation prompt;
- GSM8K uses normalized final-answer exact match;
- IFEval uses Google's official strict checker pinned to the revision in
  `configs/semantic_eval.yaml`;
- Server Fault responses go through a separate blinded pairwise judge pass
  using a pinned Mistral-family GPTQ model. Each routed-versus-control pair is
  judged in both answer orders; disagreements are reported as inconsistent
  rather than forced into a win. A deterministic 15-item package is retained
  for manual review.

The scorer reports per-condition accuracy and paired 10,000-sample bootstrap
confidence intervals for routed minus base, oracle, and wrong-expert scores.
No outcome claim will be made until systems judging and the predeclared Stage 2
exit criteria are complete.

The private dual-T4 kernel is
`bhabhihunter/qwen3-mole-semantic-generation`. It consumes immutable outputs
from the production training and independent evaluation kernels.
The dependent judge kernel is `bhabhihunter/qwen3-mole-systems-judge`.
