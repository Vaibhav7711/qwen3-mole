# Next stages after v0.1.0

## Checkpoint outcome (2026-10-19)

Stage 1 passed and Stage 2 completed but failed its quality gates. The routed
system scored below the untouched base model on code, mathematics,
instruction-following, and the eligible systems comparisons. Its combined
routed-minus-base difference was -0.339 with a 95% paired bootstrap interval
of [-0.391, -0.287]. The current four adapters must not be scaled to a larger
training run: more steps on the same objective would be an unsupported use of
compute.

The next useful experiment is a new small pilot, not Stage 3 as originally
written. Preserve the base model's general capability by lowering the adapter
learning rate, mixing general replay examples into every expert, evaluating at
frequent checkpoints, and selecting on independent semantic quality rather
than teacher-forced loss. Replace the current router's in-domain classifier
assumption with an out-of-distribution-aware fallback. Only resume the roadmap
below if that pilot beats base with a positive paired confidence interval in at
least one domain without a material regression elsewhere.

The v0.1.0 checkpoint freezes a working four-expert Qwen3-8B MoLE system,
production QLoRA training, held-out likelihood evaluation, routing, data audit,
and optimized AWQ multi-LoRA inference. Future work should proceed through the
following gates in order. A later stage should not be used to hide a failure in
an earlier one.

## Stage 1: independent evaluation set

Status: **passed**. Builder v3 froze 400 independently sourced prompts (100 per
domain), found no configured similarity-threshold violations, and recorded the
manifest SHA-256. See `docs/INDEPENDENT_EVAL_V1.md`.

Goal: replace the same-source random test split with a defensible independent
evaluation set.

Work:

- group systems examples by shared source context before splitting;
- remove prompt clusters with train similarity of 0.90 or higher;
- exclude detected derivatives of HumanEval, MBPP, GSM8K, and any benchmark
  selected for final reporting;
- create a separately sourced set with at least 100 prompts per domain;
- freeze prompts, references, provenance, licenses, and hashes before scoring.

Exit criteria:

- zero exact train/evaluation matches;
- zero evaluation prompts with train similarity of 0.90 or higher after manual
  review;
- no known derivatives of the chosen external benchmarks;
- a versioned manifest makes the split exactly reproducible.

## Stage 2: semantic and task-specific quality

Status: **failed**. Generation, automatic task scoring, and the order-swapped
systems judge completed. Routed did not beat base or wrong-expert control and
had material per-domain regressions. Human/judge agreement remains unclaimed,
but cannot change the failed base, wrong-control, and domain-regression gates.
See `docs/FINAL_INDEPENDENT_BENCHMARK.md`.

Goal: establish whether lower answer NLL corresponds to better answers.

Work:

- generate base, oracle, routed, and wrong-expert answers with identical chat
  templates, thinking mode, token budgets, and decoding;
- use at least 256 new tokens so base reasoning is not selectively truncated;
- execute code against hidden tests and score mathematics by normalized final
  answers where possible;
- use a judge from a different model family for systems and instruction, with
  blinded answer order and a fixed rubric;
- manually review a stratified sample and measure judge/human agreement;
- report paired bootstrap confidence intervals and per-domain win rates.

Exit criteria:

- routed quality beats base with a 95% paired confidence interval excluding
  zero overall;
- routed quality remains close to oracle and beats wrong-expert control;
- no domain has a material regression hidden by the overall average;
- judge agreement and all prompts, outputs, and rubrics are retained.

## Stage 3: router robustness and calibration

Status: **blocked by Stage 2**, apart from small diagnostic pilots.

Goal: make fallback behavior reliable outside the curated four-domain mix.

Work:

- evaluate mixed-domain, ambiguous, adversarial, and out-of-domain prompts;
- measure calibration error and risk-coverage curves;
- tune the fallback threshold on validation data only;
- compare TF-IDF routing with a compact embedding or neural router only if the
  new router improves quality after accounting for latency.

Exit criteria:

- a frozen threshold meets an explicit error-versus-fallback target;
- out-of-domain prompts fall back safely rather than receiving confident but
  arbitrary experts;
- router latency stays negligible relative to time to first token.

## Stage 4: production serving benchmark

Status: **engineering baseline complete; further productization deferred**.
The two-replica AWQ benchmark remains useful as a serving result, but semantic
failure prevents it from being treated as a deployable quality improvement.

Goal: convert the successful offline vLLM run into a service-level result.

Work:

- expose routing and vLLM behind one OpenAI-compatible endpoint;
- measure concurrency 1, 4, 8, and 16 with fixed prompt/output lengths;
- report p50/p95 time to first token, inter-token latency, requests per second,
  tokens per second, GPU memory, and failure rate;
- separate cold startup, warm restart, cold-adapter, and warm-adapter results;
- verify AWQ versus NF4 quality parity on the independent set;
- test one-replica failure and health-check behavior.

Exit criteria:

- routed serving meets defined latency and throughput targets at the chosen
  concurrency;
- AWQ has no material quality regression;
- the service survives adapter changes and single-replica failure cleanly.

## Stage 5: release and communication

Status: **research checkpoint ready; adapter release deferred**. Publish the
negative result and reproducible code, but do not advertise the current
adapters as an improvement over Qwen3-8B.

Goal: make the result independently usable and accurately presented.

Work:

- publish four adapter repositories with model cards, base revision, licenses,
  training configuration, data statement, and evaluation limitations;
- add a reproducible service image and a minimal routed API example;
- create a GitHub release linked to immutable benchmark summaries;
- prepare presentation figures that separate general SFT gain, specialization
  gain, routing loss, and serving performance;
- document failure history and avoid describing MoLE as a native sparse-FFN
  mixture of experts.

Exit criteria:

- a new user can reproduce routing and inference from published artifacts;
- every headline claim maps to a versioned report;
- the presentation clearly states limitations and evaluation scope.

## Priority

Stage 1 and Stage 2 are the critical research path. Stage 4 is the critical
engineering path. Do not invest in larger training runs until independent
semantic evaluation shows which domains benefit and whether more data or more
epochs are likely to help.
