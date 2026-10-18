# Final independent benchmark

Status: **complete; Stage 2 failed**.

## Decision

The current Qwen3-8B Mixture of LoRA Experts does not improve semantic answer
quality over the untouched base model. The appropriate checkpoint is to stop
scaling this training recipe and redesign the specialization objective on a
small pilot. This is a useful negative result: the earlier 46.2% reduction in
teacher-forced answer NLL did not translate into better generated answers.

## Frozen evaluation

Builder v3 supplied 400 independently sourced prompts, 100 each for code,
mathematics, systems, and instruction-following. Its output SHA-256 is
`24fa0c9a4e493d7cf0fb624d0381ca2944b7229dedbb1c443874cf04f0a10449`,
and all configured train-similarity and benchmark-derivative checks passed.

Four conditions used the same pinned Qwen3-8B-AWQ backbone: base, oracle-domain
adapter, router-selected adapter or base fallback, and a fixed wrong adapter.
Code and math used Qwen thinking mode and its recommended sampling; systems and
instruction used non-thinking mode. The final run generated all 1,600 answers
and produced 1,200 automatic task scores. Generation took 29,955.6 seconds
(8.32 hours) on two T4 workers.

## Automatic task quality

| Condition | Code | Math | Instruction | Overall |
|---|---:|---:|---:|---:|
| Base | 31% | 94% | 81% | 68.7% |
| Oracle adapter | 23% | 52% | 37% | 37.3% |
| Routed | 22% | 55% | 46% | 41.0% |
| Wrong adapter | 14% | 63% | 71% | 49.3% |

Routed minus base was -27.7 percentage points with a 95% paired bootstrap
interval of [-33.0, -22.3]. Per-domain routed-minus-base differences were -9
points for code, -39 for math, and -35 for instruction; every interval was
entirely below zero. Routed also trailed the wrong-expert control by 8.3 points,
with interval [-14.7, -2.0].

The surprising fact that routed was 3.7 points above oracle is not evidence of
successful routing. Both were far below base, and routed sometimes fell back
to base or selected a different adapter. The result instead shows that the
domain adapters themselves caused most of the semantic regression.

## Systems comparison

Systems answers were compared by a pinned Mistral-7B-Instruct-v0.3 AWQ judge
from a different model family. Every routed/control pair was shown in both
answer orders. The corrected pass completed 600 judge calls in 1,167.4 seconds.

| Pair | Eligible | Excluded | Routed wins | Ties | Control wins | Routed preference |
|---|---:|---:|---:|---:|---:|---:|
| Routed vs base | 48 | 52 | 0 | 13 | 35 | 13.5% |
| Routed vs oracle | 64 | 36 | 27 | 33 | 4 | 68.0% |
| Routed vs wrong | 67 | 33 | 20 | 46 | 1 | 64.2% |

The routed/base direction is unambiguous among eligible comparisons, but
systems numbers must be treated as secondary evidence: 121 of 300 comparisons
(40.3%) were invalid or changed mapped winner when answer order was reversed.
The retained 15-item review package has not been annotated by a human, so no
human/judge agreement or Cohen's kappa is claimed.

Human agreement cannot rescue the Stage 2 decision. Even if that one gate later
passes, the predeclared routed-vs-base, routed-vs-wrong, and no-domain-regression
gates have already failed.

## Combined result and gates

Eligible systems preferences and the three automatic domains use the same
paired difference scale: +1 routed win, 0 tie, and -1 routed loss.

| Comparison | Paired examples | Mean difference | 95% interval |
|---|---:|---:|---:|
| Routed minus base | 348 | -0.339 | [-0.391, -0.287] |
| Routed minus oracle | 364 | +0.093 | [+0.055, +0.135] |
| Routed minus wrong | 367 | -0.016 | [-0.074, +0.041] |

Only the closeness-to-oracle gate passed. The required positive lower bounds
against base and wrong did not pass, and material domain regressions occurred.
Stage 2 therefore failed regardless of the unresolved human-agreement gate.

## Router diagnosis

The router classified 60.5% of independent prompts correctly. Domain routing
was 67% for code, 81% for instruction, 94% for math, and 0% for systems. It sent
74 of 100 systems prompts to the code adapter, 11 to instruction, and 15 to
base; none went to the systems expert. This is a major independent-domain shift
from the earlier same-source router result.

Router error is not the primary failure, however. Oracle adapters scored below
base in every automatically scored domain, so perfect routing would still have
reduced quality.

## Generation diagnostics

The final protocol corrected two rejected preliminary runs: one had undersized
budgets and thinking enabled for all domains; the next used greedy thinking,
which the Qwen3 model card warns can repeat indefinitely. The accepted run used
the pinned model-card sampling policy and up to 8,192 new tokens for code and
math.

Base code remained expensive: 68 of 100 generations hit the 8,192-token limit,
versus 13 routed generations. This limits absolute code interpretation, but it
does not create a false routed advantage—base still passed 31 problems versus
22 for routed. Across all conditions, 1,416 of 1,600 generations stopped
normally and 184 reached their domain limit.

## What the project established

The engineering path works: four rank-16 QLoRA adapters can be trained on two
T4s, routed, and served with two AWQ vLLM replicas at 207.8 aggregate generated
tokens/s. The research hypothesis did not survive independent semantic
evaluation. Lower held-out NLL mostly reflected supervised adaptation to the
training distribution, not reliable domain specialization.

The next experiment should use small checkpointed pilots with lower learning
rates, general-data replay in every expert, explicit base-capability retention,
and semantic validation during training. A stronger out-of-distribution
fallback is also needed. No larger run should begin until a pilot beats base
with a positive paired confidence interval in at least one domain without
material regressions elsewhere.

Machine-readable headline results are in
`benchmarks/independent_semantic_v1/summary.json`; the exact inference and judge
configuration is in `configs/semantic_eval.yaml`.
