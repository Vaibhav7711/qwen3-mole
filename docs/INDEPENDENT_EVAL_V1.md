# Independent evaluation v1

Status: **accepted and production-verified on Kaggle**.

This stage replaces same-source random test examples with a separately sourced,
frozen evaluation suite. The suite contains 100 prompts in each expert domain:

| Domain | Source | License | Scoring path |
| --- | --- | --- | --- |
| code | LiveCodeBench v6 (January--April 2025) | benchmark code MIT; problem text retains source-platform provenance | official private-test execution |
| math | OpenAI GSM8K test | MIT | normalized final-answer match |
| systems | Server Fault via Stack Exchange preferences | CC BY-SA 4.0 | blinded judge plus manual sample |
| instruction | Google IFEval | Apache 2.0 | deterministic instruction checks |

Every source is pinned to an immutable repository revision in
`configs/eval_sources.yaml`. The builder records the source identifier,
revision, license, evaluation method, per-record content hash, maximum training
similarity, and a hash of the final JSONL file.

## Decontamination policy

The builder compares normalized evaluation prompts with the complete production
training split using character 3--5 gram TF-IDF cosine similarity. Exact matches
are always removed. The general exclusion threshold is 0.90. LiveCodeBench and
GSM8K use a stricter 0.80 threshold because the earlier audit found transformed
benchmark-like training prompts below 0.90.

Builder version 2 initially used HumanEval with a 0.70 threshold. Although its
automated gate passed, manual review still found recognizable transformed
training derivatives below that cutoff. Those results were rejected rather
than reported. Builder version 3 replaces HumanEval with the pinned `test6`
file from LiveCodeBench. Its 175 problems were published from January through
April 2025, after the dated corpora used for adapter training. The original
problem statement and starter code, without the generation wrapper, are still
compared against every production-training prompt.

Private tests are stored only in the frozen evaluation artifact. They are
never inserted into model prompts or copied into generation outputs. Scoring
joins answers back to the frozen records and uses the official LiveCodeBench
execution implementation pinned in `configs/semantic_eval.yaml`.

Candidates are deterministically ranked from their identifier, content hash,
and seed. Up to 10 times the target count per domain enters the similarity
scan, and the first 100 clean candidates are frozen. The generated prompts and
references are ignored by Git and retained in private Kaggle output; the public
repository contains the exact builder and source manifest.

## Reproduction

```bash
python -m qwen3_mole.build_eval \
  --train data/processed/full-train.jsonl \
  --output data/eval/independent-v1.jsonl \
  --manifest artifacts/eval/independent-v1-manifest.json \
  --report artifacts/eval/independent-v1-decontamination.json
```

The private Kaggle kernel is
`bhabhihunter/qwen3-mole-independent-evaluation-set`. Its output is not a
model-quality result: it only freezes and verifies the test set. Semantic
generation and task-specific scoring begin only after builder version 3 passes
its automated gates and the retained-prompt manual review.

## Accepted build

Kaggle kernel version 3 completed with 400 records and zero threshold
violations. The independently rechecked output SHA-256 is
`24fa0c9a4e493d7cf0fb624d0381ca2944b7229dedbb1c443874cf04f0a10449`.
Maximum selected train similarities were 0.55684 for code, 0.48836 for math,
0.43448 for systems, and 0.38336 for instruction following. The selected code
problems span January 4 through April 6, 2025. The non-sensitive machine-readable
result is retained in `benchmarks/independent_eval_v1/summary.json`; prompts,
references, and private tests remain in the private hashed Kaggle artifact.
