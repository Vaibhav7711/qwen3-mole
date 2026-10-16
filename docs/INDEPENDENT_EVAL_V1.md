# Independent evaluation v1

Status: **build implemented; production Kaggle verification pending**.

This stage replaces same-source random test examples with a separately sourced,
frozen evaluation suite. The suite contains 100 prompts in each expert domain:

| Domain | Source | License | Scoring path |
| --- | --- | --- | --- |
| code | OpenAI HumanEval | MIT | execute generated Python against tests |
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
are always removed. The general exclusion threshold is 0.90. HumanEval uses a
stricter 0.70 threshold and GSM8K uses 0.80 because the earlier audit found
transformed benchmark-like training prompts below 0.90.

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
`bhabhihunter/qwen3-mole-independent-evaluation-set`. Its output is not a model-quality
result: it only freezes and verifies the test set. Semantic generation and
task-specific scoring begin after this build passes its contamination gates.
