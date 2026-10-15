# Kaggle data audit v1

Kernel: `bhabhihunter/qwen3-mole-data-audit`, version 1

Final state: `COMPLETE`  
Partitions: 7,376 train, 408 validation, 408 test

## Cross-split leakage

The audit found zero exact normalized prompt matches and zero exact normalized
full-conversation matches between train and either held-out partition.

Character 3–5 gram TF-IDF nearest-neighbor matching found:

| Comparison | Similarity ≥0.90 | ≥0.95 | ≥0.98 |
| --- | ---: | ---: | ---: |
| Validation versus train | 6 | 5 | 5 |
| Test versus train | 12 | 10 | 9 |

Of the 12 test matches at 0.90 or above, 11 are systems records and one is
math. Manual inspection shows that the systems pairs generally share a long
retrieved context while asking different questions. They are not exact answer
duplicates, but they make the random record-level split less independent than
a source-document or similarity-cluster split would be.

## Common benchmark overlap

The audit loaded HumanEval, the sanitized MBPP test split, and the GSM8K test
split without errors. No training prompt reached 0.90 similarity to a catalog
prompt; maximum similarity was 0.802.

That threshold result is not proof of zero contamination. Manual inspection of
the nearest matches found transformed or translated examples resembling
HumanEval functions such as `is_prime`, `separate_paren_groups`, and
`encode_cyclic`, plus generic MBPP-style tasks. These are below the selected
near-duplicate threshold but should be excluded before evaluating on those
benchmarks themselves.

## Consequence for the final benchmark

The existing 408-example result remains useful because there is no exact
train/test leakage and only 2.9% of test prompts exceed 0.90 similarity. The
reported likelihood improvement should nevertheless be described as a
same-source held-out result, not as evidence of broad benchmark
generalization.

For publication-grade evaluation, split systems records by source context or
similarity cluster and explicitly remove derivatives of the external benchmark
catalogs before training. Machine-readable headline results are in
`benchmarks/kaggle_data_audit_v1/summary.json`.
