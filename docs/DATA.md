# Data sources and curation

Every source is revision-pinned in `configs/data_sources.yaml`. A manifest is
written with the source repository, revision, declared license, scanned and
accepted counts, and a SHA-256 content fingerprint for every accepted record.

| Expert | Dataset | Declared license | Default cap |
|---|---|---:|---:|
| code | `ise-uiuc/Magicoder-Evol-Instruct-110K` | Apache-2.0 | 5,000 |
| math | `AI-MO/NuminaMath-CoT` | Apache-2.0 | 5,000 |
| systems | `jalpan04/devops-sft-dataset` | Apache-2.0 | 5,000 |
| instruction | `HuggingFaceH4/ultrachat_200k` | MIT | 5,000 |

The DevOps source contains both JSON-encoded strings and structured JSON
answers. Its raw JSONL cannot be represented by one Arrow scalar schema, so the
pipeline uses Hugging Face's pinned Parquet conversion and renders structured
fields into readable labeled text. No answer fields are silently discarded.

## Admission policy

The builder:

1. shuffles each pinned source deterministically;
2. extracts the first complete user/assistant exchange and preserves an
   optional leading system message;
3. rejects missing, too-short, or extreme-length prompts and answers;
4. renders each candidate with the real Qwen3 tokenizer;
5. rejects conversations longer than the configured 1,024-token training
   window instead of truncating away part of the answer;
6. removes normalized exact duplicates within each source;
7. records source and content provenance in every output row.

The preparation stage then makes stratified train, validation, and untouched
test sets. Validation is for router/threshold development. Test is reserved for
the final router and model comparison.

## Verified pilot

A live 32-record-per-source pilot admitted 128 balanced records. Token-length
filtering rejected 2 code, 12 systems, and 4 instruction candidates while
finding 32 accepted records in every domain. The resulting split contained 112
train, 8 validation, and 8 test records. The validation sample is too small for
quality claims; its purpose was schema and pipeline validation.

## Remaining contamination gate

Exact fingerprints prevent exact duplicate leakage. Before the full run, add a
near-duplicate pass across all splits and compare prompts against the selected
evaluation benchmarks. The full adapters should not be trained until that gate
produces a report and removes flagged records.
