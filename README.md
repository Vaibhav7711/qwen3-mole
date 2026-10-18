# Qwen3-MoLE

Qwen3-MoLE is a Kaggle-sized experiment in specialization: one shared
`Qwen/Qwen3-8B` backbone, four small LoRA experts, and a learned prompt router.
It is designed to make expert fine-tuning and high-throughput serving practical
on Kaggle's two NVIDIA T4 GPUs.

## What we are building

```text
prompt -> router -> code | math | systems | instruction | base
                      \________ Qwen3-8B + selected LoRA ________/
```

This is a **Mixture of LoRA Experts (MoLE)** with request-level routing. The
router selects one adapter per prompt while every expert shares the same 8B
backbone. This keeps training feasible on 2 x 16 GB T4s and enables efficient
multi-adapter inference.

The first router is deliberately simple and fast: word/bigram TF-IDF followed
by balanced logistic regression. Low-confidence prompts use the untouched base
model. The modular design also supports future neural routers and soft/top-k
adapter composition experiments.

## Why Qwen3-8B

- An 8B dense backbone is large enough to be a credible general model while
  still fitting on each T4 in 4-bit form for QLoRA.
- The family has a standard Transformers/PEFT path, broad downstream tooling,
  and an Apache-2.0 model license.
- Training two independent adapters at once uses both T4s efficiently and
  maximizes experiment throughput.

## Highlights

- Four domain experts trained over an 8,192-example curated corpus.
- 43.6M trainable parameters per rank-16 expert, just 0.53% of the backbone.
- Concurrent one-expert-per-GPU QLoRA training on 2 x 16 GB T4s.
- 46.2% lower held-out answer NLL for routed inference relative to base.
- Routed NLL within 1.64% of oracle-domain expert selection.
- 207.8 aggregate generated tokens/s with two AWQ vLLM replicas.
- 66.9% higher sequential routed throughput than the PEFT serving baseline.
- Reproducible 400-prompt independent evaluation across four domains.

## Current status

- [x] Strict normalized chat-data schema and stratified split
- [x] Assistant-only loss masking
- [x] 4-bit NF4 QLoRA expert trainer
- [x] Concurrent one-expert-per-T4 launcher
- [x] Learned top-1 router with calibrated fallback threshold
- [x] Router metrics and confusion matrix
- [x] Multi-LoRA vLLM serving command
- [x] Bounded Kaggle recovery kernel with per-wave artifact preservation
- [x] Completed four-expert real-data recovery benchmark on Kaggle T4 x2
- [x] Local unit and end-to-end router smoke tests
- [x] Kaggle dual-T4 GPU smoke run (3 optimizer steps per concurrent expert)
- [x] Pinned, licensed four-domain ingestion and token-length filtering
- [x] Separate stratified train/validation/test outputs
- [x] 256-example-per-domain one-epoch real-data training run
- [x] All four complete rank-16 adapter runs
- [x] 2,048-example-per-domain production training run
- [x] Near-duplicate and benchmark-overlap audit
- [x] Independently sourced 400-prompt evaluation suite
- [x] Base-vs-expert-vs-routed quality and latency benchmark
- [x] Independent semantic answer-quality evaluation
- [x] Optimized AWQ multi-LoRA serving benchmark

The verified smoke result is documented in
[`docs/KAGGLE_SMOKE_V5.md`](docs/KAGGLE_SMOKE_V5.md). The private Kaggle run is
`vaibhav0383/qwen3-mole-gpu-smoke`, version 5.

The source catalog and curation policy are documented in
[`docs/DATA.md`](docs/DATA.md).

## Data contract

Input is JSONL. Every row has a domain and a chat with at least one user and
one assistant message:

```json
{"domain":"code","messages":[{"role":"user","content":"Write a parser."},{"role":"assistant","content":"Here is a safe implementation..."}]}
```

Allowed domains are `code`, `math`, `systems`, and `instruction`. Router text
stops before the first assistant message, which prevents answer leakage.

## Local control-path check

The local control-path check runs without downloading the model:

```bash
cd qwen3-mole
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
./scripts/smoke_pipeline.sh
```

The smoke corpus verifies the local pipeline; model metrics come from the
versioned benchmark runs.

## Kaggle: first GPU run

Create a Kaggle notebook with **2 x T4** and Internet enabled, attach the
project files, then run from the project directory:

```bash
pip install -q -r requirements-kaggle.txt
pip install -q -e .
python3 scripts/write_smoke_data.py data/smoke.jsonl
python3 -m qwen3_mole.prepare_data --input data/smoke.jsonl --validation-fraction 0.25
./scripts/train_pair.sh code math 10
./scripts/train_pair.sh systems instruction 10
```

This first stage verifies hardware and software compatibility. Confirm both
processes use separate GPUs with `nvidia-smi`. Then replace `data/smoke.jsonl`
with curated normalized data and omit the final step count:

```bash
python3 -m qwen3_mole.prepare_data --input data/all.jsonl
python3 -m qwen3_mole.train_router
./scripts/train_pair.sh code math
./scripts/train_pair.sh systems instruction
```

To build the pinned real-data mixture before those commands:

```bash
python3 -m qwen3_mole.build_data \
  --output data/raw/curated.jsonl \
  --manifest artifacts/data-manifest.json
```

Each trainer loads Qwen3-8B in 4-bit NF4, computes loss only on assistant
tokens, and saves a rank-16 adapter under `artifacts/adapters/<domain>/final`.
The default sequence length is 1024 and effective batch size is 16 per expert.

## Fast inference

For throughput-oriented serving, install a vLLM release compatible with the
notebook's CUDA/PyTorch environment and run:

```bash
./scripts/serve_vllm.sh artifacts/adapters
```

The server tensor-parallelizes one FP16 backbone across both T4s and registers
all four adapters through vLLM's multi-LoRA support. The router's chosen expert
name becomes the LoRA request name. The `base` fallback sends the request with
no LoRA adapter.

The completed bounded real-data benchmark is documented in
[`docs/KAGGLE_RECOVERY_V2.md`](docs/KAGGLE_RECOVERY_V2.md). Machine-readable
headline metrics are in
[`benchmarks/kaggle_recovery_v2/summary.json`](benchmarks/kaggle_recovery_v2/summary.json).

The completed one-epoch run is documented in
[`docs/KAGGLE_FULL_V1.md`](docs/KAGGLE_FULL_V1.md). All four experts reached 15
optimizer steps with improving validation loss; the router achieved 96.2% test
accuracy on 52 held-out prompts. Machine-readable results are in
[`benchmarks/kaggle_full_v1/summary.json`](benchmarks/kaggle_full_v1/summary.json).

The production held-out benchmark is documented in
[`docs/KAGGLE_FINAL_BENCHMARK_V2.md`](docs/KAGGLE_FINAL_BENCHMARK_V2.md). The
routed system reduced answer NLL by 46.2% relative to base and remained within
1.64% of oracle routing. Machine-readable results are in
[`benchmarks/kaggle_final_v2/summary.json`](benchmarks/kaggle_final_v2/summary.json).

The completed data-integrity and benchmark-overlap audit is documented in
[`docs/KAGGLE_DATA_AUDIT_V1.md`](docs/KAGGLE_DATA_AUDIT_V1.md). It verified
zero exact cross-split duplicates and zero ≥0.90 matches against HumanEval,
MBPP, or GSM8K.

The completed optimized inference benchmark is documented in
[`docs/KAGGLE_VLLM_V3.md`](docs/KAGGLE_VLLM_V3.md). Two independent T4 vLLM
replicas delivered 207.8 aggregate generated tokens/s for a mixed routed batch;
sequential routed serving was 66.9% faster than the PEFT baseline. Headline
metrics are in
[`benchmarks/kaggle_vllm_v3/summary.json`](benchmarks/kaggle_vllm_v3/summary.json).

The independent 400-prompt evaluation builder and its stricter benchmark-
derivative filters are documented in
[`docs/INDEPENDENT_EVAL_V1.md`](docs/INDEPENDENT_EVAL_V1.md).
The accepted builder-v3 manifest summary is in
[`benchmarks/independent_eval_v1/summary.json`](benchmarks/independent_eval_v1/summary.json).

## Evaluation design

The evaluation stack compares four conditions on the same held-out prompts:

1. untouched Qwen3-8B;
2. the oracle domain expert;
3. the automatically routed system;
4. cross-domain expert controls.

The reports cover task quality by domain, router accuracy and fallback rate,
tokens per second, time to first token, peak GPU memory, adapter-switch
overhead, and paired confidence intervals. Quality comparisons use a frozen,
model-appropriate sampling policy and independently sourced evaluation data.

## Repository layout

```text
configs/kaggle.yaml             experiment and hardware-safe defaults
scripts/train_pair.sh           launch two independent experts on two GPUs
scripts/serve_vllm.sh           tensor-parallel multi-LoRA inference
kaggle/recovery/                 bounded recovery kernel for T4 x2
kaggle/full/                     one-epoch real-data training kernel for T4 x2
src/qwen3_mole/data.py          schema, validation, splitting, router prompts
src/qwen3_mole/router.py        trained top-1 router and base fallback
src/qwen3_mole/train_expert.py  QLoRA training entry point
tests/                          CPU-only control-path tests
```

Artifacts and processed data are ignored by Git. Adapter weights should be
published separately with their base-model revision, data manifest, training
configuration, and evaluation report.
