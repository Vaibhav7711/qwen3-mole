# Qwen3-MoLE

Qwen3-MoLE is a Kaggle-sized experiment in specialization: one shared
`Qwen/Qwen3-8B` backbone, four small LoRA experts, and a learned prompt router.
It is designed for Kaggle's two NVIDIA T4 GPUs without pretending that an 8B
model can be pretrained there.

## What we are building

```text
prompt -> router -> code | math | systems | instruction | base
                      \________ Qwen3-8B + selected LoRA ________/
```

This is a **Mixture of LoRA Experts (MoLE)**. It is not a native sparse-FFN MoE
such as Mixtral. The router selects one adapter per request, so only one 8B
backbone is needed in memory. This gives us a useful, measurable MoE research
project while keeping training feasible on 2 x 16 GB T4s.

The first router is deliberately simple and fast: word/bigram TF-IDF followed
by balanced logistic regression. Low-confidence prompts use the untouched base
model. A learned neural router and soft/top-k adapter composition are later
experiments, not unmeasured claims in the baseline.

## Why Qwen3-8B

- An 8B dense backbone is large enough to be a credible general model while
  still fitting on each T4 in 4-bit form for QLoRA.
- The family has a standard Transformers/PEFT path, broad downstream tooling,
  and an Apache-2.0 model license.
- Training two independent adapters at once uses both T4s efficiently. Trying
  to split a single QLoRA training process across the two cards adds complexity
  without buying as much experiment throughput.

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
- [ ] Near-duplicate and benchmark-contamination checks
- [ ] 256-example-per-domain real-data pilot
- [ ] All four complete adapter runs
- [ ] Base-vs-expert-vs-routed quality and latency benchmark

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

The local check does not download the model:

```bash
cd qwen3-mole
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
./scripts/smoke_pipeline.sh
```

The smoke corpus exists only to verify plumbing. Its accuracy is not a model
result and must not be reported as one.

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

That is a hardware/software compatibility smoke test, not the final training
run. Confirm both processes use separate GPUs with `nvidia-smi`. Then replace
`data/smoke.jsonl` with curated normalized data and omit the final step count:

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

The completed bounded real-data benchmark and its limitations are documented in
[`docs/KAGGLE_RECOVERY_V2.md`](docs/KAGGLE_RECOVERY_V2.md). Machine-readable
headline metrics are in
[`benchmarks/kaggle_recovery_v2/summary.json`](benchmarks/kaggle_recovery_v2/summary.json).

## Evaluation gate

We do not call the experiment successful merely because loss falls. The final
benchmark must compare, on held-out prompts:

1. untouched Qwen3-8B;
2. the oracle domain expert;
3. the automatically routed system;
4. wrong-expert controls.

Report task quality by domain, router accuracy and fallback rate, tokens per
second, time to first token, peak GPU memory, and adapter-switch overhead.
Use deterministic decoding for quality comparisons. Keep an untouched test set
whose sources and near-duplicates were excluded from training.

## Repository layout

```text
configs/kaggle.yaml             experiment and hardware-safe defaults
scripts/train_pair.sh           launch two independent experts on two GPUs
scripts/serve_vllm.sh           tensor-parallel multi-LoRA inference
kaggle/recovery/                 bounded recovery kernel for T4 x2
src/qwen3_mole/data.py          schema, validation, splitting, router prompts
src/qwen3_mole/router.py        trained top-1 router and base fallback
src/qwen3_mole/train_expert.py  QLoRA training entry point
tests/                          CPU-only control-path tests
```

Artifacts and processed data are ignored by Git. Adapter weights should be
published separately with their base-model revision, data manifest, training
configuration, and evaluation report.
