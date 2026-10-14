# Kaggle full-training v1 results

Kernel: `bhabhihunter/qwen3-mole-full-train`, version 1  
Final state: `COMPLETE`  
Hardware: two NVIDIA Tesla T4 GPUs with one independent QLoRA process per GPU

## Outcome

The one-epoch Qwen3-MoLE training run completed end to end. It built 1,024
pinned real-data records, made stratified train/validation/test partitions,
trained and evaluated the prompt router, and trained code, math, systems, and
instruction LoRA experts in two concurrent waves. All subprocesses exited with
code zero, all experts reached step 15, and all four final adapter
configurations were retained in the Kaggle output.

The measured pipeline stages took 1,593.1 seconds (26 minutes 33 seconds).
This excludes any time Kaggle spent queued or starting the worker. The first
code/math wave took 805.0 seconds and the systems/instruction wave took 688.2
seconds; the remaining stages covered installation, data preparation, tests,
and router training.

## Training configuration

- Base model: `Qwen/Qwen3-8B` loaded in 4-bit NF4 for QLoRA.
- Data: 256 accepted records per domain; 230 train, 13 validation, and 13 test
  examples per domain.
- Sequence length: 1,024 tokens. No train or validation example was skipped.
- One epoch: 230 microbatches and 15 optimizer steps per expert, with gradient
  accumulation of 16.
- LoRA: rank 16, alpha 32, dropout 0.05, applied to all attention and MLP
  projection modules.
- Trainable parameters: 43,646,976 per expert, or 0.5301% of the 8.234B
  parameters visible through the quantized model.

## Expert results

| Expert | Training runtime | Mean train loss | Validation loss: step 5 | Step 10 | Step 15 | Step 5 to 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Code | 619.4 s | 1.182 | 1.252 | 1.170 | 1.149 | -8.2% |
| Math | 658.0 s | 0.766 | 0.955 | 0.759 | 0.722 | -24.4% |
| Systems | 305.8 s | 1.714 | 1.732 | 1.361 | 1.292 | -25.4% |
| Instruction | 644.8 s | 1.299 | 1.204 | 1.067 | 1.035 | -14.0% |

Validation loss declined at both evaluation intervals for every expert. That
is good evidence that the optimizer, masking, evaluation, and checkpoint paths
worked and that the adapters learned the training distribution. It is not by
itself evidence that generated answers are better than those from untouched
Qwen3-8B; that requires the generation benchmark described below.

Systems examples process substantially faster because their tokenized
sequences are shorter on average. All domains still use the same maximum
sequence length and training configuration.

## Router results

The TF-IDF/logistic-regression router trained on 920 prompts. At the fixed 0.55
confidence threshold it achieved:

| Partition | Examples | Accuracy | Base fallback rate |
| --- | ---: | ---: | ---: |
| Validation | 52 | 98.1% | 23.1% |
| Test | 52 | 96.2% | 23.1% |

On the test set, math and systems were perfect. Code and instruction each had
one prompt confused with the other. These held-out sets are much more useful
than the 12-example recovery sets, but they remain too small for a narrow
confidence interval or a broad generalization claim.

## Artifact verification

The downloaded compact output contains a run manifest and step-15 trainer state
for each expert. Each final PEFT configuration reports:

- base model `Qwen/Qwen3-8B`;
- PEFT type `LORA` and task type `CAUSAL_LM`;
- rank 16, alpha 32, and dropout 0.05;
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj` target modules.

The kernel also passed the repository unit tests before training. The generated
adapter weights, tokenizers, checkpoints, router, source manifest, and reports
remain in the private Kaggle output. Machine-readable headline metrics are in
`benchmarks/kaggle_full_v1/summary.json`.

## What this result does and does not prove

This run proves that the complete dual-T4 training design works and produces
four loadable Qwen3-8B adapters. It also provides encouraging in-distribution
loss and routing results.

It does not yet prove that the MoLE system beats base Qwen3-8B on answer
quality or that its serving path is faster. The final quality and inference
gate must compare base, oracle-domain expert, automatically routed expert, and
wrong-expert controls on untouched prompts, then measure time to first token,
decode throughput, memory, and adapter-switch overhead. Until that is run, the
correct project claim is **successful full training**, not **validated quality
improvement**.
