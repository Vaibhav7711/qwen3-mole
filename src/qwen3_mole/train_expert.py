from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import read_jsonl
from .tokenization import CausalLMCollator, tokenize_chat_with_assistant_labels


def require_training_dependencies() -> None:
    missing = []
    for module in ("accelerate", "bitsandbytes", "datasets", "peft", "transformers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "missing training dependencies: " + ", ".join(missing) + "; install requirements-kaggle.txt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Qwen3-8B LoRA expert")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args()
    require_training_dependencies()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = load_config(args.config)
    if args.domain not in config.domains:
        raise SystemExit(f"unknown domain {args.domain!r}; choose from {', '.join(config.domains)}")
    data_config = config.section("data")
    lora_config = config.section("lora")
    training = config.section("training")
    set_seed(int(training["seed"]))

    train_records = [
        record
        for record in read_jsonl(data_config["train_path"], config.domains)
        if record["domain"] == args.domain
    ]
    validation_records = [
        record
        for record in read_jsonl(data_config["validation_path"], config.domains)
        if record["domain"] == args.domain
    ]
    if not train_records or not validation_records:
        raise SystemExit(f"domain {args.domain!r} needs train and validation records")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(data_config["max_length"])

    def build_dataset(records: list[dict]) -> Dataset:
        rows = [
            tokenize_chat_with_assistant_labels(tokenizer, record["messages"], max_length)
            for record in records
        ]
        return Dataset.from_list(rows)

    train_dataset = build_dataset(train_records)
    validation_dataset = build_dataset(validation_records)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
        use_cache=False,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )
    adapter = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, adapter, adapter_name=args.domain)
    model.print_trainable_parameters()

    output_dir = Path(training["output_root"]) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "base_model": config.base_model,
        "domain": args.domain,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "config": config.raw,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        logging_steps=int(training["logging_steps"]),
        save_steps=int(training["save_steps"]),
        eval_steps=int(training["eval_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        max_grad_norm=float(training["max_grad_norm"]),
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=int(training["seed"]),
        max_steps=args.smoke_steps if args.smoke_steps is not None else -1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
