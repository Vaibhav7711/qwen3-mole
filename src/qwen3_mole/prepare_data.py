from __future__ import annotations

import argparse
from collections import Counter

from .config import load_config
from .data import read_jsonl, stratified_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split normalized MoLE JSONL")
    parser.add_argument("--input", required=True, help="JSONL with messages and domain fields")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    records = list(read_jsonl(args.input, config.domains))
    train, validation = stratified_split(records, args.validation_fraction, args.seed)
    data = config.section("data")
    write_jsonl(data["train_path"], train)
    write_jsonl(data["validation_path"], validation)
    print("train", len(train), dict(sorted(Counter(row["domain"] for row in train).items())))
    print("validation", len(validation), dict(sorted(Counter(row["domain"] for row in validation).items())))


if __name__ == "__main__":
    main()
