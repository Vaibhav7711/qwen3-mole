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
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    records = list(read_jsonl(args.input, config.domains))
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise SystemExit("validation and test fractions must sum to less than one")
    train_and_validation, test = stratified_split(records, args.test_fraction, args.seed)
    relative_validation_fraction = args.validation_fraction / (1.0 - args.test_fraction)
    train, validation = stratified_split(
        train_and_validation, relative_validation_fraction, args.seed + 1
    )
    data = config.section("data")
    write_jsonl(data["train_path"], train)
    write_jsonl(data["validation_path"], validation)
    write_jsonl(data["test_path"], test)
    print("train", len(train), dict(sorted(Counter(row["domain"] for row in train).items())))
    print("validation", len(validation), dict(sorted(Counter(row["domain"] for row in validation).items())))
    print("test", len(test), dict(sorted(Counter(row["domain"] for row in test).items())))


if __name__ == "__main__":
    main()
