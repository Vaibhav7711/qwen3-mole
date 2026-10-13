from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import load_config
from .data import prompt_text, read_jsonl
from .router import PromptRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the prompt-level expert router")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--report", help="optional JSON metrics output path")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config.section("data")
    router_config = config.section("router")
    train = list(read_jsonl(data["train_path"], config.domains))
    validation = list(read_jsonl(data["validation_path"], config.domains))
    test = list(read_jsonl(data["test_path"], config.domains))

    router = PromptRouter(
        confidence_threshold=float(router_config["confidence_threshold"]),
        max_features=int(router_config["max_features"]),
        ngram_range=(int(router_config["ngram_min"]), int(router_config["ngram_max"])),
        c=float(router_config["c"]),
    ).fit((prompt_text(record) for record in train), (record["domain"] for record in train))

    def evaluate(records: list[dict]) -> dict:
        routes = [router.route(prompt_text(record)) for record in records]
        truth = [record["domain"] for record in records]
        predictions = [route.predicted_domain for route in routes]
        return {
            "accuracy": accuracy_score(truth, predictions),
            "fallback_rate": sum(route.expert == "base" for route in routes) / len(routes),
            "examples": len(records),
            "confusion_matrix": confusion_matrix(
                truth, predictions, labels=router.classes_
            ).tolist(),
            "classification_report": classification_report(
                truth, predictions, output_dict=True, zero_division=0
            ),
        }

    validation_report = evaluate(validation)
    report = {
        "accuracy": validation_report["accuracy"],
        "fallback_rate": validation_report["fallback_rate"],
        "labels": list(router.classes_),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation": validation_report,
        "test": evaluate(test),
    }
    router.save(router_config["output_path"])
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
