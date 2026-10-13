from __future__ import annotations

import argparse
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import load_config
from .data import prompt_text, read_jsonl
from .router import PromptRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the prompt-level expert router")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config.section("data")
    router_config = config.section("router")
    train = list(read_jsonl(data["train_path"], config.domains))
    validation = list(read_jsonl(data["validation_path"], config.domains))

    router = PromptRouter(
        confidence_threshold=float(router_config["confidence_threshold"]),
        max_features=int(router_config["max_features"]),
        ngram_range=(int(router_config["ngram_min"]), int(router_config["ngram_max"])),
        c=float(router_config["c"]),
    ).fit((prompt_text(record) for record in train), (record["domain"] for record in train))

    routes = [router.route(prompt_text(record)) for record in validation]
    truth = [record["domain"] for record in validation]
    predictions = [route.predicted_domain for route in routes]
    fallback_rate = sum(route.expert == "base" for route in routes) / len(routes)
    report = {
        "accuracy": accuracy_score(truth, predictions),
        "fallback_rate": fallback_rate,
        "labels": list(router.classes_),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=router.classes_).tolist(),
        "classification_report": classification_report(truth, predictions, output_dict=True, zero_division=0),
        "train_examples": len(train),
        "validation_examples": len(validation),
    }
    router.save(router_config["output_path"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
