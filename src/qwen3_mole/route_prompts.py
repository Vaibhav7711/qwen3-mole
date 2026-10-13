from __future__ import annotations

import argparse
import json
import sys

from .router import PromptRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach expert routes to prompt JSONL")
    parser.add_argument("--router", default="artifacts/router/router.joblib")
    parser.add_argument("--input", help="JSONL path; defaults to stdin")
    args = parser.parse_args()

    router = PromptRouter.load(args.router)
    stream = open(args.input) if args.input else sys.stdin
    try:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("prompt")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} needs a non-empty prompt")
            route = router.route(text)
            record["route"] = {
                "expert": route.expert,
                "predicted_domain": route.predicted_domain,
                "confidence": route.confidence,
                "probabilities": route.probabilities,
            }
            print(json.dumps(record, ensure_ascii=False))
    finally:
        if args.input:
            stream.close()


if __name__ == "__main__":
    main()
