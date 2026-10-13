from __future__ import annotations

import json
import sys
from pathlib import Path


EXAMPLES = {
    "code": [
        ("Write a Python function that parses JSON safely.", "Use json.loads inside a try/except block."),
        ("Fix a TypeScript type error in an async function.", "Annotate the Promise return type and narrow the error."),
        ("Implement binary search in Rust.", "Use two indices and halve the search interval each iteration."),
        ("Review this SQL query for correctness.", "Check join cardinality, null behavior, and aggregation keys."),
        ("Add unit tests for a JavaScript parser.", "Cover valid input, malformed input, and boundary cases."),
        ("Refactor a recursive Python traversal.", "Use an explicit stack if recursion depth is a concern."),
    ],
    "math": [
        ("Prove that the square root of two is irrational.", "Assume a reduced fraction and derive that both terms are even."),
        ("Solve 3x + 7 = 25.", "Subtract seven and divide by three to obtain x = 6."),
        ("Differentiate x squared times sine x.", "Apply the product rule: 2x sin(x) + x^2 cos(x)."),
        ("Find the determinant of a two by two matrix.", "For entries a,b,c,d the determinant is ad-bc."),
        ("Explain Bayes theorem with an equation.", "P(A|B) = P(B|A)P(A)/P(B), when P(B) is nonzero."),
        ("Compute the sum of the first n integers.", "Pairing terms gives n(n+1)/2."),
    ],
    "systems": [
        ("Diagnose high disk latency on a Linux server.", "Inspect iostat, queue depth, saturation, and recent workload changes."),
        ("A Kubernetes pod keeps restarting. What should I check?", "Inspect pod events, container logs, probes, and memory limits."),
        ("Design a cache invalidation strategy.", "Choose ownership, TTLs, versioned keys, and an explicit invalidation path."),
        ("Why is this TCP service timing out?", "Check DNS, routing, firewall rules, listening sockets, and packet loss."),
        ("Plan a zero-downtime database migration.", "Use expand-and-contract changes with backward-compatible application releases."),
        ("Reduce tail latency in an API.", "Profile p95/p99 dependencies, bound queues, and add timeouts and backpressure."),
    ],
    "instruction": [
        ("Rewrite this paragraph in a concise professional tone.", "Remove repetition, preserve the facts, and use direct language."),
        ("Summarize a meeting into action items.", "List each decision, owner, deliverable, and due date."),
        ("Turn these notes into a polite email.", "Add a clear subject, brief context, request, and courteous close."),
        ("Explain this concept to a beginner.", "Start with the goal, define essential terms, and give one concrete example."),
        ("Create a checklist from these requirements.", "Convert every verifiable requirement into a short checkbox item."),
        ("Classify the sentiment of a customer message.", "Return the label and a brief evidence-based rationale."),
    ],
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_smoke_data.py OUTPUT.jsonl")
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for domain, examples in EXAMPLES.items():
            for prompt, answer in examples:
                row = {
                    "domain": domain,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                }
                stream.write(json.dumps(row) + "\n")
    print(f"wrote {sum(map(len, EXAMPLES.values()))} records to {output}")


if __name__ == "__main__":
    main()
