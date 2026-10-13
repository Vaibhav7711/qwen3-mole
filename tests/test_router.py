from qwen3_mole.router import PromptRouter


def test_router_learns_domains_and_falls_back() -> None:
    texts = [
        "write a Python function and debug this code",
        "implement a Rust iterator and unit tests",
        "solve this equation and prove the theorem",
        "calculate the integral and simplify the fraction",
        "diagnose CUDA memory and Linux kernel latency",
        "design a distributed database and inspect network packets",
        "rewrite this email in a friendly tone",
        "summarize this article and improve the prose",
    ]
    labels = ["code", "code", "math", "math", "systems", "systems", "instruction", "instruction"]
    router = PromptRouter(confidence_threshold=0.0, max_features=1_000).fit(texts, labels)
    assert router.route("debug a Python program").expert == "code"
    assert router.route("solve an algebra equation").expert == "math"

    strict = PromptRouter(confidence_threshold=1.0, max_features=1_000).fit(texts, labels)
    assert strict.route("ambiguous request").expert == "base"
