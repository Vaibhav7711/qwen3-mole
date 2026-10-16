from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .audit_data import normalize_text
from .data import read_jsonl


BUILDER_VERSION = 1
BLOCK_TAGS = {
    "address",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "tr",
    "ul",
}
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_code = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag in {"code", "pre"}:
            self.in_code += 1
            self.parts.append("`" if tag == "code" else "\n```\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"code", "pre"}:
            self.parts.append("`" if tag == "code" else "\n```\n")
            self.in_code = max(0, self.in_code - 1)
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    result = html.unescape("".join(parser.parts)).replace("\r", "")
    result = "\n".join(WHITESPACE.sub(" ", line).strip() for line in result.splitlines())
    return BLANK_LINES.sub("\n\n", result).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(source: dict[str, Any], source_id: str, source_url: str | None = None) -> dict[str, Any]:
    result = {
        "source": source["repository"],
        "revision": source["revision"],
        "license": source["license"],
        "source_id": source_id,
        "homepage": source["homepage"],
    }
    if source_url:
        result["source_url"] = source_url
    return result


def adapt_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    kind = source["kind"]
    domain = source["domain"]
    if kind == "humaneval":
        source_id = str(row.get("task_id", ""))
        raw_prompt = row.get("prompt")
        solution = row.get("canonical_solution")
        tests = row.get("test")
        entry_point = row.get("entry_point")
        if not all(isinstance(value, str) and value.strip() for value in (source_id, raw_prompt, solution, tests, entry_point)):
            return None
        prompt = "Complete the following Python function. Return only the completed code.\n\n" + raw_prompt.strip()
        task = {
            "canonical_solution": solution,
            "tests": tests,
            "entry_point": entry_point,
        }
        reference = raw_prompt + solution
        provenance = _provenance(source, source_id)
    elif kind == "gsm8k":
        raw_prompt = row.get("question")
        reference = row.get("answer")
        if not isinstance(raw_prompt, str) or not isinstance(reference, str):
            return None
        source_id = sha256_text(normalize_text(raw_prompt))[:16]
        prompt = raw_prompt.strip()
        match = re.search(r"####\s*([^\n]+)\s*$", reference)
        if not match:
            return None
        task = {"final_answer": match.group(1).strip()}
        provenance = _provenance(source, source_id)
    elif kind == "stackexchange":
        question = row.get("question")
        answers = row.get("answers")
        metadata = row.get("metadata")
        qid = row.get("qid")
        if not isinstance(question, str) or not isinstance(answers, list) or qid is None:
            return None
        usable = [answer for answer in answers if isinstance(answer, dict) and isinstance(answer.get("text"), str)]
        if not usable:
            return None
        chosen = max(
            usable,
            key=lambda answer: (bool(answer.get("selected")), int(answer.get("pm_score") or 0)),
        )
        prompt = html_to_text(question)
        reference = html_to_text(chosen["text"])
        if not prompt or not reference:
            return None
        source_id = str(qid)
        source_url = metadata[0] if isinstance(metadata, list) and metadata and isinstance(metadata[0], str) else None
        task = {
            "answer_id": chosen.get("answer_id"),
            "answer_score": chosen.get("pm_score"),
            "answer_selected": bool(chosen.get("selected")),
        }
        provenance = _provenance(source, source_id, source_url)
        provenance["answer_author"] = chosen.get("author")
        provenance["group_id"] = f"serverfault:{qid}"
    elif kind == "ifeval":
        raw_prompt = row.get("prompt")
        key = row.get("key")
        instruction_ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")
        if not isinstance(raw_prompt, str) or key is None or not isinstance(instruction_ids, list) or not isinstance(kwargs, list):
            return None
        prompt = raw_prompt.strip()
        reference = canonical_json({"instruction_id_list": instruction_ids, "kwargs": kwargs})
        source_id = str(key)
        task = {"instruction_id_list": instruction_ids, "kwargs": kwargs}
        provenance = _provenance(source, source_id)
    else:
        raise ValueError(f"unknown evaluation source kind: {kind}")

    record = {
        "id": f"{source['name']}:{source_id}",
        "domain": domain,
        "prompt": prompt,
        "reference": reference,
        "task": task,
        "evaluation_method": source["evaluation_method"],
        "provenance": provenance,
    }
    record["content_sha256"] = sha256_text(canonical_json(record))
    return record


def load_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required; install the train dependencies") from exc

    if source["kind"] == "stackexchange":
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("pyarrow and huggingface_hub are required for Stack Exchange") from exc
        path = hf_hub_download(
            repo_id=source["repository"],
            filename=source["data_file"],
            repo_type="dataset",
            revision=source["revision"],
        )
        rows: Iterable[dict[str, Any]] = pq.read_table(path).to_pylist()
    else:
        rows = load_dataset(
            source["repository"],
            source.get("config"),
            split=source["split"],
            revision=source["revision"],
        )
    return [record for row in rows if (record := adapt_row(dict(row), source)) is not None]


def quality_reason(record: dict[str, Any], quality: dict[str, Any]) -> str | None:
    prompt_length = len(record["prompt"])
    reference_length = len(record["reference"])
    if prompt_length < int(quality["min_prompt_chars"]):
        return "prompt_too_short"
    if prompt_length > int(quality["max_prompt_chars"]):
        return "prompt_too_long"
    if reference_length < int(quality["min_reference_chars"]):
        return "reference_too_short"
    if reference_length > int(quality["max_reference_chars"]):
        return "reference_too_long"
    return None


def nearest_train_similarities(
    train_prompts: list[str], candidate_prompts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not train_prompts or not candidate_prompts:
        raise ValueError("training and candidate prompts must both be non-empty")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform([normalize_text(value) for value in train_prompts])
    candidate_matrix = vectorizer.transform([normalize_text(value) for value in candidate_prompts])
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(train_matrix)
    distances, indices = neighbors.kneighbors(candidate_matrix)
    return (1.0 - distances[:, 0]).astype(float), indices[:, 0].astype(int)


def stable_rank(record: dict[str, Any], seed: int) -> str:
    return sha256_text(f"{seed}:{record['id']}:{record['content_sha256']}")


def training_query_text(record: dict[str, Any]) -> str:
    """Return pre-answer content without role labels for contamination checks."""
    parts = []
    for message in record["messages"]:
        if message["role"] == "assistant":
            break
        parts.append(message["content"].strip())
    return "\n".join(parts)


def build_suite(
    train_records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_prompts = [training_query_text(record) for record in train_records]
    train_normalized = {normalize_text(value) for value in train_prompts}
    candidates: list[dict[str, Any]] = []
    stats: dict[str, Counter[str]] = {domain: Counter() for domain in config["domains"]}
    seen_eval: set[str] = set()

    for record in source_records:
        domain = record["domain"]
        stats[domain]["loaded"] += 1
        reason = quality_reason(record, config["quality"])
        if reason:
            stats[domain][reason] += 1
            continue
        normalized = normalize_text(record["prompt"])
        if normalized in seen_eval:
            stats[domain]["duplicate_eval_prompt"] += 1
            continue
        seen_eval.add(normalized)
        if normalized in train_normalized:
            stats[domain]["exact_train_overlap"] += 1
            continue
        candidates.append(record)

    pool_limit = int(config["target_per_domain"]) * int(config.get("candidate_pool_multiplier", 10))
    pooled: list[dict[str, Any]] = []
    for domain in sorted(config["domains"]):
        domain_candidates = [record for record in candidates if record["domain"] == domain]
        domain_candidates.sort(key=lambda record: stable_rank(record, int(config["seed"])))
        pooled.extend(domain_candidates[:pool_limit])
        stats[domain]["candidate_pool"] = min(len(domain_candidates), pool_limit)
        stats[domain]["candidate_pool_excluded"] = max(0, len(domain_candidates) - pool_limit)
    candidates = pooled

    similarities, nearest_indices = nearest_train_similarities(
        train_prompts, [record["prompt"] for record in candidates]
    )
    retained: dict[str, list[dict[str, Any]]] = {domain: [] for domain in config["domains"]}
    review: list[dict[str, Any]] = []
    for record, similarity, train_index in zip(candidates, similarities, nearest_indices):
        domain = record["domain"]
        threshold = float(config["domains"][domain]["max_train_similarity"])
        decision = "retained" if similarity < threshold else "near_train_overlap"
        stats[domain][decision] += 1
        record = {**record, "decontamination": {"max_train_similarity": round(float(similarity), 8)}}
        if decision == "retained":
            retained[domain].append(record)
        if decision != "retained" or similarity >= min(threshold - 0.10, 0.75):
            review.append(
                {
                    "id": record["id"],
                    "domain": domain,
                    "decision": decision,
                    "threshold": threshold,
                    "similarity": round(float(similarity), 8),
                    "evaluation_prompt": record["prompt"][:500],
                    "nearest_train_index": int(train_index),
                    "nearest_train_prompt": train_prompts[int(train_index)][:500],
                }
            )

    selected: list[dict[str, Any]] = []
    target = int(config["target_per_domain"])
    seed = int(config["seed"])
    for domain in sorted(retained):
        ranked = sorted(retained[domain], key=lambda record: stable_rank(record, seed))
        if len(ranked) < target:
            raise RuntimeError(f"{domain} retained {len(ranked)} clean records; {target} required")
        selected.extend(ranked[:target])
        stats[domain]["selected"] = target
    selected.sort(key=lambda record: (record["domain"], stable_rank(record, seed)))
    report = {
        "builder_version": BUILDER_VERSION,
        "train_records": len(train_records),
        "train_prompt_sha256": sha256_text(canonical_json(sorted(train_normalized))),
        "stats": {domain: dict(sorted(values.items())) for domain, values in sorted(stats.items())},
        "review": sorted(review, key=lambda item: (-item["similarity"], item["id"])),
    }
    return selected, report


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(record) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pinned independent MoLE evaluation suite")
    parser.add_argument("--sources", default="configs/eval_sources.yaml")
    parser.add_argument("--train", required=True, help="production training JSONL")
    parser.add_argument("--output", default="data/eval/independent-v1.jsonl")
    parser.add_argument("--manifest", default="artifacts/eval/independent-v1-manifest.json")
    parser.add_argument("--report", default="artifacts/eval/independent-v1-decontamination.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.sources).read_text())
    train_records = list(read_jsonl(args.train))
    source_records: list[dict[str, Any]] = []
    source_stats = []
    for source in config["sources"]:
        records = load_source(source)
        source_records.extend(records)
        source_stats.append({**source, "loaded_records": len(records)})
        print(f"loaded {len(records)} records from {source['name']}", flush=True)

    selected, report = build_suite(train_records, source_records, config)
    count = write_jsonl(args.output, selected)
    output_hash = sha256_file(args.output)
    domain_counts = dict(sorted(Counter(record["domain"] for record in selected).items()))
    manifest = {
        "version": config["version"],
        "builder_version": BUILDER_VERSION,
        "seed": config["seed"],
        "target_per_domain": config["target_per_domain"],
        "candidate_pool_multiplier": config.get("candidate_pool_multiplier", 10),
        "records": count,
        "domain_counts": domain_counts,
        "output_sha256": output_hash,
        "train_prompt_sha256": report["train_prompt_sha256"],
        "sources": source_stats,
        "decontamination": config["domains"],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": count, "domain_counts": domain_counts, "sha256": output_hash}, indent=2))


if __name__ == "__main__":
    main()
