"""Rebuild production prompts and freeze the independent evaluation suite."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-independent-eval")
OUTPUT = Path("/kaggle/working/independent-eval-output")
STATUS = OUTPUT / "status.json"
OUTPUT.mkdir(parents=True, exist_ok=True)
status = {"complete": False, "steps": {}}


def save() -> None:
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run(name: str, command: list[str], cwd: Path) -> None:
    started = time.time()
    print("+", name, "::", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    status["steps"][name] = {
        "exit_code": result.returncode,
        "seconds": round(time.time() - started, 1),
    }
    save()
    if result.returncode:
        raise SystemExit(result.returncode)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


save()
projects = [path for path in INPUT_ROOT.rglob("pyproject.toml") if "qwen3-mole-source" in path.parts]
if len(projects) != 1:
    raise SystemExit(f"expected one source project, found {projects}")
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(projects[0].parent, WORK_ROOT)
run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], WORK_ROOT)
run("install_project", [sys.executable, "-m", "pip", "install", "-q", "-e", "."], WORK_ROOT)
run("unit_tests", [sys.executable, "-m", "pytest", "-q"], WORK_ROOT)
run(
    "build_production_data",
    [
        sys.executable,
        "-c",
        "import os,sys; from qwen3_mole.build_data import main; main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)",
        "--per-source",
        "2048",
        "--output",
        "data/raw/eval-curated.jsonl",
        "--manifest",
        str(OUTPUT / "production-data-manifest.json"),
    ],
    WORK_ROOT,
)
run(
    "prepare_production_split",
    [
        sys.executable,
        "-m",
        "qwen3_mole.prepare_data",
        "--input",
        "data/raw/eval-curated.jsonl",
        "--config",
        "configs/kaggle_full.yaml",
    ],
    WORK_ROOT,
)
run(
    "build_independent_eval",
    [
        sys.executable,
        "-m",
        "qwen3_mole.build_eval",
        "--train",
        "data/processed/full-train.jsonl",
        "--output",
        str(OUTPUT / "independent-v1.jsonl"),
        "--manifest",
        str(OUTPUT / "independent-v1-manifest.json"),
        "--report",
        str(OUTPUT / "independent-v1-decontamination.json"),
    ],
    WORK_ROOT,
)

manifest = json.loads((OUTPUT / "independent-v1-manifest.json").read_text())
report = json.loads((OUTPUT / "independent-v1-decontamination.json").read_text())
records = [json.loads(line) for line in (OUTPUT / "independent-v1.jsonl").read_text().splitlines() if line]
if manifest.get("builder_version") != 2 or report.get("builder_version") != 2:
    raise SystemExit("expected corrected evaluation builder version 2")
counts = Counter(record["domain"] for record in records)
expected = {"code": 100, "math": 100, "systems": 100, "instruction": 100}
if dict(counts) != expected:
    raise SystemExit(f"unexpected domain counts: {dict(counts)}")
if manifest["output_sha256"] != sha256_file(OUTPUT / "independent-v1.jsonl"):
    raise SystemExit("evaluation output hash does not match manifest")
thresholds = {domain: values["max_train_similarity"] for domain, values in manifest["decontamination"].items()}
violations = [
    record["id"]
    for record in records
    if record["decontamination"]["max_train_similarity"] >= thresholds[record["domain"]]
]
if violations:
    raise SystemExit(f"selected records violate contamination thresholds: {violations[:10]}")
if any(values.get("exact_train_overlap", 0) for values in report["stats"].values()):
    print("Exact overlaps were detected and excluded, as intended.", flush=True)

status["verification"] = {
    "records": len(records),
    "domain_counts": dict(sorted(counts.items())),
    "output_sha256": manifest["output_sha256"],
    "threshold_violations": len(violations),
    "maximum_selected_similarity": {
        domain: max(
            record["decontamination"]["max_train_similarity"]
            for record in records
            if record["domain"] == domain
        )
        for domain in sorted(expected)
    },
}
status["complete"] = True
save()
print("INDEPENDENT EVALUATION SET COMPLETE", json.dumps(status, sort_keys=True), flush=True)
