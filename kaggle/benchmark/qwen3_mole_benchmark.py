"""Held-out quality and latency benchmark for the trained Qwen3-MoLE system."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-benchmark")
OUTPUT = Path("/kaggle/working/benchmark-output")
STATUS = OUTPUT / "status.json"
CONFIG = "configs/kaggle_full.yaml"
OUTPUT.mkdir(parents=True, exist_ok=True)
status = {"complete": False, "steps": {}}


def save_status() -> None:
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run(name: str, command: list[str], cwd: Path | None = None) -> None:
    print("+", name, "::", " ".join(command), flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    status["steps"][name] = {"exit_code": result.returncode, "seconds": round(time.time() - started, 1)}
    save_status()
    if result.returncode:
        raise SystemExit(result.returncode)


save_status()
projects = [
    path
    for path in INPUT_ROOT.rglob("pyproject.toml")
    if "datasets" in path.parts and "qwen3-mole-source" in path.parts
]
if len(projects) != 1:
    raise SystemExit(f"expected one project source, found {projects}")
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(projects[0].parent, WORK_ROOT)

production_statuses = []
for candidate in INPUT_ROOT.rglob("status.json"):
    try:
        value = json.loads(candidate.read_text())
    except Exception:
        continue
    if value.get("complete") and value.get("records_per_domain") == 2048:
        production_statuses.append(candidate)
if len(production_statuses) != 1:
    raise SystemExit(f"expected one completed production output, found {production_statuses}")
artifact_root = production_statuses[0].parent / "artifacts/full"
adapter_root = artifact_root / "adapters"
router_path = artifact_root / "router/router.joblib"
if not router_path.exists() or not adapter_root.exists():
    raise SystemExit(f"production artifacts incomplete below {artifact_root}")

run("gpu_inventory", ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"])
run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], WORK_ROOT)
run("install_project", [sys.executable, "-m", "pip", "install", "-q", "-e", "."], WORK_ROOT)
run("unit_tests", [sys.executable, "-m", "pytest", "-q"], WORK_ROOT)
run(
    "build_data",
    [
        sys.executable,
        "-c",
        "import os,sys; from qwen3_mole.build_data import main; main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)",
        "--per-source",
        "2048",
        "--output",
        "data/raw/benchmark-curated.jsonl",
        "--manifest",
        "artifacts/benchmark-data-manifest.json",
    ],
    WORK_ROOT,
)
run(
    "prepare_data",
    [sys.executable, "-m", "qwen3_mole.prepare_data", "--input", "data/raw/benchmark-curated.jsonl", "--config", CONFIG],
    WORK_ROOT,
)

started = time.time()
processes = []
for gpu, domains in enumerate((("code", "math"), ("systems", "instruction"))):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output = OUTPUT / f"worker-{'-'.join(domains)}.json"
    command = [
        sys.executable,
        "-m",
        "qwen3_mole.benchmark",
        "--domains",
        *domains,
        "--adapter-root",
        str(adapter_root),
        "--router",
        str(router_path),
        "--test-data",
        "data/processed/full-test.jsonl",
        "--output",
        str(output),
    ]
    print("+ benchmark", domains, "::", " ".join(command), flush=True)
    processes.append((domains, subprocess.Popen(command, cwd=WORK_ROOT, env=environment)))

exit_codes = {"-".join(domains): process.wait() for domains, process in processes}
status["steps"]["benchmark"] = {
    "exit_codes": exit_codes,
    "seconds": round(time.time() - started, 1),
}
save_status()
if any(exit_codes.values()):
    raise SystemExit(max(exit_codes.values()))

workers = [json.loads(path.read_text()) for path in sorted(OUTPUT.glob("worker-*.json"))]
summary = {
    "base_model": "Qwen/Qwen3-8B",
    "conditions": ["base", "oracle", "routed", "wrong"],
    "test_examples": sum(worker["test_examples"] for worker in workers),
    "workers": workers,
}
(OUTPUT / "benchmark-report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
status["complete"] = True
save_status()
print("BENCHMARK COMPLETE", json.dumps(status, sort_keys=True), flush=True)
