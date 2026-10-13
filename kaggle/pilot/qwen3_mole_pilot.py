"""Train a bounded four-expert Qwen3-MoLE pilot on Kaggle T4 x2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-pilot")
OUTPUT = Path("/kaggle/working/pilot-output")
STATUS_PATH = OUTPUT / "status.json"


def run(name: str, command: list[str], cwd: Path | None = None) -> None:
    print("+", name, "::", " ".join(command), flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    status["steps"][name] = {
        "exit_code": result.returncode,
        "seconds": round(time.time() - started, 1),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


OUTPUT.mkdir(parents=True, exist_ok=True)
status = {"steps": {}, "records_per_domain": 256, "domains": ["code", "math", "systems", "instruction"]}
STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")

project_files = list(INPUT_ROOT.rglob("pyproject.toml"))
if len(project_files) != 1:
    raise SystemExit(f"expected one project source, found {project_files}")
source_root = project_files[0].parent
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(source_root, WORK_ROOT)

run(
    "gpu_inventory",
    ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"],
)
run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], WORK_ROOT)
run("install_project", [sys.executable, "-m", "pip", "install", "-q", "-e", "."], WORK_ROOT)
run("unit_tests", [sys.executable, "-m", "pytest", "-q"], WORK_ROOT)
run(
    "build_data",
    [
        sys.executable,
        "-m",
        "qwen3_mole.build_data",
        "--per-source",
        "256",
        "--output",
        "data/raw/curated.jsonl",
        "--manifest",
        "artifacts/pilot/data-manifest.json",
    ],
    WORK_ROOT,
)
run("prepare_data", [sys.executable, "-m", "qwen3_mole.prepare_data", "--input", "data/raw/curated.jsonl"], WORK_ROOT)
run(
    "train_router",
    [sys.executable, "-m", "qwen3_mole.train_router", "--report", "artifacts/pilot/router-report.json"],
    WORK_ROOT,
)
run("train_code_math", ["bash", "scripts/train_pair.sh", "code", "math"], WORK_ROOT)
run("train_systems_instruction", ["bash", "scripts/train_pair.sh", "systems", "instruction"], WORK_ROOT)

for relative in ("artifacts/pilot", "artifacts/router", "artifacts/adapters"):
    source = WORK_ROOT / relative
    if source.exists():
        shutil.copytree(source, OUTPUT / relative, dirs_exist_ok=True)

status["complete"] = True
STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
print("PILOT COMPLETE", json.dumps(status, sort_keys=True), flush=True)
