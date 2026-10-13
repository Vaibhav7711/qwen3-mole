"""Bounded, checkpointed recovery pilot for Kaggle T4 x2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-recovery")
OUTPUT = Path("/kaggle/working/recovery-output")
STATUS_PATH = OUTPUT / "status.json"
CONFIG = "configs/kaggle_recovery.yaml"
SMOKE_STEPS = "2"


OUTPUT.mkdir(parents=True, exist_ok=True)
status = {
    "complete": False,
    "steps": {},
    "records_per_domain": 64,
    "max_length": 512,
    "optimizer_steps_per_expert": int(SMOKE_STEPS),
    "domains": ["code", "math", "systems", "instruction"],
}


def save_status() -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run(name: str, command: list[str], cwd: Path | None = None) -> None:
    print("+", name, "::", " ".join(command), flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    status["steps"][name] = {
        "exit_code": result.returncode,
        "seconds": round(time.time() - started, 1),
    }
    save_status()
    if result.returncode:
        raise SystemExit(result.returncode)


def sync_artifacts() -> None:
    for relative in ("artifacts/recovery",):
        source = WORK_ROOT / relative
        if source.exists():
            shutil.copytree(source, OUTPUT / relative, dirs_exist_ok=True)
    save_status()


def train_pair(name: str, domain0: str, domain1: str) -> None:
    started = time.time()
    processes = []
    for gpu, domain in enumerate((domain0, domain1)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            sys.executable,
            "-m",
            "qwen3_mole.train_expert",
            "--domain",
            domain,
            "--config",
            CONFIG,
            "--smoke-steps",
            SMOKE_STEPS,
        ]
        print("+", f"train_{domain}", "::", " ".join(command), flush=True)
        processes.append((domain, subprocess.Popen(command, cwd=WORK_ROOT, env=environment)))

    exit_codes = {domain: process.wait() for domain, process in processes}
    status["steps"][name] = {
        "exit_codes": exit_codes,
        "seconds": round(time.time() - started, 1),
    }
    sync_artifacts()
    if any(exit_codes.values()):
        raise SystemExit(max(exit_codes.values()))


save_status()
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

# Some Kaggle/Python/PyArrow combinations hang during interpreter shutdown after
# streaming has completed. All output files are synchronously closed by main(),
# so exit immediately after a successful build instead of waiting on library
# cleanup threads. A real build failure still exits before os._exit(0).
run(
    "build_data",
    [
        sys.executable,
        "-c",
        (
            "import os,sys; from qwen3_mole.build_data import main; "
            "main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
        ),
        "--per-source",
        "64",
        "--output",
        "data/raw/recovery-curated.jsonl",
        "--manifest",
        "artifacts/recovery/data-manifest.json",
    ],
    WORK_ROOT,
)
run(
    "prepare_data",
    [
        sys.executable,
        "-m",
        "qwen3_mole.prepare_data",
        "--input",
        "data/raw/recovery-curated.jsonl",
        "--config",
        CONFIG,
    ],
    WORK_ROOT,
)
run(
    "train_router",
    [
        sys.executable,
        "-m",
        "qwen3_mole.train_router",
        "--config",
        CONFIG,
        "--report",
        "artifacts/recovery/router-report.json",
    ],
    WORK_ROOT,
)
sync_artifacts()
train_pair("train_code_math", "code", "math")
train_pair("train_systems_instruction", "systems", "instruction")

status["complete"] = True
sync_artifacts()
print("RECOVERY COMPLETE", json.dumps(status, sort_keys=True), flush=True)
