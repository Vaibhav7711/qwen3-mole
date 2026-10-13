"""Private Kaggle dual-T4 compatibility smoke test for Qwen3-MoLE."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SOURCE_ROOT = Path("/kaggle/input/qwen3-mole-source")
WORK_ROOT = Path("/kaggle/working/qwen3-mole")
OUTPUT = Path("/kaggle/working/smoke-output")
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
status = {"steps": {}, "requested_experts": ["code", "math"], "optimizer_steps": 3}
STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")

if not (SOURCE_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"missing project source: {SOURCE_ROOT}")
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(SOURCE_ROOT, WORK_ROOT)
status["steps"]["copy_source"] = {"exit_code": 0, "seconds": 0.0}
STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
run(
    "gpu_inventory",
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv",
    ],
)
run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], WORK_ROOT)
run("install_project", [sys.executable, "-m", "pip", "install", "-q", "-e", "."], WORK_ROOT)
run("unit_tests", [sys.executable, "-m", "pytest", "-q"], WORK_ROOT)
run("write_data", [sys.executable, "scripts/write_smoke_data.py", "data/smoke.jsonl"], WORK_ROOT)
run(
    "prepare_data",
    [
        sys.executable,
        "-m",
        "qwen3_mole.prepare_data",
        "--input",
        "data/smoke.jsonl",
        "--validation-fraction",
        "0.25",
    ],
    WORK_ROOT,
)
run("train_router", [sys.executable, "-m", "qwen3_mole.train_router"], WORK_ROOT)
run("dual_gpu_qlora", ["bash", "scripts/train_pair.sh", "code", "math", "3"], WORK_ROOT)

for relative in ("artifacts/router", "artifacts/adapters/code", "artifacts/adapters/math"):
    source = WORK_ROOT / relative
    if source.exists():
        shutil.copytree(source, OUTPUT / relative, dirs_exist_ok=True)

status["complete"] = True
STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
print("SMOKE COMPLETE", json.dumps(status, sort_keys=True), flush=True)
