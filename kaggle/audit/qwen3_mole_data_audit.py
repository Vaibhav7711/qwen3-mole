"""Rebuild the production split and audit leakage and common benchmark overlap."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-audit")
OUTPUT = Path("/kaggle/working/audit-output")
STATUS = OUTPUT / "status.json"
OUTPUT.mkdir(parents=True, exist_ok=True)
status = {"complete": False, "steps": {}}


def save() -> None:
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run(name: str, command: list[str], cwd: Path) -> None:
    started = time.time()
    print("+", name, "::", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    status["steps"][name] = {"exit_code": result.returncode, "seconds": round(time.time() - started, 1)}
    save()
    if result.returncode:
        raise SystemExit(result.returncode)


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
    "build_data",
    [
        sys.executable,
        "-c",
        "import os,sys; from qwen3_mole.build_data import main; main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)",
        "--per-source",
        "2048",
        "--output",
        "data/raw/audit-curated.jsonl",
        "--manifest",
        str(OUTPUT / "data-manifest.json"),
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
        "data/raw/audit-curated.jsonl",
        "--config",
        "configs/kaggle_full.yaml",
    ],
    WORK_ROOT,
)
run(
    "audit",
    [
        sys.executable,
        "-m",
        "qwen3_mole.audit_data",
        "--train",
        "data/processed/full-train.jsonl",
        "--validation",
        "data/processed/full-validation.jsonl",
        "--test",
        "data/processed/full-test.jsonl",
        "--output",
        str(OUTPUT / "audit-report.json"),
    ],
    WORK_ROOT,
)
status["complete"] = True
save()
print("AUDIT COMPLETE", json.dumps(status, sort_keys=True), flush=True)
