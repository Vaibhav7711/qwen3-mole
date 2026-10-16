"""Judge systems generations with two independent Mistral GPTQ replicas."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-systems-judge")
OUTPUT = Path("/kaggle/working/systems-judge-output")
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

semantic_statuses = []
for candidate in INPUT_ROOT.rglob("status.json"):
    try:
        value = json.loads(candidate.read_text())
    except Exception:
        continue
    if value.get("complete") and value.get("verification", {}).get("generation_records") == 1600:
        semantic_statuses.append(candidate)
if len(semantic_statuses) != 1:
    raise SystemExit(f"expected one semantic output, found {semantic_statuses}")
generations = semantic_statuses[0].parent / "generations-all.jsonl"

run("gpu_inventory", ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], WORK_ROOT)
run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], WORK_ROOT)
run("install_project", [sys.executable, "-m", "pip", "install", "-q", "-e", "."], WORK_ROOT)
run("unit_tests", [sys.executable, "-m", "pytest", "-q"], WORK_ROOT)
run("install_vllm", [sys.executable, "-m", "pip", "install", "-q", "vllm==0.31.0"], WORK_ROOT)
run("remove_incompatible_audio_extension", [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], WORK_ROOT)
run(
    "install_matching_torchvision",
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        "--no-deps",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "torchvision==0.28.0+cu130",
    ],
    WORK_ROOT,
)

started = time.time()
processes = []
for gpu in range(2):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    shard = OUTPUT / f"judge-shard-{gpu}.json"
    command = [
        sys.executable,
        "-m",
        "qwen3_mole.judge_systems",
        "--generations",
        str(generations),
        "--output",
        str(shard),
        "--shard-index",
        str(gpu),
        "--num-shards",
        "2",
    ]
    print("+ judge", gpu, "::", " ".join(command), flush=True)
    processes.append((gpu, subprocess.Popen(command, cwd=WORK_ROOT, env=environment)))
exit_codes = {str(gpu): process.wait() for gpu, process in processes}
status["steps"]["judge"] = {"exit_codes": exit_codes, "seconds": round(time.time() - started, 1)}
save()
if any(exit_codes.values()):
    raise SystemExit(max(exit_codes.values()))

from qwen3_mole.judge_systems import load_comparisons, select_manual_review, summarize_judgments

judgments = []
for path in sorted(OUTPUT.glob("judge-shard-*.json")):
    judgments.extend(json.loads(path.read_text())["comparisons"])
if len(judgments) != 300:
    raise SystemExit(f"expected 300 comparisons, found {len(judgments)}")
summary = summarize_judgments(judgments, 20261016)
(OUTPUT / "systems-judge-summary.json").write_text(
    json.dumps({"summary": summary, "comparisons": judgments}, indent=2, sort_keys=True) + "\n"
)

comparison_data = {
    item["comparison_id"]: item for item in load_comparisons(generations, 20261016)
}
review = select_manual_review(judgments, comparison_data, target=15)
(OUTPUT / "manual-review-sample.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
status["verification"] = {
    "comparisons": len(judgments),
    "judge_passes": sum(len(value["passes"]) for value in judgments),
    "manual_review_items": 15,
}
status["complete"] = True
save()
print("SYSTEMS JUDGE COMPLETE", json.dumps(status, sort_keys=True), flush=True)
