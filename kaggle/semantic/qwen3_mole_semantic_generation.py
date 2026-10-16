"""Generate and task-score independent base/oracle/routed/wrong answers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/qwen3-mole-semantic")
OUTPUT = Path("/kaggle/working/semantic-output")
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


save()
projects = [path for path in INPUT_ROOT.rglob("pyproject.toml") if "qwen3-mole-source" in path.parts]
if len(projects) != 1:
    raise SystemExit(f"expected one source project, found {projects}")
if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(projects[0].parent, WORK_ROOT)

production_statuses = []
evaluation_statuses = []
for candidate in INPUT_ROOT.rglob("status.json"):
    try:
        value = json.loads(candidate.read_text())
    except Exception:
        continue
    if value.get("complete") and value.get("records_per_domain") == 2048:
        production_statuses.append(candidate)
    if value.get("complete") and value.get("verification", {}).get("records") == 400:
        evaluation_statuses.append(candidate)
if len(production_statuses) != 1:
    raise SystemExit(f"expected one production output, found {production_statuses}")
if len(evaluation_statuses) != 1:
    raise SystemExit(f"expected one independent evaluation output, found {evaluation_statuses}")
artifact_root = production_statuses[0].parent / "artifacts/full"
eval_data = evaluation_statuses[0].parent / "independent-v1.jsonl"

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
run(
    "validate_vllm_import",
    [
        sys.executable,
        "-c",
        (
            "import torch,torchvision; from vllm import LLM, SamplingParams; "
            "print(torch.__version__, torch.version.cuda, torchvision.__version__)"
        ),
    ],
    WORK_ROOT,
)

started = time.time()
processes = []
for gpu, domains in enumerate((("code", "math"), ("systems", "instruction"))):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    worker_output = OUTPUT / f"generations-{'-'.join(domains)}.jsonl"
    command = [
        sys.executable,
        "-m",
        "qwen3_mole.generate_semantic",
        "--domains",
        *domains,
        "--adapter-root",
        str(artifact_root / "adapters"),
        "--router",
        str(artifact_root / "router/router.joblib"),
        "--eval-data",
        str(eval_data),
        "--output",
        str(worker_output),
    ]
    print("+ generate", domains, "::", " ".join(command), flush=True)
    processes.append((domains, subprocess.Popen(command, cwd=WORK_ROOT, env=environment)))
exit_codes = {"-".join(domains): process.wait() for domains, process in processes}
status["steps"]["generation"] = {
    "exit_codes": exit_codes,
    "seconds": round(time.time() - started, 1),
}
save()
if any(exit_codes.values()):
    raise SystemExit(max(exit_codes.values()))

generation_files = sorted(OUTPUT.glob("generations-*.jsonl"))
generations = []
for path in generation_files:
    generations.extend(json.loads(line) for line in path.read_text().splitlines() if line)
if len(generations) != 1600:
    raise SystemExit(f"expected 1600 generations, found {len(generations)}")
combined = OUTPUT / "generations-all.jsonl"
with combined.open("w") as stream:
    for record in sorted(generations, key=lambda value: (value["id"], value["condition"])):
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

ifeval_revision = "3ef52111c1fabc301880818f5a547570e926a76b"
run(
    "clone_ifeval",
    [
        "git",
        "clone",
        "--quiet",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/google-research/google-research.git",
        "ifeval-source",
    ],
    WORK_ROOT,
)
run("checkout_ifeval", ["git", "-C", "ifeval-source", "checkout", "--quiet", ifeval_revision, "--", "instruction_following_eval"], WORK_ROOT)
run(
    "install_ifeval_dependencies",
    [sys.executable, "-m", "pip", "install", "-q", "absl-py", "langdetect", "nltk", "immutabledict"],
    WORK_ROOT,
)
run(
    "download_ifeval_nltk_data",
    [sys.executable, "-m", "nltk.downloader", "punkt", "punkt_tab"],
    WORK_ROOT,
)
run(
    "score_task_domains",
    [
        sys.executable,
        "-m",
        "qwen3_mole.score_semantic",
        "--generations",
        str(combined),
        "--ifeval-root",
        str(WORK_ROOT / "ifeval-source"),
        "--output",
        str(OUTPUT / "task-scores.json"),
    ],
    WORK_ROOT,
)
scores = json.loads((OUTPUT / "task-scores.json").read_text())
expected_scored = 3 * 100 * 4
if scores["scored_records"] != expected_scored:
    raise SystemExit(f"expected {expected_scored} task scores, found {scores['scored_records']}")
status["verification"] = {
    "evaluation_records": 400,
    "generation_records": len(generations),
    "task_scored_records": scores["scored_records"],
    "systems_pending_judge_records": 400,
}
status["complete"] = True
save()
print("SEMANTIC GENERATION COMPLETE", json.dumps(status, sort_keys=True), flush=True)
