from pathlib import Path

from qwen3_mole.config import load_config


def test_project_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "kaggle.yaml"
    config = load_config(path)
    assert config.base_model == "Qwen/Qwen3-8B"
    assert config.domains == ("code", "math", "systems", "instruction")
