import pytest

from qwen3_mole.review_agreement import agreement_metrics


def test_agreement_metrics_perfect() -> None:
    result = agreement_metrics(["routed", "base", "tie"], ["routed", "base", "tie"])
    assert result["exact_agreement"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_agreement_metrics_detects_disagreement() -> None:
    result = agreement_metrics(["routed", "routed", "base"], ["base", "routed", "base"])
    assert result["exact_agreement"] == pytest.approx(2 / 3)
    assert result["cohen_kappa"] < result["exact_agreement"]


def test_agreement_requires_aligned_nonempty_labels() -> None:
    with pytest.raises(ValueError):
        agreement_metrics([], [])
