from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Route:
    expert: str
    predicted_domain: str
    confidence: float
    probabilities: dict[str, float]


class PromptRouter:
    """Fast learned baseline router with confidence-based base fallback."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        c: float = 4.0,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=True,
                        ngram_range=ngram_range,
                        max_features=max_features,
                        min_df=1,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, texts: Iterable[str], domains: Iterable[str]) -> "PromptRouter":
        text_values = list(texts)
        domain_values = list(domains)
        if len(text_values) != len(domain_values) or not text_values:
            raise ValueError("texts and domains must have the same non-zero length")
        if len(set(domain_values)) < 2:
            raise ValueError("router training requires at least two domains")
        self.pipeline.fit(text_values, domain_values)
        return self

    @property
    def classes_(self) -> np.ndarray:
        classifier = self.pipeline.named_steps["classifier"]
        return classifier.classes_

    def route(self, text: str) -> Route:
        probabilities = self.pipeline.predict_proba([text])[0]
        best_index = int(np.argmax(probabilities))
        predicted = str(self.classes_[best_index])
        confidence = float(probabilities[best_index])
        expert = predicted if confidence >= self.confidence_threshold else "base"
        return Route(
            expert=expert,
            predicted_domain=predicted,
            confidence=confidence,
            probabilities={str(label): float(value) for label, value in zip(self.classes_, probabilities)},
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "PromptRouter":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"{path} does not contain a PromptRouter")
        return value
