from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: object
    validation_accuracy: float


def _candidate_models(random_state: int) -> dict[str, object]:
    return {
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "gboost": GradientBoostingClassifier(random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def select_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    random_state: int = 7,
) -> ModelResult:
    """Fit several candidates and keep the best validation scorer."""

    best: ModelResult | None = None
    for name, model in _candidate_models(random_state).items():
        model.fit(train_x, train_y)
        pred = model.predict(valid_x)
        score = accuracy_score(valid_y, pred)
        if best is None or score > best.validation_accuracy:
            best = ModelResult(name=name, model=model, validation_accuracy=score)
    assert best is not None
    return best


def probability_series(model: object, x: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[:, 1]
    else:
        raw = model.decision_function(x)
        probs = 1 / (1 + np.exp(-raw))
    return pd.Series(probs, index=x.index, name="prob_up")

