"""
Model factory for PathSnap candidate classification.

Provides pipelines for five distinct model families:
1. Logistic Regression (Linear / Probabilistic, with StandardScaler)
2. K-Nearest Neighbors (Instance-Based, with StandardScaler)
3. Decision Tree (Tree-Based)
4. Random Forest (Ensemble / Bagging)
5. HistGradientBoosting (Gradient Boosting)

Scale-sensitive estimators (Logistic Regression, KNN) are wrapped in
`sklearn.pipeline.Pipeline` with `StandardScaler` to prevent leakage across folds.
Class weighting is configurable only for estimators natively supporting `class_weight`.
"""

from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

SUPPORTED_MODELS = (
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "HistGradientBoosting",
)


def get_model(
    name: str,
    random_state: int = 42,
    class_weight: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a single model or pipeline by name.

    Args:
        name: Name of the model family. Must be one of SUPPORTED_MODELS.
        random_state: Random seed for reproducibility.
        class_weight: Optional class weight ('balanced', dict, etc.).
                      Applied only to estimators that natively support class_weight:
                      Logistic Regression, Decision Tree, and Random Forest.
                      Ignored for KNN and HistGradientBoosting to respect their
                      estimator capabilities.
        **kwargs: Additional keyword arguments passed to the underlying classifier.

    Returns:
        Instantiated sklearn estimator or Pipeline.
    """
    if name == "Logistic Regression":
        clf_kwargs = {"max_iter": 1000, "random_state": random_state}
        if class_weight is not None:
            clf_kwargs["class_weight"] = class_weight
        clf_kwargs.update(kwargs)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**clf_kwargs)),
            ]
        )

    elif name == "KNN":
        clf_kwargs = {"n_neighbors": 5, "weights": "distance"}
        # Note: KNeighborsClassifier does not support class_weight.
        clf_kwargs.update(kwargs)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(**clf_kwargs)),
            ]
        )

    elif name == "Decision Tree":
        clf_kwargs = {"max_depth": None, "random_state": random_state}
        if class_weight is not None:
            clf_kwargs["class_weight"] = class_weight
        clf_kwargs.update(kwargs)
        return DecisionTreeClassifier(**clf_kwargs)

    elif name == "Random Forest":
        clf_kwargs = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": -1,
        }
        if class_weight is not None:
            clf_kwargs["class_weight"] = class_weight
        clf_kwargs.update(kwargs)
        return RandomForestClassifier(**clf_kwargs)

    elif name == "HistGradientBoosting":
        clf_kwargs = {
            "max_iter": 100,
            "random_state": random_state,
        }
        # Note: HistGradientBoostingClassifier does not take class_weight in __init__.
        clf_kwargs.update(kwargs)
        return HistGradientBoostingClassifier(**clf_kwargs)

    else:
        raise ValueError(
            f"Unsupported model name: '{name}'. Supported models: {list(SUPPORTED_MODELS)}"
        )


def get_models(
    random_state: int = 42,
    class_weight: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Return all five baseline classification models/pipelines.

    Args:
        random_state: Random seed for reproducibility.
        class_weight: Class weight parameter for supported estimators
                      (Logistic Regression, Decision Tree, Random Forest).
                      None produces unweighted baselines.

    Returns:
        Dictionary mapping model names to instantiated estimators or pipelines.
    """
    return {
        name: get_model(name, random_state=random_state, class_weight=class_weight)
        for name in SUPPORTED_MODELS
    }
