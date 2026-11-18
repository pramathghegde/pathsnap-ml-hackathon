"""
Production inference module for PathSnap GPS map matching candidate ranking.

Provides:
- PathSnapPredictor: High-performance, fail-safe predictor for candidate scoring and ranking.
- load_predictor: Convenience factory to load serialized model and metadata.

Design Principles:
1. Strict schema validation: Rejects inputs missing mandatory predictive features.
2. Order independence: Automatically re-aligns input columns to match trained feature order.
3. Metadata-driven positive class resolution: Determines class 1 column using model.classes_
   rather than assuming binary column index.
4. Identifier preservation: Retains trip_id, observation_id, and candidate_arc_id throughout.
5. Target label agnostic: Never requires or references ground-truth labels during inference.
6. Real-time latency optimized: Capable of scoring candidate batches in sub-millisecond time.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


DEFAULT_PREDICTIVE_FEATURES = [
    "perp_dist",
    "proj_ratio",
    "delta_heading",
    "arc_length",
    "gps_speed",
    "speed_ratio",
    "traversal_rate",
    "candidate_rank",
    "is_adjacent",
]

DEFAULT_IDENTIFIER_COLS = [
    "trip_id",
    "observation_id",
    "candidate_arc_id",
]


class PathSnapPredictor:
    """
    Candidate emission probability predictor and Top-1 ranker for PathSnap map matching.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load serialized model artifact and corresponding metadata sidecar.

        Args:
            model_path: Path to serialized .joblib model file.
            metadata_path: Path to .json metadata file. If None, inferred by replacing
                           '.joblib' with '_metadata.json'.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Resolve metadata path
        if metadata_path is None:
            self.metadata_path = self.model_path.parent / (
                self.model_path.stem + "_metadata.json"
            )
        else:
            self.metadata_path = Path(metadata_path)

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found at: {self.metadata_path}"
            )

        # 1. Load model artifact
        self.model = joblib.load(self.model_path)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                f"Loaded model of type {type(self.model).__name__} does not implement 'predict_proba()'."
            )

        # 2. Load and validate metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self._validate_metadata()

        # 3. Cache feature schema and classes
        self.feature_names: List[str] = self.metadata.get(
            "feature_names", list(DEFAULT_PREDICTIVE_FEATURES)
        )
        self.model_name: str = self.metadata.get(
            "model_name", type(self.model).__name__
        )
        self.version: str = self.metadata.get("version", "1.0.0")

        # 4. Resolve positive class index dynamically
        self.pos_class_index = self._resolve_positive_class_index()

    def _validate_metadata(self) -> None:
        """Verify presence of essential metadata attributes."""
        required_keys = ["model_name", "feature_names", "classes"]
        missing = [k for k in required_keys if k not in self.metadata]
        if missing:
            raise ValueError(
                f"Metadata at {self.metadata_path} is missing mandatory keys: {missing}"
            )

    def _resolve_positive_class_index(self) -> int:
        """
        Dynamically determine column index corresponding to positive match class (1).
        """
        if hasattr(self.model, "classes_"):
            classes = self.model.classes_
        elif hasattr(self.model, "named_steps") and hasattr(
            self.model.named_steps.get("classifier", None), "classes_"
        ):
            classes = self.model.named_steps["classifier"].classes_
        else:
            classes = np.array(self.metadata.get("classes", [0, 1]))

        # Check for integer 1 or float 1.0
        matches = np.where((classes == 1) | (classes == 1.0))[0]
        if len(matches) > 0:
            return int(matches[0])

        if len(classes) == 2:
            return 1

        raise ValueError(
            f"Cannot resolve positive match class (1) in model classes: {classes}"
        )

    def validate_features(self, df: pd.DataFrame) -> None:
        """
        Assert that all required predictive features are present and non-null.

        Args:
            df: Input candidate DataFrame.

        Raises:
            ValueError: If input is empty, missing features, or contains NaNs.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(df).__name__}."
            )

        if df.empty:
            raise ValueError("Input candidate DataFrame is empty.")

        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"Input DataFrame is missing required features: {missing_features}. "
                f"Expected features ({len(self.feature_names)}): {self.feature_names}."
            )

        # Check for NaN / infinite values in predictive features
        na_counts = df[self.feature_names].isna().sum()
        cols_with_na = na_counts[na_counts > 0].to_dict()
        if cols_with_na:
            raise ValueError(
                f"Predictive features contain missing/NaN values: {cols_with_na}"
            )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute emission match probabilities for candidate road segments.

        Automatically enforces feature alignment to the model's expected schema.

        Args:
            df: Input DataFrame containing predictive feature columns.

        Returns:
            1D numpy array of probabilities p(is_match == 1) in [0.0, 1.0].
        """
        self.validate_features(df)

        # Strictly align feature columns to trained ordering
        X_aligned = df[self.feature_names]

        # Call underlying model predict_proba
        all_probas = self.model.predict_proba(X_aligned)

        # Extract positive class probability
        pos_probas = all_probas[:, self.pos_class_index].astype(np.float64)

        return pos_probas

    def predict_candidates(
        self,
        df: pd.DataFrame,
        id_cols: Sequence[str] = ("trip_id", "observation_id", "candidate_arc_id"),
        output_prob_col: str = "match_probability",
    ) -> pd.DataFrame:
        """
        Score candidate road segments and return identifiers alongside match probabilities.

        Args:
            df: Input candidate DataFrame containing features and identifier columns.
            id_cols: List of identifier column names to preserve.
            output_prob_col: Name for generated probability column.

        Returns:
            DataFrame containing preserved identifiers and predicted probabilities.
        """
        probabilities = self.predict_proba(df)

        # Retain all available identifier columns
        present_id_cols = [c for c in id_cols if c in df.columns]
        result_df = df[present_id_cols].copy()
        result_df[output_prob_col] = probabilities

        return result_df

    def rank_candidates(
        self,
        df: pd.DataFrame,
        group_cols: Sequence[str] = ("trip_id", "observation_id"),
        id_cols: Sequence[str] = ("trip_id", "observation_id", "candidate_arc_id"),
        tie_breaker_col: Optional[str] = "perp_dist",
        ascending_tie_breaker: bool = True,
        output_prob_col: str = "match_probability",
        output_rank_col: str = "model_rank",
    ) -> pd.DataFrame:
        """
        Score and rank candidate road segments within each GPS observation.

        Ties are broken deterministically using tie_breaker_col (e.g. perp_dist).

        Args:
            df: Input candidate DataFrame.
            group_cols: Columns defining unique observation groupings.
            id_cols: Identifier columns to preserve in output.
            tie_breaker_col: Column used for secondary deterministic tie-breaking.
            ascending_tie_breaker: Whether smaller tie-breaker value is preferred.
            output_prob_col: Name for output probability column.
            output_rank_col: Name for output rank column (1 = top candidate).

        Returns:
            DataFrame with candidates sorted and ranked by predicted probability.
        """
        scored_df = self.predict_candidates(
            df=df,
            id_cols=id_cols,
            output_prob_col=output_prob_col,
        )

        # Attach tie breaker if present in original df and not already in scored_df
        if tie_breaker_col is not None and tie_breaker_col in df.columns:
            if tie_breaker_col not in scored_df.columns:
                scored_df[tie_breaker_col] = df[tie_breaker_col].values

        # Sort criteria: group ascending, probability descending, tie_breaker ascending
        sort_cols = list(group_cols) + [output_prob_col]
        ascending_list = [True] * len(group_cols) + [False]

        if tie_breaker_col is not None and tie_breaker_col in scored_df.columns:
            sort_cols.append(tie_breaker_col)
            ascending_list.append(ascending_tie_breaker)

        ranked = scored_df.sort_values(
            by=sort_cols,
            ascending=ascending_list,
            kind="mergesort",
        ).copy()

        # Compute rank per observation group (1-indexed)
        ranked[output_rank_col] = (
            ranked.groupby(list(group_cols), sort=False).cumcount() + 1
        )

        return ranked

    def select_top1_candidates(
        self,
        df: pd.DataFrame,
        group_cols: Sequence[str] = ("trip_id", "observation_id"),
        id_cols: Sequence[str] = ("trip_id", "observation_id", "candidate_arc_id"),
        tie_breaker_col: Optional[str] = "perp_dist",
        ascending_tie_breaker: bool = True,
        output_prob_col: str = "match_probability",
    ) -> pd.DataFrame:
        """
        Select the single most probable candidate road segment (Top-1) per observation.

        Args:
            df: Input candidate DataFrame.
            group_cols: Columns defining unique observations.
            id_cols: Identifier columns to preserve.
            tie_breaker_col: Column for deterministic tie-breaking.
            ascending_tie_breaker: Whether smaller tie-breaker is preferred.
            output_prob_col: Name for probability column.

        Returns:
            DataFrame containing exactly one Top-1 candidate per GPS observation.
        """
        ranked = self.rank_candidates(
            df=df,
            group_cols=group_cols,
            id_cols=id_cols,
            tie_breaker_col=tie_breaker_col,
            ascending_tie_breaker=ascending_tie_breaker,
            output_prob_col=output_prob_col,
            output_rank_col="_rank",
        )

        top1_df = ranked[ranked["_rank"] == 1].drop(columns=["_rank"]).copy()
        return top1_df.reset_index(drop=True)


def load_predictor(
    model_dir: Optional[Union[str, Path]] = None,
    model_filename: str = "pathsnap_histgbdt.joblib",
) -> PathSnapPredictor:
    """
    Convenience loader to initialize PathSnapPredictor from the artifacts directory.

    Args:
        model_dir: Directory containing model artifacts. Defaults to 'artifacts/models'.
        model_filename: Filename of the joblib model artifact.

    Returns:
        Initialized and validated PathSnapPredictor.
    """
    if model_dir is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        resolved_dir = repo_root / "artifacts" / "models"
    else:
        resolved_dir = Path(model_dir)

    model_path = resolved_dir / model_filename
    return PathSnapPredictor(model_path=model_path)
