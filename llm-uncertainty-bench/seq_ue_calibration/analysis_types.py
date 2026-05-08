"""Type definitions for the pickled prepared data produced by analysis_prepare_data.

Usage in visualization notebook::

    import pickle
    from analysis_types import PreparedData

    with open(PREPARED_DATA_PATH, "rb") as fh:
        prepared_data: PreparedData = pickle.load(fh)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from typing_extensions import TypedDict

import numpy as np
import pandas as pd


class ArithmeticMetrics(TypedDict):
    """Metrics for arithmetic / free-form answer datasets (e.g. GSM8K, SciBench)."""

    totally_correct_questions: int
    """Number of questions answered correctly in all 10 iterations."""

    accuracy_questions: float
    """Fraction of questions that are totally correct."""

    accuracy: float
    """Overall per-item accuracy across all iterations."""

    different_answer_count: Dict[int, int]
    """Frequency distribution: number_of_unique_answers → occurrence_count."""

    different_answer_count_mean: float
    """Weighted mean of unique-answer counts per question."""

    different_answer_count_std: float
    """Weighted standard deviation of unique-answer counts per question."""


class MCMetrics(TypedDict):
    """Metrics for multiple-choice question (MCQA) datasets (e.g. MMLU)."""

    # --- Binary classification (yes/no per choice) ---
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    accuracy: float
    f1: float

    # --- Choice / question level ---
    totally_correct_choices: int
    """Number of choices answered correctly in all 10 iterations."""

    accuracy_choices: float
    """Fraction of totally-correct choices."""

    totally_correct_questions: int
    """Number of questions where all 4 choices are totally correct."""

    accuracy_questions: float
    """Fraction of totally-correct questions."""


class CalibrationEntry(TypedDict):
    """Calibration and discrimination metrics for one UQ method on one model+dataset."""

    bin_confidences: np.ndarray
    """Centre confidence value of each calibration bin."""

    bucket_accuracies: np.ndarray
    """Mean accuracy of items falling into each bin."""

    bucket_counts: np.ndarray
    """Number of items in each bin."""

    correct: np.ndarray
    """Boolean correctness array (after dropping NaN UQ scores)."""

    certainties: np.ndarray
    """UQ certainty scores, after optional inversion for uncertainty-type methods."""

    ece: float
    """Expected Calibration Error."""

    auroc: float
    """Area Under the ROC Curve."""

    accuracy: float
    """Overall accuracy for items with a valid UQ score."""

    average_certainty: float
    """Mean certainty score across all items."""

    normalized_entropy: float
    """Normalized entropy of the bucket-count distribution (15 buckets)."""

    normalized_entropy_100: float
    """Normalized entropy of the bucket-count distribution (100 buckets)."""

    invalid_uq_method_scores: int
    """Number of rows dropped because the UQ method produced NaN."""

    relplot_diagram: Any
    """Precomputed relplot diagram object; pass directly to ``relplot.plot_rel_diagram``."""


class ModelDatasetEntry(TypedDict):
    """All prepared data for a single (model, dataset) pair."""

    df: pd.DataFrame
    """Merged per-item DataFrame.
    For MC datasets this also contains boolean columns y_pred and y_true."""

    is_arithmetic: bool
    """True for arithmetic/free-form datasets; False for multiple-choice datasets."""

    total_items: int
    """Total number of rows in the raw merged DataFrame (before any filtering)."""

    invalid_answers: int
    """Rows removed by filter_valid_answers + dropna(is_correct)."""

    # Only present when is_arithmetic is True:
    arithmetic_metrics: ArithmeticMetrics

    # Only present when is_arithmetic is False:
    mc_metrics: MCMetrics

    cal: Dict[str, Optional[CalibrationEntry]]
    """Calibration entries keyed by UQ method id (e.g. "verbalized", "p_true").
    Value is None when computation failed (e.g. all-NaN scores, AUROC error)."""


#: Top-level type of the pickled prepared data file.
#: ``prepared_data[dataset_id][model_id]`` → :class:`ModelDatasetEntry`
PreparedData = Dict[str, Dict[str, ModelDatasetEntry]]
