"""Type definitions for the pickled prepared-data structure.

The data-preparation notebook (``analysis_prepare_data``) pickles its output to
``resources_struct_decoding/prepared_data.pkl``.  This module documents the
exact shape of that pickle so that the visualization notebook can be written
and reasoned about without running the preparation step first.

All types are described with :mod:`typing` and :class:`typing.TypedDict` where
practical; for nested dicts keyed by runtime strings (model/dataset IDs) plain
``Dict`` annotations are used.

Pickle top-level keys
---------------------
.. code-block:: python

    {
        "cal_data":                   CalData,
        "labels_chosen_proportion":   ModelLabelProportions,
        "actual_correct_label_count": Dict[int, int],
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class CalibrationMetrics(TypedDict, total=False):
    """Metrics produced by :func:`analysis_utils.calibration_compute.compute_calibration_metrics`.

    All float fields are ``None`` when the model output for this cell was missing.
    """

    bin_confidences: List[float]
    """Mid-point confidence of each histogram bucket."""

    bucket_accuracies: List[float]
    """Fraction of correct predictions in each bucket."""

    bucket_counts: List[int]
    """Number of items in each bucket."""

    ece: float
    """Expected Calibration Error."""

    auroc: float
    """Area Under the ROC Curve."""

    accuracy: float
    """Fraction of items where the chosen label is correct."""

    normalized_entropy: float
    """Entropy of bucket-count distribution, normalised to [0, 1]."""

    total_items: int
    """Total number of items processed."""

    invalid_answers: int
    """Items where no probability mass was assigned to any choice."""

    average_certainty: float
    """Mean probability assigned to the chosen label."""

    label_prob_median: float
    label_prob_iqr: float
    label_prob_sum_mean: float
    label_prob_sum_median: float
    label_prob_sum_std: float
    label_prob_sum_iqr: float

    y_true: List[bool]
    """Per-item correctness flag (chosen label == correct answer)."""

    y_pred: List[float]
    """Per-item confidence of the chosen label."""


# ---------------------------------------------------------------------------
# Per-item DataFrame
# ---------------------------------------------------------------------------

#: Columns present in the trimmed per-item DataFrame stored in
#: ``cal_data[norm][pname][ds_id][m_id]["per_item_df"]``.
#:
#: * ``id``              – item identifier from the benchmark CSV
#: * ``correct_answer``  – integer index 0–3 of the ground-truth choice
#: * ``probs_raw``       – list[4] of raw token probabilities for A/B/C/D
#: * ``probs_norm``      – list[4] of L1-normalised probabilities
#: * ``chosen_label``    – integer index 0–3 of the argmax label
#: * ``is_correct``      – bool (chosen_label == correct_answer)
#: * ``chosen_prob_raw`` – raw probability of the chosen label
#: * ``chosen_prob_norm``– normalised probability of the chosen label
PER_ITEM_DF_COLUMNS = [
    "id",
    "correct_answer",
    "probs_raw",
    "probs_norm",
    "chosen_label",
    "is_correct",
    "chosen_prob_raw",
    "chosen_prob_norm",
]


# ---------------------------------------------------------------------------
# Cell entry
# ---------------------------------------------------------------------------

class CellEntry(TypedDict, total=False):
    """Dict stored at ``cal_data[norm][pname][ds_id][m_id]``."""

    chosen_labels: Optional[CalibrationMetrics]
    """Metrics computed using only the single highest-probability label per question."""

    all_labels: Optional[CalibrationMetrics]
    """Metrics computed treating all four labels as separate prediction events."""

    per_item_df: Optional[pd.DataFrame]
    """Trimmed per-item DataFrame (see :data:`PER_ITEM_DF_COLUMNS`)."""


# ---------------------------------------------------------------------------
# Top-level types
# ---------------------------------------------------------------------------

#: ``CalData[norm_key][prompt_basename][dataset_id][model_id]`` → :class:`CellEntry`
CalData = Dict[str, Dict[str, Dict[str, Dict[str, CellEntry]]]]

#: ``ModelLabelProportions[model_id][label_index]`` → proportion in [0, 1].
#: Proportions across the four labels (0–3) sum to 1.0 for each model.
ModelLabelProportions = Dict[str, Dict[int, float]]


class PreparedData(TypedDict):
    """Shape of the top-level pickle produced by ``analysis_prepare_data``."""

    cal_data: CalData
    """Full calibration metrics, indexed by norm × prompt × dataset × model."""

    labels_chosen_proportion: ModelLabelProportions
    """Fraction of probability mass each model assigns to each choice position."""

    actual_correct_label_count: Dict[int, int]
    """Ground-truth count of each choice index (0–3) across all datasets."""
