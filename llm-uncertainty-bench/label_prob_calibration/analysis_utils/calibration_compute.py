"""Calibration metric computation for the label_prob_calibration experiment.

This module provides the core data-preparation functions that transform
raw model output DataFrames into calibration metrics dicts and per-item
DataFrames.  All functions are pure (no side effects, no file I/O) and
designed to be called from the data-preparation notebook.

Key entry points:

- :func:`compute_all_variants` — **preferred** high-performance path that parses
  the label-probability DataFrame exactly **once** and returns all four
  (normalize × chosen_only) calibration metrics dicts from shared numpy arrays.
- :func:`compute_calibration_metrics` — legacy single-variant path, kept for
  backward compatibility.
- :func:`build_per_item_df` — builds a trimmed per-item DataFrame with full
  label-probability distributions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibration_visualization import (
    calculate_calibration_data,
    calculate_ece,
    calculate_normalized_entropy,
)


def build_per_item_df(ground_truth_df, label_prob_df):
    """Build a trimmed per-item DataFrame with full label-probability distributions.

    Merges each row of *label_prob_df* with its ground-truth record to produce
    a compact, analysis-ready DataFrame.  Text-heavy benchmark columns
    (``question``, ``choices``, free-text answers, etc.) are intentionally
    excluded; only numerically/logically relevant fields are retained.

    Stored columns:

    - ``id``: item identifier (string or multi-index tuple).
    - ``correct_answer``: ground-truth label index (int, 0–3 for MCQA).
    - ``probs_raw``: un-normalised per-choice probability vector (list[float, 4]).
    - ``probs_norm``: L1-normalised per-choice probability vector (list[float, 4]).
      All zeros when the model assigned no probability mass.
    - ``chosen_label``: argmax of ``probs_norm`` (int), or ``None`` when all probs
      are zero (invalid/abstained answer).
    - ``is_correct``: whether ``chosen_label == correct_answer`` (bool; False for
      invalid answers).
    - ``chosen_prob_raw``: raw probability of the chosen label (float or ``None``
      for invalid answers).
    - ``chosen_prob_norm``: normalised probability of the chosen label (float or
      ``None`` for invalid answers).

    Args:
        ground_truth_df: Ground-truth DataFrame indexed by item id, with at least
            a ``correct_answer`` column (integer label index 0–3).
        label_prob_df: Raw model output DataFrame with ``id`` and
            ``confidence_per_choice`` columns.  Rows whose ``id`` is absent
            from *ground_truth_df* are silently skipped.

    Returns:
        pd.DataFrame: One row per matched item (missing ids skipped).
    """
    rows = []
    for _, row in label_prob_df.iterrows():
        if row["id"] not in ground_truth_df.index:
            continue
        mc_task = ground_truth_df.loc[row["id"]]
        probs_raw = np.array(row["confidence_per_choice"], dtype=float)
        probs_norm = (probs_raw / probs_raw.sum()
                      if probs_raw.sum() > 0
                      else np.zeros_like(probs_raw))
        chosen = int(np.argmax(probs_norm)) if probs_norm.sum() > 0 else None
        rows.append({
            "id": row["id"],
            "correct_answer": mc_task.correct_answer,
            "probs_raw": probs_raw.tolist(),
            "probs_norm": probs_norm.tolist(),
            "chosen_label": chosen,
            "is_correct": (chosen == mc_task.correct_answer) if chosen is not None else False,
            "chosen_prob_raw": float(probs_raw[chosen]) if chosen is not None else None,
            "chosen_prob_norm": float(probs_norm[chosen]) if chosen is not None else None,
        })
    return pd.DataFrame(rows)


def align_predictions_with_ground_truth(ground_truth_df, label_prob_df,
                                        chosen_only=True, normalize=True):
    """Align model label-probability rows with ground-truth answers.

    Iterates over *label_prob_df*, skipping rows whose ``id`` has no match in
    *ground_truth_df*.  Per row, optionally L1-normalises the probability
    vector and collects either the single argmax confidence (``chosen_only=True``)
    or all per-label confidences (``chosen_only=False``).

    Args:
        ground_truth_df: Ground-truth DataFrame indexed by item id, with a
            ``correct_answer`` column (integer label index).
        label_prob_df: Model output DataFrame with ``id`` and
            ``confidence_per_choice`` columns.
        chosen_only: If ``True``, record only the argmax choice per question.
        normalize: If ``True``, L1-normalise each per-choice probability vector.

    Returns:
        tuple:
            correct (list[bool]):    Whether each recorded prediction is correct.
            certainties (list[float]): Corresponding confidence values.
            accuracy (float):        Fraction of all rows where argmax == correct
                                     label.
            invalid_count (int):     Rows where all probabilities are zero
                                     (only counted when ``chosen_only`` is True).
    """
    correct, certainties = [], []
    n_correct = 0
    invalid_count = 0

    for _, row in label_prob_df.iterrows():
        if row["id"] not in ground_truth_df.index:
            continue
        mc_task = ground_truth_df.loc[row["id"]]
        probs = np.array(row["confidence_per_choice"], dtype=float)
        if normalize and probs.sum() > 0:
            probs /= probs.sum()
        chosen = int(np.argmax(probs)) if probs.sum() > 0 else None

        if chosen is not None and chosen == mc_task.correct_answer:
            n_correct += 1

        if chosen_only:
            if chosen is None:
                invalid_count += 1
            else:
                correct.append(chosen == mc_task.correct_answer)
                certainties.append(probs[chosen])
        else:
            for i, p in enumerate(probs):
                correct.append(i == mc_task.correct_answer)
                certainties.append(p)

    accuracy = n_correct / len(label_prob_df) if len(label_prob_df) > 0 else 0.0
    return correct, certainties, accuracy, invalid_count


def compute_calibration_metrics(ground_truth_df, label_prob_df,
                                chosen_only=True, normalize=True, n_bins=15,
                                return_arrays=False):
    """Compute the full calibration metrics dict for one (prompt, dataset, model) cell.

    Combines alignment, calibration-curve binning, and label-probability summary
    statistics into a single metrics dict consumed by the visualisation layer.

    Args:
        ground_truth_df: Ground-truth DataFrame (see :func:`align_predictions_with_ground_truth`).
        label_prob_df: Raw model output DataFrame.
        chosen_only: Use only the argmax label per question.
        normalize: L1-normalise confidence vectors before computing metrics.
        n_bins: Number of calibration histogram bins.
        return_arrays: If ``True``, also return the raw ``(correct, certainties)``
            arrays alongside the metrics dict (useful for computing relplot
            diagrams before dropping the arrays).

    Returns:
        dict, or ``(dict, correct, certainties)`` when *return_arrays* is ``True``.
    """
    correct, certainties, accuracy, invalid_count = align_predictions_with_ground_truth(
        ground_truth_df, label_prob_df, chosen_only=chosen_only, normalize=normalize
    )
    bin_confidences, bucket_accuracies, bucket_counts, bucket_acc_ci, ece_ci = \
        calculate_calibration_data(correct, certainties, n_bins)
    ece = calculate_ece(bin_confidences, bucket_accuracies, bucket_counts)
    bucket_counts_100 = calculate_calibration_data(correct, certainties, 100, compute_ci=False)[2]
    try:
        auroc = roc_auc_score(correct, certainties)
    except Exception:
        auroc = "undefined"

    # Per-question sum of raw label probabilities (kept ≥ 0; not capped at 1
    # so that models whose 4 token-probs sum > 1 are represented faithfully).
    sum_series = (
        label_prob_df["confidence_per_choice"]
        .apply(lambda probs: max(0.0, sum(probs)))
    )

    metrics = {
        "is_normalized": normalize,
        "bin_confidences": bin_confidences,
        "bucket_accuracies": bucket_accuracies,
        "bucket_counts": bucket_counts,
        "bucket_acc_ci": bucket_acc_ci,
        "ece": ece,
        "ece_ci": ece_ci,
        "auroc": auroc,
        "accuracy": accuracy,
        "label_prob_sum_mean": sum_series.mean(),
        "label_prob_sum_median": sum_series.quantile(0.5),
        "label_prob_sum_std": sum_series.std(),
        "label_prob_sum_iqr": sum_series.quantile(0.75) - sum_series.quantile(0.25),
        "normalized_entropy": calculate_normalized_entropy(bucket_counts),
        "normalized_entropy_100": calculate_normalized_entropy(bucket_counts_100),
        "total_items": len(label_prob_df),
        "invalid_answers": invalid_count,
    }

    if certainties:
        metrics["label_prob_median"] = np.median(certainties)
        metrics["label_prob_q25"] = float(np.percentile(certainties, 25))
        metrics["label_prob_q75"] = float(np.percentile(certainties, 75))
        metrics["label_prob_iqr"] = metrics["label_prob_q75"] - metrics["label_prob_q25"]
        metrics["label_prob_min"] = float(np.min(certainties))
        metrics["label_prob_max"] = float(np.max(certainties))
        metrics["average_certainty"] = np.mean(certainties)
        dist = 1.0 - np.array(certainties)
        metrics["dist_to_1_mean"] = float(np.mean(dist))
        metrics["dist_to_1_std"] = float(np.std(dist))
    else:
        for key in ("label_prob_median", "label_prob_q25", "label_prob_q75",
                    "label_prob_iqr", "label_prob_min", "label_prob_max",
                    "average_certainty", "dist_to_1_mean", "dist_to_1_std"):
            metrics[key] = "undefined"

    if return_arrays:
        return metrics, correct, certainties
    return metrics


# ---------------------------------------------------------------------------
# High-performance multi-variant helper
# ---------------------------------------------------------------------------

def _compute_metrics_from_arrays(correct_arr, certainties_arr, label_prob_df, n_bins,
                                 normalize, chosen_only):
    """Build a calibration metrics dict from pre-aligned numpy arrays.

    Internal helper used by :func:`compute_all_variants`.  The caller has
    already performed id-filtering, matrix stacking, normalisation and
    argmax/ravel steps; this function only computes the aggregated statistics.

    Args:
        correct_arr (np.ndarray): Boolean array of per-prediction correctness.
        certainties_arr (np.ndarray): Confidence values matching *correct_arr*.
        label_prob_df (pd.DataFrame): Full (pre-filtered or original) raw output
            DataFrame — used only to compute per-question probability-sum stats.
        n_bins (int): Number of calibration histogram bins.
        normalize (bool): Whether these arrays were derived from normalised probs.
        chosen_only (bool): Whether these arrays represent argmax predictions only.

    Returns:
        dict: Calibration metrics dict (same schema as :func:`compute_calibration_metrics`).
    """
    correct_list = correct_arr.tolist()
    certainties_list = certainties_arr.tolist()

    bin_confidences, bucket_accuracies, bucket_counts, bucket_acc_ci, ece_ci = \
        calculate_calibration_data(correct_list, certainties_list, n_bins)
    ece = calculate_ece(bin_confidences, bucket_accuracies, bucket_counts)

    # Additional 100-bin normalised entropy (used alongside the n_bins variant in tables).
    bucket_counts_100 = calculate_calibration_data(correct_list, certainties_list, 100, compute_ci=False)[2]
    try:
        auroc = roc_auc_score(correct_list, certainties_list)
    except Exception:
        auroc = "undefined"

    # Per-question sum of raw label probabilities (kept ≥ 0; not capped at 1
    # so that models whose 4 token-probs sum > 1 are represented faithfully).
    sum_series = (
        label_prob_df["confidence_per_choice"]
        .apply(lambda probs: max(0.0, sum(probs)))
    )

    n_total = len(label_prob_df)
    # invalid_count is only meaningful for chosen_only=True
    invalid_count = int(n_total - len(certainties_list)) if chosen_only else 0
    # accuracy: fraction of *all* rows (including invalid) where argmax == correct_answer
    # - for chosen_only: valid_correct / n_total
    # - for all labels:  positive labels / (n_total * 4), but we follow legacy convention
    accuracy = float(np.sum(correct_arr)) / len(correct_arr) if len(correct_arr) > 0 else 0.0

    metrics = {
        "is_normalized": normalize,
        "bin_confidences": bin_confidences,
        "bucket_accuracies": bucket_accuracies,
        "bucket_counts": bucket_counts,
        "bucket_acc_ci": bucket_acc_ci,
        "ece": ece,
        "ece_ci": ece_ci,
        "auroc": auroc,
        "accuracy": accuracy,
        "label_prob_sum_mean": sum_series.mean(),
        "label_prob_sum_median": sum_series.quantile(0.5),
        "label_prob_sum_std": sum_series.std(),
        "label_prob_sum_iqr": sum_series.quantile(0.75) - sum_series.quantile(0.25),
        "normalized_entropy": calculate_normalized_entropy(bucket_counts),
        "normalized_entropy_100": calculate_normalized_entropy(bucket_counts_100),
        "total_items": n_total,
        "invalid_answers": invalid_count,
    }

    if len(certainties_list) > 0:
        metrics["label_prob_median"] = float(np.median(certainties_arr))
        metrics["label_prob_q25"] = float(np.percentile(certainties_arr, 25))
        metrics["label_prob_q75"] = float(np.percentile(certainties_arr, 75))
        metrics["label_prob_iqr"] = metrics["label_prob_q75"] - metrics["label_prob_q25"]
        metrics["label_prob_min"] = float(np.min(certainties_arr))
        metrics["label_prob_max"] = float(np.max(certainties_arr))
        metrics["average_certainty"] = float(np.mean(certainties_arr))
        dist = 1.0 - certainties_arr
        metrics["dist_to_1_mean"] = float(np.mean(dist))
        metrics["dist_to_1_std"] = float(np.std(dist))
    else:
        for key in ("label_prob_median", "label_prob_q25", "label_prob_q75",
                    "label_prob_iqr", "label_prob_min", "label_prob_max",
                    "average_certainty", "dist_to_1_mean", "dist_to_1_std"):
            metrics[key] = "undefined"

    return metrics


def compute_all_variants(ground_truth_df, label_prob_df, n_bins=15):
    """Compute calibration metrics for all four (normalize × chosen_only) variants at once.

    This is the **preferred** entry point for the data-loading loop.  It parses
    ``label_prob_df`` exactly once (id filtering, matrix stacking, normalisation,
    argmax, correctness derivation) using vectorised numpy operations, then
    derives all four metric dicts from the shared precomputed arrays.

    This is significantly faster than calling :func:`compute_calibration_metrics`
    four times because:

    * The label-probability matrix is stacked from the DataFrame **once** instead
      of four times.
    * The ``id → correct_answer`` mapping is built once via ``.to_dict()``.
    * Normalised and raw probability matrices are derived in a single vectorised
      step.
    * All four ``(correct, certainties)`` pairs are computed via numpy ops
      (argmax, boolean compare, ravel) rather than Python-level ``iterrows()``.

    The returned ``_arrays`` dict exposes the raw ``(correct, certainties)``
    numpy arrays for each key so the caller can pass them directly to
    ``rp.prepare_rel_diagram`` without repeating any work.

    Args:
        ground_truth_df (pd.DataFrame): Ground-truth DataFrame indexed by item id,
            with at least a ``correct_answer`` column (integer label index 0–3).
        label_prob_df (pd.DataFrame): Raw model output DataFrame with ``id`` and
            ``confidence_per_choice`` columns.
        n_bins (int): Number of calibration histogram bins (default ``15``).

    Returns:
        tuple[dict, dict]:
            - **metrics_dict** ``{key: metrics_dict}`` — one entry per variant key
              (``"norm_chosen"``, ``"norm_all"``, ``"raw_chosen"``, ``"raw_all"``).
              Each value is a calibration metrics dict as returned by
              :func:`compute_calibration_metrics`.
            - **arrays_dict** ``{key: (correct_arr, certainties_arr)}`` — raw numpy
              arrays for each variant, for use with ``rp.prepare_rel_diagram``.
    """
    # ------------------------------------------------------------------ #
    # 1. Build id → correct_answer lookup (O(1) per lookup afterwards)    #
    # ------------------------------------------------------------------ #
    id_to_answer = ground_truth_df["correct_answer"].to_dict()
    valid_id_set = set(id_to_answer.keys())

    # ------------------------------------------------------------------ #
    # 2. Filter rows whose id exists in ground_truth_df (vectorised)      #
    # ------------------------------------------------------------------ #
    valid_mask = label_prob_df["id"].apply(lambda x: x in valid_id_set)
    valid_df = label_prob_df[valid_mask]

    if len(valid_df) == 0:
        # Return empty-but-valid metrics for all variants
        empty_arr = np.array([], dtype=float)
        empty_metrics = _compute_metrics_from_arrays(
            np.array([], dtype=bool), empty_arr, label_prob_df, n_bins,
            normalize=True, chosen_only=True,
        )
        empty_entry = {k: empty_metrics for k in ("norm_chosen", "norm_all", "raw_chosen", "raw_all")}
        empty_arrays = {k: (np.array([], dtype=bool), empty_arr)
                        for k in ("norm_chosen", "norm_all", "raw_chosen", "raw_all")}
        return empty_entry, empty_arrays

    # ------------------------------------------------------------------ #
    # 3. Stack confidence_per_choice into a (N, 4) matrix                 #
    # ------------------------------------------------------------------ #
    ids = valid_df["id"].tolist()
    raw_mat = np.stack(
        valid_df["confidence_per_choice"].apply(lambda x: np.asarray(x, dtype=float)).values
    )  # shape (N, 4)

    # ------------------------------------------------------------------ #
    # 4. Normalised matrix — rows that sum to 0 are left as zeros         #
    # ------------------------------------------------------------------ #
    row_sums = raw_mat.sum(axis=1, keepdims=True)  # (N, 1)
    norm_mat = np.where(row_sums > 0, raw_mat / row_sums, 0.0)  # (N, 4)

    # ------------------------------------------------------------------ #
    # 5. Per-row ground-truth correct labels                              #
    # ------------------------------------------------------------------ #
    correct_labels = np.array([id_to_answer[i] for i in ids], dtype=int)  # (N,)

    # ------------------------------------------------------------------ #
    # 6. Derive (correct, certainties) for all 4 variants                 #
    # ------------------------------------------------------------------ #
    results = {}
    arrays = {}

    for mat, normalize in ((norm_mat, True), (raw_mat, False)):
        # chosen_only=True — argmax prediction per question
        chosen_idx = np.argmax(mat, axis=1)  # (N,) -- 0 when all zeros (invalid)
        valid_rows = row_sums.ravel() > 0    # boolean mask: True = valid prediction
        # certainty = probability assigned to argmax choice
        chosen_certainties = mat[np.arange(len(mat)), chosen_idx]  # (N,)
        chosen_correct = (chosen_idx == correct_labels)             # (N,)

        # Restrict to valid rows only (invalid rows are those with all-zero probs)
        c_arr = chosen_correct[valid_rows]
        cert_arr = chosen_certainties[valid_rows]

        key = "norm_chosen" if normalize else "raw_chosen"
        results[key] = _compute_metrics_from_arrays(
            c_arr, cert_arr, label_prob_df, n_bins, normalize=normalize, chosen_only=True,
        )
        # Override invalid_answers to include truly invalid rows
        results[key]["invalid_answers"] = int((~valid_rows).sum())
        arrays[key] = (c_arr, cert_arr)

        # chosen_only=False — all 4 label predictions per question
        # Correctness: True for position == correct_label, False for the other 3
        all_correct = (
            np.arange(4)[np.newaxis, :] == correct_labels[:, np.newaxis]
        ).ravel()  # (N*4,)
        all_certainties = mat.ravel()  # (N*4,)

        key_all = "norm_all" if normalize else "raw_all"
        results[key_all] = _compute_metrics_from_arrays(
            all_correct, all_certainties, label_prob_df, n_bins,
            normalize=normalize, chosen_only=False,
        )
        # accuracy for "all" variant: fraction of correct labels (== overall accuracy)
        # Recalculate as per-question argmax accuracy for consistency with chosen variant
        results[key_all]["accuracy"] = float(np.mean(chosen_correct[valid_rows])) if valid_rows.any() else 0.0
        arrays[key_all] = (all_correct, all_certainties)

    return results, arrays

