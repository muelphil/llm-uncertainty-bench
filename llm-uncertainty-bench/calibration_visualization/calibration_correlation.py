import numpy as np
from scipy.stats import pearsonr


def calculate_calibration_correlation(bin_confidences, bucket_accuracies, bucket_counts):
    """Compute the Pearson correlation between bin-confidence midpoints and bucket accuracies.

    Empty bins (``bucket_counts == 0``) are excluded before computing the
    correlation.  All non-empty bins contribute equally (i.e. the correlation
    is *not* weighted by bin count, unlike ECE).

    Args:
        bin_confidences (numpy.ndarray): Midpoint confidence value for each bin.
            Shape: (n_bins,).
        bucket_accuracies (numpy.ndarray): Empirical accuracy in each bin.
            Range: [0, 1].  Shape: (n_bins,).
        bucket_counts (numpy.ndarray): Number of items in each bin.
            Shape: (n_bins,).

    Returns:
        float: Pearson *r* for the non-empty bins.  Returns ``float('nan')``
        when fewer than two non-empty bins exist (correlation is undefined).

    Notes:
        * A value of *r* = 1 indicates that accuracy rises perfectly in step
          with confidence (ideal calibration).
        * *r* = 0 indicates no linear relationship.
        * Negative values indicate that higher confidence is associated with
          lower accuracy (miscalibration).
    """
    bin_confidences = np.asarray(bin_confidences, dtype=float)
    bucket_accuracies = np.asarray(bucket_accuracies, dtype=float)
    bucket_counts = np.asarray(bucket_counts, dtype=int)

    mask = bucket_counts > 0
    if mask.sum() < 2:
        return float("nan")

    r, _ = pearsonr(bin_confidences[mask], bucket_accuracies[mask])
    return float(r)
