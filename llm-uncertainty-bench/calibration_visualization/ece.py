import numpy as np


def calculate_ece(bin_confidences, bucket_accuracies, bucket_counts):
    """
    Computes the Expected Calibration Error (ECE) for binned predictions.

    The ECE is a weighted average of the difference between confidence
    estimates and the observed accuracy in each bin. It quantifies
    how well model confidence scores align with actual correctness.

    Args:
        bin_confidences (numpy.ndarray): The representative confidence value
            for each bin (e.g., bin midpoint or average predicted confidence).
            Shape: (num_bins,)
        bucket_accuracies (numpy.ndarray): The empirical accuracy (fraction
            of correct predictions) within each bin. Range: [0, 1].
            Shape: (num_bins,)
        bucket_counts (numpy.ndarray): The number of predictions falling
            into each bin. Shape: (num_bins,)

    Returns:
        float: The Expected Calibration Error (ECE), a value in [0, 1].
            Lower values indicate better calibration.

    Formula:
        ECE = sum_i ( | acc_i - conf_i | * (n_i / N) )
            where acc_i = accuracy in bin i
                  conf_i = confidence representative for bin i
                  n_i    = number of items in bin i
                  N      = total number of items

    Example:
        >>> bin_confidences = np.array([0.1, 0.3, 0.5, 0.7])
        >>> bucket_accuracies = np.array([0.2, 0.25, 0.55, 0.65])
        >>> bucket_counts = np.array([50, 100, 80, 70])
        >>> calculate_ece(bin_confidences, bucket_accuracies, bucket_counts)
        0.03392857142857143
    """
    return np.sum(np.abs(bucket_accuracies - bin_confidences) * bucket_counts) / np.sum(bucket_counts)
