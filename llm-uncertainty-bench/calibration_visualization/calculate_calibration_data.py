import numpy as np


def calculate_calibration_data(correct, certainty, n_buckets=10) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    Computes calibration statistics for an LLM's confidence estimates.

    This function sorts the given correctness and confidence values into `n_buckets`
    and calculates:
    - The number of samples per bucket.
    - The proportion of correct answers per bucket.
    - The Expected Calibration Error (ECE).

    Args:
        - correct (list of bool):
            A list of boolean values indicating whether each response was correct (`True`) or incorrect (`False`).
        - certainty (list of float):
            A list of confidence scores (ranging from 0.0 to 1.0) representing the model's certainty in each response.
        - n_buckets (int, optional):
            The number of bins to divide the confidence range [0,1] into. Defaults to 10.

    Returns:
        - bin_confidences (numpy.ndarray):
          The midpoints of each confidence bin, representing the expected confidence per bucket.
        - bucket_accuracies (numpy.ndarray):
          The percentage of correct answers in each bucket.
        - bucket_counts (numpy.ndarray):
          The number of items sorted into each bucket.
    """
    bins = np.linspace(0, 1, n_buckets + 1)
    bucket_counts = np.zeros(n_buckets, dtype=int)
    bucket_correct = np.zeros(n_buckets)

    # Assign data to buckets
    indices = np.array([min(x, n_buckets - 1) for x in np.digitize(certainty, bins) - 1])
    assert np.all((indices >= 0) & (indices <= n_buckets - 1)), "Values out of range"

    for i, idx in enumerate(indices):
        if 0 <= idx < n_buckets:
            bucket_counts[idx] += 1
            bucket_correct[idx] += correct[i]

    # Calculate accuracy per bucket and ECE
    bucket_accuracies = np.divide(bucket_correct, bucket_counts, where=bucket_counts > 0,
                                  out=np.zeros_like(bucket_counts, dtype=float))
    bin_confidences = (bins[:-1] + bins[1:]) / 2  # Midpoint of each bin
    return bin_confidences, bucket_accuracies, bucket_counts


def calculate_calibration_data_discrete(correct, certainty, n_buckets=11):
    """
    Computes calibration statistics for a metric that only ever emits
    n_buckets discrete certainty values equally spaced between 0 and 1.

    Bins are centered on [0, w, 2w, ..., 1], with w = 1/(n_buckets-1),
    so that the first bin covers [−w/2, +w/2], the next [w/2, 3w/2], …, and
    the last [1−w/2, 1+w/2].

    Args:
        - correct (list of bool):
            True/False for each sample.
        - certainty (list of float):
            Confidence scores; should all lie exactly on the grid
            {0.0, w, 2w, …, 1.0}.
        - n_buckets (int, optional):
            Number of discrete confidence levels (and bins).
            Defaults to 11 (i.e. [0.0,0.1,…,1.0]).

    Returns:
        - bin_confidences (np.ndarray, shape=(n_buckets,)):
            The bin centers: [0, w, 2w, …, 1].
        - bucket_accuracies (np.ndarray, shape=(n_buckets,)):
            Percent correct in each bin (0 for empty bins).
        - bucket_counts (np.ndarray, shape=(n_buckets,)):
            Number of samples falling into each bin.
    """
    correct = np.array(correct, dtype=int)
    certainty = np.array(certainty, dtype=float)

    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2 for a discrete calibration grid")

    # 1) midpoints at 0, w, 2w, …, 1
    bin_confidences = np.linspace(0.0, 1.0, n_buckets)

    # 2) half‑width of each bin
    w = 1.0 / (n_buckets - 1)

    # 3) bin edges: [-w/2, +w/2, 3w/2, …, 1+w/2]
    edges = np.concatenate([
        bin_confidences - w/2,
        [bin_confidences[-1] + w/2]
    ])

    # 4) allocate counters
    bucket_counts = np.zeros(n_buckets, dtype=int)
    bucket_correct = np.zeros(n_buckets, dtype=int)

    # 5) assign each sample to its bin
    #    digitize: returns index i such that edges[i-1] <= x < edges[i]
    indices = np.digitize(certainty, edges) - 1
    for i, idx in enumerate(indices):
        if 0 <= idx < n_buckets:
            bucket_counts[idx] += 1
            bucket_correct[idx] += correct[i]

    # 6) compute accuracy (safely divide, zero out empty bins)
    bucket_accuracies = np.divide(
        bucket_correct,
        bucket_counts,
        out=np.zeros_like(bucket_counts, dtype=float),
        where=(bucket_counts > 0),
    )

    return bin_confidences, bucket_accuracies, bucket_counts