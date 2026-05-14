import numpy as np


def _bootstrap_ci(correct, certainty, bin_fn, n_bootstrap, confidence):
    """Compute bootstrap confidence intervals for bucket accuracies and ECE.

    Resamples ``(correct[i], certainty[i])`` pairs *n_bootstrap* times with
    replacement, calls *bin_fn* on each resample, and returns the
    ``confidence``-level percentile interval for each bucket accuracy and for
    the ECE.

    Args:
        correct: Array-like of bool/int correctness values.
        certainty: Array-like of float confidence scores.
        bin_fn: Callable ``(correct, certainty) -> (bin_confs, bucket_accs, bucket_counts)``
            — one of the two binning functions with its parameters already bound.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level, e.g. 0.95.

    Returns:
        tuple:
            bucket_acc_ci (np.ndarray, shape (n_buckets, 2)):
                Columns are [lower, upper] absolute accuracy CI boundaries
                for each bucket.
            ece_ci (tuple[float, float]):
                ``(lower, upper)`` absolute ECE CI boundaries.
    """
    correct_arr = np.asarray(correct, dtype=float)
    certainty_arr = np.asarray(certainty, dtype=float)
    n = len(correct_arr)

    alpha = (1.0 - confidence) / 2.0
    lo_pct, hi_pct = 100.0 * alpha, 100.0 * (1.0 - alpha)

    rng = np.random.default_rng(seed=42)
    boot_accs = []
    boot_eces = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        c_boot = correct_arr[idx]
        s_boot = certainty_arr[idx]
        bin_confs_b, bucket_accs_b, bucket_counts_b = bin_fn(c_boot, s_boot)
        boot_accs.append(bucket_accs_b)
        ece_b = (np.sum(np.abs(bucket_accs_b - bin_confs_b) * bucket_counts_b)
                 / np.sum(bucket_counts_b))
        boot_eces.append(ece_b)

    boot_accs_arr = np.array(boot_accs)   # shape (n_bootstrap, n_buckets)
    boot_eces_arr = np.array(boot_eces)   # shape (n_bootstrap,)

    bucket_acc_ci = np.stack([
        np.percentile(boot_accs_arr, lo_pct, axis=0),
        np.percentile(boot_accs_arr, hi_pct, axis=0),
    ], axis=1)

    ece_ci = (float(np.percentile(boot_eces_arr, lo_pct)),
              float(np.percentile(boot_eces_arr, hi_pct)))

    return bucket_acc_ci, ece_ci


def calculate_calibration_data(correct, certainty, n_buckets=10,
                                compute_ci=True, n_bootstrap=1000, confidence=0.95):
    """
    Computes calibration statistics for an LLM's confidence estimates.

    This function sorts the given correctness and confidence values into `n_buckets`
    and calculates the number of samples per bucket, the proportion of correct
    answers per bucket, and optionally bootstrap confidence intervals.

    Args:
        - correct (list of bool):
            A list of boolean values indicating whether each response was correct.
        - certainty (list of float):
            A list of confidence scores (ranging from 0.0 to 1.0).
        - n_buckets (int, optional):
            The number of bins to divide the confidence range [0,1] into. Defaults to 10.
        - compute_ci (bool, optional):
            Whether to compute bootstrap confidence intervals. Defaults to True.
        - n_bootstrap (int, optional):
            Number of bootstrap resamples. Defaults to 1000.
        - confidence (float, optional):
            Confidence level for the interval, e.g. 0.95. Defaults to 0.95.

    Returns:
        5-tuple:
        - bin_confidences (numpy.ndarray): Midpoints of each confidence bin.
        - bucket_accuracies (numpy.ndarray): Fraction correct in each bucket.
        - bucket_counts (numpy.ndarray): Number of items in each bucket.
        - bucket_acc_ci (numpy.ndarray or None): Shape (n_buckets, 2) with
          [lower, upper] absolute accuracy CI boundaries per bucket.
          None when compute_ci=False.
        - ece_ci (tuple[float, float] or None): (lower, upper) absolute ECE CI
          boundaries. None when compute_ci=False.
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

    if compute_ci:
        def _bin_fn(c, s):
            bc = np.zeros(n_buckets, dtype=int)
            bcorr = np.zeros(n_buckets)
            idx = np.array([min(x, n_buckets - 1) for x in np.digitize(s, bins) - 1])
            for i2, idx2 in enumerate(idx):
                if 0 <= idx2 < n_buckets:
                    bc[idx2] += 1
                    bcorr[idx2] += c[i2]
            ba = np.divide(bcorr, bc, where=bc > 0, out=np.zeros_like(bc, dtype=float))
            return bin_confidences, ba, bc

        bucket_acc_ci, ece_ci = _bootstrap_ci(correct, certainty, _bin_fn,
                                               n_bootstrap, confidence)
    else:
        bucket_acc_ci, ece_ci = None, None

    return bin_confidences, bucket_accuracies, bucket_counts, bucket_acc_ci, ece_ci


def calculate_calibration_data_discrete(correct, certainty, n_buckets=11,
                                        compute_ci=True, n_bootstrap=1000, confidence=0.95):
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
        - compute_ci (bool, optional):
            Whether to compute bootstrap confidence intervals. Defaults to True.
        - n_bootstrap (int, optional):
            Number of bootstrap resamples. Defaults to 1000.
        - confidence (float, optional):
            Confidence level for the interval, e.g. 0.95. Defaults to 0.95.

    Returns:
        5-tuple:
        - bin_confidences (np.ndarray, shape=(n_buckets,)):
            The bin centers: [0, w, 2w, …, 1].
        - bucket_accuracies (np.ndarray, shape=(n_buckets,)):
            Percent correct in each bin (0 for empty bins).
        - bucket_counts (np.ndarray, shape=(n_buckets,)):
            Number of samples falling into each bin.
        - bucket_acc_ci (np.ndarray or None): Shape (n_buckets, 2) with
          [lower, upper] absolute accuracy CI boundaries per bucket.
          None when compute_ci=False.
        - ece_ci (tuple[float, float] or None): (lower, upper) absolute ECE CI
          boundaries. None when compute_ci=False.
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

    if compute_ci:
        def _bin_fn(c, s):
            bc = np.zeros(n_buckets, dtype=int)
            bcorr = np.zeros(n_buckets, dtype=int)
            idx = np.digitize(s, edges) - 1
            for i2, idx2 in enumerate(idx):
                if 0 <= idx2 < n_buckets:
                    bc[idx2] += 1
                    bcorr[idx2] += int(c[i2])
            ba = np.divide(bcorr, bc, out=np.zeros_like(bc, dtype=float), where=(bc > 0))
            return bin_confidences, ba, bc

        bucket_acc_ci, ece_ci = _bootstrap_ci(correct, certainty, _bin_fn,
                                               n_bootstrap, confidence)
    else:
        bucket_acc_ci, ece_ci = None, None

    return bin_confidences, bucket_accuracies, bucket_counts, bucket_acc_ci, ece_ci