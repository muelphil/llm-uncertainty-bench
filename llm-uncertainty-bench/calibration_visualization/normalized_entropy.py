import numpy as np
from scipy.stats import entropy as scipy_entropy


def calculate_normalized_entropy(bucket_counts):
    probabilities = np.array(bucket_counts) / sum(bucket_counts)
    num_buckets = len(bucket_counts)

    if num_buckets <= 1:  # Avoid division by zero when only one bucket exists
        return 0.0

    return scipy_entropy(probabilities, base=2) / np.log2(num_buckets)