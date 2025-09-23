import matplotlib.pyplot as plt
import numpy as np

def plot_bucket_counts(bin_confidences, bucket_counts, ax=None):
    """
    Plots a bar chart showing the number of samples in each confidence bucket.

    Args:
        bin_confidences (numpy.ndarray): The midpoints of each confidence bin.
        bucket_counts (numpy.ndarray): The number of items sorted into each bucket.
        ax (matplotlib.axes.Axes, optional): The axes to plot onto. If not provided, a new figure is created.

    Returns:
        matplotlib.pyplot or None: Returns plt if ax is not provided, otherwise None.
    """
    # Create the plot if no ax is passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bucket counts
    ax.bar(bin_confidences, bucket_counts, width=1.0 / len(bin_confidences), color="lightblue", edgecolor="black")

    # Labeling the axes and title
    ax.set_xlabel("Confidence Bins")
    ax.set_ylabel("Number of Samples")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(bucket_counts) * 1.1)
    ax.set_title("Bucket Counts")

    if ax is None:
        return plt