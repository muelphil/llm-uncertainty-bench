import matplotlib.pyplot as plt
import numpy as np

def shorten_number(num: float) -> str:
    """
    Shortens large numbers to human-readable format.
    Examples:
        259312 -> '259.3K'
        1_000_000 -> '1.0M'
        123 -> '123'
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(int(num)) if num.is_integer() else f"{num:.1f}"

def ease_out(x: float, k: float = 2.0) -> float:
    """
    Ease-out function with tunable 'ease strength' via parameter k.

    Parameters:
        x (float): Input value in [0, 1]
        k (float): Easing exponent > 0 (higher means stronger ease-out)

    Returns:
        float: Eased value in [0, 1]
    """
    if not 0 <= x <= 1:
        raise ValueError("Input x must be in the range [0, 1]")
    if k <= 0:
        raise ValueError("Easing parameter k must be > 0")

    return 1 - (1 - x) ** k

def plot_calibration_curve(bin_confidences, bucket_accuracies, bucket_counts, colormap="Blues",
                           title=None, ax=None, xlabel="Confidence Bins", ylabel="Accuracy in Bin",
                           total_item_count=None, fontsize=18, tick_fontsize=12, count_fontsize=8, ece=None):
    """
    Plots a reliability diagram showing model calibration.

    The x-axis represents confidence bins, while the y-axis represents
    the proportion of correct answers in each bin. Bar color intensity
    reflects the number of samples in each bin. A dotted identity line
    (f(x) = x) is also shown for reference, and the Expected Calibration
    Error (ECE) is displayed in the upper-left corner.

    Args:
        bin_confidences (numpy.ndarray): The midpoints of each confidence bin.
        bucket_accuracies (numpy.ndarray): The percentage of correct answers in each bucket.
        bucket_counts (numpy.ndarray): The number of items sorted into each bucket.
        colormap (str, optional): The base color for the bars. Defaults to "Blues".
        title (str, optional): The title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot onto. If not provided, a new figure is created.

    Returns:
        matplotlib.pyplot or None: Returns plt if ax is not provided, otherwise None.

    Example:
        >>> bin_confidences = np.array([0.05, 0.15, 0.25])
        >>> bucket_accuracies = np.array([0.8, 0.6, 0.5])
        >>> bucket_counts = np.array([10, 20, 5])
        >>> plot_calibration_curve(bin_confidences, bucket_accuracies, bucket_counts, ece)
    """
    # Create the plot if no ax is passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize bucket sizes for color intensity
    bucket_count_max = bucket_counts.max()
    if total_item_count is not None:
        bucket_count_max = max(bucket_count_max, total_item_count/len(bin_confidences))
    if bucket_count_max > 0:
        norm_counts = bucket_counts / bucket_count_max
    else:
        norm_counts = np.zeros_like(bucket_counts)

    cmap = plt.colormaps[colormap]  # Updated to avoid deprecation warning
    colors = [cmap(c * 0.8 + 0.2) for c in norm_counts]  # Adjust color range

    bin_width = bin_confidences[1] - bin_confidences[0]
    # Create bars
    bars = ax.bar(bin_confidences, bucket_accuracies, width=bin_width, color=colors, edgecolor="black")

    # Plot the identity line (f(x) = x)
    ax.plot([0, 2], [0, 2], linestyle="--", color="gray")

    # Add bucket counts as labels
    for bar, count in zip(bars, bucket_counts):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        label = shorten_number(count)

        if height > 0.5:  # place inside the bar
            y = height - 0.05  # just below top
            va = "top"
            color = "white"
        else:  # place above the bar
            y = height + 0.02
            va = "bottom"
            color = "black"

        ax.text(x, y, label, ha="center", va=va, fontsize=count_fontsize, color=color)


    # Display the ECE in the upper left corner
    if ece is not None:
        ax.text(0.02, 0.98, f"ECE:\n{ece:.4f}", transform=ax.transAxes, fontsize=24,
                verticalalignment='top', horizontalalignment='left', color="black")

    y_ticks = np.arange(0.2, 1.1, 0.2)
    # Labeling the axes and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if bin_confidences[0] == 0:
        bin_width = bin_confidences[1] - bin_confidences[0]
        ax.set_xlim(-0.5*bin_width, 1+0.5*bin_width)
        ax.set_ylim(0, 1+bin_width)
        x_ticks = np.arange(0.0, 1.1, 0.2)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        x_ticks = np.arange(0.2, 1.1, 0.2)

    # Set x-axis and y-axis ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if title:
        ax.set_title(title)

    if ax is None:
        return plt
