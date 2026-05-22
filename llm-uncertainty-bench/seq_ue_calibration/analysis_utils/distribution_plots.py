"""Grid-plot utilities for visualising per-model score distributions.

Functions take pre-aggregated data (value-count Series, bucket count arrays)
and render them as grids of donut charts or bar charts, one subplot per model.
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_donut_row_with_other(verbalized_metadata, models, threshold_pct=8,
                               figsize_per_chart=(4, 4), color_dict=None, ncols=4):
    """Draw a grid of donut charts showing verbalized confidence distributions.

    Labels whose share is below *threshold_pct* % of the total are collapsed
    into a single ``Other`` slice.  A consistent colour palette is built
    across all charts so the same confidence value always maps to the same
    colour.

    Args:
        verbalized_metadata: Dict of ``model_id`` → ``pd.Series`` of value
            counts (as returned by ``Series.value_counts()``).
        models: List of model dicts with ``"id"`` and ``"shortname"`` keys;
            used to map model IDs to display names.
        threshold_pct: Minimum percentage a label must reach to get its own
            slice; smaller labels are merged into ``Other``.
        figsize_per_chart: ``(width, height)`` in inches for each subplot.
        color_dict: Optional mapping of label value → matplotlib colour
            override.  Labels not present here are assigned from ``"tab10"``.
        ncols: Maximum number of columns in the output grid.

    Returns:
        matplotlib.figure.Figure
    """
    color_dict = color_dict or {}

    # Collect labels visible in at least one chart so the palette is stable.
    all_labels = set()
    for vc in verbalized_metadata.values():
        total = vc.sum()
        mask = (vc / total * 100) >= threshold_pct
        all_labels |= set(vc.index[mask])
    all_labels = sorted(all_labels)

    cmap = plt.get_cmap("tab10")
    other_labels = [lbl for lbl in all_labels if lbl not in color_dict]
    color_map = {lbl: cmap(i % 10) for i, lbl in enumerate(other_labels)}
    color_map.update(color_dict)
    other_color = "lightgrey"

    model_ids = list(verbalized_metadata.keys())
    n = len(model_ids)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_chart[0] * ncols, figsize_per_chart[1] * nrows),
    )
    axes = axes.ravel() if n > 1 else [axes]

    for ax, model_id in zip(axes, model_ids):
        model = next(m for m in models if model_id in m["id"])
        vc = verbalized_metadata[model_id]
        total = vc.sum()
        mask = (vc / total * 100) >= threshold_pct
        big_labels = list(vc.index[mask])
        big_sizes = vc.loc[big_labels].values
        other_size = total - big_sizes.sum()

        sizes = list(big_sizes) + [other_size]
        colors = [color_map[lbl] for lbl in big_labels] + [other_color]
        labels_for_plot = [str(lbl) for lbl in big_labels] + [f"Other ({len(vc) - len(big_labels)})"]

        ax.pie(sizes, labels=labels_for_plot, colors=colors,
               autopct="%1.1f%%", pctdistance=0.74,
               startangle=90, counterclock=False, wedgeprops=dict(width=0.5))
        ax.set_title(model["shortname"], pad=22, fontweight="bold")
        ax.axis("equal")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout(pad=3.0)
    return fig


def plot_bucket_distribution(ptrue_bucket_counts, models, figsize_per_chart=(4, 3), ncols=4):
    """Plot a grid of bar charts showing the P(True) bucket-count distribution per model.

    Each subplot shows how many items fall into each confidence bin for the
    P(True) metric (summed across all datasets).  Bars are coloured by count
    intensity using a blue (instruct) or green (reasoning) colour map.

    Args:
        ptrue_bucket_counts: Dict of ``model_id`` → ``np.ndarray`` of bucket
            counts (all arrays must have the same length).
        models: List of model dicts with ``"name"`` and ``"type"`` keys; used
            to select the colour palette per model.
        figsize_per_chart: ``(width, height)`` in inches for each subplot.
        ncols: Maximum number of columns in the output grid.

    Returns:
        matplotlib.figure.Figure
    """
    model_ids = list(ptrue_bucket_counts.keys())
    n_models = len(model_ids)
    n_bins = len(next(iter(ptrue_bucket_counts.values())))
    bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins

    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_chart[0] * ncols, figsize_per_chart[1] * nrows),
    )
    axes = axes.ravel() if n_models > 1 else [axes]

    for idx, (ax, model_id) in enumerate(zip(axes, model_ids)):
        model_type = next(m for m in models if model_id in m["name"])["type"]
        cmap = plt.colormaps["Blues" if model_type == "instruct" else "Greens"]
        counts = ptrue_bucket_counts[model_id]
        norm_counts = counts / counts.max()
        colors = [cmap(c * 0.8 + 0.2) for c in norm_counts]

        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.bar(bin_centers, counts, width=1.0 / n_bins,
               color=colors, edgecolor="black", zorder=3)
        ax.set_title(model_id, fontsize=18, fontweight="bold", y=1.03)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence", fontsize=16)
        if idx % ncols == 0:
            ax.set_ylabel("Count", fontsize=16)

    for ax in axes[n_models:]:
        ax.axis("off")
    plt.tight_layout(pad=3.0)
    return fig
