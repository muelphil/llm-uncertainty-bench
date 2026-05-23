"""Calibration-correlation swarm / violin plot.

Provides ``plot_calibration_correlation_swarm``, which visualises the
Pearson correlation between bin-confidence midpoints and bucket accuracies
for every (model, dataset) pair, grouped by UQ method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from util.config import apply_matplotlib_defaults


#: Scatter colours by model type, matching the project-wide convention.
_TYPE_COLORS = {
    "instruct":  "tab:blue",
    "reasoning": "tab:green",
}


def _beeswarm_offsets(y_values, y_radius, xy_aspect, max_half_width=None):
    """Return per-point horizontal offsets for a non-overlapping beeswarm column.

    Points are sorted by y value and placed at the smallest horizontal offset
    that prevents visual overlap, treating each point as an ellipse with
    semi-axes ``(xy_aspect * y_radius, y_radius)`` in data coordinates.
    When *max_half_width* is given, offsets exceeding that bound are skipped;
    points that cannot be placed without overlap inside the bound fall back to
    x = 0 (centre, accepting overlap) rather than bleeding into adjacent groups.

    Args:
        y_values: 1-D numpy array of y positions (data coordinates).
        y_radius: Effective point radius in y data units.
        xy_aspect: Ratio ``x_radius / y_radius``.  Set to the ratio of
            data-units-per-inch in y vs x so that the collision ellipse
            maps to a circle on screen.
        max_half_width: Maximum allowed absolute x offset (data units).
            ``None`` means no limit.

    Returns:
        np.ndarray: x offsets in data coordinates, same length and order
        as *y_values*.
    """
    ry = float(y_radius)
    rx = xy_aspect * ry
    dx_step = 2.0 * rx

    order = np.argsort(y_values)
    offsets = np.zeros(len(y_values))
    placed_xy = []

    for idx in order:
        y = y_values[idx]
        placed = False
        for col in range(100):
            candidates = [0.0] if col == 0 else [col * dx_step, -col * dx_step]
            for x_try in candidates:
                if max_half_width is not None and abs(x_try) > max_half_width:
                    continue
                if all(
                    (x_try - px) ** 2 / rx ** 2 + (y - py) ** 2 / ry ** 2 >= 4.0
                    for px, py in placed_xy
                ):
                    offsets[idx] = x_try
                    placed_xy.append((x_try, y))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            offsets[idx] = 0.0
            placed_xy.append((0.0, y))

    return offsets


def plot_calibration_correlation_swarm(
    prepared_data,
    models,
    datasets,
    uq_methods,
    figures_dir,
    narrow=False,
):
    """Plot calibration correlation (Pearson r) by UQ method.

    For each UQ method a grey violin shows the distribution of Pearson
    correlations across all (model, dataset) pairs.  Individual points are
    placed on top and coloured by model type.  A solid bar marks the median
    for each method, drawn last so it is never occluded by scatter points.

    Two layout modes are available via *narrow*:

    * ``narrow=False`` (default): wide figure, random horizontal jitter,
      suitable for full-page or slide use.
    * ``narrow=True``: compact figure (~5.5 in wide) optimised for
      two-column paper layouts.  Uses a deterministic beeswarm placement
      so that points never overlap (and never bleed into neighbouring
      violin groups), uses line-broken x-axis labels (spaces replaced by
      newlines), shows only ±1 / ±0.5 / 0 y-ticks, and places the legend
      in a single row below the x-axis labels.

    Args:
        prepared_data: Nested dict ``prepared_data[dataset_id][model_id]``
            as produced by the data-preparation notebook.  Each entry must
            contain a ``"cal"`` sub-dict keyed by UQ method ID, with a
            ``"calibration_correlation"`` float value (or ``None`` on
            failure).
        models: List of model dicts with keys ``"id"`` and ``"type"``.
        datasets: List of dataset dicts with key ``"id"``.
        uq_methods: List of UQ method dicts with keys ``"id"`` and
            ``"label"``.
        figures_dir: ``pathlib.Path`` (or str) pointing to the directory
            where figures are saved.  Unused inside this function — saving
            is intentionally left to the caller.
        narrow: When ``True``, use the compact paper-column layout.
            Defaults to ``False``.

    Returns:
        matplotlib.figure.Figure: The produced figure.
    """
    apply_matplotlib_defaults()

    n_methods = len(uq_methods)
    x_positions = list(range(n_methods))

    # ------------------------------------------------------------------ #
    # Collect per-method lists of (r, model_type) pairs                   #
    # ------------------------------------------------------------------ #
    method_data = {}
    for uq in uq_methods:
        uq_id = uq["id"]
        points = []
        for dataset in datasets:
            ds_id = dataset["id"]
            for model in models:
                mid = model["id"]
                cal_entry = prepared_data[ds_id][mid]["cal"].get(uq_id)
                if cal_entry is None:
                    continue
                r = cal_entry.get("calibration_correlation")
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                points.append((float(r), model["type"]))
        method_data[uq_id] = points

    # ------------------------------------------------------------------ #
    # Layout parameters                                                    #
    # ------------------------------------------------------------------ #
    if narrow:
        figsize         = (5.5, 4.5)
        violin_width    = 0.7
        dot_s           = 22
        dot_alpha       = 0.75
        xtick_fontsize  = 13
        ytick_fontsize  = 11
        ylabel_fontsize = 12
        median_hw       = 0.19   # half-width of median bar in data units
        median_lw       = 2.5
        # Beeswarm geometry for figsize=(5.5, 4.5), subplots_adjust below:
        #   axis ≈ 4.68 in wide × 3.15 in tall
        #   x: 4 data units → 4.68/4 = 1.17 in/unit × 72 = 84.2 pt/unit
        #   y: 2.24 data units → 3.15/2.24 = 1.41 in/unit × 72 = 101 pt/unit
        #   dot_s=22 → diameter ≈ √22 ≈ 4.7 pt → radius ≈ 2.35 pt
        #   y_radius = 2.35/101 ≈ 0.023 data units
        #   xy_aspect = 101/84.2 ≈ 1.20
        _beeswarm_y_radius  = 0.023
        _beeswarm_xy_aspect = 1.20
        _beeswarm_max_hw    = 0.21   # cap: just inside violin half-width (0.25)
    else:
        figsize         = (max(n_methods * 2.2, 6), 5)
        violin_width    = 0.6
        dot_s           = 28
        dot_alpha       = 0.85
        xtick_fontsize  = 12
        ytick_fontsize  = None
        ylabel_fontsize = 12
        median_hw       = 0.22
        median_lw       = 2.0

    # ------------------------------------------------------------------ #
    # Draw – three separate passes so median bars always appear on top     #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(seed=0)

    # Pass 1: violins
    for x_pos, uq in zip(x_positions, uq_methods):
        r_values = np.array([p[0] for p in method_data[uq["id"]]])
        if len(r_values) < 2:
            continue
        parts = ax.violinplot(
            [r_values],
            positions=[x_pos],
            widths=violin_width,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#cccccc")
            body.set_edgecolor("#888888")
            body.set_alpha(0.8)
            body.set_linewidth(0.8)

    # Pass 2: scatter points
    for x_pos, uq in zip(x_positions, uq_methods):
        points = method_data[uq["id"]]
        if not points:
            continue
        r_values = np.array([p[0] for p in points])
        types    = [p[1] for p in points]

        if narrow:
            x_offsets = _beeswarm_offsets(
                r_values, _beeswarm_y_radius, _beeswarm_xy_aspect,
                max_half_width=_beeswarm_max_hw,
            )
        else:
            x_offsets = rng.uniform(-0.12, 0.12, size=len(r_values))

        for r_val, m_type, x_off in zip(r_values, types, x_offsets):
            ax.scatter(
                x_pos + x_off, r_val,
                color=_TYPE_COLORS.get(m_type, "tab:grey"),
                s=dot_s, zorder=5, alpha=dot_alpha,
                edgecolors="none",
            )

    # Pass 3: median bars – drawn last so they are never occluded
    for x_pos, uq in zip(x_positions, uq_methods):
        points = method_data[uq["id"]]
        if not points:
            continue
        median_r = float(np.median([p[0] for p in points]))
        ax.hlines(
            median_r, x_pos - median_hw, x_pos + median_hw,
            colors="#222222", linewidths=median_lw, zorder=6,
        )

    # ------------------------------------------------------------------ #
    # Reference lines & axis limits                                        #
    # ------------------------------------------------------------------ #
    ax.axhline( 0.0, color="black",   linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline( 1.0, color="#aaaaaa", linewidth=0.6, linestyle=":",  zorder=1)
    ax.axhline(-1.0, color="#aaaaaa", linewidth=0.6, linestyle=":",  zorder=1)

    ax.set_xlim(-0.6, n_methods - 0.4)
    ax.set_ylim(-1.12, 1.12)

    # ------------------------------------------------------------------ #
    # Axis labels & ticks                                                  #
    # ------------------------------------------------------------------ #
    ax.set_xticks(x_positions)

    if narrow:
        # Break labels on spaces so they stack vertically and stay centred.
        wrapped = [uq["label"].replace(" ", "\n") for uq in uq_methods]
        ax.set_xticklabels(wrapped, fontsize=xtick_fontsize,
                           ha="center", va="top", multialignment="center")
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels([f"{v:.1f}" for v in [-1, -0.5, 0, 0.5, 1]],
                           fontsize=ytick_fontsize)
        ax.set_ylabel("Pearson $r$", fontsize=ylabel_fontsize)
    else:
        ax.set_xticklabels([uq["label"] for uq in uq_methods],
                           fontsize=xtick_fontsize)
        ax.set_ylabel("Pearson $r$ (confidence vs. accuracy)",
                      fontsize=ylabel_fontsize)

    # ------------------------------------------------------------------ #
    # Legend                                                               #
    # ------------------------------------------------------------------ #
    legend_handles = [
        mpatches.Patch(color=color, label=m_type.capitalize())
        for m_type, color in _TYPE_COLORS.items()
    ]
    if narrow:
        ax.legend(
            handles=legend_handles,
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=len(legend_handles),
            frameon=False,
        )
        # tight_layout first so matplotlib knows the label heights, then
        # override bottom to give room for the wrapped labels + legend.
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.24)
    else:
        ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
        plt.tight_layout()

    return fig
