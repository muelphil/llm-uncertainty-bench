"""
Grid-of-subplots calibration plot builder.

``generate_grid_plot`` arranges a 2-D grid of calibration subplots.
Rows correspond to datasets and columns to models (or metrics), depending
on the calling experiment.  Optionally a stats table is appended below each
calibration row.

The function delegates actual subplot and table drawing to caller-supplied
callables (``plot_subplot_fn``, ``plot_table_fn``), keeping the layout logic
experiment-agnostic.
"""

import numpy as np
import matplotlib.pyplot as plt

from util.plot_empty import plot_empty


def generate_grid_plot(
    cell_data,
    row_titles,
    col_titles,
    plot_subplot_fn,
    plot_table_fn=None,
    sharex=False,
    sharey=False,
    with_table=False,
    show_axis_labels=False,
    skip_annotation=False,
    row_col_titles_font_size=36,
    title_font_size=44,
    plot_title=None,
    subplot_aspect=1,
):
    """
    Build a grid of calibration subplots and return the ``plt`` module.

    Layout
    ------
    - Rows map to entries in *cell_data*; columns map to entries within each row.
    - When *with_table* is ``True`` every calibration row is followed by a
      narrower stats-table row (height ratio 1 : 0.15).
    - Row and column titles are annotated as rotated text / subplot titles.

    Args:
        cell_data (list[list[tuple]]): 2-D list of ``(data_dict, model_type)``
            tuples.  ``data_dict`` is passed verbatim to *plot_subplot_fn*.
            If the first element is ``None``, a placeholder is rendered.
        row_titles (list[str]): Labels shown on the left of each row.
        col_titles (list[str]): Labels shown above each column.
        plot_subplot_fn (callable): ``(ax, data, model_type) -> None``.
            Renders a single calibration subplot on *ax*.
        plot_table_fn (callable | None): ``(ax, data) -> None``.
            Renders a stats table on *ax*.  Required when *with_table* is
            ``True``.
        sharex (bool): Share the x-axis across all subplots.
        sharey (bool): Share the y-axis across all subplots.
        with_table (bool): Append a table row below each calibration row.
        show_axis_labels (bool): Always show x/y axis labels (default: hide
            redundant labels on inner subplots).
        skip_annotation (bool): Omit row/column title annotations entirely.
        row_col_titles_font_size (int): Font size for row and column headers.
        title_font_size (int): Font size for the overall figure title.
        subplot_aspect (int | float | None): Aspect ratio applied to each
            calibration subplot after drawing.  Pass ``None`` to skip setting
            the aspect ratio (required for relplot subplots, which manage their
            own layout).  Default is ``1`` (square, matching the legacy
            calibration-curve style).
        plot_title (str | None): Overall figure title; omitted when ``None``.

    Returns:
        The ``matplotlib.pyplot`` module (caller can call ``.savefig`` /
        ``.show`` directly).
    """
    original_n_rows = len(cell_data)
    n_cols = len(cell_data[0]) if original_n_rows else 0
    figsize = (n_cols * 6, original_n_rows * (6 + (2.0 if with_table else 0)))

    n_rows = original_n_rows * 2 if with_table else original_n_rows
    if with_table:
        height_ratios = []
        for _ in range(original_n_rows):
            height_ratios.extend([1, 0.15])
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey,
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey)

    # Normalise axes to a 2-D array for uniform indexing.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i in range(original_n_rows):
        for j in range(n_cols):
            subplot_ax = axes[2 * i if with_table else i, j]
            data, model_type = cell_data[i][j]
            if data is None:
                plot_empty(subplot_ax)
            else:
                try:
                    plot_subplot_fn(subplot_ax, data, model_type)
                    if subplot_aspect is not None:
                        subplot_ax.set_aspect(subplot_aspect)
                except Exception:
                    print(f"subplot unplottable at row={i}, col={j}")
                    plot_empty(subplot_ax)

            if with_table and data is not None and plot_table_fn is not None:
                plot_table_fn(axes[2 * i + 1, j], data)

            if not with_table and not show_axis_labels:
                if i < n_rows - 1:
                    subplot_ax.xaxis.label.set_color("none")
                if j > 0:
                    subplot_ax.yaxis.label.set_color("none")

            if i == 0 and not skip_annotation:
                subplot_ax.set_title(
                    col_titles[j],
                    fontsize=row_col_titles_font_size,
                    fontweight="bold",
                    y=1.05,
                )

        if not skip_annotation:
            front_ax = axes[2 * i if with_table else i, 0]
            front_ax.annotate(
                row_titles[i],
                xy=(-0.2, 0.5),
                xycoords="axes fraction",
                fontsize=row_col_titles_font_size,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
            )

    if plot_title and not skip_annotation:
        fig.text(0.5, 1.01, plot_title, ha="center", va="bottom",
                 fontsize=title_font_size, fontweight="bold")

    plt.tight_layout()
    return plt
