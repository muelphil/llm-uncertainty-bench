"""Calibration plot helpers: subplot renderers and grid-plot builders.

Wraps the lower-level ``generate_grid_plot`` and ``plot_calibration_curve``
utilities with project-standard renderers and layouts.
"""

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from util.config import CALIBRATION_PLOT_COLORS
from util.generate_grid_plot import generate_grid_plot
from util.plot_empty import plot_empty
from calibration_visualization import plot_calibration_curve

#: Single colours for relplot reliability diagrams, keyed by model type.
RELPLOT_COLORS = {"instruct": "tab:blue", "reasoning": "tab:green", "base": "tab:orange"}


def plot_calibration_subplot(ax, data_item, model_type):
    """Render a calibration curve on *ax*.

    Delegates to ``plot_calibration_curve`` with the colour palette that
    corresponds to *model_type*.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        data_item: Calibration metrics dict with keys ``"bin_confidences"``,
            ``"bucket_accuracies"``, ``"bucket_counts"``, ``"ece"``, and
            optionally ``"bucket_acc_ci"`` and ``"ece_ci"``.
        model_type: Model type string (``"instruct"`` or ``"reasoning"``);
            selects the colour palette via ``CALIBRATION_PLOT_COLORS``.
    """
    plot_calibration_curve(
        data_item["bin_confidences"], data_item["bucket_accuracies"], data_item["bucket_counts"],
        ax=ax, colormap=CALIBRATION_PLOT_COLORS[model_type],
        fontsize=20, tick_fontsize=18,
        xlabel="Confidence Bins", ylabel="Accuracy in Bin",
        ece=data_item.get("ece"),
        bucket_acc_ci=data_item.get("bucket_acc_ci"),
        ece_ci=data_item.get("ece_ci"),
    )


def plot_calibration_stats_table(ax, data_item, total_items=None, invalid_answers=None,
                                  smooth_ece=None, smooth_ece_ci_width=None):
    """Render a per-subplot statistics table on *ax* (axis is turned off).

    Displays ECE, optionally smECE (with ± confidence interval), AUROC,
    invalid-answer and invalid-score counts, normalised entropy of bucket
    counts, and overall accuracy.

    When *total_items* and *invalid_answers* are ``None`` they are read from
    ``data_item`` (backward-compatible with the legacy cal_data structure where
    those fields were merged into each calibration-metrics dict).

    Args:
        ax: Matplotlib ``Axes`` to draw on; its axis display is disabled.
        data_item: Calibration metrics dict with at least keys ``"ece"``,
            ``"auroc"``, ``"invalid_uq_method_scores"``, ``"normalized_entropy"``,
            and ``"accuracy"``.  Keys ``"total_items"`` and ``"invalid_answers"``
            are also read if the corresponding keyword arguments are ``None``.
        total_items: Total number of items for the dataset/model entry.
            Falls back to ``data_item["total_items"]`` when ``None``.
        invalid_answers: Number of items with invalid answers.
            Falls back to ``data_item["invalid_answers"]`` when ``None``.
        smooth_ece: Smooth ECE scalar from a relplot reliability diagram
            (``relplot_diagram["ce"]``).  When provided, an extra "smECE" row
            is inserted immediately after the "ECE" row.
        smooth_ece_ci_width: Half-width of the 95 % bootstrap confidence
            interval for smECE (``relplot_diagram["ce_ci_width"]``).  When
            provided alongside *smooth_ece*, the row shows
            ``"X.XXXX ± Y.YYYY"``; otherwise just ``"X.XXXX"``.
    """
    _total   = total_items   if total_items   is not None else data_item["total_items"]
    _invalid = invalid_answers if invalid_answers is not None else data_item["invalid_answers"]
    _valid   = _total - _invalid

    table_rows = [
        ["ECE", f"{data_item['ece']:.4f}"],
    ]
    ece_ci = data_item.get("ece_ci")
    if ece_ci is not None:
        table_rows.append(["ECE CI", f"[{ece_ci[0]:.4f}\u2013{ece_ci[1]:.4f}]"])
    if smooth_ece is not None:
        if smooth_ece_ci_width is not None:
            smece_str = f"{float(smooth_ece):.4f} ± {float(smooth_ece_ci_width):.4f}"
        else:
            smece_str = f"{float(smooth_ece):.4f}"
        table_rows.append(["smECE", smece_str])
    table_rows += [
        ["AUROC", f"{data_item['auroc']:.4f}"],
        ["Invalid Answers", f"{_invalid}/{_total}"],
        ["Invalid UQ Method Scores",
         f"{data_item['invalid_uq_method_scores']}/{_valid}"],
        ["Norm. Entropy of Bucket Counts", f"{data_item['normalized_entropy']:.4f}"],
        ["Accuracy", f"{data_item['accuracy']:.4f}"],
    ]
    table = ax.table(cellText=table_rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (_, col), cell in table.get_celld().items():
        if col == 0:
            cell.PAD = 0.02
            cell.get_text().set_ha("right")
            cell.set_width(0.65)
        elif col == 1:
            cell.PAD = 0.04
            cell.get_text().set_ha("left")
            cell.set_width(0.35)
    ax.axis("off")

def plot_relplot_subplot(ax, data_item, model_type):
    """Render a relplot reliability diagram on *ax* using a pre-computed diagram.

    The diagram object must already be stored in ``data_item["relplot_diagram"]``
    (produced by ``relplot.prepare_rel_diagram`` in the data-preparation
    notebook) so that this function never re-runs the expensive kernel fitting
    step.

    The colour is chosen from ``RELPLOT_COLORS`` based on *model_type*
    (``"instruct"`` → blue, ``"reasoning"`` → green, ``"base"`` → orange).

    Axis labels and ticks are overridden after the relplot call to match the
    style of ``plot_calibration_subplot`` (font size 20/18, no 0.0 tick,
    human-readable axis names).

    Global ``mpl.rcParams`` are saved and restored around the relplot call
    because ``relplot.plot_rel_diagram`` with ``use_default_style=True``
    internally calls ``mpl.rc_file_defaults()`` which would otherwise corrupt
    the global matplotlib state and break any plots rendered afterward.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        data_item: Calibration metrics dict; must contain
            ``"relplot_diagram"`` (returned by ``rp.prepare_rel_diagram``).
        model_type: Model type string (``"instruct"``, ``"reasoning"``, or
            ``"base"``); selects the diagram colour via ``RELPLOT_COLORS``.
    """
    import relplot as rp
    diagram = data_item["relplot_diagram"]
    color = RELPLOT_COLORS.get(model_type, "tab:blue")

    # Save and restore rcParams: relplot's set_default_style() calls
    # mpl.rc_file_defaults() which resets *all* rcParams to system defaults,
    # corrupting the global matplotlib state (font family, sizes, etc.).
    saved_params = {k: v for k, v in mpl.rcParams.items()}
    rp.plot_rel_diagram(diagram, fig=ax.get_figure(), ax=ax, color=color,
                        plot_labels=False)
    mpl.rcParams.update(saved_params)

    # Custom ticks: remove 0.0 to avoid the overlapping corner tick.
    custom_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(labelsize=18)

    # Axis labels matching the calibration-plot style.
    ax.set_xlabel("Confidence Scores", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)


def build_calibration_grid_by_model(cal_data, model, datasets, metrics,
                                    with_table, skip_annotation=False,
                                    with_title=True, subplot_fn=None,
                                    subplot_aspect=1, **kwargs):
    """Build a calibration grid (dataset rows × metric columns) for one model.

    Reads pre-computed calibration data from *cal_data* and delegates to
    ``generate_grid_plot`` using the standard subplot and stats-table renderers.

    Args:
        cal_data: Nested dict ``cal_data[model_id][dataset_id][metric_id]``
            of calibration metrics dicts (or ``None`` on failure).
        model: Model dict with at least ``"id"``, ``"shortname"``, and
            ``"type"`` keys.
        datasets: Ordered list of dataset dicts with ``"id"`` keys.
        metrics: Ordered list of metric dicts with ``"id"`` and ``"label"`` keys.
        with_table: If ``True``, include per-subplot statistics tables.
        skip_annotation: Forwarded to ``generate_grid_plot``.
        with_title: If ``True``, set a figure-level title with the model name.
        subplot_fn: Callable ``(ax, data_item, model_type) -> None`` used to
            draw each subplot.  Defaults to ``plot_calibration_subplot``.  Pass
            ``plot_relplot_subplot`` for kernel-smoothed reliability diagrams.
        subplot_aspect: Forwarded to ``generate_grid_plot``.  Pass ``None``
            when *subplot_fn* manages its own layout (e.g. relplot).
        **kwargs: Additional keyword arguments forwarded to ``generate_grid_plot``.

    Returns:
        matplotlib.figure.Figure
    """
    cell_data = [
        [(cal_data[model["id"]][ds["id"]][m["id"]], model["type"]) for m in metrics]
        for ds in datasets
    ]
    row_titles = [ds["id"].replace("_", "-") for ds in datasets]
    col_titles = [m["label"] for m in metrics]
    plot_title = f"Calibration Plots for Model {model['shortname']}" if with_title else None
    _subplot_fn = subplot_fn if subplot_fn is not None else plot_calibration_subplot
    return generate_grid_plot(
        cell_data, row_titles, col_titles,
        plot_subplot_fn=_subplot_fn,
        plot_table_fn=plot_calibration_stats_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        subplot_aspect=subplot_aspect,
        **kwargs,
    )


def build_calibration_grid_by_uq_method(cal_data, models, datasets, uq_method,
                                     with_table, skip_annotation=False,
                                     with_title=True, subplot_fn=None,
                                     subplot_aspect=1, **kwargs):
    """Build a calibration grid (dataset rows × model columns) for one uq_method.

    Reads pre-computed calibration data from *cal_data* and delegates to
    ``generate_grid_plot`` using the standard subplot and stats-table renderers.

    Args:
        cal_data: Nested dict ``cal_data[model_id][dataset_id][uq_method_id]``
            of calibration uq_methods dicts (or ``None`` on failure).
        models: Ordered list of model dicts with ``"id"``, ``"shortname"``,
            and ``"type"`` keys.
        datasets: Ordered list of dataset dicts with ``"id"`` keys.
        uq_method: Metric dict with ``"id"`` and ``"label"`` keys.
        with_table: If ``True``, include per-subplot statistics tables.
        skip_annotation: Forwarded to ``generate_grid_plot``.
        with_title: If ``True``, set a figure-level title with the uq_method name.
        subplot_fn: Callable ``(ax, data_item, model_type) -> None`` used to
            draw each subplot.  Defaults to ``plot_calibration_subplot``.  Pass
            ``plot_relplot_subplot`` for kernel-smoothed reliability diagrams.
        subplot_aspect: Forwarded to ``generate_grid_plot``.  Pass ``None``
            when *subplot_fn* manages its own layout (e.g. relplot).
        **kwargs: Additional keyword arguments forwarded to ``generate_grid_plot``.

    Returns:
        matplotlib.figure.Figure
    """
    cell_data = [
        [(cal_data[m["id"]][ds["id"]][uq_method["id"]], m["type"]) for m in models]
        for ds in datasets
    ]
    row_titles = [ds["id"].replace("_", "-") for ds in datasets]
    col_titles = [m["shortname"] for m in models]
    plot_title = f"Calibration Plots for Uncertainty Metric {uq_method['label']}" if with_title else None
    _subplot_fn = subplot_fn if subplot_fn is not None else plot_calibration_subplot
    return generate_grid_plot(
        cell_data, row_titles, col_titles,
        plot_subplot_fn=_subplot_fn,
        plot_table_fn=plot_calibration_stats_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        subplot_aspect=subplot_aspect,
        **kwargs,
    )


def assemble_cal_grid(
    subplot_fns_2d,
    row_titles,
    col_titles,
    table_fns_2d=None,
    plot_title=None,
    subplot_size=(6, 6),
    subplot_aspect=1,
    skip_annotation=False,
    row_col_titles_font_size=36,
    title_font_size=44,
):
    """Assemble a grid of pre-built subplot callables into a Figure.

    Each callable must accept a single ``ax`` argument and render itself onto
    it — typically a ``functools.partial`` created during the subplot
    pre-building step.  ``plot_empty`` can be used for missing/failed entries.

    Layout mirrors ``generate_grid_plot``: calibration rows optionally
    followed by narrower table rows (height ratio 1 : 0.15).

    Args:
        subplot_fns_2d: 2-D list ``subplot_fns_2d[row][col]`` of callables
            ``(ax) -> None``.  Use ``plot_empty`` for cells that should be
            rendered as empty placeholders.
        row_titles: List of row title strings (one per data row).
        col_titles: List of column title strings (one per column).
        table_fns_2d: Optional 2-D list of table callables with the same
            shape as *subplot_fns_2d*.  When provided, a narrow table row is
            inserted after each data row.
        plot_title: Optional figure-level title string.
        subplot_size: ``(width, height)`` in inches for each calibration cell.
            Table rows use 2 extra inches of total figure height per data row.
        subplot_aspect: Aspect ratio applied to each calibration ``Axes``
            after drawing.  Pass ``None`` to skip (required for relplot
            subplots that manage their own layout).
        skip_annotation: When ``True``, omit all row/column title annotations
            and the figure title.
        row_col_titles_font_size: Font size for row and column headers.
        title_font_size: Font size for the figure-level title.

    Returns:
        matplotlib.figure.Figure
    """
    with_table = table_fns_2d is not None
    n_rows_data = len(subplot_fns_2d)
    n_cols = len(subplot_fns_2d[0]) if n_rows_data else 0
    w, h = subplot_size
    figsize = (n_cols * w, n_rows_data * (h + (2.0 if with_table else 0)))

    if with_table:
        height_ratios = []
        for _ in range(n_rows_data):
            height_ratios.extend([1, 0.15])
        fig, axes = plt.subplots(
            n_rows_data * 2, n_cols, figsize=figsize,
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        fig, axes = plt.subplots(n_rows_data, n_cols, figsize=figsize)

    # Normalise axes to a 2-D array for uniform indexing.
    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_rows_data == 1 else axes.reshape(-1, 1)

    for i in range(n_rows_data):
        for j in range(n_cols):
            subplot_row = 2 * i if with_table else i
            ax = axes[subplot_row, j]
            fn = subplot_fns_2d[i][j]
            try:
                fn(ax)
                if subplot_aspect is not None:
                    ax.set_aspect(subplot_aspect)
            except Exception:
                print(f"subplot unplottable at row={i}, col={j}")
                plot_empty(ax)

            if with_table:
                table_ax = axes[2 * i + 1, j]
                try:
                    table_fns_2d[i][j](table_ax)
                except Exception:
                    print(f"table unplottable at row={i}, col={j}")
                    plot_empty(table_ax)

            if not skip_annotation and i == 0:
                ax.set_title(
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
    return fig
