"""Calibration plot helpers: subplot renderers and grid-plot builders.

Wraps the lower-level ``generate_grid_plot`` and ``plot_calibration_curve``
utilities with project-standard renderers and layouts.
"""

from util.config import CALIBRATION_PLOT_COLORS
from util.generate_grid_plot import generate_grid_plot
from calibration_visualization import plot_calibration_curve


def plot_calibration_subplot(ax, data_item, model_type):
    """Render a calibration curve on *ax*.

    Delegates to ``plot_calibration_curve`` with the colour palette that
    corresponds to *model_type*.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        data_item: Calibration metrics dict with keys ``"bin_confidences"``,
            ``"bucket_accuracies"``, and ``"bucket_counts"``.
        model_type: Model type string (``"instruct"`` or ``"reasoning"``);
            selects the colour palette via ``CALIBRATION_PLOT_COLORS``.
    """
    plot_calibration_curve(
        data_item["bin_confidences"], data_item["bucket_accuracies"], data_item["bucket_counts"],
        ax=ax, colormap=CALIBRATION_PLOT_COLORS[model_type],
        fontsize=20, tick_fontsize=18,
        xlabel="Confidence Bins", ylabel="Accuracy in Bin",
    )


def plot_calibration_stats_table(ax, data_item):
    """Render a per-subplot statistics table on *ax* (axis is turned off).

    Displays ECE, AUROC, invalid answer / metric-score counts, normalised
    entropy of bucket counts, and overall accuracy.

    Args:
        ax: Matplotlib ``Axes`` to draw on; its axis display is disabled.
        data_item: Calibration metrics dict with keys ``"ece"``, ``"auroc"``,
            ``"invalid_answers"``, ``"total_items"``, ``"invalid_uq_method_scores"``,
            ``"normalized_entropy"``, and ``"accuracy"``.
    """
    table_rows = [
        ["ECE", f"{data_item['ece']:.4f}"],
        ["AUROC", f"{data_item['auroc']:.4f}"],
        ["Invalid Answers", f"{data_item['invalid_answers']}/{data_item['total_items']}"],
        ["Invalid UQ Method Scores",
         f"{data_item['invalid_uq_method_scores']}/{data_item['total_items'] - data_item['invalid_answers']}"],
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
    """Render a relplot reliability diagram on *ax*.

    Uses ``relplot.rel_diagram`` (kernel-smoothed reliability diagram with
    smECE) instead of the project's custom binned calibration curve.
    Requires ``"correct"`` and ``"certainties"`` arrays in *data_item* —
    these are stored in ``cal_data`` by ``compute_calibration_metrics``.

    The confidence band (bootstrapped 95 % CI around the regression line) is
    disabled by default for speed.  To enable it, change
    ``plot_confidence_band=False`` to ``plot_confidence_band=True``; each
    subplot will then run ~200 bootstrap iterations.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        data_item: Calibration metrics dict; must contain ``"correct"``
            (1-D int/bool array) and ``"certainties"`` (1-D float array in
            [0, 1]).
        model_type: Unused; present to satisfy the
            ``plot_subplot_fn(ax, data, model_type)`` interface required by
            ``generate_grid_plot``.
    """
    import relplot as rp
    # Use prepare + plot separately so we can pass the existing ax's figure.
    # Calling rp.rel_diagram(ax=ax) without fig would trigger a new figure
    # creation (relplot's guard: `if ax is None or fig is None: ...`).
    diagram = rp.prepare_rel_diagram(
        f=data_item["certainties"],
        y=data_item["correct"].astype(float),
        plot_confidence_band=False,  # set to True to enable bootstrap CI bands (slow, ~200 iters)
        report_CE_std=False,         # set to True to show ±CI on smECE label (requires bootstrapping)
    )
    rp.plot_rel_diagram(diagram, fig=ax.get_figure(), ax=ax)


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
