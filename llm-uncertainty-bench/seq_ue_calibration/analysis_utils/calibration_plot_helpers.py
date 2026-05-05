"""Calibration plot helpers: subplot renderers and grid-plot builders.

Wraps the lower-level ``generate_grid_plot`` and ``plot_calibration_curve``
utilities with project-standard renderers and layouts.

Note: ``config`` and ``plot_calibration_curve`` are resolved via the sys.path
entries added at the top of ``analysis.md`` before this module is imported.
"""

from config import CALIBRATION_PLOT_COLORS
from generate_grid_plot import generate_grid_plot
from plot_calibration_curve import plot_calibration_curve


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


def build_calibration_grid_by_model(cal_data, model, datasets, metrics,
                                    with_table, skip_annotation=False,
                                    with_title=True, **kwargs):
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
    return generate_grid_plot(
        cell_data, row_titles, col_titles,
        plot_subplot_fn=plot_calibration_subplot,
        plot_table_fn=plot_calibration_stats_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        **kwargs,
    )


def build_calibration_grid_by_uq_method(cal_data, models, datasets, uq_method,
                                     with_table, skip_annotation=False,
                                     with_title=True, **kwargs):
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
    return generate_grid_plot(
        cell_data, row_titles, col_titles,
        plot_subplot_fn=plot_calibration_subplot,
        plot_table_fn=plot_calibration_stats_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        **kwargs,
    )
