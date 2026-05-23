"""Calibration plot helpers for the label_prob_calibration experiment.

Provides high-level functions for building and saving calibration grids
(dataset × model) from pre-computed ``cal_data`` dicts, as well as the
normalization-comparison plot and relplot reliability diagrams.

``cal_data[pname][ds_id][m_id]`` is a dict with four keys for the 2×2
(normalize × chosen_only) combinations:

* ``"norm_chosen"``  – normalised probabilities, chosen-label only
* ``"norm_all"``     – normalised probabilities, all 4 labels
* ``"raw_chosen"``   – raw (un-normalised) probabilities, chosen-label only
* ``"raw_all"``      – raw (un-normalised) probabilities, all 4 labels

Each leaf dict contains binned calibration metrics plus a ``"relplot_diagram"``
key produced by ``relplot.prepare_rel_diagram``.

All functions delegate the low-level grid layout to
:func:`util.generate_grid_plot.generate_grid_plot`.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product

from calibration_visualization import plot_calibration_curve
from util.config import CALIBRATION_PLOT_COLORS
from util.generate_grid_plot import generate_grid_plot

#: Single colours for relplot reliability diagrams, keyed by model type.
RELPLOT_COLORS = {"instruct": "tab:blue", "reasoning": "tab:green", "base": "tab:orange"}


def _cal_key(normalize: bool, chosen_only: bool) -> str:
    """Return the ``cal_data[…][m_id]`` leaf key for this combination.

    Args:
        normalize: Whether L1 normalisation was applied.
        chosen_only: Whether only the argmax label is used.

    Returns:
        str: One of ``"norm_chosen"``, ``"norm_all"``, ``"raw_chosen"``, ``"raw_all"``.
    """
    return f"{'norm' if normalize else 'raw'}_{'chosen' if chosen_only else 'all'}"


def make_calibration_subplot_fn(ece_in_plot=False):
    """Return a grid-cell rendering function for calibration curves.

    The returned callable matches the ``plot_subplot_fn(ax, data, model_type)``
    signature expected by :func:`util.generate_grid_plot.generate_grid_plot`.

    Args:
        ece_in_plot: If ``True``, annotate each calibration curve with its ECE
            value.

    Returns:
        Callable: ``render_calibration_subplot(ax, data, model_type) -> None``.
    """
    def render_calibration_subplot(ax, data, model_type):
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            return
        plot_calibration_curve(
            data["bin_confidences"], data["bucket_accuracies"], data["bucket_counts"],
            ax=ax, colormap=CALIBRATION_PLOT_COLORS[model_type],
            fontsize=14, tick_fontsize=10,
            xlabel="Confidence Bins", ylabel="Accuracy in Bin",
            ece=data["ece"] if ece_in_plot else None,
            bucket_acc_ci=data.get("bucket_acc_ci"),
            ece_ci=data.get("ece_ci") if ece_in_plot else None,
        )
    return render_calibration_subplot


def plot_relplot_subplot(ax, data, model_type):
    """Render a relplot reliability diagram on *ax* using the pre-computed diagram.

    The ``relplot_diagram`` key in *data* must be the object returned by
    ``relplot.prepare_rel_diagram`` (stored during the data-loading loop).

    Global ``mpl.rcParams`` are saved and restored around the relplot call
    because ``relplot.plot_rel_diagram`` internally calls
    ``mpl.rc_file_defaults()`` which would otherwise corrupt the global
    matplotlib state (font family, sizes, background colour, etc.) and break
    subsequent plots.

    Axis labels and ticks are overridden after the relplot call to match the
    style of :func:`make_calibration_subplot_fn` (no 0.0 corner tick,
    human-readable axis names at consistent font sizes).

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        data: Calibration metrics dict; must contain a ``"relplot_diagram"`` key
            produced by ``rp.prepare_rel_diagram``.
        model_type: Model type string (``"instruct"``, ``"reasoning"``, or
            ``"base"``); selects the diagram colour via ``RELPLOT_COLORS``.
    """
    import relplot as rp

    diagram = data.get("relplot_diagram") if data is not None else None
    if diagram is None:
        ax.text(0.5, 0.5, "relplot N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        return

    color = RELPLOT_COLORS.get(model_type, "tab:blue")

    # Save and restore rcParams: relplot's set_default_style() calls
    # mpl.rc_file_defaults() which resets ALL rcParams to system defaults,
    # corrupting the global matplotlib state.
    saved_params = {k: v for k, v in mpl.rcParams.items()}
    rp.plot_rel_diagram(diagram, fig=ax.get_figure(), ax=ax, color=color,
                        plot_labels=False)
    mpl.rcParams.update(saved_params)

    # Custom ticks: remove 0.0 to avoid the overlapping corner tick.
    custom_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(custom_ticks)
    ax.set_yticks(custom_ticks)
    ax.tick_params(labelsize=10)

    # Axis labels matching the calibration-plot style.
    ax.set_xlabel("Confidence Scores", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)


def render_metrics_table(ax, data):
    """Render a per-subplot summary-statistics table on *ax*.

    Displays ECE, optionally smooth ECE with ± CI (when ``"relplot_diagram"``
    is present), AUROC, normalised entropy, invalid-answer count, accuracy,
    and label-probability statistics in a two-column matplotlib table.

    Args:
        ax: The matplotlib ``Axes`` to draw on (axis display is turned off).
        data: Calibration metrics dict (output of
            :func:`analysis_utils.calibration_compute.compute_calibration_metrics`).
            When a ``"relplot_diagram"`` key is present, ``smECE ± CI`` is
            inserted immediately after the ECE row.
    """
    from analysis_config import safe_format

    table_rows = [["ECE", safe_format(data["ece"])]]
    ece_ci = data.get("ece_ci")
    if ece_ci is not None:
        table_rows.append(["ECE CI", f"[{ece_ci[0]:.4f}\u2013{ece_ci[1]:.4f}]"])

    # smECE row: read from the embedded relplot_diagram when available.
    # rp_diag = data.get("relplot_diagram")
    # if rp_diag is not None:
    #     smece = rp_diag.get("ce")
    #     ci_width = rp_diag.get("ce_ci_width")
    #     if smece is not None:
    #         if ci_width is not None:
    #             smece_str = f"{float(smece):.4f} ± {float(ci_width):.4f}"
    #         else:
    #             smece_str = f"{float(smece):.4f}"
    #         table_rows.append(["smECE", smece_str])

    table_rows += [
        ["AUROC", safe_format(data["auroc"])],
        ["Norm. Entropy (15 bins)", safe_format(data["normalized_entropy"])],
        ["Norm. Entropy (100 bins)", safe_format(data.get("normalized_entropy_100", "N/A"))],
        ["Invalid Answer Count", f"{data['invalid_answers']}/{data['total_items']}"],
        ["Accuracy", safe_format(data["accuracy"])],
        ["Median of Label Probabilities", safe_format(data["label_prob_median"])],
        ["IQR of Label Probabilities", safe_format(data["label_prob_iqr"])],
        # ["Median of Sum of Label Probabilities", safe_format(data["label_prob_sum_median"])],
        # ["IQR of Sum of Label Probabilities", safe_format(data["label_prob_sum_iqr"])],
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


def build_calibration_grid(prompt, datasets, models, cal_data,
                           chosen_only, normalize, with_table,
                           skip_annotation=False, with_title=True,
                           ece_in_plot=False, subplot_fn=None, **kwargs):
    """Build a dataset × model calibration grid for one configuration.

    Reads pre-computed metrics from *cal_data* and assembles the grid via
    :func:`util.generate_grid_plot.generate_grid_plot`.

    Args:
        prompt: Prompt design dict with a ``basename`` and ``label`` key.
        datasets: List of dataset dicts (grid rows).
        models: List of model dicts (grid columns).
        cal_data: Nested metrics dict:
            ``cal_data[pname][ds_id][m_id][cal_key]`` where *cal_key*
            is produced by :func:`_cal_key`.
        chosen_only: If ``True``, use ``chosen``; otherwise ``all``.
        normalize: If ``True``, use the ``norm`` prefix.
        with_table: If ``True``, render a metrics table below each curve.
        skip_annotation: If ``True``, omit row/column title annotations.
        with_title: If ``True``, add a descriptive plot title.
        ece_in_plot: If ``True``, annotate each curve with its ECE value.
            Ignored when *subplot_fn* is provided.
        subplot_fn: Custom subplot renderer; defaults to
            :func:`make_calibration_subplot_fn`.  Pass
            :func:`plot_relplot_subplot` for reliability diagrams.
        **kwargs: Forwarded to :func:`util.generate_grid_plot.generate_grid_plot`.

    Returns:
        matplotlib.figure.Figure
    """
    key = _cal_key(normalize=normalize, chosen_only=chosen_only)

    def _safe_get(pname, ds_id, m_id, k):
        entry = cal_data.get(pname, {}).get(ds_id, {}).get(m_id)
        return entry.get(k) if isinstance(entry, dict) else None

    cell_data = [
        [
            (_safe_get(prompt["basename"], ds["id"], m["id"], key), m["type"])
            for m in models
        ]
        for ds in datasets
    ]

    plot_title = (
        f"Calibration of "
        f"{'Normalized' if normalize else 'Non-Normalized'} "
        f"Label Probabilities for {prompt['label']} "
        f"{'(Most Probable Label per Question Only)' if chosen_only else '(All Labels)'}"
    ) if with_title else None

    _subplot_fn = (subplot_fn if subplot_fn is not None
                   else make_calibration_subplot_fn(ece_in_plot=ece_in_plot))

    return generate_grid_plot(
        cell_data,
        row_titles=[ds["id"] for ds in datasets],
        col_titles=[m["shortname"] for m in models],
        plot_subplot_fn=_subplot_fn,
        plot_table_fn=render_metrics_table if with_table else None,
        with_table=with_table,
        skip_annotation=skip_annotation,
        plot_title=plot_title,
        **kwargs,
    )


def save_all_calibration_grids(prompt_designs, datasets, models, cal_data, figures_dir, extensions=['svg']):
    """Save calibration grid SVGs for all prompt × normalisation × label combinations.

    Standard binned-calibration and relplot variants are both saved.  File names:

    * ``cal_plot_prompt{idx}_table{0|1}_chosenonly{0|1}_norm{0|1}_mmlu_physics.svg``
    * ``cal_plot_prompt{idx}_table{0|1}_chosenonly{0|1}_norm{0|1}_mmlu_physics_relplot.svg``

    All files are written to ``figures_dir / "full_plots/"``.

    Args:
        prompt_designs: List of prompt design dicts.
        datasets: List of dataset dicts (grid rows).
        models: List of model dicts (grid columns).
        cal_data: Pre-computed metrics dict (see :func:`build_calibration_grid`).
        figures_dir: :class:`pathlib.Path` to the figures output directory.
    """
    for prompt in prompt_designs:
        for chosen_only, with_table, normalize in product([True, False], repeat=3):
            tag = (
                f"cal_plot_prompt{prompt['idx']}"
                f"_struct_dec_table{int(with_table)}"
                f"_chosenonly{int(chosen_only)}"
                f"_norm{int(normalize)}"
            )
            print(f"Generating {prompt['basename']}/{tag}")
            try:
                plot = build_calibration_grid(
                    prompt, datasets, models, cal_data,
                    chosen_only=chosen_only, normalize=normalize,
                    with_table=with_table, with_title=True,
                    row_col_titles_font_size=26,
                    ece_in_plot=True,
                )
                for ext in extensions:
                    plot.savefig(
                        figures_dir / f"full_plots/{tag}.{ext}",
                        bbox_inches="tight",
                    )
                plt.close(plot)
                # Relplot variant for all combinations.
                # rp_plot = build_calibration_grid(
                #     prompt, datasets, models, cal_data,
                #     chosen_only=chosen_only, normalize=normalize,
                #     with_table=with_table, with_title=True,
                #     subplot_fn=plot_relplot_subplot,
                #     row_col_titles_font_size=26,
                # )
                # for ext in extensions:
                #     rp_plot.savefig(
                #         figures_dir / f"full_plots/{tag}_mmlu_physics_relplot.{ext}",
                #         bbox_inches="tight",
                #     )
                # plt.close(rp_plot)
            except Exception as e:
                print(f"  Failed: {e}")


def plot_normalization_comparison(model, datasets, prompt, cal_data, figures_dir):
    """Plot calibration curves without vs with L1 probability normalisation.

    Rows correspond to the two normalisation modes (raw on top, normalised on
    bottom); columns to datasets.  The figure is saved to
    ``figures_dir / normalization_effect_prompt{idx}_{model_id}.svg``.

    Args:
        model: Model dict to visualise.
        datasets: List of dataset dicts (one column per dataset).
        prompt: Prompt design dict.
        cal_data: Pre-computed metrics dict (see :func:`build_calibration_grid`).
        figures_dir: :class:`pathlib.Path` to the figures output directory.
    """
    cell_data = [
        [
            (
                (cal_data.get(prompt["basename"], {}).get(ds["id"], {}).get(model["id"]) or {})
                .get(_cal_key(normalize, chosen_only=True)),
                model["type"]
            )
            for ds in datasets
        ]
        for normalize in [False, True]
    ]
    plot = generate_grid_plot(
        cell_data,
        row_titles=["Without Normalization", "With Normalization"],
        col_titles=[ds["id"] for ds in datasets],
        plot_subplot_fn=make_calibration_subplot_fn(ece_in_plot=False),
        sharex=False, sharey=False, with_table=False,
        show_axis_labels=True, plot_title=None,
        row_col_titles_font_size=28,
    )
    plot.savefig(
        figures_dir / f"normalization_effect_prompt{prompt['idx']}_{model['id']}.svg",
        bbox_inches="tight",
    )
    plot.show()
