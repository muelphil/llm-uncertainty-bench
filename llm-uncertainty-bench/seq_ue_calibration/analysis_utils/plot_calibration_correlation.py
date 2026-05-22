"""Swarm + violin plot of calibration correlations across models/datasets per UQ method.

Produces one figure with one column per UQ method.  Every (model, dataset) pair
that has a valid ``calibration_correlation`` value contributes one point, coloured
by model type.  A violin background summarises the distribution; individual points
are overlaid as a strip (jittered) plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_TYPE_COLORS = {"base": "tab:orange", "instruct": "tab:blue", "reasoning": "tab:green"}
_JITTER_SEED = 0
_JITTER_WIDTH = 0.18


def _collect_correlations(prepared_data: dict, models: list, datasets: list,
                           uq_methods: list) -> dict:
    """Gather calibration_correlation values per UQ method.

    Returns:
        dict mapping ``uq_method_id`` → list of ``(correlation, model_type)`` tuples.
        Entries with NaN or None calibration entries are silently skipped.
    """
    result: dict = {uq["id"]: [] for uq in uq_methods}
    for model in models:
        mid  = model["id"]
        mtype = model["type"]
        for dataset in datasets:
            ds_id = dataset["id"]
            entry = prepared_data.get(ds_id, {}).get(mid)
            if entry is None:
                continue
            for uq in uq_methods:
                cal = entry.get("cal", {}).get(uq["id"])
                if cal is None:
                    continue
                r = cal.get("calibration_correlation")
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                result[uq["id"]].append((float(r), mtype))
    return result


def plot_calibration_correlation_swarm(
    prepared_data: dict,
    models: list,
    datasets: list,
    uq_methods: list,
    figures_dir: Path,
    filename_stem: str = "calibration_correlation_swarm",
) -> plt.Figure:
    """Swarm + violin plot of calibration correlations, one column per UQ method.

    Each point represents one (model, dataset) pair and is coloured by model type
    (base=orange, instruct=blue, reasoning=green).  A neutral-grey violin shows
    the overall distribution shape behind the individual points.

    Args:
        prepared_data: ``prepared_data[dataset_id][model_id]`` → ModelDatasetEntry,
            as loaded from the prepared-data pickle.  Each calibration entry must
            contain the ``"calibration_correlation"`` field (computed in
            analysis_prepare_data).
        models: List of model dicts with ``"id"`` and ``"type"`` keys.
        datasets: List of dataset dicts with ``"id"`` keys.
        uq_methods: List of UQ method dicts with ``"id"`` and ``"label"`` keys.
        figures_dir: Directory where SVG and PNG outputs are saved.
        filename_stem: Filename base (without extension).

    Returns:
        matplotlib.figure.Figure
    """
    figures_dir = Path(figures_dir)
    correlations = _collect_correlations(prepared_data, models, datasets, uq_methods)

    n_methods = len(uq_methods)
    fig, ax = plt.subplots(figsize=(2.8 * n_methods + 1.2, 5))

    rng = np.random.default_rng(_JITTER_SEED)

    violin_data   = []
    violin_pos    = []

    for col_idx, uq in enumerate(uq_methods):
        vals_types = correlations[uq["id"]]
        if not vals_types:
            violin_data.append([0.0])   # placeholder so violin doesn't crash
        else:
            violin_data.append([v for v, _ in vals_types])
        violin_pos.append(col_idx)

    # Violin background (neutral grey, no colour meaning here)
    parts = ax.violinplot(
        violin_data,
        positions=violin_pos,
        widths=0.6,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#cccccc")
        body.set_edgecolor("#888888")
        body.set_alpha(0.55)
        body.set_linewidth(1.0)

    # Scatter points — one per (model, dataset) pair — with jitter
    for col_idx, uq in enumerate(uq_methods):
        vals_types = correlations[uq["id"]]
        if not vals_types:
            continue
        for r, mtype in vals_types:
            color = _TYPE_COLORS.get(mtype, "grey")
            jitter = rng.uniform(-_JITTER_WIDTH, _JITTER_WIDTH)
            ax.scatter(
                col_idx + jitter, r,
                color=color, s=28, alpha=0.78,
                linewidths=0.4, edgecolors="white",
                zorder=3,
            )

    # Median line per violin
    for col_idx, uq in enumerate(uq_methods):
        vals = [v for v, _ in correlations[uq["id"]]]
        if not vals:
            continue
        med = float(np.median(vals))
        ax.plot(
            [col_idx - 0.22, col_idx + 0.22], [med, med],
            color="black", linewidth=2.0, zorder=4,
        )

    # Formatting
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([uq["label"] for uq in uq_methods], rotation=0, ha="center", fontsize=11)
    ax.set_ylabel("Calibration Correlation (Pearson r)", fontsize=12)
    ax.set_xlim(-0.65, n_methods - 0.35)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1, color="grey", linestyle=":",  linewidth=0.8, alpha=0.4, zorder=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_title("Calibration Correlation per UQ Method", fontsize=13)

    # Legend (instruct and reasoning only — no base models in this experiment)
    legend_handles = [
        mpatches.Patch(color="tab:blue",   label="Instruct",  alpha=0.78),
        mpatches.Patch(color="tab:green",  label="Reasoning", alpha=0.78),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

    plt.tight_layout()
    fig.savefig(figures_dir / f"{filename_stem}.svg", bbox_inches="tight")
    fig.savefig(figures_dir / f"{filename_stem}.png", bbox_inches="tight")
    return fig
