"""ECE swarm / violin plot by UQ method.

Provides ``plot_ece_swarm``, which visualises the Expected Calibration Error
for every (model, dataset) pair, grouped by UQ method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from util.config import apply_matplotlib_defaults


#: Scatter colours by model type, matching the project-wide convention.
_TYPE_COLORS = {
    "instruct":  "tab:blue",
    "reasoning": "tab:green",
}


def plot_ece_swarm(
    prepared_data,
    models,
    datasets,
    uq_methods,
    figures_dir,
):
    """Plot ECE (Expected Calibration Error) by UQ method.

    For each UQ method a grey violin shows the distribution of ECE values
    across all (model, dataset) pairs.  Individual points are jittered
    horizontally and coloured by model type.  A horizontal bar marks the
    median for each method.

    A dashed reference line is drawn at ECE = 0 (perfect calibration).

    The figure is saved as ``ece_swarm.{svg,pdf,png}`` under *figures_dir*.

    Args:
        prepared_data: Nested dict ``prepared_data[dataset_id][model_id]``
            as produced by the data-preparation notebook.  Each entry must
            contain a ``"cal"`` sub-dict keyed by UQ method ID, with an
            ``"ece"`` float value (or ``None`` on failure).
        models: List of model dicts with keys ``"id"`` and ``"type"``.
        datasets: List of dataset dicts with key ``"id"``.
        uq_methods: List of UQ method dicts with keys ``"id"`` and
            ``"label"``.
        figures_dir: ``pathlib.Path`` (or str) pointing to the directory
            where figures are saved.

    Returns:
        matplotlib.figure.Figure: The produced figure.
    """
    apply_matplotlib_defaults()

    figures_dir = Path(figures_dir)

    n_methods = len(uq_methods)
    x_positions = list(range(n_methods))

    # ------------------------------------------------------------------ #
    # Collect per-method lists of (ece, model_type) pairs                 #
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
                ece = cal_entry.get("ece")
                if ece is None or (isinstance(ece, float) and np.isnan(ece)):
                    continue
                points.append((float(ece), model["type"]))
        method_data[uq_id] = points

    # ------------------------------------------------------------------ #
    # Draw                                                                 #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(max(n_methods * 2.2, 6), 5))

    rng = np.random.default_rng(seed=0)

    for x_pos, uq in zip(x_positions, uq_methods):
        uq_id = uq["id"]
        points = method_data[uq_id]
        if not points:
            continue

        values = np.array([p[0] for p in points])
        types  = [p[1] for p in points]

        # Grey violin
        if len(values) >= 2:
            parts = ax.violinplot(
                [values],
                positions=[x_pos],
                widths=0.6,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#cccccc")
                body.set_edgecolor("#888888")
                body.set_alpha(0.8)
                body.set_linewidth(0.8)

        # Median bar
        median_val = float(np.median(values))
        ax.hlines(median_val, x_pos - 0.22, x_pos + 0.22,
                  colors="#444444", linewidths=2.0, zorder=4)

        # Jittered scatter coloured by model type
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        for val, m_type, jit in zip(values, types, jitter):
            color = _TYPE_COLORS.get(m_type, "tab:grey")
            ax.scatter(
                x_pos + jit, val,
                color=color,
                s=28, zorder=5, alpha=0.85,
                edgecolors="none",
            )

    # ------------------------------------------------------------------ #
    # Cosmetics                                                            #
    # ------------------------------------------------------------------ #
    # Reference line at ECE = 0 (perfect calibration)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([uq["label"] for uq in uq_methods], fontsize=12)
    ax.set_ylabel("ECE (Expected Calibration Error)", fontsize=12)
    ax.set_ylim(-0.02, None)
    ax.set_xlim(-0.6, n_methods - 0.4)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=m_type.capitalize())
        for m_type, color in _TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(figures_dir / f"ece_swarm.{ext}", bbox_inches="tight")

    return fig
