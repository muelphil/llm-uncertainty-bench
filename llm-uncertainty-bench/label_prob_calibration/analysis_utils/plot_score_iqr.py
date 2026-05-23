"""Joint IQR comparison plot: model-type × dataset.

Single figure showing the interquartile range (Q25–Q75) and median per
model type per dataset.  Values are aggregated by averaging across all models
of the same type for each dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


_TYPE_COLORS = {"base": "tab:orange", "instruct": "tab:blue", "reasoning": "tab:green"}
_TYPE_ORDER   = ["base", "instruct", "reasoning"]
_TYPE_LABELS  = {"base": "Base", "instruct": "Instruct", "reasoning": "Reasoning"}
_TYPE_OFFSETS = {"base": -0.18, "instruct": 0.0, "reasoning": 0.18}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dedup_models(models: list) -> list:
    seen: set = set()
    out = []
    for m in models:
        if m["id"] not in seen:
            out.append(m)
            seen.add(m["id"])
    return out


def _aggregate_iqr(
    cal_data: dict,
    models_dedup: list,
    pname: str,
    target_datasets: list,
) -> dict:
    """Return ``agg[ds_id][mtype] = (q1, median, q3)`` or None if no data."""
    agg: dict = {}
    for ds in target_datasets:
        ds_id = ds["id"]
        agg[ds_id] = {}
        for mtype in _TYPE_ORDER:
            q1s, meds, q3s = [], [], []
            for m in models_dedup:
                if m.get("type") != mtype:
                    continue
                entry = (cal_data[pname][ds_id][m["id"]] or {}).get("norm_chosen")
                if entry is None:
                    continue
                q1 = entry.get("label_prob_q25")
                md = entry.get("label_prob_median")
                q3 = entry.get("label_prob_q75")
                if all(isinstance(v, (int, float)) for v in (q1, md, q3)):
                    q1s.append(q1)
                    meds.append(md)
                    q3s.append(q3)
            if q1s:
                agg[ds_id][mtype] = (np.mean(q1s), np.mean(meds), np.mean(q3s))
            else:
                agg[ds_id][mtype] = None
    return agg


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_score_iqr(
    cal_data: dict,
    models: list,
    prompt_designs: list,
    target_datasets: list,
    figures_dir: Path,
    filename_stem: str = "uncertainty_score_iqr",
) -> plt.Figure:
    """Plot a joint IQR comparison: one figure, model-type × dataset.

    For each dataset, three slightly-offset vertical IQR markers are drawn
    (orange = base, blue = instruct, green = reasoning).  Each marker shows
    a thick line from Q25 to Q75 with a dot at the median.  Values are
    aggregated by averaging Q25/median/Q75 across all models of the same type
    for each dataset.

    Parameters
    ----------
    cal_data:
        ``cal_data[pname][ds_id][m_id]`` → metrics dict.
    models:
        MODELS list (de-duplicated internally by id).
    prompt_designs:
        Prompt design dicts; ``prompt_designs[0]`` is used.
    target_datasets:
        ``[{"id": ..., "label": ...}, ...]``.
    figures_dir:
        Directory for saved figures.
    filename_stem:
        Filename without extension.

    Returns
    -------
    plt.Figure
    """
    figures_dir = Path(figures_dir)
    pname = prompt_designs[0]["basename"]
    models_dedup = _dedup_models(models)
    agg = _aggregate_iqr(cal_data, models_dedup, pname, target_datasets)

    n_ds = len(target_datasets)
    fig, ax = plt.subplots(figsize=(max(6, n_ds * 2.0), 5))

    for ds_idx, ds in enumerate(target_datasets):
        ds_id = ds["id"]
        for mtype in _TYPE_ORDER:
            val = agg[ds_id].get(mtype)
            if val is None:
                continue
            q1, med, q3 = val
            x = ds_idx + _TYPE_OFFSETS[mtype]
            color = _TYPE_COLORS[mtype]
            # IQR bar
            ax.plot([x, x], [q1, q3], color=color, linewidth=4, solid_capstyle="round", zorder=3)
            # Median dot
            ax.scatter([x], [med], color=color, s=60, zorder=4, edgecolors="black", linewidths=0.8)

    ax.set_xlim(-0.6, n_ds - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(n_ds))
    ax.set_xticklabels([ds["label"] for ds in target_datasets], fontsize=12)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylabel("Normalised Chosen-Label Probability", fontsize=12)
    ax.set_title(
        "IQR per Model Type per Dataset (norm_chosen)\n"
        "Bar = Q25–Q75, dot = median; averaged across models of each type",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    legend_handles = [
        mpatches.Patch(color=_TYPE_COLORS[t], label=_TYPE_LABELS[t])
        for t in _TYPE_ORDER
    ]
    ax.legend(handles=legend_handles, fontsize=11, loc="lower right")

    plt.tight_layout()
    fig.savefig(figures_dir / f"{filename_stem}.svg", bbox_inches="tight")
    fig.savefig(figures_dir / f"{filename_stem}.png", bbox_inches="tight")
    return fig
