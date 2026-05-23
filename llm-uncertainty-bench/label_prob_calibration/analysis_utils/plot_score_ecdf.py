"""ECDF plots for norm_chosen uncertainty score distributions per model type.

Produces three figures (one per type: base / instruct / reasoning), each
showing one approximate ECDF line per model.  The ECDF is constructed from
the five-number summary (min / Q25 / median / Q75 / max) averaged across the
supplied target datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_models(models: list) -> list:
    seen: set = set()
    out = []
    for m in models:
        if m["id"] not in seen:
            out.append(m)
            seen.add(m["id"])
    return out


def _model_ecdf_points(
    cal_data: dict,
    m_id: str,
    pname: str,
    target_ds_ids: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_values, y_fractions) for a 5-point ECDF approximation.

    Averages the five-number summary across all target datasets with valid data.
    Returns ``None`` if no data is available.
    """
    keys = ("label_prob_min", "label_prob_q25", "label_prob_median",
            "label_prob_q75", "label_prob_max")
    accum = {k: [] for k in keys}
    for ds_id in target_ds_ids:
        entry = (cal_data[pname][ds_id][m_id] or {}).get("norm_chosen")
        if entry is None:
            continue
        if not all(isinstance(entry.get(k), (int, float)) for k in keys):
            continue
        for k in keys:
            accum[k].append(entry[k])
    if not accum["label_prob_median"]:
        return None
    x = np.array([np.mean(accum[k]) for k in keys])
    y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    return x, y


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_score_ecdf(
    cal_data: dict,
    models: list,
    prompt_designs: list,
    target_datasets: list,
    figures_dir: Path,
    filename_stem: str = "uncertainty_score_ecdf",
) -> list[tuple[str, plt.Figure]]:
    """Plot three ECDF figures, one per model type.

    Each figure shows one line per model (filtered to that type), with the
    ECDF approximated from the five-number summary averaged across
    ``target_datasets``.  The legend listing model shortnames is placed to
    the right of the axes.

    Parameters
    ----------
    cal_data:
        ``cal_data[pname][ds_id][m_id]`` → metrics dict.
    models:
        MODELS list (de-duplicated internally by id).
    prompt_designs:
        Prompt design dicts; ``prompt_designs[0]`` is used.
    target_datasets:
        ``[{"id": ..., "label": ...}, ...]`` — datasets to average across.
    figures_dir:
        Directory for saved figures.
    filename_stem:
        Filename prefix without extension.  Each figure is saved as
        ``{filename_stem}_{type}.svg`` and ``.png``.

    Returns
    -------
    list of (type_name, Figure) tuples
    """
    figures_dir = Path(figures_dir)
    pname = prompt_designs[0]["basename"]
    models_dedup = _dedup_models(models)
    target_ds_ids = [ds["id"] for ds in target_datasets]

    type_labels = {"base": "Base", "instruct": "Instruct", "reasoning": "Reasoning"}
    results = []

    for mtype in ("base", "instruct", "reasoning"):
        type_models = [m for m in models_dedup if m.get("type") == mtype]
        if not type_models:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        for m in type_models:
            pts = _model_ecdf_points(cal_data, m["id"], pname, target_ds_ids)
            if pts is None:
                continue
            x, y = pts
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, label=m["shortname"])

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Normalised Chosen-Label Probability", fontsize=12)
        ax.set_ylabel("Cumulative Fraction", fontsize=12)
        ax.set_title(
            f"ECDF of Uncertainty Scores — {type_labels.get(mtype, mtype)} Models\n"
            f"(5-point approx. averaged across datasets)",
            fontsize=12,
        )
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(
            fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left",
            borderaxespad=0, frameon=True,
        )

        plt.tight_layout()
        fig.savefig(figures_dir / f"{filename_stem}_{mtype}.svg", bbox_inches="tight")
        fig.savefig(figures_dir / f"{filename_stem}_{mtype}.png", bbox_inches="tight")
        results.append((mtype, fig))

    return results
