"""Per-dataset boxplot grid for norm_chosen uncertainty score distributions.

Single figure with one row per target dataset and a shared x-axis.
A single family rectangle spans the full figure height (all rows) per family,
with the family name annotated above the topmost row.
A dashed line connects the median of base → instruct → reasoning models
within each family.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_TYPE_ORDER  = ["base", "instruct", "reasoning"]
_TYPE_COLORS = {"base": "tab:orange", "instruct": "tab:blue", "reasoning": "tab:green"}
_BOX_W       = 0.5
_FAMILY_PAD  = 0.4   # padding on each side of family rectangle (< BOX_W → visible gaps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_models(models: list) -> list:
    """Return *models* with duplicate ids removed (first occurrence kept)."""
    seen: set = set()
    out = []
    for m in models:
        if m["id"] not in seen:
            out.append(m)
            seen.add(m["id"])
    return out


def _collect_box_data(cal_data: dict, models_dedup: list, pname: str, ds_id: str):
    """Return (ordered_models, box_data_dict) for one dataset."""
    ordered, data = [], {}
    for m in models_dedup:
        m_id = m["id"]
        entry = (cal_data[pname][ds_id][m_id] or {}).get("norm_chosen")
        if entry is None:
            continue
        if not all(
            isinstance(entry.get(k), (int, float))
            for k in ("label_prob_min", "label_prob_q25", "label_prob_median",
                      "label_prob_q75", "label_prob_max")
        ):
            continue
        ordered.append(m)
        data[m_id] = {
            "whislo": entry["label_prob_min"],
            "q1":     entry["label_prob_q25"],
            "med":    entry["label_prob_median"],
            "q3":     entry["label_prob_q75"],
            "whishi": entry["label_prob_max"],
            "type":   m["type"],
        }
    return ordered, data


def _data_x_to_fig_frac(ax: plt.Axes, fig: plt.Figure, x_data: float) -> float:
    """Convert a data-space x value to figure-fraction x coordinate."""
    x_disp = ax.transData.transform((x_data, 0))[0]
    return fig.transFigure.inverted().transform((x_disp, 0))[0]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_score_distributions(
    cal_data: dict,
    models: list,
    model_families: list,
    prompt_designs: list,
    target_datasets: list,
    figures_dir: Path,
    filename_stem: str = "uncertainty_score_dist_grid",
) -> plt.Figure:
    """Plot a combined boxplot grid: one row per dataset, shared x-axis.

    Each model family is represented by a single rectangle spanning all rows
    in figure coordinates, with the family name shown above the topmost row.

    Parameters
    ----------
    cal_data:
        ``cal_data[pname][ds_id][m_id]`` → metrics dict.
    models:
        MODELS list (may contain duplicate ids; de-duplicated internally).
    model_families:
        List of ``{"name": ..., "models": [...]}`` dicts.
    prompt_designs:
        Prompt design dicts; ``prompt_designs[0]`` is used.
    target_datasets:
        ``[{"id": ..., "label": ...}, ...]`` — one row per entry.
    figures_dir:
        Directory for saved figures.
    filename_stem:
        Filename without extension.

    Returns
    -------
    plt.Figure
    """
    figures_dir = Path(figures_dir)
    pname       = prompt_designs[0]["basename"]
    models_dedup = _dedup_models(models)
    name_to_id  = {m["name"]: m["id"] for m in models}

    # Global x-axis order: models present in at least one target dataset
    present_ids: set = set()
    for ds in target_datasets:
        for m in models_dedup:
            entry = (cal_data[pname][ds["id"]][m["id"]] or {}).get("norm_chosen")
            if entry and isinstance(entry.get("label_prob_median"), (int, float)):
                present_ids.add(m["id"])

    global_order = [m for m in models_dedup if m["id"] in present_ids]
    n_models     = len(global_order)
    g_pos        = {m["id"]: i for i, m in enumerate(global_order)}

    n_rows = len(target_datasets)
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(max(10, n_models * 1.1), 2.8 * n_rows),
        squeeze=False,
    )

    # Make axes backgrounds transparent so figure-level family rectangles show through
    for ax in axes.flat:
        ax.set_facecolor("none")

    # Per-row: boxplots + dashed family median lines
    for row_idx, ds in enumerate(target_datasets):
        ax     = axes[row_idx, 0]
        ds_id  = ds["id"]
        _, box_data = _collect_box_data(cal_data, models_dedup, pname, ds_id)

        cap_w = _BOX_W * 0.3
        for m in global_order:
            m_id = m["id"]
            if m_id not in box_data:
                continue
            i     = g_pos[m_id]
            d     = box_data[m_id]
            color = _TYPE_COLORS[d["type"]]
            # Whisker lines
            ax.plot([i, i], [d["whislo"], d["q1"]],    color="black", linewidth=1, zorder=2)
            ax.plot([i, i], [d["q3"],     d["whishi"]], color="black", linewidth=1, zorder=2)
            # Whisker caps
            ax.plot([i - cap_w, i + cap_w], [d["whislo"], d["whislo"]],
                    color="black", linewidth=1, zorder=2)
            ax.plot([i - cap_w, i + cap_w], [d["whishi"], d["whishi"]],
                    color="black", linewidth=1, zorder=2)
            # IQR box
            ax.add_patch(mpatches.FancyBboxPatch(
                (i - _BOX_W / 2, d["q1"]), _BOX_W, d["q3"] - d["q1"],
                boxstyle="square,pad=0",
                facecolor=color, edgecolor="black", linewidth=1, alpha=0.75, zorder=3,
            ))
            # Median line
            ax.plot([i - _BOX_W / 2, i + _BOX_W / 2], [d["med"], d["med"]],
                    color="black", linewidth=2, zorder=4)

        # Dashed line connecting medians within each family (base→instruct→reasoning)
        for family in model_families:
            members = [
                (name_to_id[n], box_data[name_to_id[n]])
                for n in family["models"]
                if name_to_id.get(n) in box_data and name_to_id.get(n) in g_pos
            ]
            members.sort(
                key=lambda x: _TYPE_ORDER.index(x[1]["type"])
                if x[1]["type"] in _TYPE_ORDER else 99
            )
            if len(members) < 2:
                continue
            xs = [g_pos[fid] for fid, _ in members]
            ys = [d["med"] for _, d in members]
            ax.plot(xs, ys, color="dimgrey", linestyle="--", linewidth=1.8, zorder=5)

        # Row formatting
        ax.set_xlim(-0.7, n_models - 0.3)
        ax.set_ylim(0.0, 1.1)
        ax.set_yticks(np.linspace(0, 1, 6))   # 0.0 0.2 0.4 0.6 0.8 1.0 — no 1.1 tick
        ax.set_ylabel("Certainty Scores", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
        ax.text(
            -0.03, 0.5, ds["label"],
            transform=ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )

    # x-ticks only on bottom axis
    bottom_ax = axes[-1, 0]
    bottom_ax.set_xticks(range(n_models))
    bottom_ax.set_xticklabels(
        [m["shortname"] for m in global_order],
        rotation=45, ha="right", fontsize=10,
    )

    # Legend: outside, lower-right of bottommost plot
    legend_handles = [
        mpatches.Patch(color="tab:orange", label="Base",         alpha=0.75),
        mpatches.Patch(color="tab:blue",   label="Instruct",     alpha=0.75),
        mpatches.Patch(color="tab:green",  label="Reasoning",    alpha=0.75),
        mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor="lightgrey", edgecolor="dimgrey", linestyle="--",
            linewidth=1.5, alpha=0.4, label="Model Family",
        ),
    ]
    bottom_ax.legend(
        handles=legend_handles,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.59),
        ncol=4,
        borderaxespad=0,
        frameon=True,
    )

    # Reserve top space for family name labels
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Finalise layout so transforms are accurate, then draw figure-level family rects
    fig.canvas.draw()

    top_ax    = axes[0, 0]
    top_bbox  = top_ax.get_position()
    bot_bbox  = bottom_ax.get_position()

    for family in model_families:
        fam_positions = [
            g_pos[name_to_id[n]]
            for n in family["models"]
            if name_to_id.get(n) in g_pos
        ]
        if not fam_positions:
            continue

        x_lo_data = min(fam_positions) - _FAMILY_PAD
        x_hi_data = max(fam_positions) + _FAMILY_PAD
        x_lo_fig  = _data_x_to_fig_frac(top_ax, fig, x_lo_data)
        x_hi_fig  = _data_x_to_fig_frac(top_ax, fig, x_hi_data)
        y_lo_fig  = bot_bbox.y0 - 0.003
        y_hi_fig  = top_bbox.y1 + 0.003

        rect = mpatches.Rectangle(
            (x_lo_fig, y_lo_fig),
            x_hi_fig - x_lo_fig,
            y_hi_fig - y_lo_fig,
            facecolor="lightgrey", alpha=0.2,
            edgecolor="dimgrey", linestyle="--", linewidth=1.5,
            clip_on=False,
        )
        rect.set_transform(fig.transFigure)
        fig.add_artist(rect)

        # Family name above the topmost row
        x_center_fig = (x_lo_fig + x_hi_fig) / 2
        fig.text(
            x_center_fig, y_hi_fig + 0.008,
            family["name"],
            ha="center", va="bottom",
            fontsize=9, color="black",
        )

    fig.savefig(figures_dir / f"{filename_stem}.svg", bbox_inches="tight")
    fig.savefig(figures_dir / f"{filename_stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{filename_stem}.png", bbox_inches="tight")
    return fig

