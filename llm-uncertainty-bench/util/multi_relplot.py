#
# Multi-reliability-diagram overlay plotting function.
# This is a self-contained module - import without modifying relplot.
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.collections import LineCollection

__all__ = ['plot_multi_rel_diagram', 'rel_multi_diagram']


def _apply_style():
    """Apply seaborn + relplot-compatible default styling."""
    sns.set_style("whitegrid")
    pal = sns.color_palette("pastel")
    sns.set_palette(pal, color_codes=True)
    mpl.rcParams.update({
        "axes.edgecolor": "0.5",
        "font.size": 14,
        "legend.frameon": False,
        "patch.force_edgecolor": False,
        "figure.figsize": [6.0, 6.0],
        "axes.titlepad": 20,
    })


def _color_sequence(n, cmap='viridis'):
    """Return n distinct RGB tuples from a matplotlib colormap."""
    cm = plt.get_cmap(cmap)
    return [cm(i / (n - 1) if n > 1 else 0.5)[:3] for i in range(n)]


def plot_multi_rel_diagram(
    diagrams,
    ax=None,
    fig=None,
    use_default_style=True,
    plot_diagonal=True,
    plot_ticks=False,
    tick_data=None,
    color_cmap='tab20',
    line_alpha=0.35,
    base_linewidth=1.2,
    max_linewidth=4.0,
    show_legend=False,
    legend_labels=None,
    ece_key='ce',
    ece_offset=0.035,
):
    """Plot multiple reliability diagrams overlaid on a single axes.

    Each diagram dict must contain at least 'mu' (regression curve) and
    'density' (density estimate) from relplot.prepare_rel_diagram().

    Line widths are density-weighted (thicker where more scores are)
    but with a less pronounced range than relplot's default.
    Each line is slightly transparent for overlay visibility.
    """
    n = len(diagrams)
    assert n >= 1, "Need at least one diagram to plot"

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if use_default_style:
        _apply_style()

    colors_rgb = _color_sequence(n, cmap=color_cmap)

    for i, diag in enumerate(diagrams):
        t = diag.get("mesh", np.linspace(0, 1, 1000))
        mu = diag["mu"]
        density = diag.get("density", None)

        if density is not None:
            # Density-weighted line widths via LineCollection
            points = np.column_stack([t, mu])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            dens_norm = density / np.max(density)
            lw = np.interp(dens_norm, [0, 1], [base_linewidth, max_linewidth])

            c = colors_rgb[i]
            lc = LineCollection(
                segments,
                colors=[tuple(c) for _ in segments],
                linewidths=lw,
                linestyle="-",
                alpha=line_alpha,
                joinstyle="round",
            )
            ax.add_collection(lc)
        else:
            ax.plot(t, mu, color=colors_rgb[i], alpha=line_alpha, lw=base_linewidth)

        # smECE annotation
        if ece_key and ece_key in diag:
            txt = f"d{i}: smECE={diag[ece_key]:.3f}"
            if legend_labels and i < len(legend_labels):
                txt += f" ({legend_labels[i]})"
            ax.text(
                0.02, 0.9 - i * ece_offset,
                txt, fontsize=7, va='top', ha='left',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7),
                family='monospace',
            )

    # Tick marks
    if plot_ticks and tick_data is not None:
        assert len(tick_data) == n, "tick_data must have same length as diagrams"
        for f_samp, y_samp in tick_data:
            ax.vlines(f_samp[y_samp == 0], -0.015, 0.015,
                      color='black', alpha=0.2, zorder=3)
            ax.vlines(f_samp[y_samp == 1], -0.015, 0.015,
                      color='black', alpha=0.35, zorder=3)

    # Legend
    if show_legend and legend_labels:
        handles = [
            mpl.lines.Line2D([0], [0], color=colors_rgb[i],
                             lw=base_linewidth, alpha=line_alpha,
                             label=legend_labels[i])
            for i in range(n)
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("f" if use_default_style else "predicted probability")
    ax.set_ylabel(
        r"$\mathbb{E}[ y \mid f ]$" if use_default_style else "E[y | f]",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if plot_diagonal:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    return fig, ax


def rel_multi_diagram(diagrams, fig=None, ax=None, **kwargs):
    """Convenience wrapper."""
    return plot_multi_rel_diagram(diagrams, fig=fig, ax=ax, **kwargs)
