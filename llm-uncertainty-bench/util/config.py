"""
Shared visual configuration for both experiments.

Call ``apply_matplotlib_defaults()`` at the top of each notebook to apply
the standard font and SVG settings. The ``CALIBRATION_PLOT_COLORS`` mapping
translates a model type string to the matplotlib colormap name used for
calibration curve plots.
"""

import matplotlib.pyplot as plt


# Colormap name per model type, used in calibration curve plots.
CALIBRATION_PLOT_COLORS = {
    "instruct": "Blues",
    "reasoning": "Greens",
    "base": "Oranges",
}


def apply_matplotlib_defaults():
    """Apply shared font, SVG, and colour settings to matplotlib's global rcParams.

    Call this at the top of each notebook cell that creates figures.  It is
    especially important after any call to ``relplot.plot_rel_diagram``, which
    internally resets *all* rcParams via ``mpl.rc_file_defaults()``.

    IDE kernels running in dark-mode (e.g. IntelliJ / VS Code dark themes) can
    inject a dark matplotlib style into the kernel at startup.  This function
    overrides those settings so that all figures are rendered with a white
    background and black foreground regardless of the host IDE theme.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    # Embed fonts as text rather than paths so SVG files remain editable.
    plt.rcParams["svg.fonttype"] = "none"
    # Explicit light-background / dark-foreground overrides — these counteract
    # any dark-mode style injected by the IDE kernel at startup.
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["grid.color"] = "lightgrey"
