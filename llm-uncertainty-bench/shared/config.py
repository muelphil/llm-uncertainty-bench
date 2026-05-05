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
    """Apply shared font and SVG settings to matplotlib's global rcParams."""
    plt.rcParams["font.family"] = "Times New Roman"
    # Embed fonts as text rather than paths so SVG files remain editable.
    plt.rcParams["svg.fonttype"] = "none"
