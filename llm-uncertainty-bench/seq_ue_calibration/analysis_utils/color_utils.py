"""Matplotlib colour manipulation and font-scaling utilities."""

import matplotlib as mpl


def adjust_color(color_name, factor=0.7, lighten=True):
    """Blend a matplotlib named colour towards white (lighten) or black (darken).

    Uses linear interpolation between the original colour and the target
    (white or black).

    Args:
        color_name: Any matplotlib-recognised colour string (e.g. ``"tab:blue"``).
        factor: Blend weight for the *original* colour in [0, 1].
            0 → fully white/black; 1 → original colour unchanged.
        lighten: If ``True`` blend towards white; if ``False`` towards black.

    Returns:
        tuple[float, float, float]: Adjusted RGB tuple with values in [0, 1].
    """
    rgb = mpl.colors.to_rgb(color_name)
    target = (1, 1, 1) if lighten else (0, 0, 0)
    return tuple(factor * c + (1 - factor) * t for c, t in zip(rgb, target))


def scale_fonts(factor):
    """Scale all matplotlib font-size rcParams by *factor*.

    Multiplies every ``mpl.rcParams`` entry whose key contains
    ``"font.size"`` or ends with ``"fontsize"`` by *factor*.  Useful for
    temporarily increasing text size before saving a high-resolution figure.

    Args:
        factor: Multiplicative scaling factor (e.g. ``1.3`` to increase by 30 %).
    """
    for key, val in mpl.rcParams.items():
        if "font.size" in key or key.endswith("fontsize"):
            if isinstance(val, (int, float)):
                mpl.rcParams[key] = val * factor
