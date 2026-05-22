"""
Placeholder subplot renderer.

``plot_empty`` fills a matplotlib ``Axes`` with a light grey background and a
centred message.  It is used as a fallback wherever calibration data is
unavailable for a given model/dataset combination.
"""


def plot_empty(ax, message="No Data Available"):
    """
    Render a grey placeholder on *ax* with a centred *message*.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        message (str): Text to display in the centre of the axes.
    """
    ax.set_facecolor("#f0f0f0")
    ax.text(
        0.5, 0.5, message,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
        color="#555555",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_frame_on(False)
