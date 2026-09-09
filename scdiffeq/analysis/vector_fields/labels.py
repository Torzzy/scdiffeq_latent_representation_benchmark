import numpy as np


def plot_state_labels(
    ax,
    X,
    labels,
):
    """
    Plot state labels at cluster centroids.
    """

    labels = np.asarray(
        labels,
    )

    for state in np.unique(labels):

        mask = labels == state

        if mask.sum() < 50:
            continue

        center = X[mask].mean(
            axis=0,
        )

        ax.text(
            center[0],
            center[1],
            str(state),
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                fc="white",
                alpha=0.8,
                ec="none",
            ),
        )