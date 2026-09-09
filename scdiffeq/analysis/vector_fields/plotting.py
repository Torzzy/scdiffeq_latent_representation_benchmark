from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .labels import (
    plot_state_labels,
)


def plot_vector_field(
    X,
    drift2d,
    xx,
    yy,
    U,
    V,
    speed,
    state_labels,
    output_path,
    method,
    latent_dim,
    hidden_dim,
):
    """
    Plot vector field.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vmax = np.nanpercentile(
        speed,
        95,
    )

    speed = np.clip(
        speed / (vmax + 1e-12),
        0,
        1,
    )

    cell_norm = np.linalg.norm(
        drift2d,
        axis=1,
    )

    fig, ax = plt.subplots(
        figsize=(8, 8),
    )

    pts = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=cell_norm,
        cmap="viridis",
        s=5,
        alpha=0.45,
        linewidth=0,
        rasterized=True,
        zorder=1,
    )

    plt.colorbar(
        pts,
        ax=ax,
        label="Drift magnitude",
    )

    if state_labels is not None:

        plot_state_labels(
            ax,
            X,
            state_labels,
        )

    ax.streamplot(
        xx,
        yy,
        U,
        V,
        density=2.0,
        color=speed,
        cmap="Greys",
        linewidth=0.4 + 1.8 * speed,
        arrowsize=0.8,
        maxlength=4,
        integration_direction="forward",
        zorder=10,
    )

    ax.set_title(
        f"{method} — latent={latent_dim} — hidden={hidden_dim}"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close(fig)