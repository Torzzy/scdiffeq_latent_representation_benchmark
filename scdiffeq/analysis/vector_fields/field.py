import numpy as np

from sklearn.neighbors import (
    BallTree,
)


def compute_grid_field(
    X,
    V,
    grid_size=70,
    n_neighbors=200,
):
    """
    Continuous vector field similar to scvelo.
    """

    xmin, ymin = X.min(0)
    xmax, ymax = X.max(0)

    pad = 0.03

    dx = xmax - xmin
    dy = ymax - ymin

    xmin -= pad * dx
    xmax += pad * dx

    ymin -= pad * dy
    ymax += pad * dy

    gx = np.linspace(
        xmin,
        xmax,
        grid_size,
    )

    gy = np.linspace(
        ymin,
        ymax,
        grid_size,
    )

    xx, yy = np.meshgrid(
        gx,
        gy,
    )

    tree = BallTree(
        X,
    )

    grid_points = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    dist, ind = tree.query(
        grid_points,
        k=n_neighbors,
    )

    sigma = np.median(
        dist,
    )

    weights = np.exp(
        -(dist**2)
        / (2 * sigma**2)
    )

    weights /= (
        weights.sum(
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )

    field = (
        V[ind]
        * weights[..., None]
    ).sum(
        axis=1,
    )

    U = field[:, 0].reshape(
        xx.shape,
    )

    W = field[:, 1].reshape(
        xx.shape,
    )

    speed = np.sqrt(
        U**2 + W**2,
    )

    density = weights.sum(
        axis=1,
    ).reshape(
        xx.shape,
    )

    mask = density < np.percentile(
        density,
        20,
    )

    U[mask] = np.nan
    W[mask] = np.nan

    return (
        xx,
        yy,
        U,
        W,
        speed,
    )