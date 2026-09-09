import numpy as np
import umap

from sklearn.neighbors import (
    NearestNeighbors,
)


def project_drift_to_umap(
    Z,
    drift,
    n_neighbors=40,
):
    """
    Project latent drift vectors onto a common UMAP embedding.
    """

    mapper = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        random_state=0,
    )

    X = mapper.fit_transform(
        Z,
    )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
    ).fit(
        Z,
    )

    _, neigh = nn.kneighbors(
        Z,
    )

    drift2d = np.zeros(
        (len(Z), 2),
        dtype=np.float32,
    )

    for i in range(len(Z)):

        dz = (
            Z[neigh[i, 1:]]
            - Z[i]
        )

        dx = (
            X[neigh[i, 1:]]
            - X[i]
        )

        if len(dz) < 5:
            continue

        A, *_ = np.linalg.lstsq(
            dz,
            dx,
            rcond=None,
        )

        drift2d[i] = (
            drift[i] @ A
        )

    drift2d = drift2d[
        neigh[:, 1:]
    ].mean(
        axis=1,
    )

    return X, drift2d