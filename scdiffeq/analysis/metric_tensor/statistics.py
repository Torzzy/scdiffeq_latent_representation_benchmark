import os

import numpy as np


def compute_metric_statistics(
    G,
    neighbors,
    energy_threshold=0.999,
):
    """Compute summary statistics from the metric tensor.

    Parameters
    ----------
    G
        Metric tensor with shape ``(N, D, D)``.
    neighbors
        Indices of the k-nearest neighbours for each cell.
    energy_threshold
        Fraction of spectral energy used to define the effective rank.

    Returns
    -------
    dict
        Metric norms, eigenvalues, conditioning measures, energy ranks,
        and local metric variation statistics.
    """
    G_norm = np.linalg.norm(
        G,
        axis=(1, 2),
    )

    eigvals = np.linalg.eigvalsh(
        G,
    )

    eigvals = np.clip(
        eigvals,
        0,
        None,
    )

    eig_min = eigvals[:, 0]
    eig_max = eigvals[:, -1]

    conditioning = (
        eig_max
        / np.maximum(
            eig_min,
            1e-12,
        )
    )

    eigvals_desc = eigvals[:, ::-1]

    cumulative = np.cumsum(
        eigvals_desc,
        axis=1,
    )

    total_energy = cumulative[:, -1:]

    cumulative /= np.maximum(
        total_energy,
        1e-12,
    )

    energy_rank = (
        np.argmax(
            cumulative >= energy_threshold,
            axis=1,
        )
        + 1
    )

    effective_conditioning = np.empty(
        len(G),
        dtype=np.float32,
    )

    for i in range(len(G)):
        r = energy_rank[i]
        lam = eigvals_desc[i]

        effective_conditioning[i] = (
            lam[0]
            / np.maximum(
                lam[r - 1],
                1e-12,
            )
        )

    G_unit = (
        G
        / (
            G_norm[:, None, None]
            + 1e-12
        )
    )

    variation = np.empty(
        len(G),
        dtype=np.float32,
    )

    for i in range(len(G)):
        variation[i] = np.linalg.norm(
            G_unit[neighbors[i]]
            - G_unit[i],
            axis=(1, 2),
        ).mean()

    del G_unit

    return {
        "G_norm": G_norm,
        "eigvals": eigvals,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "conditioning": conditioning,
        "energy_rank": energy_rank,
        "effective_conditioning": effective_conditioning,
        "energy_rank_mean":
            energy_rank.mean(),
        "energy_rank_std":
            energy_rank.std(),
        "effective_conditioning_mean":
            effective_conditioning.mean(),
        "effective_conditioning_std":
            effective_conditioning.std(),
        "metric_variation_mean":
            variation.mean(),
    }


def cleanup_metric_tensor(
    G,
    memmap_path,
):
    """Release the metric tensor and remove its temporary memmap file.

    Parameters
    ----------
    G
        Metric tensor or NumPy memmap to release.
    memmap_path
        Path to the temporary metric tensor file.
    """
    del G

    if os.path.exists(
        memmap_path,
    ):
        os.remove(
            memmap_path,
        )