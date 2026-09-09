import numpy as np


def build_pair_dict(
    adata,
    clone_key="X_clone",
    time_key="time_info",
):
    """Build lineage pairs between source cells and future populations.

    Parameters
    ----------
    adata
        Annotated data matrix containing clone assignments and time points.
    clone_key
        Key in ``adata.obsm`` containing the one-hot clone encoding.
    time_key
        Column in ``adata.obs`` containing the cell time points.

    Returns
    -------
    dict
        Dictionary containing source cell indices, future cell indices at
        times 4 and 6, and the corresponding clone identifiers.
    """
    clone = (
        adata.obsm[clone_key]
        .toarray()
        .argmax(axis=1)
    )

    time = np.asarray(
        adata.obs[time_key]
    )

    src_idx = []
    t1_idx = []
    t2_idx = []
    src_clone = []

    unique_clones = np.unique(
        clone
    )

    for cl in unique_clones:
        cells = np.where(
            clone == cl
        )[0]

        t0 = cells[
            time[cells] == 2
        ]

        t1 = cells[
            time[cells] == 4
        ]

        t2 = cells[
            time[cells] == 6
        ]

        if len(t0) == 0:
            continue

        if len(t1) == 0:
            continue

        if len(t2) == 0:
            continue

        for c in t0:
            src_idx.append(
                c
            )

            t1_idx.append(
                t1.copy()
            )

            t2_idx.append(
                t2.copy()
            )

            src_clone.append(
                cl
            )

    return {
        "src_idx": np.asarray(
            src_idx
        ),
        "t1_idx": t1_idx,
        "t2_idx": t2_idx,
        "clone": np.asarray(
            src_clone
        ),
    }


if __name__ == "__main__":
    import scanpy as sc

    adata = sc.read_h5ad(
        "../../data/processed/larry_pca.h5ad"
    )

    print(
        adata.obsm.keys()
    )