import numpy as np
import umap


def project_clone_prediction(
    adata,
    latent_key,
    z0,
    gt_t1,
    gt_t2,
    pred_t1,
    pred_t2,
):
    """Compute a common UMAP projection for observed and simulated cells.

    Parameters
    ----------
    adata
        Annotated data matrix containing the latent representation.
    latent_key
        Key in ``adata.obsm`` containing the latent representation.
    z0
        Initial latent state to project.
    gt_t1
        Ground-truth latent states at the first evaluation time point.
    gt_t2
        Ground-truth latent states at the second evaluation time point.
    pred_t1
        Simulated latent states at the first evaluation time point.
    pred_t2
        Simulated latent states at the second evaluation time point.

    Returns
    -------
    dict
        Two-dimensional UMAP coordinates for the reference latent space,
        initial state, ground-truth populations, and simulated populations.
    """
    latent = np.asarray(
        adata.obsm[latent_key],
        dtype=np.float32,
    )

    mapper = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        random_state=0,
    )

    latent_2d = mapper.fit_transform(
        latent,
    )

    return {
        "latent": latent_2d,
        "z0": mapper.transform(
            z0,
        ),
        "gt_t1": mapper.transform(
            gt_t1,
        ),
        "gt_t2": mapper.transform(
            gt_t2,
        ),
        "pred_t1": mapper.transform(
            pred_t1,
        ),
        "pred_t2": mapper.transform(
            pred_t2,
        ),
    }