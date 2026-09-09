import os
import tempfile

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .metric import metric_tensor


def compute_metric_tensor(
    model,
    adata,
    latent_key="X_latent",
    n_cells=20_000,
    batch_size=128,
    k=20,
    device="cuda",
):
    """Compute the Fisher metric tensor for a subset of cells.

    Parameters
    ----------
    model
        Trained model containing the decoder module.
    adata
        Annotated data matrix containing the latent representation.
    latent_key
        Key in ``adata.obsm`` containing the latent representation.
    n_cells
        Maximum number of cells used for the metric computation.
    batch_size
        Number of cells processed per batch.
    k
        Number of nearest neighbours computed for each cell.
    device
        Device used for metric tensor computation.

    Returns
    -------
    G : numpy.memmap
        Fisher metric tensor with shape ``(n_cells, latent_dim, latent_dim)``.
    z : numpy.ndarray
        Latent coordinates of the selected cells.
    neighbors : numpy.ndarray
        Indices of the ``k`` nearest neighbours in latent space.
    memmap_path : str
        Path to the temporary file containing ``G``.
    """
    module = model.module

    rng = np.random.default_rng(0)

    if len(adata) > n_cells:
        idx = rng.choice(
            len(adata),
            n_cells,
            replace=False,
        )
    else:
        idx = np.arange(
            len(adata)
        )

    adata = adata[idx]

    z = np.asarray(
        adata.obsm[latent_key],
        dtype=np.float32,
    )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
    )

    nn.fit(z)

    neighbors = nn.kneighbors(
        return_distance=False,
    )[:, 1:]

    loader = model._make_data_loader(
        adata=adata,
        batch_size=batch_size,
        shuffle=False,
    )

    n = len(adata)
    d = z.shape[1]

    memmap_path = os.path.join(
        tempfile.gettempdir(),
        "metric_tensor.dat",
    )

    if os.path.exists(memmap_path):
        os.remove(memmap_path)

    G = np.memmap(
        memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, d, d),
    )

    start = 0

    with torch.no_grad():
        for tensors in loader:
            x = tensors["X"].to(
                device
            )

            batch = tensors["batch"].to(
                device
            )

            bs = x.shape[0]

            z_batch = torch.tensor(
                z[start:start + bs],
                dtype=torch.float32,
                device=device,
            )

            library = torch.log(
                x.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )

            G_batch = metric_tensor(
                module,
                z_batch,
                library,
                batch,
            )

            G[start:start + bs] = (
                G_batch
                .cpu()
                .numpy()
            )

            start += bs

            del (
                x,
                batch,
                z_batch,
                library,
                G_batch,
            )

            torch.cuda.empty_cache()

    return (
        G,
        z,
        neighbors,
        memmap_path,
    )