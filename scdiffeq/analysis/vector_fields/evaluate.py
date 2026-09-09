import scanpy as sc
import torch

from .model import (
    load_vector_field_model,
)

from .drift import (
    compute_drift,
)

from .projection import (
    project_drift_to_umap,
)

from .field import (
    compute_grid_field,
)

from .plotting import (
    plot_vector_field,
)


@torch.no_grad()
def evaluate_vector_field(
    dataset_path,
    checkpoint_path,
    output_path,
    latent_dim,
    hidden_dim,
    method,
    device="cuda",
):

    adata = sc.read_h5ad(
        dataset_path,
    )

    Z = adata.obsm[
        "X_latent"
    ].astype("float32")

    t = torch.tensor(
        adata.obs[
            "time_info"
        ].astype(float).values,
        dtype=torch.float32,
        device=device,
    )

    model = load_vector_field_model(
        checkpoint_path,
        latent_dim,
        hidden_dim,
        device=device,
    )

    drift = compute_drift(
        model,
        Z,
        t,
    )

    X, drift2d = project_drift_to_umap(
        Z,
        drift,
    )

    xx, yy, U, V, speed = compute_grid_field(
        X,
        drift2d,
    )

    labels = (
        adata.obs["state_info"]
        if "state_info" in adata.obs
        else None
    )

    plot_vector_field(
        X=X,
        drift2d=drift2d,
        xx=xx,
        yy=yy,
        U=U,
        V=V,
        speed=speed,
        state_labels=labels,
        output_path=output_path,
        method=method,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )

    model.cpu()

    del model

    torch.cuda.empty_cache()