from pathlib import Path

import scanpy as sc
import torch

from scdiffeq.data.dataloader import (
    build_dataloaders,
)
from scdiffeq.data.pair_dict import (
    build_pair_dict,
)
from scdiffeq.losses.sinkhorn import (
    sinkhorn,
)
from scdiffeq.models.sde import (
    LatentSDE,
)
from scdiffeq.simulations.euler_maruyama import (
    EulerMaruyamaSimulator,
)
from scdiffeq.trainer.train_model import (
    train_model,
)
from scdiffeq.utils.seed import (
    seed_everything,
)


def main():
    """Train an SDE model on a FlatVI latent representation."""
    seed_everything(0)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    latent_key = "X_latent"

    adata = sc.read_h5ad(
        "../data/processed/"
        "larry_flatvi_20.h5ad",
    )

    pair_dict = build_pair_dict(
        adata,
    )

    train_loader, val_loader = build_dataloaders(
        adata=adata,
        pair_dict=pair_dict,
        latent_key=latent_key,
        batch_size=64,
        val_fraction=0.2,
    )

    latent_dim = adata.obsm[
        latent_key
    ].shape[1]

    model = LatentSDE(
        latent_dim=latent_dim,
    ).to(device)

    simulator = EulerMaruyamaSimulator(
        sde=model,
        dt=0.05,
    )

    train_model(
        model=model,
        simulator=simulator,
        sinkhorn=sinkhorn,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(
            "../manifold_results"
        ),
        epochs=200,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()