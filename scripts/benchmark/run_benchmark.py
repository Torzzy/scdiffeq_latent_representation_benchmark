import gc
from pathlib import Path

import pandas as pd
import scanpy as sc
import torch

from scdiffeq.benchmark.evaluate import (
    evaluate_model,
)
from scdiffeq.data.dataloader import (
    build_dataloaders,
)
from scdiffeq.data.pair_dict import (
    build_pair_dict,
)
from scdiffeq.losses.sinkhorn import (
    sinkhorn,
)
from scdiffeq.models.sde import LatentSDE
from scdiffeq.simulations.euler_maruyama import (
    EulerMaruyamaSimulator,
)
from scdiffeq.trainer.train_model import (
    train_model,
)
from scdiffeq.utils.seed import (
    seed_everything,
)


LATENT_METHODS = [
    "flatvi",
    "pca",
    "scvi",
    "velovi",
]

LATENT_DIMS = [
    10,
    20,
    50,
    100,
    200,
]

HIDDEN_DIMS = [
    64,
    128,
    256,
]

SEEDS = [
    0,
    1,
    2,
]


def already_done(
    df,
    method,
    latent_dim,
    hidden_dim,
    seed,
):
    """Check whether a benchmark configuration has already been evaluated.

    Parameters
    ----------
    df
        Benchmark dataframe containing completed configurations.
    method
        Latent representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the SDE model.
    seed
        Random seed.

    Returns
    -------
    bool
        Whether the configuration is already present in the dataframe.
    """
    if len(df) == 0:
        return False

    mask = (
        (df["method"] == method)
        & (df["latent_dim"] == latent_dim)
        & (df["hidden_dim"] == hidden_dim)
        & (df["seed"] == seed)
    )

    return mask.any()


def main():
    """Run the latent SDE benchmark across model configurations."""
    processed_dir = Path(
        "../../data/processed"
    )

    results_dir = Path(
        "../../results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        results_dir
        / "benchmark.csv"
    )

    if csv_path.exists():
        benchmark = pd.read_csv(
            csv_path
        )
    else:
        benchmark = pd.DataFrame()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for method in LATENT_METHODS:
        for latent_dim in LATENT_DIMS:
            h5ad_path = (
                processed_dir
                / f"larry_{method}_{latent_dim}.h5ad"
            )

            if not h5ad_path.exists():
                continue

            adata = sc.read_h5ad(
                h5ad_path
            )

            pair_dict = build_pair_dict(
                adata
            )

            train_loader, val_loader = (
                build_dataloaders(
                    adata,
                    pair_dict,
                    latent_key="X_latent",
                    num_workers=8,
                )
            )

            for hidden_dim in HIDDEN_DIMS:
                for seed in SEEDS:
                    if already_done(
                        benchmark,
                        method,
                        latent_dim,
                        hidden_dim,
                        seed,
                    ):
                        print(
                            f"[SKIP] {method} "
                            f"latent={latent_dim} "
                            f"hidden={hidden_dim} "
                            f"seed={seed}"
                        )

                        continue

                    print(
                        f"[RUN ] {method} "
                        f"latent={latent_dim} "
                        f"hidden={hidden_dim} "
                        f"seed={seed}"
                    )

                    seed_everything(
                        seed
                    )

                    model = LatentSDE(
                        latent_dim=latent_dim,
                        hidden=hidden_dim,
                    ).to(device)

                    simulator = (
                        EulerMaruyamaSimulator(
                            model,
                            dt=0.05,
                        )
                    )

                    model_dir = (
                        results_dir
                        / "models"
                        / method
                        / f"latent_{latent_dim}"
                        / f"hidden_{hidden_dim}"
                        / f"seed_{seed}"
                    )

                    model_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    try:
                        model, history = (
                            train_model(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                sinkhorn=sinkhorn,
                                output_dir=model_dir,
                                simulator=simulator,
                            )
                        )
                    except Exception:
                        continue

                    metrics = evaluate_model(
                        model=model,
                        loader=val_loader,
                    )

                    row = pd.DataFrame(
                        [
                            {
                                "method": method,
                                "latent_dim": latent_dim,
                                "hidden_dim": hidden_dim,
                                "seed": seed,
                                **metrics,
                            }
                        ]
                    )

                    benchmark = pd.concat(
                        [
                            benchmark,
                            row,
                        ],
                        ignore_index=True,
                    )

                    benchmark.to_csv(
                        csv_path,
                        index=False,
                    )

                    del model, history

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()