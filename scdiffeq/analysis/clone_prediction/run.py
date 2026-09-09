import scanpy as sc

from scdiffeq.data.dataloader import (
    build_dataloaders,
)
from scdiffeq.data.pair_dict import (
    build_pair_dict,
)

from .evaluate import (
    evaluate_clone_prediction,
)


def run_clone_prediction_analysis(
    paths,
    raw_df,
    n_examples=5,
):
    """Generate qualitative clone prediction figures.

    Parameters
    ----------
    paths
        Analysis paths containing datasets, checkpoints, and output
        directories.
    raw_df
        Benchmark dataframe containing model configurations.
    n_examples
        Number of clones to evaluate for each selected configuration.
    """
    output_dir = (
        paths.analysis
        / "clone_prediction"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs = (
        raw_df[
            [
                "method",
                "latent_dim",
                "hidden_dim",
                "seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "method",
                "latent_dim",
                "hidden_dim",
                "seed",
            ]
        )
    )

    for _, cfg in configs.iterrows():
        method = cfg.method
        latent_dim = int(cfg.latent_dim)
        hidden_dim = int(cfg.hidden_dim)
        seed = int(cfg.seed)

        # Restrict qualitative analysis to the reference configuration.
        if seed != 0:
            continue

        if hidden_dim != 256:
            continue

        dataset_path = paths.dataset(
            method,
            latent_dim,
        )

        if not dataset_path.exists():
            continue

        print(
            f"[Clone prediction] "
            f"{method} "
            f"latent={latent_dim}"
        )

        adata = sc.read_h5ad(
            dataset_path,
        )

        pair_dict = build_pair_dict(
            adata,
        )

        train_loader, _ = build_dataloaders(
            adata,
            pair_dict,
            latent_key="X_latent",
            seed=seed,
        )

        checkpoint = paths.checkpoint(
            method,
            latent_dim,
            hidden_dim,
            seed,
        )

        if not checkpoint.exists():
            continue

        model_dir = (
            output_dir
            / method
            / f"latent_{latent_dim}"
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for clone_id in range(
            n_examples,
        ):
            evaluate_clone_prediction(
                adata=adata,
                loader=train_loader,
                checkpoint_path=checkpoint,
                output_dir=model_dir,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                clone_id=clone_id,
            )