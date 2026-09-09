import gc

import pandas as pd
import scanpy as sc
import torch

from scdiffeq.analysis.utils import (
    load_vae_model,
)

from .evaluate import (
    evaluate_metric_tensor,
)


def run_metric_tensor_analysis(
    paths,
    raw_df,
):
    """Compute metric tensor analysis for benchmark models.

    Only the reference configuration is evaluated: seed ``0`` and
    ``hidden_dim=256``. PCA and VeloVI are skipped.

    Parameters
    ----------
    paths
        Analysis paths containing datasets, processed data, and output
        directories.
    raw_df
        Benchmark dataframe containing model configurations.

    Returns
    -------
    pandas.DataFrame
        Metric tensor summary for the evaluated models.
    """
    output_dir = (
        paths.analysis
        / "metric_tensor"
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

    rows = []

    for _, cfg in configs.iterrows():
        method = cfg.method
        latent_dim = int(cfg.latent_dim)
        hidden_dim = int(cfg.hidden_dim)
        seed = int(cfg.seed)

        if seed != 0:
            continue

        if hidden_dim != 256:
            continue

        if method == "pca" or method == "velovi":
            continue

        dataset_path = paths.dataset(
            method,
            latent_dim,
        )

        if not dataset_path.exists():
            continue

        print(
            f"[Metric tensor] "
            f"{method} "
            f"latent={latent_dim}"
        )

        adata = sc.read_h5ad(
            dataset_path,
        )

        model = load_vae_model(
            adata=adata,
            latent_dim=latent_dim,
            method=method,
            processed_dir=paths.processed,
        )

        model.module.eval()
        model.module.to(
            "cuda",
        )

        stats = evaluate_metric_tensor(
            adata=adata,
            model=model,
            output_dir=(
                output_dir
                / method
                / f"latent_{latent_dim}"
            ),
            n_cells=10_000,
        )

        rows.append(
            {
                "method": method,
                "latent_dim": latent_dim,
                **stats,
            }
        )

        pd.DataFrame(
            rows,
        ).to_csv(
            output_dir / "summary.csv",
            index=False,
        )

        model.module.cpu()

        del stats
        del model
        del adata

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    summary = pd.DataFrame(
        rows,
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    return summary