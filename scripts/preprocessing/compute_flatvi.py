from pathlib import Path

import scanpy as sc

from flat_vi_like.flat_metrics_callback import (
    FlatMetricsCallback,
)
from flat_vi_like.flat_model import (
    FlatModel,
)
from scripts.preprocessing.global_preprocessing_steps import (
    preprocess_adata,
)


def compute_flatvi(
    input_path,
    output_path,
    model_dir,
    latent_dim=50,
    epochs=400,
    lambda_metric=0.1,
):
    """Train a FlatVI model and save the latent representation.

    Parameters
    ----------
    input_path
        Path to the input AnnData file.
    output_path
        Path where the processed AnnData object is saved.
    model_dir
        Directory where the trained FlatVI model is saved.
    latent_dim
        Dimensionality of the latent representation.
    epochs
        Maximum number of training epochs.
    lambda_metric
        Weight of the metric regularization term.
    """
    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    model_dir = Path(
        model_dir
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adata = sc.read_h5ad(
        input_path
    )

    adata = preprocess_adata(
        adata
    )

    FlatModel.setup_anndata(
        adata,
        layer=None,
    )

    model = FlatModel(
        adata,
        n_latent=latent_dim,
        lambda_metric=lambda_metric,
    )

    model.train(
        max_epochs=epochs,
        early_stopping=True,
        callbacks=[
            FlatMetricsCallback()
        ],
    )

    adata.obsm[
        "X_latent"
    ] = (
        model.get_latent_representation()
        .astype(
            "float32"
        )
    )

    model.save(
        model_dir,
        overwrite=True,
    )

    adata.write(
        output_path
    )

    print(
        f"FlatVI latent saved to {output_path}"
    )


if __name__ == "__main__":
    compute_flatvi(
        input_path=(
            "../../data/raw/"
            "larry_invitro_adata_sub_raw.h5ad"
        ),
        output_path=(
            "../../tmp/"
            "adata_flatvi_200.h5ad"
        ),
        model_dir=(
            "../../tmp/model_test"
        ),
        latent_dim=100,
        epochs=400,
        lambda_metric=1.0,
    )