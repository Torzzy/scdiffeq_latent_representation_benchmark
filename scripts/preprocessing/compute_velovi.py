from pathlib import Path

import scanpy as sc
import scvi
import scipy.sparse as sp

from scripts.preprocessing.global_preprocessing_steps import (
    preprocess_adata,
)


def compute_velovi(
    input_path,
    output_path,
    model_dir="data/models",
    latent_dim=50,
    epochs=400,
):
    """Compute a VELOVI latent representation.

    Parameters
    ----------
    input_path
        Path to the input AnnData file.
    output_path
        Path where the processed AnnData object is saved.
    model_dir
        Directory where the trained VELOVI model is saved.
    latent_dim
        Dimensionality of the latent representation.
    epochs
        Maximum number of training epochs.
    """
    adata = sc.read_h5ad(input_path)

    adata = preprocess_adata(
        adata,
    )

    if "spliced" not in adata.layers:
        raise ValueError(
            "adata.layers['spliced'] missing."
        )

    if "unspliced" not in adata.layers:
        raise ValueError(
            "adata.layers['unspliced'] missing."
        )

    if sp.issparse(
        adata.layers["spliced"]
    ):
        adata.layers["spliced"] = (
            adata.layers["spliced"]
            .toarray()
            .astype("float32")
        )

    if sp.issparse(
        adata.layers["unspliced"]
    ):
        adata.layers["unspliced"] = (
            adata.layers["unspliced"]
            .toarray()
            .astype("float32")
        )

    scvi.external.VELOVI.setup_anndata(
        adata,
        spliced_layer="spliced",
        unspliced_layer="unspliced",
    )

    model = scvi.external.VELOVI(
        adata,
        n_latent=latent_dim,
    )

    model.train(
        max_epochs=epochs,
        early_stopping=True,
    )

    adata.obsm[
        "X_latent"
    ] = model.get_latent_representation()

    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        model_dir,
        overwrite=True,
    )

    adata.write_h5ad(
        output_path,
    )

    print(
        f"Saved latent to {output_path}"
    )
    print(
        f"Saved model  to {model_dir}"
    )


if __name__ == "__main__":
    adata = sc.read_h5ad(
        "../../data/raw/"
        "larry_invitro_adata_sub_raw.h5ad"
    )

    adata = preprocess_adata(
        adata
    )

    print(
        adata.shape
    )

    print(
        type(
            adata.layers["spliced"]
        )
    )

    print(
        type(
            adata.layers["unspliced"]
        )
    )

    print(
        adata.layers["spliced"].shape
    )

    print(
        adata.layers["unspliced"].shape
    )