from pathlib import Path

import scanpy as sc
import scvi

from scripts.preprocessing.global_preprocessing_steps import (
    preprocess_adata,
)


def compute_scvi(
    input_path,
    output_path,
    model_dir="data/models",
    latent_dim=50,
    epochs=400,
):
    """Train an scVI model and save the latent representation.

    Parameters
    ----------
    input_path
        Path to the input AnnData file.
    output_path
        Path where the processed AnnData object is saved.
    model_dir
        Directory where the trained scVI model is saved.
    latent_dim
        Dimensionality of the latent representation.
    epochs
        Maximum number of training epochs.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adata = sc.read_h5ad(input_path)

    adata = preprocess_adata(adata)

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=latent_dim,
        gene_likelihood="nb",
    )

    model.train(
        max_epochs=epochs,
        early_stopping=True,
    )

    adata.obsm[
        "X_latent"
    ] = model.get_latent_representation()

    model.save(
        model_dir,
        overwrite=True,
    )

    adata.write(
        output_path,
    )

    print(
        f"scVI latent saved to {output_path}"
    )