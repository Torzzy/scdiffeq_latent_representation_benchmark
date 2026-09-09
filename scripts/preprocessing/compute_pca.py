import pickle
from pathlib import Path

import scanpy as sc

from scripts.preprocessing.global_preprocessing_steps import (
    preprocess_adata,
)


def compute_pca(
    input_path,
    output_path,
    model_dir,
    latent_dim=50,
):
    """Compute a PCA representation and save the processed data.

    Parameters
    ----------
    input_path
        Path to the input AnnData file.
    output_path
        Path where the processed AnnData object is saved.
    model_dir
        Directory where the PCA model information is saved.
    latent_dim
        Dimensionality of the PCA representation.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adata = sc.read_h5ad(input_path)

    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )

    sc.pp.log1p(adata)

    adata = preprocess_adata(adata)

    sc.pp.scale(
        adata,
        zero_center=True,
    )

    sc.pp.pca(
        adata,
        n_comps=latent_dim,
        zero_center=True,
    )

    adata.obsm[
        "X_latent"
    ] = (
        adata.obsm["X_pca"]
        .astype("float32")
    )

    adata.write(
        output_path
    )

    with open(
        model_dir / "pca.pkl",
        "wb",
    ) as f:
        pickle.dump(
            adata.uns["pca"],
            f,
        )

    print(
        f"PCA latent saved to {output_path}"
    )