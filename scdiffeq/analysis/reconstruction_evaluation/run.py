from __future__ import annotations

import gc
from pathlib import Path

import scanpy as sc

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    LATENT_DIMS,
    METHODS,
    MODELS_DIR,
    RECONSTRUCTION_SEED,
    RESULTS_FILE,
)

from evaluate import (
    evaluate_and_save,
)

from models import (
    load_scvi,
    encode_scvi,
    decode_scvi,

    load_velovi,
    encode_velovi,
    decode_velovi,

    load_pca,
    encode_pca,
    decode_pca,

    load_flatvi,
    encode_flatvi,
    decode_flatvi,
)


def model_path(
    method: str,
    latent_dim: int,
) -> Path:

    return (
        MODELS_DIR
        / f"{method.lower()}_{latent_dim}"
    )


def dataset_path(
    method: str,
    latent_dim: int,
) -> Path:

    return (
        DATASET_DIR
        / (
            f"larry_"
            f"{method.lower()}_"
            f"{latent_dim}.h5ad"
        )
    )


# ============================================================
# SCVI
# ============================================================

def run_scvi(
    adata,
    latent_dim: int,
):

    model = load_scvi(
        model_path(
            "scvi",
            latent_dim,
        ),
        adata,
    )

    z = encode_scvi(
        model,
        adata,
        BATCH_SIZE,
    )

    if z.shape[1] != latent_dim:
        raise ValueError(
            f"SCVI latent dimension mismatch: "
            f"expected={latent_dim}, "
            f"got={z.shape[1]}"
        )

    reconstructed = decode_scvi(
        model,
        z,
        adata,
        BATCH_SIZE,
    )

    return evaluate_and_save(
        method="SCVI",
        latent_dim=latent_dim,
        seed=RECONSTRUCTION_SEED,
        adata=adata,
        reconstructed=reconstructed,
        output_file=RESULTS_FILE,
    )


# ============================================================
# VELOVI
# ============================================================

def run_velovi(
    adata,
    latent_dim: int,
):

    model = load_velovi(
        model_path(
            "velovi",
            latent_dim,
        ),
        adata,
    )

    z = encode_velovi(
        model,
        adata,
        BATCH_SIZE,
    )

    if z.shape[1] != latent_dim:
        raise ValueError(
            f"VELOVI latent dimension mismatch: "
            f"expected={latent_dim}, "
            f"got={z.shape[1]}"
        )

    reconstructed = decode_velovi(
        model,
        z,
    )

    return evaluate_and_save(
        method="VELOVI",
        latent_dim=latent_dim,
        seed=RECONSTRUCTION_SEED,
        adata=adata,
        reconstructed=reconstructed,
        output_file=RESULTS_FILE,
    )


# ============================================================
# PCA
# ============================================================

def run_pca(
    adata,
    latent_dim: int,
):

    model = load_pca(
        model_path(
            "pca",
            latent_dim,
        ),
        adata,
    )

    z = encode_pca(
        model,
        adata,
    )

    if z.shape[1] != latent_dim:
        raise ValueError(
            f"PCA latent dimension mismatch: "
            f"expected={latent_dim}, "
            f"got={z.shape[1]}"
        )

    reconstructed = decode_pca(
        model,
        z,
        adata,
    )

    return evaluate_and_save(
        method="PCA",
        latent_dim=latent_dim,
        seed=None,
        adata=adata,
        reconstructed=reconstructed,
        output_file=RESULTS_FILE,
    )


# ============================================================
# FlatVI
# ============================================================

def run_flatvi(
    adata,
    latent_dim: int,
):

    model = load_flatvi(
        model_path(
            "flatvi",
            latent_dim,
        ),
        adata,
    )

    z = encode_flatvi(
        model,
        adata,
    )

    if z.shape[1] != latent_dim:
        raise ValueError(
            f"FlatVI latent dimension mismatch: "
            f"expected={latent_dim}, "
            f"got={z.shape[1]}"
        )

    reconstructed = decode_flatvi(
        model,
        z,
    )

    return evaluate_and_save(
        method="FlatVI",
        latent_dim=latent_dim,
        seed=None,
        adata=adata,
        reconstructed=reconstructed,
        output_file=RESULTS_FILE,
    )


# ============================================================
# Main
# ============================================================

def run_reconstruction_evaluation_analysis():

    print("=" * 70)
    print("RECONSTRUCTION EVALUATION")
    print("=" * 70)

    print(
        f"Dataset directory: {DATASET_DIR}"
    )

    print(
        f"Output: {RESULTS_FILE}"
    )

    for method in METHODS:

        for latent_dim in LATENT_DIMS:

            print()
            print("-" * 70)

            print(
                f"{method} | "
                f"latent_dim={latent_dim}"
            )

            adata = None

            try:

                adata_path = dataset_path(
                    method,
                    latent_dim,
                )

                print(
                    f"Dataset: {adata_path}"
                )

                adata = sc.read_h5ad(
                    adata_path
                )

                print(
                    f"Dataset shape: "
                    f"{adata.shape}"
                )

                if method == "scvi":

                    result = run_scvi(
                        adata,
                        latent_dim,
                    )

                elif method == "velovi":

                    result = run_velovi(
                        adata,
                        latent_dim,
                    )

                elif method == "pca":

                    result = run_pca(
                        adata,
                        latent_dim,
                    )

                elif method == "flatvi":

                    result = run_flatvi(
                        adata,
                        latent_dim,
                    )

                else:

                    raise ValueError(
                        f"Unknown method: {method}"
                    )

                print(result)

            except Exception as error:

                print(
                    f"FAILED: {method} | "
                    f"latent_dim={latent_dim}"
                )

                print(
                    f"{type(error).__name__}: "
                    f"{error}"
                )

            finally:

                if adata is not None:
                    del adata

                gc.collect()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Results saved to: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    run_reconstruction_evaluation_analysis()