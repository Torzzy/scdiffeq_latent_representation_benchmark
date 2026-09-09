from pathlib import Path

from scripts.preprocessing.compute_flatvi import (
    compute_flatvi,
)
from scripts.preprocessing.compute_pca import (
    compute_pca,
)
from scripts.preprocessing.compute_scvi import (
    compute_scvi,
)
from scripts.preprocessing.compute_velovi import (
    compute_velovi,
)


RAW_PATH = Path(
    "../../data/raw/"
    "larry_invitro_adata_sub_raw.h5ad"
)
PROCESSED_DIR = Path(
    "../../data/processed"
)

LATENT_DIMS = [
    10,
    20,
    50,
    100,
    200,
]


def main():
    """Compute latent representations for all methods and dimensions."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods = [
        (
            "flatvi",
            compute_flatvi,
        ),
        (
            "pca",
            compute_pca,
        ),
        (
            "scvi",
            compute_scvi,
        ),
        (
            "velovi",
            compute_velovi,
        ),
    ]

    for method_name, compute_fn in methods:
        print(
            "=" * 80
        )
        print(
            method_name.upper()
        )
        print(
            "=" * 80
        )

        for latent_dim in LATENT_DIMS:
            print(
                f"latent_dim = {latent_dim}"
            )

            output_path = (
                PROCESSED_DIR
                / f"larry_{method_name}_{latent_dim}.h5ad"
            )

            if output_path.exists():
                print(
                    "Already computed."
                )
                continue

            model_dir = (
                PROCESSED_DIR
                / "models"
                / f"{method_name}_{latent_dim}"
            )

            compute_fn(
                input_path=RAW_PATH,
                output_path=output_path,
                model_dir=model_dir,
                latent_dim=latent_dim,
            )


if __name__ == "__main__":
    main()