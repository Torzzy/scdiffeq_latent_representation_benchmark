from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from config import (
    EPS,
    LAYER,
)

from metrics import (
    compute_distances,
)

from preprocessing import (
    prepare_expression,
    to_dense,
)


def load_observed_expression(
    adata,
) -> torch.Tensor:
    """
    Load the observed expression matrix
    from the configured AnnData layer.
    """

    if LAYER not in adata.layers:
        raise ValueError(
            f"Layer '{LAYER}' is missing from AnnData."
        )

    observed = to_dense(
        adata.layers[LAYER]
    )

    observed = torch.as_tensor(
        observed,
        dtype=torch.float32,
    )

    if observed.ndim != 2:
        raise ValueError(
            "Observed expression must be 2D, "
            f"got shape={observed.shape}."
        )

    if not torch.isfinite(observed).all():
        raise FloatingPointError(
            "Observed expression contains NaN or Inf."
        )

    return observed


def _check_normalized(
    x: torch.Tensor,
    name: str,
    atol: float = 1e-5,
) -> None:
    """
    Verify that every cell sums to one.
    """

    library = x.sum(
        dim=1
    )

    if not torch.allclose(
        library,
        torch.ones_like(library),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} is not normalized per cell."
        )


def evaluate_reconstruction(
    adata,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """
    Compare observed and reconstructed cells.

    Both matrices are subjected to exactly the same
    post-processing before distance calculation.
    """

    observed = load_observed_expression(
        adata
    )

    reconstructed = (
        reconstructed
        .detach()
        .cpu()
        .float()
    )

    if reconstructed.shape != observed.shape:
        raise ValueError(
            "Shape mismatch: "
            f"observed={observed.shape}, "
            f"reconstructed={reconstructed.shape}"
        )

    # --------------------------------------------------------
    # Common benchmark preprocessing
    # --------------------------------------------------------

    observed = prepare_expression(
        observed,
        EPS,
    )

    reconstructed = prepare_expression(
        reconstructed,
        EPS,
    )

    _check_normalized(
        observed,
        "Observed expression",
    )

    _check_normalized(
        reconstructed,
        "Reconstructed expression",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return compute_distances(
        observed,
        reconstructed,
    )


def evaluate_and_save(
    method: str,
    latent_dim: int,
    seed: int | None,
    adata,
    reconstructed: torch.Tensor,
    output_file: Path | None = None,
) -> dict[str, float]:
    """
    Evaluate one encoder-decoder reconstruction
    and optionally append the result to CSV.
    """

    metrics = evaluate_reconstruction(
        adata=adata,
        reconstructed=reconstructed,
    )

    result = {
        "method": method,
        "latent_dim": latent_dim,
        "seed": seed,
        **metrics,
    }

    if output_file is not None:

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = pd.DataFrame(
            [result]
        )

        if output_file.exists():
            dataframe.to_csv(
                output_file,
                mode="a",
                header=False,
                index=False,
            )
        else:
            dataframe.to_csv(
                output_file,
                index=False,
            )

    return result