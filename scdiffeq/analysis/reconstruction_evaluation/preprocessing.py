from __future__ import annotations

import numpy as np
import torch


def to_dense(
    x,
) -> np.ndarray:
    """
    Convert sparse or dense data to NumPy.
    """

    if hasattr(
        x,
        "toarray",
    ):
        x = x.toarray()

    return np.asarray(x)


def clip_negative(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Replace negative expression values with zero.
    """

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Expression matrix contains NaN or Inf."
        )

    return x.clamp_min(0.0)


def normalize_per_cell(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize each cell so that its expression
    profile sums to one.
    """

    x = clip_negative(x)

    library = x.sum(
        dim=1,
        keepdim=True,
    )

    if torch.any(library <= eps):
        raise ValueError(
            "At least one cell has zero expression "
            "after negative-value clipping."
        )

    x = x / library

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Normalized expression contains NaN or Inf."
        )

    return x


def prepare_expression(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Common post-processing used for every method:

        expression
        -> clip negative values
        -> L1 normalization
    """

    return normalize_per_cell(
        x,
        eps=eps,
    )