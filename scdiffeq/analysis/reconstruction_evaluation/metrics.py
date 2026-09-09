from __future__ import annotations

import torch


def l2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Per-cell Euclidean distance.
    """

    return torch.linalg.vector_norm(
        x - y,
        dim=1,
    )


def cosine_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-cell cosine distance:

        1 - cosine similarity
    """

    x_norm = (
        torch.linalg.vector_norm(
            x,
            dim=1,
            keepdim=True,
        )
        .clamp_min(eps)
    )

    y_norm = (
        torch.linalg.vector_norm(
            y,
            dim=1,
            keepdim=True,
        )
        .clamp_min(eps)
    )

    similarity = (
        (x * y).sum(
            dim=1,
            keepdim=True,
        )
        / (x_norm * y_norm)
    )

    return (
        1.0
        - similarity.squeeze(1)
    )


def hellinger_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Hellinger-style squared distance:

        sum_i (sqrt(x_i) - sqrt(y_i))^2

    Inputs must be non-negative.
    """

    if torch.any(x < 0) or torch.any(y < 0):
        raise ValueError(
            "Hellinger distance requires "
            "non-negative inputs."
        )

    return (
        (
            torch.sqrt(x)
            - torch.sqrt(y)
        )
        .pow(2)
        .sum(dim=1)
    )


def compute_distances(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """
    Compute per-cell reconstruction distances.

    Inputs must already be:
        - finite
        - non-negative
        - normalized per cell
    """

    if observed.shape != reconstructed.shape:
        raise ValueError(
            "Shape mismatch: "
            f"observed={observed.shape}, "
            f"reconstructed={reconstructed.shape}"
        )

    if torch.any(observed < 0):
        raise ValueError(
            "Observed expression contains negative values."
        )

    if torch.any(reconstructed < 0):
        raise ValueError(
            "Reconstructed expression contains negative values."
        )

    l2 = l2_distance(
        observed,
        reconstructed,
    )

    cosine = cosine_distance(
        observed,
        reconstructed,
    )

    hellinger = hellinger_distance(
        observed,
        reconstructed,
    )

    return {
        "l2_mean": l2.mean().item(),
        "l2_std": l2.std(unbiased=False).item(),

        "cosine_mean": cosine.mean().item(),
        "cosine_std": cosine.std(unbiased=False).item(),

        "hellinger_mean": hellinger.mean().item(),
        "hellinger_std": hellinger.std(unbiased=False).item(),
    }