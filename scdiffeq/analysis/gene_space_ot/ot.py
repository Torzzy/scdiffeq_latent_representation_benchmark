from __future__ import annotations

import math

import torch


GENE_OT_DISTANCES = (
    "l2",
    "cosine",
    "hellinger",
)


def _validate_point_clouds(
    predicted,
    target,
):
    """Validate two gene-space point clouds.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud with shape ``(N, G)``.
    target
        Target gene-space point cloud with shape ``(M, G)``.

    Raises
    ------
    ValueError
        If the inputs have incompatible or empty shapes, or contain
        negative values.
    FloatingPointError
        If either input contains NaN or Inf.
    """
    if predicted.ndim != 2:
        raise ValueError(
            f"predicted must have shape (N, G), "
            f"got {tuple(predicted.shape)}"
        )

    if target.ndim != 2:
        raise ValueError(
            f"target must have shape (M, G), "
            f"got {tuple(target.shape)}"
        )

    if predicted.shape[1] != target.shape[1]:
        raise ValueError(
            "Predicted and target gene dimensions differ: "
            f"{predicted.shape[1]} vs {target.shape[1]}"
        )

    if predicted.shape[0] == 0:
        raise ValueError(
            "predicted contains zero cells."
        )

    if target.shape[0] == 0:
        raise ValueError(
            "target contains zero cells."
        )

    if not torch.isfinite(
        predicted
    ).all():
        raise FloatingPointError(
            "Predicted gene expression contains "
            "NaN or Inf."
        )

    if not torch.isfinite(
        target
    ).all():
        raise FloatingPointError(
            "Target gene expression contains "
            "NaN or Inf."
        )

    if torch.any(
        predicted < 0
    ):
        raise ValueError(
            "Predicted gene expression contains "
            "negative values."
        )

    if torch.any(
        target < 0
    ):
        raise ValueError(
            "Target gene expression contains "
            "negative values."
        )


def l2_cost_matrix(
    predicted,
    target,
):
    """Compute the pairwise Euclidean cost matrix.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud with shape ``(N, G)``.
    target
        Target gene-space point cloud with shape ``(M, G)``.

    Returns
    -------
    torch.Tensor
        Pairwise Euclidean distances with shape ``(N, M)``.
    """
    _validate_point_clouds(
        predicted,
        target,
    )

    return torch.cdist(
        predicted,
        target,
        p=2,
    )


def cosine_cost_matrix(
    predicted,
    target,
    eps=1e-12,
):
    """Compute the pairwise cosine-distance cost matrix.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud with shape ``(N, G)``.
    target
        Target gene-space point cloud with shape ``(M, G)``.
    eps
        Numerical stability constant used when normalizing vectors.

    Returns
    -------
    torch.Tensor
        Pairwise cosine distances with shape ``(N, M)``.
    """
    _validate_point_clouds(
        predicted,
        target,
    )

    pred_norm = torch.linalg.vector_norm(
        predicted,
        ord=2,
        dim=1,
        keepdim=True,
    )

    target_norm = torch.linalg.vector_norm(
        target,
        ord=2,
        dim=1,
        keepdim=True,
    )

    predicted_unit = (
        predicted
        / (
            pred_norm
            + eps
        )
    )

    target_unit = (
        target
        / (
            target_norm
            + eps
        )
    )

    similarity = (
        predicted_unit
        @ target_unit.T
    )

    # Protect against small floating-point violations of [-1, 1].
    similarity = torch.clamp(
        similarity,
        min=-1.0,
        max=1.0,
    )

    cost = (
        1.0
        - similarity
    )

    return torch.clamp(
        cost,
        min=0.0,
    )


def hellinger_cost_matrix(
    predicted,
    target,
):
    """Compute the pairwise Hellinger-distance cost matrix.

    The Hellinger distance is appropriate for the normalized gene
    profiles because each profile represents a probability distribution.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud with shape ``(N, G)``.
    target
        Target gene-space point cloud with shape ``(M, G)``.

    Returns
    -------
    torch.Tensor
        Pairwise Hellinger distances with shape ``(N, M)``.
    """
    _validate_point_clouds(
        predicted,
        target,
    )

    sqrt_pred = torch.sqrt(
        torch.clamp(
            predicted,
            min=0.0,
        )
    )

    sqrt_target = torch.sqrt(
        torch.clamp(
            target,
            min=0.0,
        )
    )

    return (
        torch.cdist(
            sqrt_pred,
            sqrt_target,
            p=2,
        )
        / math.sqrt(2.0)
    )


def gene_cost_matrix(
    predicted,
    target,
    distance,
):
    """Compute a pairwise gene-space cost matrix.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    target
        Target gene-space point cloud.
    distance
        Ground cost. Supported values are ``"l2"``, ``"cosine"``,
        and ``"hellinger"``.

    Returns
    -------
    torch.Tensor
        Pairwise cost matrix.

    Raises
    ------
    ValueError
        If the requested distance is not supported.
    """
    distance = str(
        distance
    ).lower()

    if distance == "l2":
        return l2_cost_matrix(
            predicted,
            target,
        )

    if distance == "cosine":
        return cosine_cost_matrix(
            predicted,
            target,
        )

    if distance == "hellinger":
        return hellinger_cost_matrix(
            predicted,
            target,
        )

    raise ValueError(
        f"Unknown gene-space distance: {distance}. "
        f"Available: {GENE_OT_DISTANCES}"
    )


def _sinkhorn_transport_cost(
    cost_matrix,
    *,
    epsilon=0.05,
    max_iter=300,
    tol=1e-7,
):
    """Compute balanced entropic OT for two uniform distributions.

    The Sinkhorn iterations are performed in the log domain for
    numerical stability.

    Parameters
    ----------
    cost_matrix
        Pairwise cost matrix with shape ``(N, M)``.
    epsilon
        Entropic regularization strength.
    max_iter
        Maximum number of Sinkhorn iterations.
    tol
        Convergence threshold for the dual potentials.

    Returns
    -------
    torch.Tensor
        Scalar regularized transport cost.

    Raises
    ------
    ValueError
        If the cost matrix is invalid or ``epsilon`` is not positive.
    FloatingPointError
        If the cost matrix or resulting transport plan contains
        non-finite values.
    """
    if cost_matrix.ndim != 2:
        raise ValueError(
            "cost_matrix must have shape (N, M)."
        )

    if not torch.isfinite(
        cost_matrix
    ).all():
        raise FloatingPointError(
            "Cost matrix contains NaN or Inf."
        )

    if torch.any(
        cost_matrix < 0
    ):
        raise ValueError(
            "Cost matrix contains negative values."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be strictly positive."
        )

    n, m = cost_matrix.shape

    # Float64 improves numerical stability during Sinkhorn iterations.
    C = cost_matrix.to(
        dtype=torch.float64,
    )

    device = C.device
    dtype = C.dtype

    log_a = torch.full(
        (n,),
        -math.log(n),
        device=device,
        dtype=dtype,
    )

    log_b = torch.full(
        (m,),
        -math.log(m),
        device=device,
        dtype=dtype,
    )

    f = torch.zeros(
        n,
        device=device,
        dtype=dtype,
    )

    g = torch.zeros(
        m,
        device=device,
        dtype=dtype,
    )

    epsilon = float(
        epsilon
    )

    for iteration in range(
        max_iter
    ):
        f_old = f

        f = epsilon * (
            log_a
            - torch.logsumexp(
                (
                    g[None, :]
                    - C
                )
                / epsilon,
                dim=1,
            )
        )

        g = epsilon * (
            log_b
            - torch.logsumexp(
                (
                    f[:, None]
                    - C
                )
                / epsilon,
                dim=0,
            )
        )

        if iteration % 10 == 0:
            delta = (
                f
                - f_old
            ).abs().max()

            if delta.item() < tol:
                break

    log_plan = (
        f[:, None]
        + g[None, :]
        - C
    ) / epsilon

    plan = torch.exp(
        log_plan
    )

    if not torch.isfinite(
        plan
    ).all():
        raise FloatingPointError(
            "Sinkhorn transport plan contains NaN/Inf."
        )

    transport_cost = (
        plan
        * C
    ).sum()

    return transport_cost


def _sinkhorn_divergence(
    x,
    y,
    *,
    distance,
    epsilon=0.05,
    max_iter=300,
    tol=1e-7,
):
    """Compute the debiased Sinkhorn divergence.

    Parameters
    ----------
    x
        First gene-space point cloud.
    y
        Second gene-space point cloud.
    distance
        Ground cost used to construct the pairwise cost matrices.
    epsilon
        Entropic regularization strength.
    max_iter
        Maximum number of Sinkhorn iterations.
    tol
        Convergence threshold for the dual potentials.

    Returns
    -------
    torch.Tensor
        Scalar debiased Sinkhorn divergence.
    """
    cost_xy = gene_cost_matrix(
        x,
        y,
        distance,
    )

    cost_xx = gene_cost_matrix(
        x,
        x,
        distance,
    )

    cost_yy = gene_cost_matrix(
        y,
        y,
        distance,
    )

    ot_xy = _sinkhorn_transport_cost(
        cost_xy,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    ot_xx = _sinkhorn_transport_cost(
        cost_xx,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    ot_yy = _sinkhorn_transport_cost(
        cost_yy,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    divergence = (
        ot_xy
        - 0.5 * ot_xx
        - 0.5 * ot_yy
    )

    # Small negative values can arise from floating-point error.
    return torch.clamp(
        divergence,
        min=0.0,
    )


def gene_space_ot(
    predicted,
    target,
    *,
    distance="l2",
    epsilon=0.05,
    max_iter=300,
    tol=1e-7,
):
    """Compute the Sinkhorn divergence in normalized gene space.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    target
        Target gene-space point cloud.
    distance
        Ground cost. Supported values are ``"l2"``, ``"cosine"``,
        and ``"hellinger"``.
    epsilon
        Entropic regularization strength.
    max_iter
        Maximum number of Sinkhorn iterations.
    tol
        Convergence threshold for the dual potentials.

    Returns
    -------
    float
        Scalar Sinkhorn divergence.
    """
    value = _sinkhorn_divergence(
        predicted,
        target,
        distance=distance,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    return float(
        value.item()
    )


def gene_space_ot_l2(
    predicted,
    target,
    **kwargs,
):
    """Compute gene-space Sinkhorn divergence with L2 cost.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    target
        Target gene-space point cloud.
    **kwargs
        Additional arguments passed to ``gene_space_ot``.

    Returns
    -------
    float
        Scalar Sinkhorn divergence.
    """
    return gene_space_ot(
        predicted,
        target,
        distance="l2",
        **kwargs,
    )


def gene_space_ot_cosine(
    predicted,
    target,
    **kwargs,
):
    """Compute gene-space Sinkhorn divergence with cosine cost.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    target
        Target gene-space point cloud.
    **kwargs
        Additional arguments passed to ``gene_space_ot``.

    Returns
    -------
    float
        Scalar Sinkhorn divergence.
    """
    return gene_space_ot(
        predicted,
        target,
        distance="cosine",
        **kwargs,
    )


def gene_space_ot_hellinger(
    predicted,
    target,
    **kwargs,
):
    """Compute gene-space Sinkhorn divergence with Hellinger cost.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    target
        Target gene-space point cloud.
    **kwargs
        Additional arguments passed to ``gene_space_ot``.

    Returns
    -------
    float
        Scalar Sinkhorn divergence.
    """
    return gene_space_ot(
        predicted,
        target,
        distance="hellinger",
        **kwargs,
    )


def normalized_gene_space_ot(
    predicted,
    initial,
    target,
    *,
    distance="l2",
    epsilon=0.05,
):
    """Compute relative gene-space OT against an initial-state baseline.

    Parameters
    ----------
    predicted
        Predicted gene-space point cloud.
    initial
        Initial gene-space point cloud used as the baseline.
    target
        Target gene-space point cloud.
    distance
        Ground cost. Supported values are ``"l2"``, ``"cosine"``,
        and ``"hellinger"``.
    epsilon
        Entropic regularization strength.

    Returns
    -------
    float
        Ratio between predicted-to-target OT and initial-to-target OT.
        Lower values indicate better agreement with the target.
    """
    model_ot = gene_space_ot(
        predicted,
        target,
        distance=distance,
        epsilon=epsilon,
    )

    baseline_ot = gene_space_ot(
        initial,
        target,
        distance=distance,
        epsilon=epsilon,
    )

    return (
        model_ot
        / (
            baseline_ot
            + 1e-8
        )
    )