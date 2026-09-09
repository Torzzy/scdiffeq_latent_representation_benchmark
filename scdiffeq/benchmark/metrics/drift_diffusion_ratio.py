import torch


@torch.no_grad()
def drift_diffusion_ratio(
    model,
    trajectory,
    times,
    eps=1e-8,
):
    """Compute the relative contribution of drift and diffusion.

    Parameters
    ----------
    model
        SDE model providing drift and diffusion functions.
    trajectory
        Latent trajectories with shape ``(T, B, S, D)``.
    times
        Time points corresponding to the trajectory steps.
    eps
        Numerical stability constant used in the ratio computation.

    Returns
    -------
    dict
        Summary statistics for the drift-to-total and diffusion
        magnitudes.
    """
    T, B, S, D = trajectory.shape

    drift_norms = []
    diffusion_norms = []

    for k in range(T):
        z = trajectory[k].reshape(
            B * S,
            D,
        )

        drift = model.f(
            times[k],
            z,
        )

        diffusion = model.g(
            float(k),
            z,
        )

        drift_norms.append(
            torch.norm(
                drift,
                dim=1,
            )
        )

        diffusion_norms.append(
            torch.norm(
                diffusion,
                dim=1,
            )
        )

    drift_norms = torch.stack(
        drift_norms,
    )

    diffusion_norms = torch.stack(
        diffusion_norms,
    )

    ratio = drift_norms / (
        drift_norms
        + diffusion_norms
        + eps
    )

    return {
        "ratio_mean":
            ratio.mean().item(),
        "ratio_std":
            ratio.std().item(),
        "ratio_max":
            ratio.max().item(),
        "drift_mean":
            drift_norms.mean().item(),
        "drift_max":
            drift_norms.max().item(),
        "diffusion_mean":
            diffusion_norms.mean().item(),
        "diffusion_max":
            diffusion_norms.max().item(),
    }