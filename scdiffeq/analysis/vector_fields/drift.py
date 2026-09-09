import torch


@torch.no_grad()
def compute_drift(
    model,
    Z,
    t,
):
    """
    Compute latent drift for every cell.
    """

    z = torch.from_numpy(
        Z,
    ).float().to(
        t.device,
    )

    drift = model.f(
        t,
        z,
    )

    return drift.cpu().numpy()