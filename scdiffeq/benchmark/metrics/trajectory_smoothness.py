import torch


def trajectory_length(
    trajectory,
):
    """
    trajectory

        (T,B,D)
    """

    steps = trajectory[1:] - trajectory[:-1]

    length = torch.norm(
        steps,
        dim=2,
    )

    return length.mean().item()