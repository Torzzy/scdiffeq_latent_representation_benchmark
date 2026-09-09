import torch


def mean_future_distance(
    pred,
    target,
):
    """
    Distance between predicted population mean
    and ground-truth population mean.
    """

    pred_mean = pred.mean(0)

    target_mean = target.mean(0)

    return torch.norm(
        pred_mean - target_mean
    ).item()