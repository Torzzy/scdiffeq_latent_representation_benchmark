import torch


def covariance_trace(
    population,
):

    cov = torch.cov(
        population.T
    )

    return torch.trace(
        cov
    ).item()