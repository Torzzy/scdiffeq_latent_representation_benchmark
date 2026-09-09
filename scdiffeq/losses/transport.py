import torch


def squared_euclidean_cost(
    x,
    y,
):
    """
    Cost matrix

        C_ij = ||x_i-y_j||²

    Parameters
    ----------
    x : (N,d)

    y : (M,d)

    Returns
    -------
    (N,M)
    """

    return torch.cdist(
        x,
        y,
    ).pow(2)


def euclidean_cost(
    x,
    y,
):
    """
    Euclidean transport cost.
    """

    return torch.cdist(
        x,
        y,
    )


def cosine_cost(
    x,
    y,
):
    """
    Cosine distance.

        1-cos(x,y)
    """

    x = torch.nn.functional.normalize(
        x,
        dim=1,
    )

    y = torch.nn.functional.normalize(
        y,
        dim=1,
    )

    return 1.0 - x @ y.T