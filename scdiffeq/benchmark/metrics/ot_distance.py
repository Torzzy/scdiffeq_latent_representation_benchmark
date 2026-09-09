from geomloss import SamplesLoss

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=0.05,
)


def ot_distance(
    pred,
    target,
):
    """
    Sinkhorn distance between two point clouds.
    """
    return sinkhorn(
        pred,
        target,
    ).item()